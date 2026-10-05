# SPDX-License-Identifier: Apache-2.0
"""MiniCPM-o duplex token sampling over the thinker states of one batch."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F

from sglang_omni.models.minicpm_o.special_tokens import MiniCPMOSpecialTokenIds
from sglang_omni.models.minicpm_o.thinker_state import MiniCPMOThinkerSessionState

# note: one row of a unit batch: the session's thinker state, its generation step, listen forcing.
UnitRow = tuple[MiniCPMOThinkerSessionState, int, bool]


def build_forbidden_token_index(
    special_tokens: MiniCPMOSpecialTokenIds, vocab_size: int, device: torch.device
) -> torch.Tensor:
    """Rows the second-stage sample never picks, resolved once instead of per step."""
    forbidden_token_ids = sorted(
        token_id
        for token_id in {special_tokens.chunk_eos, *special_tokens.forbidden}
        if token_id < vocab_size
    )
    return torch.tensor(forbidden_token_ids, dtype=torch.long, device=device)


def filter_top_k_top_p(
    logits: torch.Tensor, *, top_k: Sequence[int], top_p: Sequence[float]
) -> torch.Tensor:
    """Apply each row's top-k and top-p to a ``[rows, vocab]`` tensor; values outside (0, vocab) or (0, 1) leave the row alone."""
    vocab_size = logits.shape[-1]
    filtered_logits = logits.clone()
    active_k = [k for k in top_k if 0 < k < vocab_size]
    if active_k:
        kept = torch.tensor(
            [k if 0 < k < vocab_size else 1 for k in top_k], device=logits.device
        )
        thresholds = torch.topk(filtered_logits, max(active_k), dim=-1).values.gather(
            1, (kept - 1).unsqueeze(1)
        )
        enabled = torch.tensor(
            [0 < k < vocab_size for k in top_k], device=logits.device
        ).unsqueeze(1)
        filtered_logits.masked_fill_(
            enabled & (filtered_logits < thresholds), -torch.inf
        )
    else:
        pass
    if any(0.0 < p < 1.0 for p in top_p):
        sorted_logits, sorted_indices = torch.sort(
            filtered_logits, descending=True, dim=-1
        )
        cumulative_probabilities = torch.cumsum(
            F.softmax(sorted_logits, dim=-1), dim=-1
        )
        limits = torch.tensor(
            [p if 0.0 < p < 1.0 else torch.inf for p in top_p], device=logits.device
        )
        should_remove = cumulative_probabilities > limits.unsqueeze(1)
        should_remove[:, 1:] = should_remove[:, :-1].clone()
        should_remove[:, 0] = False
        filtered_logits.scatter_(
            1, sorted_indices, sorted_logits.masked_fill(should_remove, -torch.inf)
        )
    else:
        pass
    return filtered_logits


def duplex_sample(
    logits: torch.Tensor,
    units: Sequence[UnitRow],
    *,
    special_tokens: MiniCPMOSpecialTokenIds,
    forbidden_token_index: torch.Tensor,
) -> list[int]:
    """Apply the two-stage unit sampler to every row of a batch with two device reads."""
    token_ids: list[int] = [-1] * len(units)
    active: list[int] = []
    for index, (state, generation_step, is_listen_forced) in enumerate(units):
        if generation_step >= state.sampling.max_new_tokens_per_unit - 1:
            token_ids[index] = special_tokens.chunk_eos
        elif generation_step == 0 and is_listen_forced:
            token_ids[index] = special_tokens.listen
        else:
            active.append(index)
    if not active:
        return token_ids
    else:
        pass
    rows = logits.index_select(0, torch.tensor(active, device=logits.device)).float()
    # note (Junnan Li): The first sample must use the unscaled model distribution.
    first_picks = torch.stack(
        (rows.argmax(dim=-1), torch.multinomial(F.softmax(rows, dim=-1), 1)[:, 0])
    ).tolist()
    second: list[int] = []
    for position, index in enumerate(active):
        greedy = units[index][0].sampling.greedy
        if first_picks[0 if greedy else 1][position] == special_tokens.chunk_eos:
            token_ids[index] = special_tokens.chunk_eos
        else:
            second.append(position)
    if not second:
        return token_ids
    else:
        pass
    rows = rows[second]
    states = [units[active[position]][0] for position in second]
    rows[:, forbidden_token_index] = -torch.inf
    # note (Junnan Li): Matches the checkpoint sampler, which ignores the logit sign.
    repeated = [
        (row, token_id)
        for row, state in enumerate(states)
        if state.sampling.repetition_penalty != 1.0
        for token_id in set(
            state.generated_history[-state.sampling.repetition_window_size :]
        )
    ]
    if repeated:
        repeated_index = torch.tensor(repeated, device=rows.device)
        penalties = torch.tensor(
            [states[row].sampling.repetition_penalty for row, _ in repeated],
            device=rows.device,
        )
        rows[repeated_index[:, 0], repeated_index[:, 1]] /= penalties
    else:
        pass
    settings = torch.tensor(
        [
            (
                state.sampling.listen_prob_scale,
                state.sampling.temperature if state.sampling.temperature > 0 else 1.0,
            )
            for state in states
        ],
        device=rows.device,
    )
    rows[:, special_tokens.listen] *= settings[:, 0]
    filtered_logits = filter_top_k_top_p(
        rows / settings[:, 1].unsqueeze(1),
        top_k=[state.sampling.top_k for state in states],
        top_p=[state.sampling.top_p for state in states],
    )
    second_picks = torch.stack(
        (
            rows.argmax(dim=-1),
            torch.multinomial(F.softmax(filtered_logits, dim=-1), 1)[:, 0],
        )
    ).tolist()
    for row, (position, state) in enumerate(zip(second, states, strict=True)):
        sampling = state.sampling
        greedy = sampling.greedy or sampling.temperature <= 0
        candidate_token_id = second_picks[0 if greedy else 1][row]
        # note (Junnan Li): History retains controls before the mid-turn listen rewrite.
        history = state.generated_history
        history.append(candidate_token_id)
        del history[: -sampling.repetition_window_size]
        if candidate_token_id == special_tokens.listen and not state.is_turn_ended:
            candidate_token_id = special_tokens.tts_bos
        else:
            pass
        if candidate_token_id == special_tokens.turn_eos:
            state.is_turn_ended = True
        elif candidate_token_id not in special_tokens.chunk_terminators:
            state.is_turn_ended = False
        else:
            pass
        token_ids[active[position]] = candidate_token_id
    return token_ids
