# SPDX-License-Identifier: Apache-2.0
"""MiniCPM-o duplex token sampling over the thinker states of one batch."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F

from sglang_omni.models.minicpm_o.special_tokens import MiniCPMOSpecialTokenIds
from sglang_omni.models.minicpm_o.thinker_state import MiniCPMOThinkerSessionState

# note (0xtoward): a batch row is its session's thinker state, generation step and listen forcing.
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


def to_device(values: list, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """Copy host values to the device without waiting for the work already queued there."""
    host_values = torch.tensor(values, dtype=dtype)
    if device.type == "cuda":
        return host_values.pin_memory().to(device, non_blocking=True)
    else:
        return host_values.to(device)


def filter_top_k_top_p(
    logits: torch.Tensor,
    *,
    top_k: torch.Tensor,
    top_p: torch.Tensor,
    max_top_k: int,
    has_top_p: bool,
) -> torch.Tensor:
    """Apply each row's top-k and top-p; values outside (0, vocab) or (0, 1) leave the row alone.

    max_top_k and has_top_p are host-side summaries of top_k and top_p, so disabled filters launch nothing.
    """
    vocab_size = logits.shape[-1]
    filtered_logits = logits.clone()
    if max_top_k > 0:
        enabled = (top_k > 0) & (top_k < vocab_size)
        kept = torch.where(enabled, top_k, 1)
        thresholds = torch.topk(filtered_logits, max_top_k, dim=-1).values.gather(
            1, (kept - 1).unsqueeze(1)
        )
        filtered_logits.masked_fill_(
            enabled.unsqueeze(1) & (filtered_logits < thresholds), -torch.inf
        )
    else:
        pass
    if has_top_p:
        sorted_logits, sorted_indices = torch.sort(
            filtered_logits, descending=True, dim=-1
        )
        cumulative_probabilities = torch.cumsum(
            F.softmax(sorted_logits, dim=-1), dim=-1
        )
        limits = torch.where((top_p > 0.0) & (top_p < 1.0), top_p, torch.inf)
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
    """Apply the two-stage unit sampler to every row of a batch with one device read."""
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
    states = [units[index][0] for index in active]
    vocab_size = logits.shape[-1]
    count = len(active)
    penalized = [
        (row, token_id, state.sampling.repetition_penalty)
        for row, state in enumerate(states)
        if state.sampling.repetition_penalty != 1.0
        for token_id in set(
            state.generated_history[-state.sampling.repetition_window_size :]
        )
    ]
    top_k = [state.sampling.top_k for state in states]
    top_p = [state.sampling.top_p for state in states]
    # note (0xtoward): one copy per dtype carries every row's settings, so the device never waits on a host read.
    integers = to_device(
        active
        + top_k
        + [row for row, _, _ in penalized]
        + [token_id for _, token_id, _ in penalized],
        torch.long,
        logits.device,
    )
    floats = to_device(
        [float(state.sampling.greedy) for state in states]
        + [
            float(state.sampling.greedy or state.sampling.temperature <= 0)
            for state in states
        ]
        + [
            state.sampling.temperature if state.sampling.temperature > 0 else 1.0
            for state in states
        ]
        + [state.sampling.listen_prob_scale for state in states]
        + top_p
        + [penalty for _, _, penalty in penalized],
        torch.float32,
        logits.device,
    )
    rows = logits.index_select(0, integers[:count]).float()
    # note (Junnan Li): The first sample must use the unscaled model distribution.
    first_picks = torch.where(
        floats[:count] > 0,
        rows.argmax(dim=-1),
        torch.multinomial(F.softmax(rows, dim=-1), 1)[:, 0],
    )
    rows[:, forbidden_token_index] = -torch.inf
    if penalized:
        pairs = len(penalized)
        penalty_rows = integers[2 * count : 2 * count + pairs]
        penalty_tokens = integers[2 * count + pairs :]
        # note (Junnan Li): Matches the checkpoint sampler, which ignores the logit sign.
        rows[penalty_rows, penalty_tokens] /= floats[5 * count :]
    else:
        pass
    rows[:, special_tokens.listen] *= floats[3 * count : 4 * count]
    temperatures = floats[2 * count : 3 * count].unsqueeze(1)
    row_top_k = integers[count : 2 * count]
    row_top_p = floats[4 * count : 5 * count]
    max_top_k = max((k for k in top_k if 0 < k < vocab_size), default=0)
    has_top_p = any(0.0 < p < 1.0 for p in top_p)
    if max_top_k > 0 and all(0 < k < vocab_size for k in top_k):
        # note (0xtoward): every row keeps at most its top-k, so top-p and the draw only need the top-k candidates.
        candidate_logits, candidate_ids = torch.topk(
            rows / temperatures, max_top_k, dim=-1
        )
        ranks = torch.arange(max_top_k, device=rows.device)
        candidate_logits = candidate_logits.masked_fill(
            ranks.unsqueeze(0) >= row_top_k.unsqueeze(1), -torch.inf
        )
        if has_top_p:
            cumulative_probabilities = torch.cumsum(
                F.softmax(candidate_logits, dim=-1), dim=-1
            )
            limits = torch.where(
                (row_top_p > 0.0) & (row_top_p < 1.0), row_top_p, torch.inf
            )
            should_remove = cumulative_probabilities > limits.unsqueeze(1)
            should_remove[:, 1:] = should_remove[:, :-1].clone()
            should_remove[:, 0] = False
            candidate_logits = candidate_logits.masked_fill(should_remove, -torch.inf)
        else:
            pass
        sampled = candidate_ids.gather(
            1, torch.multinomial(F.softmax(candidate_logits, dim=-1), 1)
        )[:, 0]
    else:
        filtered_logits = filter_top_k_top_p(
            rows / temperatures,
            top_k=row_top_k,
            top_p=row_top_p,
            max_top_k=max_top_k,
            has_top_p=has_top_p,
        )
        sampled = torch.multinomial(F.softmax(filtered_logits, dim=-1), 1)[:, 0]
    second_picks = torch.where(
        floats[count : 2 * count] > 0, rows.argmax(dim=-1), sampled
    )
    picks = torch.stack((first_picks, second_picks)).tolist()
    for position, (index, state) in enumerate(zip(active, states, strict=True)):
        if picks[0][position] == special_tokens.chunk_eos:
            token_ids[index] = special_tokens.chunk_eos
            continue
        else:
            pass
        candidate_token_id = picks[1][position]
        sampling = state.sampling
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
        token_ids[index] = candidate_token_id
    return token_ids
