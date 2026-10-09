# SPDX-License-Identifier: Apache-2.0
"""MiniCPM-o duplex token sampling with each session's sampler state on the device."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from sglang_omni.models.minicpm_o.native_config import MiniCPMODuplexSampling
from sglang_omni.models.minicpm_o.special_tokens import MiniCPMOSpecialTokenIds


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


@dataclass(kw_only=True)
class DuplexUnitStart:
    """Host facts for a unit whose first sample this step draws."""

    sampling_slot: int
    sampling: MiniCPMODuplexSampling
    is_new_session: bool
    generation_steps: int
    is_listen_forced: bool


@dataclass(kw_only=True)
class DuplexSamplerState:
    """Every session's duplex sampler state, one device row per session sampling slot.

    history keeps a session's recent second-stage picks right-aligned, oldest first; the
    value vocab marks an empty position. Unit counters restart with each unit.
    """

    history: torch.Tensor
    window_sizes: torch.Tensor
    is_turn_ended: torch.Tensor
    unit_steps: torch.Tensor
    unit_budgets: torch.Tensor
    is_listen_forced: torch.Tensor
    is_unit_done: torch.Tensor
    is_greedy: torch.Tensor
    is_second_stage_greedy: torch.Tensor
    temperatures: torch.Tensor
    top_ks: torch.Tensor
    top_ps: torch.Tensor
    repetition_penalties: torch.Tensor
    listen_scales: torch.Tensor
    vocab: int

    @classmethod
    def allocate(
        cls, slots: int, window: int, vocab: int, device: torch.device
    ) -> DuplexSamplerState:
        return cls(
            history=torch.full((slots, window), vocab, dtype=torch.long, device=device),
            window_sizes=torch.ones(slots, dtype=torch.long, device=device),
            is_turn_ended=torch.ones(slots, dtype=torch.bool, device=device),
            unit_steps=torch.zeros(slots, dtype=torch.long, device=device),
            unit_budgets=torch.ones(slots, dtype=torch.long, device=device),
            is_listen_forced=torch.zeros(slots, dtype=torch.bool, device=device),
            is_unit_done=torch.zeros(slots, dtype=torch.bool, device=device),
            is_greedy=torch.zeros(slots, dtype=torch.bool, device=device),
            is_second_stage_greedy=torch.zeros(slots, dtype=torch.bool, device=device),
            temperatures=torch.ones(slots, dtype=torch.float32, device=device),
            top_ks=torch.full((slots,), -1, dtype=torch.long, device=device),
            top_ps=torch.ones(slots, dtype=torch.float32, device=device),
            repetition_penalties=torch.ones(slots, dtype=torch.float32, device=device),
            listen_scales=torch.ones(slots, dtype=torch.float32, device=device),
            vocab=vocab,
        )

    def widen_history(self, window: int) -> None:
        """Keep every session's history when a session asks for a longer window."""
        slots, current_window = self.history.shape
        if window > current_window:
            history = torch.full(
                (slots, window),
                self.vocab,
                dtype=torch.long,
                device=self.history.device,
            )
            history[:, window - current_window :] = self.history
            self.history = history
        else:
            pass

    def start_units(self, units: list[DuplexUnitStart]) -> None:
        """Restart the unit counters, and clear the rows of sessions that just opened."""
        device = self.history.device
        slots = to_device([unit.sampling_slot for unit in units], torch.long, device)
        self.unit_steps[slots] = to_device(
            [unit.generation_steps for unit in units], torch.long, device
        )
        self.unit_budgets[slots] = to_device(
            [unit.sampling.max_new_tokens_per_unit for unit in units],
            torch.long,
            device,
        )
        self.is_listen_forced[slots] = to_device(
            [unit.is_listen_forced for unit in units], torch.bool, device
        )
        self.is_unit_done[slots] = False
        new_sessions = [unit for unit in units if unit.is_new_session]
        if new_sessions:
            samplings = [unit.sampling for unit in new_sessions]
            new_slots = to_device(
                [unit.sampling_slot for unit in new_sessions], torch.long, device
            )
            self.history[new_slots] = self.vocab
            self.is_turn_ended[new_slots] = True
            self.window_sizes[new_slots] = to_device(
                [sampling.repetition_window_size for sampling in samplings],
                torch.long,
                device,
            )
            self.is_greedy[new_slots] = to_device(
                [sampling.greedy for sampling in samplings], torch.bool, device
            )
            self.is_second_stage_greedy[new_slots] = to_device(
                [
                    sampling.greedy or sampling.temperature <= 0
                    for sampling in samplings
                ],
                torch.bool,
                device,
            )
            self.temperatures[new_slots] = to_device(
                [
                    sampling.temperature if sampling.temperature > 0 else 1.0
                    for sampling in samplings
                ],
                torch.float32,
                device,
            )
            self.top_ks[new_slots] = to_device(
                [sampling.top_k for sampling in samplings], torch.long, device
            )
            self.top_ps[new_slots] = to_device(
                [sampling.top_p for sampling in samplings], torch.float32, device
            )
            self.repetition_penalties[new_slots] = to_device(
                [sampling.repetition_penalty for sampling in samplings],
                torch.float32,
                device,
            )
            self.listen_scales[new_slots] = to_device(
                [sampling.listen_prob_scale for sampling in samplings],
                torch.float32,
                device,
            )
        else:
            pass

    def sample(
        self,
        logits: torch.Tensor,
        slots: torch.Tensor,
        samplings: list[MiniCPMODuplexSampling],
        *,
        special_tokens: MiniCPMOSpecialTokenIds,
        forbidden_token_index: torch.Tensor,
    ) -> torch.Tensor:
        """Draw every row's next token and advance its session's state without a host read.

        samplings are the rows' session settings; their host copy fixes the draw's shapes.
        """
        rows = logits.float()
        count, vocab_size = rows.shape
        row_top_ks = [sampling.top_k for sampling in samplings]
        max_top_k = max((k for k in row_top_ks if 0 < k < vocab_size), default=0)
        has_top_p = any(0.0 < sampling.top_p < 1.0 for sampling in samplings)
        # note (Junnan Li): The first sample must use the unscaled model distribution.
        first_picks = torch.where(
            self.is_greedy[slots],
            rows.argmax(dim=-1),
            torch.multinomial(F.softmax(rows, dim=-1), 1)[:, 0],
        )
        rows[:, forbidden_token_index] = -torch.inf
        history = self.history[slots]
        window = history.shape[1]
        is_in_window = torch.arange(window, device=rows.device).unsqueeze(0) >= (
            window - self.window_sizes[slots]
        ).unsqueeze(1)
        is_recent = torch.zeros(
            count, self.vocab + 1, dtype=torch.bool, device=rows.device
        )
        is_recent.scatter_(1, torch.where(is_in_window, history, self.vocab), True)
        # note (Junnan Li): Matches the checkpoint sampler, which ignores the logit sign.
        rows = torch.where(
            is_recent[:, : self.vocab],
            rows / self.repetition_penalties[slots].unsqueeze(1),
            rows,
        )
        rows[:, special_tokens.listen] *= self.listen_scales[slots]
        temperatures = self.temperatures[slots].unsqueeze(1)
        top_ks = self.top_ks[slots]
        top_ps = self.top_ps[slots]
        if all(0 < k < vocab_size for k in row_top_ks):
            # note (0xtoward): every row keeps at most its top-k, so top-p and the draw only need the top-k candidates.
            candidate_logits, candidate_ids = torch.topk(
                rows / temperatures, max_top_k, dim=-1
            )
            ranks = torch.arange(max_top_k, device=rows.device)
            candidate_logits = candidate_logits.masked_fill(
                ranks.unsqueeze(0) >= top_ks.unsqueeze(1), -torch.inf
            )
            if has_top_p:
                cumulative_probabilities = torch.cumsum(
                    F.softmax(candidate_logits, dim=-1), dim=-1
                )
                limits = torch.where((top_ps > 0.0) & (top_ps < 1.0), top_ps, torch.inf)
                should_remove = cumulative_probabilities > limits.unsqueeze(1)
                should_remove[:, 1:] = should_remove[:, :-1].clone()
                should_remove[:, 0] = False
                candidate_logits = candidate_logits.masked_fill(
                    should_remove, -torch.inf
                )
            else:
                pass
            sampled = candidate_ids.gather(
                1, torch.multinomial(F.softmax(candidate_logits, dim=-1), 1)
            )[:, 0]
        else:
            filtered_logits = filter_top_k_top_p(
                rows / temperatures,
                top_k=top_ks,
                top_p=top_ps,
                max_top_k=max_top_k,
                has_top_p=has_top_p,
            )
            sampled = torch.multinomial(F.softmax(filtered_logits, dim=-1), 1)[:, 0]
        second_picks = torch.where(
            self.is_second_stage_greedy[slots], rows.argmax(dim=-1), sampled
        )
        unit_steps = self.unit_steps[slots]
        is_turn_ended = self.is_turn_ended[slots]
        is_unit_done = self.is_unit_done[slots]
        is_budget_spent = unit_steps >= self.unit_budgets[slots] - 1
        is_listen_forced = (unit_steps == 0) & self.is_listen_forced[slots]
        is_chunk_closed = first_picks == special_tokens.chunk_eos
        candidate_ids = torch.where(
            (second_picks == special_tokens.listen) & ~is_turn_ended,
            special_tokens.tts_bos,
            second_picks,
        )
        token_ids = torch.where(
            is_budget_spent,
            special_tokens.chunk_eos,
            torch.where(
                is_listen_forced,
                special_tokens.listen,
                torch.where(is_chunk_closed, special_tokens.chunk_eos, candidate_ids),
            ),
        )
        # note (0xtoward): a row whose unit already closed is a step drawn past its end, so it changes nothing.
        is_sampled = ~(
            is_budget_spent | is_listen_forced | is_chunk_closed | is_unit_done
        )
        is_chunk_terminator = (
            (token_ids == special_tokens.listen)
            | (token_ids == special_tokens.chunk_eos)
            | (token_ids == special_tokens.chunk_tts_eos)
        )
        # note (Junnan Li): History retains controls before the mid-turn listen rewrite.
        self.history[slots] = torch.where(
            is_sampled.unsqueeze(1),
            torch.cat((history[:, 1:], second_picks.unsqueeze(1)), dim=1),
            history,
        )
        self.is_turn_ended[slots] = torch.where(
            is_sampled,
            (token_ids == special_tokens.turn_eos)
            | (is_chunk_terminator & is_turn_ended),
            is_turn_ended,
        )
        self.unit_steps[slots] = unit_steps + 1
        self.is_unit_done[slots] = is_unit_done | is_chunk_terminator
        return token_ids
