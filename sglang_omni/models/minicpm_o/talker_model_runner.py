# SPDX-License-Identifier: Apache-2.0
"""MiniCPM-o talker runner: condition-embeds prefill + windowed rep penalty."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from sglang_omni.model_runner.base import (
    ModelRunner,
    rank_shared_unseeded_sampling_seed,
)
from sglang_omni.model_runner.prefill_inputs import (
    OmniPrefillInputs,
    attach_omni_prefill_inputs,
)
from sglang_omni.models.minicpm_o.talker_session import TalkerUnitRequestData
from sglang_omni.sampling.seed import SAMPLING_SEED_MASK, resolve_row_seed
from sglang_omni.scheduling.sglang_backend.request_data import (
    SGLangARRequestData,
    session_prefill_rows,
)
from sglang_omni.scheduling.types import SchedulerRequest

if TYPE_CHECKING:
    from sglang.srt.layers.logits_processor import LogitsProcessorOutput
    from sglang.srt.managers.schedule_batch import ScheduleBatch
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
else:
    pass

# note (MayDomine): the checkpoint penalizes only the most recent 16 codec tokens.
REP_PENALTY_WINDOW = 16


@dataclass(kw_only=True)
class TalkerSlotState:
    """Decode history and sampling inputs per request slot, kept on the GPU.

    A decode step reads and updates them on the device, so it needs no host token
    history and no host-to-device copy, which lets the talker run async decode.
    Token i of a request sits at windows[slot, i % REP_PENALTY_WINDOW].
    """

    windows: torch.Tensor
    generated: torch.Tensor
    min_new_tokens: torch.Tensor
    penalties: torch.Tensor
    seeds: torch.Tensor
    vocab: int
    eos_id: int

    @classmethod
    def allocate(
        cls, slots: int, vocab: int, eos_id: int, device: torch.device
    ) -> TalkerSlotState:
        return cls(
            windows=torch.full(
                (slots, REP_PENALTY_WINDOW), vocab, dtype=torch.long, device=device
            ),
            generated=torch.zeros(slots, dtype=torch.long, device=device),
            min_new_tokens=torch.zeros(slots, dtype=torch.long, device=device),
            penalties=torch.ones(slots, dtype=torch.float32, device=device),
            seeds=torch.zeros(slots, dtype=torch.long, device=device),
            vocab=vocab,
            eos_id=eos_id,
        )

    def reset(self, rows: torch.Tensor, requests: list[SchedulerRequest]) -> None:
        """Load the recent tokens and sampling inputs of newly prefilled requests."""
        count = len(requests)
        windows = torch.full((count, REP_PENALTY_WINDOW), self.vocab, dtype=torch.long)
        generated = torch.zeros(count, dtype=torch.long)
        min_new_tokens = torch.zeros(count, dtype=torch.long)
        penalties = torch.ones(count, dtype=torch.float32)
        seeds = torch.zeros(count, dtype=torch.long)
        for row, sched_req in enumerate(requests):
            req = sched_req.data.req
            length = len(req.output_ids)
            for index in range(max(0, length - REP_PENALTY_WINDOW), length):
                windows[row, index % REP_PENALTY_WINDOW] = int(req.output_ids[index])
            generated[row] = length
            inputs = sched_req.data.talker_model_inputs
            min_new_tokens[row] = int(inputs.get("min_new_tokens", 0))
            penalties[row] = float(inputs.get("rep_penalty", 1.0))
            seed = req.sampling_params.sampling_seed
            if seed is None:
                seed = rank_shared_unseeded_sampling_seed(sched_req, row)
            elif not (0 <= seed <= SAMPLING_SEED_MASK):
                seed = resolve_row_seed(seed)
                req.sampling_params.sampling_seed = seed
            else:
                pass
            seeds[row] = seed
        device = rows.device
        self.windows[rows] = windows.to(device)
        self.generated[rows] = generated.to(device)
        self.min_new_tokens[rows] = min_new_tokens.to(device)
        self.penalties[rows] = penalties.to(device)
        self.seeds[rows] = seeds.to(device)

    def apply(self, logits: torch.Tensor, rows: torch.Tensor) -> None:
        """Apply the windowed frequency penalty and hold EOS until min_new_tokens."""
        windows = self.windows[rows]
        counts = torch.zeros(
            len(rows), self.vocab + 1, dtype=torch.float32, device=logits.device
        )
        counts.scatter_add_(1, windows, torch.ones_like(windows, dtype=torch.float32))
        counts = counts[:, : self.vocab]
        factors = self.penalties[rows].unsqueeze(1) ** counts
        scores = logits.to(torch.float32)
        penalized = torch.where(scores < 0, scores * factors, scores / factors)
        logits.copy_(torch.where(counts > 0, penalized, scores).to(logits.dtype))
        eos_logits = logits[:, self.eos_id]
        logits[:, self.eos_id] = torch.where(
            self.generated[rows] < self.min_new_tokens[rows],
            torch.full_like(eos_logits, float("-inf")),
            eos_logits,
        )

    def append(self, rows: torch.Tensor, next_token_ids: torch.Tensor) -> None:
        generated = self.generated[rows]
        self.windows[rows, generated % REP_PENALTY_WINDOW] = next_token_ids.long()
        self.generated[rows] = generated + 1


class MiniCPMOTalkerModelRunner(ModelRunner):
    """Prefill codec conditions and apply a frequency penalty over recent tokens."""

    slot_state: TalkerSlotState | None = None

    def before_prefill(
        self,
        forward_batch: ForwardBatch,
        schedule_batch: ScheduleBatch,
        requests: list[SchedulerRequest],
    ) -> None:
        """Prepare request embeddings; schedule_batch follows the runner interface."""
        parts: list[torch.Tensor] = []
        for sched_req in requests:
            data = sched_req.data
            if isinstance(data, TalkerUnitRequestData):
                parts.append(
                    session_prefill_rows(
                        data, self.model.emb_code, self.model.emb_code.weight.device
                    )
                )
                continue
            else:
                pass
            tensor = data.prefill_input_embeds
            if tensor is None:
                raise RuntimeError(
                    "MiniCPM-o talker prefill requires condition embeddings"
                )
            else:
                pass
            req = data.req
            prefix_len = len(req.prefix_indices)
            end = prefix_len + int(req.extend_range.length)
            prompt_len = int(tensor.shape[0])
            if prefix_len < prompt_len:
                parts.append(tensor[prefix_len : min(end, prompt_len)])
            else:
                pass
            if end > prompt_len:
                # note (MayDomine): retracted requests replay already-generated tokens.
                fill_ids = req.get_fill_ids()
                generated = torch.tensor(
                    fill_ids[max(prefix_len, prompt_len) : end],
                    dtype=torch.long,
                    device=self.model.emb_code.weight.device,
                )
                parts.append(self.model.emb_code(generated))
            else:
                pass
        input_embeds = torch.cat(parts, dim=0).to(
            device=forward_batch.input_ids.device,
            dtype=self.model.emb_code.weight.dtype,
        )
        expected_rows = int(forward_batch.input_ids.shape[0])
        if input_embeds.shape[0] != expected_rows:
            raise RuntimeError(
                "Talker prefill embeds must align with forward input_ids: "
                f"got {input_embeds.shape[0]} rows for {expected_rows} input ids"
            )
        else:
            pass
        attach_omni_prefill_inputs(
            forward_batch,
            OmniPrefillInputs(
                input_embeds=input_embeds,
                input_embeds_are_projected=True,
            ),
        )

    def sample_next_token_ids(
        self,
        logits_output: LogitsProcessorOutput,
        forward_batch: ForwardBatch,
        schedule_batch: ScheduleBatch | None,
        requests: list[SchedulerRequest],
    ) -> torch.Tensor:
        if any(
            isinstance(sched_req.data, TalkerUnitRequestData) for sched_req in requests
        ):
            # note (0xtoward): duplex units keep the host window over their session.
            return super().sample_next_token_ids(
                logits_output, forward_batch, schedule_batch, requests
            )
        else:
            pass
        assert not any(sched_req.data.return_logprob for sched_req in requests)
        logits = logits_output.next_token_logits[: len(requests)]
        rows = forward_batch.req_pool_indices[: len(requests)]
        if self.slot_state is None:
            self.slot_state = TalkerSlotState.allocate(
                self.tp_worker.model_runner.req_to_token_pool.req_to_token.shape[0],
                logits.shape[1],
                next(iter(requests[0].data.req.eos_token_ids)),
                logits.device,
            )
        else:
            pass
        if forward_batch.forward_mode.is_extend():
            self.slot_state.reset(rows, requests)
        else:
            pass
        self.apply_codec_suppress_tokens(logits_output, requests)
        self.slot_state.apply(logits, rows)
        sampling_info = forward_batch.sampling_info
        if any(
            sched_req.data.req.sampling_params.sampling_seed is not None
            for sched_req in requests
        ):
            self.validate_seeded_sampling_supported(sampling_info)
            sampling_info.sampling_seed = self.slot_state.seeds[rows]
        else:
            pass
        next_token_ids = self.tp_worker.model_runner.sample(
            logits_output, forward_batch
        )
        self.slot_state.append(rows, next_token_ids[: len(requests)])
        return next_token_ids

    def process_sampling_logits(
        self, logits_output: LogitsProcessorOutput, requests: list[SchedulerRequest]
    ) -> None:
        logits = logits_output.next_token_logits
        if logits is None or logits.ndim != 2:
            return
        else:
            pass
        vocab = logits.shape[1]
        device = logits.device
        penalized_rows: list[int] = []
        penalties: list[float] = []
        windows: list[list[int]] = []
        for row_idx, sched_req in enumerate(requests):
            data = sched_req.data
            penalty = float(data.talker_model_inputs.get("rep_penalty", 1.0))
            if penalty == 1.0:
                continue
            else:
                pass
            window = [
                tok
                for tok in map(int, data.req.output_ids[-REP_PENALTY_WINDOW:])
                if 0 <= tok < vocab
            ]
            if not window:
                continue
            else:
                pass
            penalized_rows.append(row_idx)
            penalties.append(penalty)
            windows.append(window)
        if not penalized_rows:
            return
        else:
            pass
        # note (MayDomine): a dummy vocabulary bin excludes ragged-window padding.
        num = len(windows)
        window_ids = torch.full((num, REP_PENALTY_WINDOW), vocab, dtype=torch.long)
        for i, window in enumerate(windows):
            window_ids[i, : len(window)] = torch.tensor(window, dtype=torch.long)
        window_ids = window_ids.to(device)
        counts = torch.zeros(num, vocab + 1, dtype=torch.float32, device=device)
        counts.scatter_add_(
            1, window_ids, torch.ones_like(window_ids, dtype=torch.float32)
        )
        counts = counts[:, :vocab]
        alphas = (
            torch.tensor(penalties, dtype=torch.float32, device=device).unsqueeze(1)
            ** counts
        )
        rows_t = torch.tensor(penalized_rows, dtype=torch.long, device=device)
        orig_dtype = logits.dtype
        scores = logits[rows_t].to(torch.float32)
        penalized = torch.where(scores < 0, scores * alphas, scores / alphas)
        scores = torch.where(counts > 0, penalized, scores)
        logits[rows_t] = scores.to(orig_dtype)

    def on_request_finished(self, request_id: str, data: SGLangARRequestData) -> None:
        if isinstance(data, TalkerUnitRequestData):
            # note (Junnan Li): The last sample has no KV and is not committed by chunk TTS.
            data.req.output_ids = data.req.output_ids[:-1]
            data.req.finished_len = len(data.req.output_ids)
        else:
            pass
        super().on_request_finished(request_id, data)
