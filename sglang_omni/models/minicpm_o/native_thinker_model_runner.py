# SPDX-License-Identifier: Apache-2.0
"""MiniCPM-o thinker model runner."""

from __future__ import annotations

import torch
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.utils import GenerationBatchResult
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode, ForwardBatch

from sglang_omni.model_runner.model_worker import ModelWorker
from sglang_omni.model_runner.prefill_inputs import (
    OmniPrefillInputs,
    attach_omni_prefill_inputs,
)
from sglang_omni.models.minicpm_o.duplex_sampler import (
    DuplexSamplerState,
    DuplexUnitStart,
    build_forbidden_token_index,
    to_device,
)
from sglang_omni.models.minicpm_o.special_tokens import (
    MiniCPMOSpecialTokenIds,
    resolve_special_token_ids,
)
from sglang_omni.models.minicpm_o.thinker_model_runner import (
    MiniCPMOThinkerModelRunner as OfflineThinkerModelRunner,
)
from sglang_omni.models.minicpm_o.thinker_state import DuplexUnitRequestData
from sglang_omni.scheduling.sglang_backend.output_processor import SGLangOutputProcessor
from sglang_omni.scheduling.sglang_backend.request_data import session_prefill_rows
from sglang_omni.scheduling.types import (
    RequestOutput,
    SchedulerOutput,
    SchedulerRequest,
)


class MiniCPMOThinkerModelRunner(OfflineThinkerModelRunner):
    """Run duplex units on the shared MiniCPM-o runner with media history."""

    def __init__(
        self, tp_worker: ModelWorker, output_processor: SGLangOutputProcessor
    ) -> None:
        # note (ruinique): duplex_sample owns the native runner's termination tokens.
        super().__init__(tp_worker, output_processor, eos_token_ids=[])
        self.special_tokens: MiniCPMOSpecialTokenIds | None = None
        self.forbidden_token_index: torch.Tensor | None = None
        self.sampler_state: DuplexSamplerState | None = None

    def custom_prefill_forward(
        self,
        forward_batch: ForwardBatch,
        schedule_batch: ScheduleBatch,
        requests: list[SchedulerRequest],
    ) -> GenerationBatchResult | None:
        embed_tokens = self.embed_tokens
        last_token_id = embed_tokens.num_embeddings - 1
        rows = [
            session_prefill_rows(
                request.data,
                lambda token_ids: embed_tokens(token_ids.clamp(0, last_token_id)),
                forward_batch.input_ids.device,
            )
            for request in requests
        ]
        attach_omni_prefill_inputs(
            forward_batch,
            OmniPrefillInputs(input_embeds=torch.cat(rows, dim=0)),
        )
        return None

    def sample_before_post_prefill(
        self,
        forward_batch: ForwardBatch,
        schedule_batch: ScheduleBatch,
        requests: list[SchedulerRequest],
    ) -> bool:
        return True

    def sample_before_post_decode(
        self,
        forward_batch: ForwardBatch,
        schedule_batch: ScheduleBatch,
        requests: list[SchedulerRequest],
    ) -> bool:
        return True

    def lookahead_eligible(self, batch: ScheduleBatch) -> bool:
        # note (Junnan Li): Lookahead would sample before repetition history advances.
        return False

    def resolve_special_tokens(
        self, data: DuplexUnitRequestData
    ) -> MiniCPMOSpecialTokenIds:
        if self.special_tokens is None:
            self.special_tokens = resolve_special_token_ids(
                data.req.tokenizer,
                bad_token_ids=tuple(data.req.tokenizer.bad_token_ids),
            )
        else:
            pass
        return self.special_tokens

    def sample_next_token_ids(
        self,
        logits_output: LogitsProcessorOutput,
        forward_batch: ForwardBatch,
        schedule_batch: ScheduleBatch,
        requests: list[SchedulerRequest],
    ) -> torch.Tensor:
        logits = logits_output.next_token_logits[: len(requests)]
        vocab_size = logits.shape[-1]
        special_tokens = self.resolve_special_tokens(requests[0].data)
        if self.forbidden_token_index is None:
            self.forbidden_token_index = build_forbidden_token_index(
                special_tokens, vocab_size, logits.device
            )
        else:
            pass
        states = [request.data.thinker_state for request in requests]
        samplings = [state.sampling for state in states]
        window = max(sampling.repetition_window_size for sampling in samplings)
        if self.sampler_state is None:
            self.sampler_state = DuplexSamplerState.allocate(
                self.tp_worker.model_runner.req_to_token_pool.size,
                window,
                vocab_size,
                logits.device,
            )
        else:
            self.sampler_state.widen_history(window)
        if forward_batch.forward_mode.is_extend():
            self.sampler_state.start_units(
                [
                    DuplexUnitStart(
                        sampling_slot=request.data.thinker_state.sampling_slot,
                        sampling=request.data.thinker_state.sampling,
                        is_new_session=request.data.thinker_state.is_sampling_slot_fresh,
                        generation_steps=request.data.generation_steps,
                        is_listen_forced=request.data.is_listen_forced,
                    )
                    for request in requests
                ]
            )
            for state in states:
                state.is_sampling_slot_fresh = False
        else:
            pass
        token_ids = self.sampler_state.sample(
            logits,
            to_device(
                [state.sampling_slot for state in states], torch.long, logits.device
            ),
            samplings,
            special_tokens=special_tokens,
            forbidden_token_index=self.forbidden_token_index,
        )
        for request in requests:
            data = request.data
            if data.is_listen_forced and data.generation_steps == 0:
                data.thinker_state.force_listen_counter += 1
            else:
                pass
        return token_ids

    # note (Junnan Li): FULL capture must match decode graphs and retain talker conditioning.
    def requested_capture_hidden_mode_prefill(
        self, schedule_batch: ScheduleBatch, requests: list[SchedulerRequest]
    ) -> CaptureHiddenMode:
        return CaptureHiddenMode.FULL

    def requested_capture_hidden_mode_decode(
        self, schedule_batch: ScheduleBatch, requests: list[SchedulerRequest]
    ) -> CaptureHiddenMode:
        return CaptureHiddenMode.FULL

    def post_process_outputs(
        self,
        result: GenerationBatchResult,
        scheduler_output: SchedulerOutput,
        outputs: dict[str, RequestOutput],
    ) -> None:
        """Pair generated tokens with their next-step hidden states for the talker."""
        conditioned: list[tuple[DuplexUnitRequestData, int, bool]] = []
        hidden_states: list[torch.Tensor] = []
        for scheduler_request in scheduler_output.requests:
            request_output = outputs[scheduler_request.request_id]
            data = scheduler_request.data
            sampled_token_id = int(request_output.data)
            special_tokens = self.resolve_special_tokens(data)
            pending_token_id = data.pending_unit_token
            if pending_token_id is not None and data.generation_steps >= 2:
                hidden_state = request_output.extra["hidden_states"]
                hidden_states.append(
                    hidden_state.reshape(-1, hidden_state.shape[-1])[-1]
                )
                conditioned.append(
                    (
                        data,
                        pending_token_id,
                        pending_token_id == special_tokens.turn_eos,
                    )
                )
            else:
                pass
            if sampled_token_id in special_tokens.chunk_terminators:
                data.pending_unit_token = None
            else:
                if data.generation_steps > 0:
                    data.generated_unit_ids.append(sampled_token_id)
                else:
                    pass
                data.pending_unit_token = sampled_token_id
        if hidden_states:
            # note (0xtoward): the rows stay on the device until their unit ends, so a step reads nothing back.
            step_hidden_states = torch.stack(hidden_states).detach()
            for (data, token_id, is_turn_end), hidden_state in zip(
                conditioned, step_hidden_states, strict=True
            ):
                data.talker_conditions.append((token_id, hidden_state, is_turn_end))
        else:
            pass

    def on_request_finished(
        self, request_id: str, req_data: DuplexUnitRequestData
    ) -> None:
        """Move the unit's talker conditions to the host with one copy."""
        conditions = req_data.talker_conditions
        if conditions:
            host_hidden_states = torch.stack(
                [hidden_state for _, hidden_state, _ in conditions]
            ).to("cpu")
            req_data.talker_conditions = [
                (token_id, hidden_state, is_turn_end)
                for (token_id, _, is_turn_end), hidden_state in zip(
                    conditions, host_hidden_states, strict=True
                )
            ]
        else:
            pass
        super().on_request_finished(request_id, req_data)


__all__ = [
    "MiniCPMOThinkerModelRunner",
]
