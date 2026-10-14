# SPDX-License-Identifier: Apache-2.0
"""Bridge pipeline units to streaming sessions."""

from __future__ import annotations

import logging
import time
from array import array
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Protocol

import torch
from sglang.srt.managers.io_struct import (
    CloseSessionReqInput,
    OpenSessionReqInput,
    SessionParams,
    TokenizedGenerateReqInput,
)
from sglang.srt.managers.schedule_batch import FINISH_ABORT, Req
from sglang.srt.session.session_controller import Session, SessionController
from sglang.srt.utils.common import ceil_align

from sglang_omni.admission import (
    REQUEST_TO_TOKEN_SLOTS_RESERVED_FOR_RETAINED_KV,
    QueueFullError,
)
from sglang_omni.profiler.event_recorder import get_active_stage
from sglang_omni.proto.request import OmniRequest, StagePayload
from sglang_omni.proto.session import (
    SessionIdentity,
    SessionOperation,
    TimedChunk,
    find_session_operation,
)
from sglang_omni.scheduling.message import OutgoingMessage
from sglang_omni.scheduling.sglang_backend.request_data import (
    EmbeddingSpan,
    SGLangARRequestData,
)
from sglang_omni.scheduling.types import RequestOutput

SESSION_STRING_LENGTH_LIMIT_CHARACTERS = 0


logger = logging.getLogger(__name__)


class SessionKV(Protocol):
    req_pool_idx: int | None
    cache_protected_len: int
    kv_committed_len: int
    kv_allocated_len: int
    holds_kv: bool


class SessionSlot(Protocol):
    kv: SessionKV


class RequestToTokenPool(Protocol):
    free_slots: list[int]
    req_to_token: torch.Tensor


class SessionTreeCache(Protocol):
    slots: Mapping[str, SessionSlot]
    req_to_token_pool: RequestToTokenPool
    page_size: int

    def evictable_size(self) -> int:
        pass

    def free_kv_row(self, kv: SessionKV, ranges: list[tuple[int, int]]) -> None:
        pass


class LayerKVCache(Protocol):
    size: int
    page_size: int
    start_layer: int
    layer_num: int

    def get_key_buffer(self, layer_id: int) -> torch.Tensor:
        pass

    def get_value_buffer(self, layer_id: int) -> torch.Tensor:
        pass


class TokenToKVPoolAllocator(Protocol):
    def available_size(self) -> int:
        pass

    def get_kvcache(self) -> LayerKVCache:
        pass


class ModelVocabulary(Protocol):
    vocab_size: int


class BridgeScheduler(Protocol):
    """The scheduler surface the bridge drives; OmniScheduler satisfies it."""

    session_controller: SessionController
    tree_cache: SessionTreeCache
    req_to_token_pool: RequestToTokenPool
    token_to_kv_pool_allocator: TokenToKVPoolAllocator
    model_config: ModelVocabulary
    waiting_queue: list[Req]
    chunked_req: Req | None
    max_running_requests: int
    max_req_len: int

    def abort(self, request_id: str) -> None:
        pass

    def release_request_kv_cache(self, req: Req) -> None:
        pass

    def run_abort_callback(self, request_id: str) -> None:
        pass

    def synchronize_launched_decode(self) -> None:
        pass

    def resolve_pending_async(self) -> None:
        pass


def is_close_request(payload: StagePayload) -> bool:
    """Return whether the payload carries a session close."""
    operation_metadata = find_session_operation(payload.request.metadata)
    if operation_metadata is None:
        return False
    else:
        return operation_metadata.operation == "close"


@dataclass(frozen=True, kw_only=True)
class ARSessionPreparation:
    reset_history: bool = False
    bypass_generation: bool = False


@dataclass(frozen=True, kw_only=True)
class HistoryCut:
    """Retained session positions [start, end) to drop before the next unit."""

    start: int
    end: int
    # test only (xp): further disjoint ranges after [start, end), ascending
    more: tuple[tuple[int, int], ...] = ()

    def __post_init__(self) -> None:
        if not 0 <= self.start < self.end:
            raise ValueError(f"invalid history cut [{self.start}, {self.end})")
        else:
            pass
        for (_, previous_end), (start, end) in zip(self.ranges, self.ranges[1:]):
            if not previous_end < start < end:
                raise ValueError(f"invalid history cut ranges {self.ranges}")
            else:
                pass

    @property
    def ranges(self) -> list[tuple[int, int]]:
        return [(self.start, self.end), *self.more]

    @property
    def length(self) -> int:
        return sum(end - start for start, end in self.ranges)


class ARSessionAdapter:
    """Convert unit inputs and outputs without mutating streaming session state."""

    def open(self, session_identity: SessionIdentity, request: OmniRequest) -> None:
        """Initialize auxiliary model history after the streaming session is created."""

    def close(self, session_identity: SessionIdentity) -> None:
        """Release auxiliary history after streaming session work and KV are released."""

    def prepare_unit(
        self,
        session_identity: SessionIdentity,
        chunk: TimedChunk,
        payload: StagePayload,
    ) -> ARSessionPreparation:
        """Select a fresh model turn or relay a unit that needs no generation."""
        return ARSessionPreparation()

    def finish_input(
        self, session_identity: SessionIdentity, payload: StagePayload
    ) -> StagePayload | None:
        """Return a relayed EOS payload, or None to use normal generation."""
        return None

    def build(
        self,
        session_identity: SessionIdentity,
        chunk: TimedChunk,
        payload: StagePayload,
    ) -> SGLangARRequestData:
        raise NotImplementedError

    def plan_history_cut(
        self,
        session_identity: SessionIdentity,
        request_data: SGLangARRequestData,
        history_length: int,
        history_capacity: int,
    ) -> HistoryCut | None:
        """Choose retained history to drop before this unit joins the session.

        The unit can still finish when at most history_capacity tokens stay ahead of it.
        """
        return None

    def shift_key_positions(self, keys: torch.Tensor, distance: int) -> torch.Tensor:
        """Re-encode cached keys distance positions earlier, in the cache dtype."""
        raise NotImplementedError

    def result(
        self, session_identity: SessionIdentity, request_data: SGLangARRequestData
    ) -> StagePayload:
        raise NotImplementedError

    def stream(
        self,
        session_identity: SessionIdentity,
        request_data: SGLangARRequestData,
        output: RequestOutput,
    ) -> Iterable[TimedChunk]:
        return ()

    def flush(
        self, session_identity: SessionIdentity, request_data: SGLangARRequestData
    ) -> Iterable[TimedChunk]:
        return ()


@dataclass(kw_only=True)
class SessionUnit:
    request_id: str
    session_identity: SessionIdentity
    stages: tuple[str, ...]
    chunk: TimedChunk
    # Note (chenyang):
    # There are multiple requests in SGLang Omni. Let's make a distinction between them.
    # payload.request is the user's OmniRequest: inputs, params, and metadata. It is not queued.
    # The session scheduler creates session_request with Session.create_req and queues that same
    # SGLang Req. create_req builds it from token ids produced from the OmniRequest, joins tokens
    # from earlier turns, and binds it to the streaming session.
    session_request: Req | None = None
    is_enqueued: bool = False
    # note (Junnan Li): Only completed units commit their spans to retained history.
    embedding_spans: list[EmbeddingSpan] = field(default_factory=list)


@dataclass(kw_only=True)
class BridgeSession:
    session_identity: SessionIdentity
    unit: SessionUnit | None = None
    # note (Junnan Li): KV rebuilds need embeddings from every completed unit.
    embedding_spans: list[EmbeddingSpan] = field(default_factory=list)


class ARSessionBridge:
    """Map pipeline open, append, and close onto one SGLang streaming session."""

    def __init__(
        self, bridge_scheduler: BridgeScheduler, adapter: ARSessionAdapter
    ) -> None:
        self.bridge_scheduler = bridge_scheduler
        self.adapter = adapter
        self.sessions: dict[str, BridgeSession] = {}
        self.units_by_request_id: dict[str, SessionUnit] = {}
        self.cancelling_request_id: str | None = None

    def drain(self) -> None:
        self.bridge_scheduler.synchronize_launched_decode()
        self.bridge_scheduler.resolve_pending_async()

    def apply_operation(
        self, payload: StagePayload, operation: SessionOperation
    ) -> StagePayload:
        session_identity = operation.session_identity
        operation_kind = operation.operation
        session = self.sessions.get(session_identity.id)
        if (
            session is not None
            and session_identity.open_index != session.session_identity.open_index
        ):
            raise ValueError("stale session open index")
        elif operation_kind == "open":
            if session is not None:
                raise ValueError("session already opened")
            elif len(self.sessions) >= self.bridge_scheduler.max_running_requests:
                raise QueueFullError()
            elif self.open_streaming_session(session_identity.id):
                self.sessions[session_identity.id] = BridgeSession(
                    session_identity=session_identity
                )
                self.adapter.open(session_identity, payload.request)
                payload.data = {"opened": True}
            else:
                raise ValueError("streaming session open failed")
        elif operation_kind == "close":
            if session is not None:
                self.close_session(session)
            else:
                pass
                # note (chenyang): coding style, each if should be closed by else.
            payload.data = {"closed": True}
        else:
            raise ValueError("unknown session operation")
        return payload

    def accept(self, payload: StagePayload, operation: SessionOperation) -> SessionUnit:
        chunk = operation.chunk
        assert (
            chunk is not None
        ), f"session append {payload.request_id} requires a chunk"
        session_identity = operation.session_identity
        session = self.sessions.get(session_identity.id)
        if session is None or session_identity != session.session_identity:
            raise ValueError("unknown or stale session open index")
        else:
            if session.unit is None:
                session.unit = SessionUnit(
                    request_id=payload.request_id,
                    session_identity=session_identity,
                    stages=operation.stages,
                    chunk=chunk,
                )
                self.units_by_request_id[payload.request_id] = session.unit
                return session.unit
            elif session.unit.request_id != payload.request_id:
                raise ValueError("session already has an active request")
            else:
                return session.unit

    def prepare_unit(self, unit: SessionUnit, payload: StagePayload) -> bool:
        preparation = self.adapter.prepare_unit(
            unit.session_identity, unit.chunk, payload
        )
        if preparation.reset_history:
            session_id = unit.session_identity.id
            self.sessions[session_id].embedding_spans.clear()
            self.release_streaming_session(session_id)
            if not self.open_streaming_session(session_id):
                raise RuntimeError("native session reset failed")
            else:
                pass
        else:
            pass
        return preparation.bypass_generation

    def open_streaming_session(self, session_id: str) -> bool:
        result = self.bridge_scheduler.session_controller.open(
            OpenSessionReqInput(
                session_id=session_id,
                capacity_of_str_len=SESSION_STRING_LENGTH_LIMIT_CHARACTERS,
                streaming=True,
                timeout=None,
            )
        )
        return result.success

    def release_streaming_session(self, session_id: str) -> None:
        self.drain()
        self.bridge_scheduler.session_controller.close(
            CloseSessionReqInput(session_id=session_id)
        )
        if (
            self.bridge_scheduler.session_controller.get(session_id) is not None
            or session_id in self.bridge_scheduler.tree_cache.slots
        ):
            raise RuntimeError("streaming session close is still pending")
        else:
            pass

    def create_session_request(
        self, payload: StagePayload, request_data: SGLangARRequestData
    ) -> None:
        self.drain()
        unit = self.units_by_request_id[payload.request_id]
        adapter_request = request_data.req
        assert (
            adapter_request.rid == payload.request_id
        ), f"session adapter changed request ID: expected {payload.request_id}, got {adapter_request.rid}"
        if (
            request_data.prefill_input_embeds is not None
            or request_data.decode_input_embeds
            or request_data.input_embeds_are_projected
            or adapter_request.input_embeds is not None
            or adapter_request.multimodal_inputs is not None
        ):
            raise ValueError(
                "history-aware session embedding and multimodal inputs are not supported"
            )
        else:
            pass
        native_session = self.bridge_scheduler.session_controller.get(
            unit.session_identity.id
        )
        session = self.sessions[unit.session_identity.id]
        if native_session.req_nodes:
            [last_node] = native_session.req_nodes.values()
            last_request = last_node.req
            # note (0xtoward): Session.create_req puts the committed ids and the last outputs, up to max_new_tokens, ahead of this unit.
            history_length = native_session.committed_origin_len + len(
                last_request.output_ids[: last_request.sampling_params.max_new_tokens]
            )
        else:
            history_length = 0
        # note (0xtoward): SGLang grants the whole max_new_tokens only while input plus output stays below max_req_len.
        history_capacity = (
            self.bridge_scheduler.max_req_len
            - 1
            - len(adapter_request.origin_input_ids)
            - int(adapter_request.sampling_params.max_new_tokens or 0)
        )
        # test only (xp): key statistics of retained units for the knorm / keydiff windows
        self.adapter.xp_scorer = lambda bounds, policy: self.score_history_units(
            native_session, bounds, policy
        )
        history_cut = self.adapter.plan_history_cut(
            unit.session_identity, request_data, history_length, history_capacity
        )
        if history_cut is not None:
            self.drop_history(session, native_session, history_cut)
        else:
            pass
        tokenized_input = TokenizedGenerateReqInput(
            rid=adapter_request.rid,
            input_text=None,
            input_ids=array("q", adapter_request.origin_input_ids),
            input_embeds=None,
            mm_inputs=None,
            token_type_ids=None,
            sampling_params=adapter_request.sampling_params,
            logprob_start_len=adapter_request.logprob_start_len,
            session_params=SessionParams(id=native_session.session_id),
            stream=adapter_request.stream,
            return_logprob=adapter_request.return_logprob,
            return_sampling_mask=adapter_request.return_sampling_mask,
            lora_id=adapter_request.lora_id,
            custom_logit_processor=adapter_request.custom_logit_processor,
            require_reasoning=adapter_request.require_reasoning,
            return_hidden_states=adapter_request.return_hidden_states,
            return_routed_experts=adapter_request.return_routed_experts,
            routed_experts_start_len=adapter_request.routed_experts_start_len,
            priority=adapter_request.priority,
            routing_key=adapter_request.routing_key,
            extra_key=adapter_request.extra_key,
            cache_salt=adapter_request.cache_salt,
            http_worker_ipc=adapter_request.http_worker_ipc,
            top_logprobs_num=adapter_request.logprob.top_logprobs_num,
            token_ids_logprob=adapter_request.logprob.token_ids_logprob,
        )
        session_request = native_session.create_req(
            tokenized_input,
            adapter_request.tokenizer,
            self.bridge_scheduler.model_config.vocab_size,
            eos_token_ids=adapter_request.eos_token_ids,
        )
        if session_request.to_finish is not None:
            raise ValueError("native session rejected append")
        else:
            pass
        unit.session_request = session_request
        # note (Junnan Li): Native requests prepend retained turns to this unit's ids.
        offset = len(session_request.origin_input_ids) - len(
            adapter_request.origin_input_ids
        )
        request_data.session_history_length = offset
        unit.embedding_spans = [
            EmbeddingSpan(
                start=span.start + offset,
                end=span.end + offset,
                input_embeds=span.input_embeds,
            )
            for span in request_data.unit_embedding_spans
        ]
        if any(span.start < 0 for span in unit.embedding_spans):
            raise ValueError("embedding span precedes the native sequence")
        else:
            pass
        # note (Junnan Li): Placeholder ids under embedding spans do not identify their rows, so such a unit stays out of the radix tree.
        session_request.skip_radix_cache_insert = bool(unit.embedding_spans)
        request_data.session_embedding_spans = [
            *session.embedding_spans,
            *unit.embedding_spans,
        ]
        session_request.logprob_start_len = adapter_request.logprob_start_len
        session_request._omni_prompt_cache_key = getattr(
            adapter_request, "_omni_prompt_cache_key", None
        )  # noqa: leading-underscore
        request_data.req = session_request
        request_data.stage_payload = payload

    def drop_history(
        self, session: BridgeSession, native_session: Session, cut: HistoryCut
    ) -> None:
        """Drop the retained position ranges of cut from an idle session; later positions move down."""
        started = time.perf_counter()
        ranges = cut.ranges
        dropped_length = cut.length
        [last_node] = native_session.req_nodes.values()
        last_request = last_node.req
        assert ranges[-1][1] <= min(
            native_session.committed_origin_len,
            native_session.committed_unpadded_len,
            native_session.committed_fill_len,
        ), f"history cut {ranges} leaves the committed token ids"
        slot = self.bridge_scheduler.tree_cache.slots.get(native_session.session_id)
        # note (0xtoward): After an abort the slot is gone and the next unit re-prefills the shortened ids.
        if slot is not None and slot.kv.holds_kv:
            self.drop_retained_kv(slot.kv, cut)
        else:
            pass
        # note (0xtoward): The unpadded ids usually alias the origin ids, so each array is cut once.
        token_id_arrays = {
            id(token_ids): token_ids
            for token_ids in (
                last_request.origin_input_ids,
                last_request.origin_input_ids_unpadded,
                last_request.full_untruncated_fill_ids,
            )
        }
        for token_ids in token_id_arrays.values():
            for range_start, range_end in reversed(ranges):
                del token_ids[range_start:range_end]
        native_session.committed_origin_len -= dropped_length
        native_session.committed_unpadded_len -= dropped_length
        native_session.committed_fill_len -= dropped_length
        kept_spans: list[EmbeddingSpan] = []
        for span in session.embedding_spans:
            shift = 0
            is_dropped = False
            for range_start, range_end in ranges:
                assert not (
                    span.start < range_start < span.end
                    or span.start < range_end < span.end
                ), f"embedding span [{span.start}, {span.end}) crosses the history cut"
                if span.start >= range_end:
                    shift += range_end - range_start
                elif span.start >= range_start and span.end <= range_end:
                    is_dropped = True
                else:
                    pass
            if is_dropped:
                pass
            elif shift:
                kept_spans.append(
                    EmbeddingSpan(
                        start=span.start - shift,
                        end=span.end - shift,
                        input_embeds=span.input_embeds,
                    )
                )
            else:
                kept_spans.append(span)
        session.embedding_spans = kept_spans
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        else:
            pass
        logger.info(
            "Session %s: dropped history [%d, %d), %d tokens kept; xp ranges=%d dropped=%d ms=%.1f",
            native_session.session_id,
            ranges[0][0],
            ranges[-1][1],
            native_session.committed_origin_len,
            len(ranges),
            dropped_length,
            (time.perf_counter() - started) * 1000,
        )

    def drop_retained_kv(self, kv: SessionKV, cut: HistoryCut) -> None:
        """Move the KV kept after the cut down inside the session row and free the row tail."""
        tree_cache = self.bridge_scheduler.tree_cache
        ranges = cut.ranges
        assert (
            ranges[0][0] >= kv.cache_protected_len
        ), f"history cut at {ranges[0][0]} reaches the tree-owned prefix of {kv.cache_protected_len} tokens"
        assert (
            ranges[-1][1] <= kv.kv_committed_len
        ), f"history cut ends at {ranges[-1][1]}, past the committed KV of {kv.kv_committed_len} tokens"
        kept_length = kv.kv_committed_len - cut.length
        row = tree_cache.req_to_token_pool.req_to_token[kv.req_pool_idx]
        # (source start, source end, positions it moves down)
        segments: list[tuple[int, int, int]] = []
        cursor, distance = ranges[0][0], 0
        for range_start, range_end in ranges:
            if range_start > cursor:
                segments.append((cursor, range_start, distance))
            else:
                pass
            distance += range_end - range_start
            cursor = range_end
        if kv.kv_committed_len > cursor:
            segments.append((cursor, kv.kv_committed_len, distance))
        else:
            pass
        if segments:
            source_positions = torch.cat(
                [
                    torch.arange(start, end, device=row.device)
                    for start, end, _ in segments
                ]
            )
            distances: int | torch.Tensor = (
                segments[0][2]
                if len(segments) == 1
                else torch.cat(
                    [
                        torch.full(
                            (end - start,),
                            shift,
                            dtype=torch.int64,
                            device=row.device,
                        )
                        for start, end, shift in segments
                    ]
                )
            )
            source_slots = row[source_positions].to(torch.int64)
            target_slots = row[ranges[0][0] : kept_length].to(torch.int64)
            assert source_slots.numel() == target_slots.numel()
            kv_cache = self.bridge_scheduler.token_to_kv_pool_allocator.get_kvcache()
            # note (0xtoward): Rows keep their slots, so pages stay whole and captured graphs keep their buffers; only the KV moves.
            for layer_id in range(
                kv_cache.start_layer, kv_cache.start_layer + kv_cache.layer_num
            ):
                key_buffer = kv_cache.get_key_buffer(layer_id)
                value_buffer = kv_cache.get_value_buffer(layer_id)
                assert (
                    key_buffer.shape[0] == kv_cache.size + kv_cache.page_size
                ), "history cut needs one KV row per token slot"
                key_buffer.index_copy_(
                    0,
                    target_slots,
                    self.adapter.shift_key_positions(
                        key_buffer.index_select(0, source_slots), distances
                    ),
                )
                value_buffer.index_copy_(
                    0, target_slots, value_buffer.index_select(0, source_slots)
                )
        else:
            pass
        tail_start = ceil_align(kept_length, tree_cache.page_size)
        if tail_start < kv.kv_allocated_len:
            tree_cache.free_kv_row(kv, [(tail_start, kv.kv_allocated_len)])
        else:
            pass
        kv.kv_allocated_len = kept_length
        kv.kv_committed_len = kept_length

    def score_history_units(
        self, native_session: Session, bounds: list[tuple[int, int]], policy: str
    ) -> list[float]:
        """test only (xp): one score per retained unit [start, end) from its cached keys, every fourth layer.

        knorm is the mean key L2 norm; keydiff is the mean cosine similarity of the position-free keys to their mean.
        A higher score drops first.
        """
        slot = self.bridge_scheduler.tree_cache.slots.get(native_session.session_id)
        if not bounds or slot is None or not slot.kv.holds_kv:
            return [0.0] * len(bounds)
        else:
            pass
        first, last = bounds[0][0], bounds[-1][1]
        row = self.bridge_scheduler.tree_cache.req_to_token_pool.req_to_token[
            slot.kv.req_pool_idx
        ]
        slots = row[first:last].to(torch.int64)
        positions = torch.arange(first, last, device=row.device)
        kv_cache = self.bridge_scheduler.token_to_kv_pool_allocator.get_kvcache()
        layers = range(
            kv_cache.start_layer, kv_cache.start_layer + kv_cache.layer_num, 4
        )
        token_scores = torch.zeros(last - first, device=row.device, dtype=torch.float32)
        for layer_id in layers:
            keys = (
                kv_cache.get_key_buffer(layer_id)
                .index_select(0, slots)
                .to(torch.float32)
            )
            if policy == "knorm":
                token_scores += keys.norm(dim=-1).mean(dim=-1)
            else:
                keys = self.adapter.shift_key_positions(keys, positions)
                token_scores += torch.nn.functional.cosine_similarity(
                    keys, keys.mean(dim=0, keepdim=True), dim=-1
                ).mean(dim=-1)
        token_scores /= len(layers)
        return torch.stack(
            [token_scores[start - first : end - first].mean() for start, end in bounds]
        ).tolist()

    def release_append_unit(self, request_id: str) -> None:
        unit = self.units_by_request_id.pop(request_id, None)
        if unit is not None:
            streaming_session = self.bridge_scheduler.session_controller.get(
                unit.session_identity.id
            )
            if streaming_session is not None and unit.session_request is not None:
                streaming_session.abort_req()
            else:
                pass
            self.sessions[unit.session_identity.id].unit = None
        else:
            pass

    def cancel(self, request_id: str) -> None:
        unit = self.units_by_request_id[request_id]
        session_request = unit.session_request
        previous_cancelling_request_id = self.cancelling_request_id
        self.cancelling_request_id = request_id
        try:
            if not unit.is_enqueued:
                # note (Junnan Li): Before enqueue, release_append_unit must preserve the prior unit's KV.
                self.bridge_scheduler.abort(request_id)
                self.release_append_unit(request_id)
            else:
                assert (
                    session_request is not None
                ), f"enqueued session request {request_id} has no Req"
                is_queued = any(
                    queued_request is session_request
                    for queued_request in self.bridge_scheduler.waiting_queue
                )
                if is_queued:
                    # note (Junnan Li): Failed prefill admission may have restored the prior slot.
                    session_request.detach_kv()
                    session_request.session = None
                    self.bridge_scheduler.abort(request_id)
                    self.drain()
                else:
                    self.bridge_scheduler.abort(request_id)
                    self.drain()
                    session_request.finished_reason = FINISH_ABORT()
                    self.bridge_scheduler.release_request_kv_cache(session_request)
                    # note (Junnan Li): Batch selection must filter rows and tensors together.
                    if self.bridge_scheduler.chunked_req is session_request:
                        self.bridge_scheduler.chunked_req = None
                    else:
                        pass
                    if session_request.omni_data is not None:
                        self.bridge_scheduler.run_abort_callback(request_id)
                        session_request.omni_data = None
                    else:
                        pass
                self.release_append_unit(request_id)
        finally:
            self.cancelling_request_id = previous_cancelling_request_id

    def check_session_capacity(self, request_id: str) -> str | None:
        unit = self.units_by_request_id[request_id]
        session_request = unit.session_request
        assert (
            session_request is not None
        ), f"session capacity check {request_id} requires a session request"
        cache = self.bridge_scheduler.tree_cache
        slot = cache.slots.get(unit.session_identity.id)
        retained_kv_tokens = slot.kv.kv_allocated_len if slot is not None else 0
        free_request_slots = self.bridge_scheduler.req_to_token_pool.free_slots
        unallocated_request_count = sum(
            active_append_unit is not unit
            and active_append_unit.session_request is not None
            and not active_append_unit.session_request.kv.holds_kv
            and active_append_unit.session_identity.id not in cache.slots
            for active_append_unit in self.units_by_request_id.values()
        )
        if (
            not retained_kv_tokens
            and len(free_request_slots)
            <= unallocated_request_count
            + REQUEST_TO_TOKEN_SLOTS_RESERVED_FOR_RETAINED_KV
        ):
            capacity_message = (
                "session request-to-token slots exhausted "
                "(one slot kept for a session that already holds KV)"
            )
        else:
            required_kv_tokens = len(session_request.origin_input_ids) + int(
                session_request.sampling_params.max_new_tokens or 0
            )
            reserved_kv_tokens = 0
            for active_append_unit in self.units_by_request_id.values():
                if (
                    active_append_unit is unit
                    or active_append_unit.session_request is None
                ):
                    continue
                else:
                    active_session_request = active_append_unit.session_request
                    allocated_kv_tokens = active_session_request.kv.kv_allocated_len
                    reserved_kv_tokens += max(
                        0,
                        len(active_session_request.origin_input_ids)
                        + int(
                            active_session_request.sampling_params.max_new_tokens or 0
                        )
                        - allocated_kv_tokens,
                    )
            available_kv_tokens = (
                self.bridge_scheduler.token_to_kv_pool_allocator.available_size()
                + cache.evictable_size()
            )
            if (
                required_kv_tokens - retained_kv_tokens + reserved_kv_tokens
                > available_kv_tokens
            ):
                capacity_message = "session KV capacity exhausted"
            else:
                capacity_message = None
        return capacity_message

    def count_row_reusing_requests(
        self, waiting_queue: list[Req], free_request_rows: int
    ) -> int:
        """Count queued units that reuse their slot's row, up to the first request no free row is left for."""
        slots = self.bridge_scheduler.tree_cache.slots
        fresh_request_count = 0
        row_reusing_request_count = 0
        for queued_request in waiting_queue:
            session = queued_request.session
            if session is not None and session.streaming:
                slot = slots.get(session.session_id)
            else:
                slot = None
            # note (Junnan Li): An aborted unit does not take its slot's row.
            if (
                slot is not None
                and slot.kv.holds_kv
                and queued_request.to_finish is None
            ):
                row_reusing_request_count += 1
            elif fresh_request_count < free_request_rows:
                fresh_request_count += 1
            else:
                break
        return row_reusing_request_count

    def stream_messages(
        self,
        request_id: str,
        request_data: SGLangARRequestData,
        output: RequestOutput | None = None,
        *,
        should_flush: bool = False,
    ) -> Iterable[OutgoingMessage]:
        unit = self.units_by_request_id[request_id]
        if get_active_stage() != unit.stages[-1]:
            return
        else:
            if should_flush:
                chunks = self.adapter.flush(unit.session_identity, request_data)
            else:
                assert (
                    output is not None
                ), f"session stream {request_id} requires request output"
                chunks = self.adapter.stream(
                    unit.session_identity, request_data, output
                )
            for chunk in chunks:
                yield OutgoingMessage(
                    request_id=request_id,
                    type="stream",
                    data=chunk.to_dict(),
                    metadata={"modality": chunk.modality},
                )

    def complete(self, request_id: str) -> None:
        unit = self.units_by_request_id.pop(request_id, None)
        if unit is not None:
            session = self.sessions[unit.session_identity.id]
            session.unit = None
            session.embedding_spans.extend(unit.embedding_spans)
        else:
            pass

    def close_streaming_session(self, session: BridgeSession) -> None:
        if session.unit is not None:
            self.bridge_scheduler.abort(session.unit.request_id)
        else:
            pass
        self.release_streaming_session(session.session_identity.id)

    def close_session(self, session: BridgeSession) -> None:
        self.close_streaming_session(session)
        self.adapter.close(session.session_identity)
        self.sessions.pop(session.session_identity.id)

    def close_open_sessions(self) -> None:
        adapter_failures: dict[str, Exception] = {}
        for session in list(self.sessions.values()):
            self.close_streaming_session(session)
            session_id = session.session_identity.id
            try:
                self.adapter.close(session.session_identity)
            except Exception as adapter_error:
                adapter_failures[session_id] = adapter_error
            else:
                self.sessions.pop(session_id)
        if not adapter_failures:
            return
        else:
            failure_text = "; ".join(
                f"{session_id}: {adapter_error}"
                for session_id, adapter_error in adapter_failures.items()
            )
            raise RuntimeError(
                f"AR session adapter cleanup failed for {failure_text}"
            ) from next(iter(adapter_failures.values()))
