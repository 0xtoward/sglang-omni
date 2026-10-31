# SPDX-License-Identifier: Apache-2.0
"""Drama 3 vocoder: Fish codec decoding of sampled frames, streamed in fixed windows."""

from __future__ import annotations

import bisect
import collections
import time
from dataclasses import dataclass, field

import torch

from sglang_omni.models.fishaudio_drama3.request_builders import Drama3GenerationResult
from sglang_omni.models.fishaudio_s2_pro.fish_speech.models.dac.modded_dac import DAC
from sglang_omni.pipeline.stage.stream_queue import StreamItem
from sglang_omni.platforms import current_platform
from sglang_omni.platforms.device_graph import ReplayableGraph
from sglang_omni.proto.request import StagePayload
from sglang_omni.scheduling.audio_stream import (
    FinalAudioData,
    StreamVocoderState,
    WindowDecoder,
    build_stream_vocoder_chunk,
    finish_stream_window,
    stream_window_codes,
)
from sglang_omni.scheduling.audio_vocoder import StreamingVocoderScheduler
from sglang_omni.scheduling.message import OutgoingMessage


def generation_result_of(payload: StagePayload) -> Drama3GenerationResult:
    """The engine's result in a vocoder payload, which must hold an audio frame."""
    result = payload.data
    if not isinstance(result, Drama3GenerationResult):
        raise TypeError(
            f"Drama 3 vocoder payloads carry Drama3GenerationResult; got "
            f"{type(result).__name__}"
        )
    elif result.frames.shape[0] == 0:
        raise ValueError(
            f"Request {payload.request_id}: Drama 3 generated no audio frames"
        )
    else:
        return result


def non_streaming_decode_groups(
    frame_counts: list[int], max_padded_frames_per_decode: int | None
) -> list[list[int]]:
    if max_padded_frames_per_decode is None or (
        len(frame_counts) * max(frame_counts) <= max_padded_frames_per_decode
    ):
        return [list(range(len(frame_counts)))]
    else:
        groups: list[list[int]] = []
        for index in sorted(
            range(len(frame_counts)), key=frame_counts.__getitem__, reverse=True
        ):
            if (
                groups
                and (len(groups[-1]) + 1) * frame_counts[groups[-1][0]]
                <= max_padded_frames_per_decode
            ):
                groups[-1].append(index)
            else:
                groups.append([index])
        return groups


class WindowGraphDecoder:
    def __init__(
        self,
        decode: WindowDecoder,
        *,
        num_codebooks: int,
        frame_length: int,
        device: torch.device,
        window_lengths: tuple[int, ...],
        batch_sizes: tuple[int, ...] = (1,),
    ) -> None:
        graph_backend = current_platform.get_device_graph_backend(device)
        if graph_backend is None:
            raise RuntimeError(
                f"no device graph backend for {device}; set "
                f"stages.vocoder.factory.window_cuda_graphs to false to decode eagerly"
            )
        else:
            pass
        self.decode = decode
        self.frame_length = frame_length
        self.window_lengths = tuple(sorted(window_lengths))
        self.batch_sizes = tuple(sorted(batch_sizes))
        shapes = [
            (batch_size, window_length)
            for batch_size in reversed(self.batch_sizes)
            for window_length in reversed(self.window_lengths)
        ]
        self.static_indices: dict[tuple[int, int], torch.Tensor] = {}
        self.static_audio: dict[tuple[int, int], torch.Tensor] = {}
        self.graphs: dict[tuple[int, int], ReplayableGraph] = {}
        device_module = torch.get_device_module(device)
        current_stream = device_module.current_stream(device)
        capture_stream = device_module.Stream(device=device)
        capture_stream.wait_stream(current_stream)
        with torch.no_grad(), device_module.device(device):
            with device_module.stream(capture_stream):
                for batch_size, window_length in shapes:
                    decode(
                        torch.zeros(
                            (batch_size, num_codebooks, window_length),
                            dtype=torch.long,
                            device=device,
                        )
                    )
            capture_stream.synchronize()
            graph_pool = device_module.graph_pool_handle()
            for shape in shapes:
                batch_size, window_length = shape
                static_indices = torch.zeros(
                    (batch_size, num_codebooks, window_length),
                    dtype=torch.long,
                    device=device,
                )
                with graph_backend.capture(
                    pool=graph_pool, stream=capture_stream
                ) as graph:
                    self.static_audio[shape] = decode(static_indices)
                self.static_indices[shape] = static_indices
                self.graphs[shape] = graph
        current_stream.wait_stream(capture_stream)
        device_module.synchronize(device)

    def __call__(self, indices: torch.Tensor) -> torch.Tensor:
        """Decode [batch, codebooks, frames] codes; the codec is causal, so the zero frames that pad a window to a captured
        length leave its own samples unchanged."""
        row_count = int(indices.shape[0])
        frame_count = int(indices.shape[-1])
        if row_count > self.batch_sizes[-1] or frame_count > self.window_lengths[-1]:
            return self.decode(indices)
        else:
            pass
        shape = (
            self.batch_sizes[bisect.bisect_left(self.batch_sizes, row_count)],
            self.window_lengths[bisect.bisect_left(self.window_lengths, frame_count)],
        )
        static_indices = self.static_indices[shape]
        static_indices[:row_count, :, :frame_count].copy_(indices)
        static_indices[:row_count, :, frame_count:].zero_()
        static_indices[row_count:].zero_()
        self.graphs[shape].replay()
        return self.static_audio[shape][
            :row_count, ..., : frame_count * self.frame_length
        ].clone()


@dataclass(kw_only=True)
class PendingStreamFrames:
    """Frames a stream sent that are not decoded yet, and when each complete window among them became ready."""

    frames: list[torch.Tensor] = field(default_factory=list)
    frame_count: int = 0
    ready_times: collections.deque[float] = field(default_factory=collections.deque)
    # When the stream ended; its tail and final result go out after its complete windows.
    done_time: float | None = None


class Drama3VocoderScheduler(StreamingVocoderScheduler):
    """Decode complete windows regardless of incoming frame chunk sizes.

    With first_window_priority, chunks are only queued on arrival and complete windows decode one per step: every stream's
    first window before any later window, each group in the order its windows became complete. A stream's windows keep
    their boundaries and order, so its audio is the same; only the order across streams changes.
    """

    def __init__(
        self,
        codec: DAC,
        *,
        device: str,
        stream_stride: int,
        stream_followup_stride_ramp: tuple[int, ...],
        stream_followup_stride: int,
        stream_overlap_tokens: int,
        stream_crossfade_samples: int,
        max_batch_size: int,
        max_batch_wait_ms: int,
        decode_window: WindowDecoder | None = None,
        max_padded_frames_per_decode: int | None = None,
        first_window_priority: bool = False,
        stream_window_batch_size: int = 1,
    ) -> None:
        if not all(stride > 0 for stride in stream_followup_stride_ramp):
            raise ValueError(
                f"stream_followup_stride_ramp entries must be > 0; got "
                f"{stream_followup_stride_ramp}"
            )
        elif stream_window_batch_size > 1 and not first_window_priority:
            raise ValueError(
                "stream_window_batch_size > 1 decodes queued windows together, so it needs first_window_priority"
            )
        else:
            pass
        super().__init__(
            codec,
            device=device,
            stream_stride=stream_stride,
            stream_followup_stride=stream_followup_stride,
            stream_overlap_tokens=stream_overlap_tokens,
            stream_crossfade_samples=stream_crossfade_samples,
            max_batch_size=max_batch_size,
            max_batch_wait_ms=max_batch_wait_ms,
            decode_window=decode_window,
        )
        self.stream_followup_stride_ramp = tuple(stream_followup_stride_ramp)
        self.stream_window_counts: dict[str, int] = {}
        self.max_padded_frames_per_decode = max_padded_frames_per_decode
        self.first_window_priority = first_window_priority
        self.stream_window_batch_size = stream_window_batch_size
        self.pending_frames: dict[str, PendingStreamFrames] = {}
        # note (0xtoward): streams with a complete window to decode, so the per-message readiness check stays O(1).
        self.ready_request_ids: set[str] = set()

    def stride_after_window(self, window_count: int) -> int:
        if window_count < len(self.stream_followup_stride_ramp):
            return self.stream_followup_stride_ramp[window_count]
        else:
            return self.stream_followup_stride

    def followup_stride(self, request_id: str) -> int:
        return self.stride_after_window(self.stream_window_counts.get(request_id, 0))

    def window_end_frame(self, window_count: int) -> int:
        """Frames a stream has sent when its window number window_count (from 0) is complete."""
        end_frame = self.stream_stride
        for index in range(window_count):
            end_frame += self.stride_after_window(index)
        return end_frame

    def queue_frames(self, request_id: str, frame_codes: torch.Tensor) -> None:
        pending = self.pending_frames.setdefault(request_id, PendingStreamFrames())
        pending.frames.append(frame_codes)
        pending.frame_count += int(frame_codes.shape[1])
        state = self.stream_states.get(request_id)
        received_frame_count = (
            0 if state is None else state.total_tokens
        ) + pending.frame_count
        window_count = self.stream_window_counts.get(request_id, 0) + len(
            pending.ready_times
        )
        now = time.monotonic()
        while received_frame_count >= self.window_end_frame(window_count):
            pending.ready_times.append(now)
            self.ready_request_ids.add(request_id)
            window_count += 1

    def take_pending_frames(
        self, pending: PendingStreamFrames, frame_count: int
    ) -> torch.Tensor:
        frames = (
            torch.cat(pending.frames, dim=1)
            if len(pending.frames) > 1
            else pending.frames[0]
        )
        pending.frames = (
            [frames[:, frame_count:]] if frames.shape[1] > frame_count else []
        )
        pending.frame_count -= frame_count
        return frames[:, :frame_count]

    def has_ready_work(self) -> bool:
        return bool(self.ready_request_ids)

    def run_ready_step(self) -> None:
        """Decode the next complete windows, up to stream_window_batch_size streams at once: first windows first, then the
        windows that became complete earliest."""
        with self.state_lock:
            ranked = sorted(
                (
                    self.stream_window_counts.get(request_id, 0) > 0,
                    (
                        self.pending_frames[request_id].ready_times[0]
                        if self.pending_frames[request_id].ready_times
                        else self.pending_frames[request_id].done_time
                    ),
                    request_id,
                )
                for request_id in self.ready_request_ids
            )
            if ranked and not self.pending_frames[ranked[0][2]].ready_times:
                self.finish_queued_stream(ranked[0][2])
                return
            else:
                pass
            windows: list[tuple[str, torch.Tensor]] = []
            for _, _, request_id in ranked:
                if len(windows) == self.stream_window_batch_size:
                    break
                elif not self.pending_frames[request_id].ready_times:
                    continue
                else:
                    pass
                pending = self.pending_frames[request_id]
                state = self.stream_states.get(request_id)
                window_frame_count = self.window_end_frame(
                    self.stream_window_counts.get(request_id, 0)
                ) - (0 if state is None else state.total_tokens)
                pending.ready_times.popleft()
                if not pending.ready_times and pending.done_time is None:
                    self.ready_request_ids.discard(request_id)
                else:
                    pass
                windows.append(
                    (request_id, self.take_pending_frames(pending, window_frame_count))
                )
            if not windows:
                return
            elif len(windows) == 1:
                request_id, frame_codes = windows[0]
                try:
                    messages = self.stream_frames(request_id, frame_codes)
                except Exception as exc:
                    self.emit_error(request_id, exc)
                    self.abort(request_id)
                    return
            else:
                try:
                    messages = self.decode_windows_together(windows)
                except Exception as exc:
                    for request_id, _ in windows:
                        self.emit_error(request_id, exc)
                        self.abort(request_id)
                    return
            for message in messages:
                if not self.is_aborted(message.request_id):
                    self.outbox.put(message)
                else:
                    pass

    def decode_windows_together(
        self, windows: list[tuple[str, torch.Tensor]]
    ) -> list[OutgoingMessage]:
        """What stream_frames does for one complete window per stream, with the windows padded to one length and decoded
        in one call; the codec is causal, so each window's samples do not see the padding.
        """
        prepared = []
        for request_id, frame_codes in windows:
            state = self.stream_states.setdefault(request_id, StreamVocoderState())
            state.codes.append(
                frame_codes.detach().to(
                    device=self.device, dtype=torch.long, non_blocking=True
                )
            )
            state.total_tokens += int(frame_codes.shape[1])
            codebook_codes, overlap_token_count = stream_window_codes(
                state,
                device=self.device,
                stream_overlap_tokens=self.stream_overlap_tokens,
            )
            prepared.append((request_id, state, codebook_codes, overlap_token_count))
        frame_counts = [int(codes.shape[-1]) for _, _, codes, _ in prepared]
        batch = torch.zeros(
            (len(prepared), prepared[0][2].shape[0], max(frame_counts)),
            dtype=torch.long,
            device=self.device,
        )
        for row, (_, _, codes, _) in enumerate(prepared):
            batch[row, :, : codes.shape[-1]] = codes
        with torch.no_grad():
            audio = (
                self.codec.from_indices
                if self.decode_window is None
                else self.decode_window
            )(batch)
        messages = []
        for row, (request_id, state, codes, overlap_token_count) in enumerate(prepared):
            audio_chunk = finish_stream_window(
                state,
                audio[row, 0, : frame_counts[row] * self.codec.frame_length].float(),
                overlap_token_count=overlap_token_count,
                codec=self.codec,
                stream_overlap_tokens=self.stream_overlap_tokens,
                stream_crossfade_samples=self.stream_crossfade_samples,
                is_final=False,
            )
            state.next_vocode_tokens = state.total_tokens + self.followup_stride(
                request_id
            )
            self.stream_window_counts[request_id] = (
                self.stream_window_counts.get(request_id, 0) + 1
            )
            if audio_chunk is not None:
                messages.append(
                    OutgoingMessage(
                        request_id=request_id,
                        type="stream",
                        data=audio_chunk,
                        metadata={"modality": "audio"},
                    )
                )
            else:
                pass
        return messages

    def stream_frames(
        self, request_id: str, frame_codes: torch.Tensor
    ) -> list[OutgoingMessage]:
        """Audio chunks for frames [1 + num_codebooks, frames] given in order."""
        state = self.stream_states.setdefault(request_id, StreamVocoderState())
        messages: list[OutgoingMessage] = []
        incoming_frame_count = frame_codes.shape[1]
        first_frame_index = 0
        while first_frame_index < incoming_frame_count:
            window_end_frame = state.next_vocode_tokens or self.stream_stride
            remaining_window_frame_count = window_end_frame - state.total_tokens
            if incoming_frame_count - first_frame_index < remaining_window_frame_count:
                end_frame_index = incoming_frame_count
                is_window_end = False
            else:
                end_frame_index = first_frame_index + remaining_window_frame_count
                is_window_end = True
            audio_chunk = build_stream_vocoder_chunk(
                state,
                frame_codes[:, first_frame_index:end_frame_index],
                codec=self.codec,
                device=self.device,
                stream_stride=self.stream_stride,
                stream_followup_stride=self.followup_stride(request_id),
                stream_overlap_tokens=self.stream_overlap_tokens,
                stream_crossfade_samples=self.stream_crossfade_samples,
                decode_window=self.decode_window,
            )
            if is_window_end:
                self.stream_window_counts[request_id] = (
                    self.stream_window_counts.get(request_id, 0) + 1
                )
            else:
                pass
            if audio_chunk is not None:
                messages.append(
                    OutgoingMessage(
                        request_id=request_id,
                        type="stream",
                        data=audio_chunk,
                        metadata={"modality": "audio"},
                    )
                )
            else:
                pass
            first_frame_index = end_frame_index
        return messages

    def on_stream_chunk(
        self, request_id: str, chunk: StreamItem
    ) -> list[OutgoingMessage]:
        frame_codes = chunk.data
        if not isinstance(frame_codes, torch.Tensor):
            raise TypeError(
                f"Drama 3 stream chunk for {request_id!r} must carry a torch.Tensor, "
                f"got {type(frame_codes).__name__}"
            )
        elif self.first_window_priority:
            self.queue_frames(request_id, frame_codes)
            return []
        else:
            return self.stream_frames(request_id, frame_codes)

    def on_stream_done(self, request_id: str) -> list[OutgoingMessage] | None:
        """Stream the frames no chunk carried, skipping those already streamed, then
        flush the tail and send the final result. With first_window_priority, the frames are queued and the stream
        completes in a later step, after its complete windows."""
        frames = generation_result_of(self.payloads[request_id]).frames
        pending = self.pending_frames.get(request_id)
        streamed_frame_count = self.stream_states.setdefault(
            request_id, StreamVocoderState()
        ).total_tokens + (0 if pending is None else pending.frame_count)
        if streamed_frame_count > frames.shape[0]:
            raise RuntimeError(
                f"Request {request_id}: {streamed_frame_count} frames streamed but "
                f"the result holds {frames.shape[0]}"
            )
        elif self.first_window_priority:
            # note (0xtoward): decoding the rest here, in message order, would let streams that finish generating early
            # jump ahead of every queued first window.
            if streamed_frame_count < frames.shape[0]:
                self.queue_frames(request_id, frames[streamed_frame_count:].T)
            else:
                pass
            self.pending_frames.setdefault(
                request_id, PendingStreamFrames()
            ).done_time = time.monotonic()
            self.ready_request_ids.add(request_id)
            return None
        else:
            return self.stream_frames(
                request_id, frames[streamed_frame_count:].T
            ) + super().on_stream_done(request_id)

    def finish_queued_stream(self, request_id: str) -> None:
        """Decode a finished stream's tail and send its final result, once its complete windows are decoded."""
        pending = self.pending_frames.pop(request_id)
        self.ready_request_ids.discard(request_id)
        try:
            messages = (
                self.stream_frames(
                    request_id, self.take_pending_frames(pending, pending.frame_count)
                )
                if pending.frame_count > 0
                else []
            ) + super().on_stream_done(request_id)
        except Exception as exc:
            self.emit_error(request_id, exc)
            self.abort(request_id)
            return
        self.complete_stream_request(request_id, messages)

    def validate_non_streaming_payload(self, payload: StagePayload) -> None:
        generation_result_of(payload)

    def clear_stream_state(self, request_id: str) -> None:
        super().clear_stream_state(request_id)
        self.stream_window_counts.pop(request_id, None)
        self.pending_frames.pop(request_id, None)
        self.ready_request_ids.discard(request_id)

    def final_result_data(self, payload: StagePayload) -> FinalAudioData:
        result = generation_result_of(payload)
        completion_tokens = int(result.frames.shape[0])
        final_data: FinalAudioData = {
            "modality": "audio",
            "sample_rate": int(self.codec.sample_rate),
            "usage": {
                "prompt_tokens": result.prompt_row_count,
                "completion_tokens": completion_tokens,
                "total_tokens": result.prompt_row_count + completion_tokens,
            },
        }
        if result.finish_reason is not None:
            final_data["finish_reason"] = result.finish_reason
        else:
            pass
        return final_data

    def vocode_payloads(self, payloads: list[StagePayload]) -> list[StagePayload]:
        codebook_codes = [
            generation_result_of(payload).frames[:, 1:].T for payload in payloads
        ]
        audio_by_index: dict[int, torch.Tensor] = {}
        for group in non_streaming_decode_groups(
            [int(codes.shape[-1]) for codes in codebook_codes],
            self.max_padded_frames_per_decode,
        ):
            audio_by_index.update(
                zip(
                    group,
                    self.decode_codebook_codes([codebook_codes[i] for i in group]),
                    strict=True,
                )
            )
        audios = [audio_by_index[index] for index in range(len(payloads))]
        return [
            StagePayload(
                request_id=payload.request_id,
                request=payload.request,
                data={**self.final_result_data(payload), "audio_data": audio.tolist()},
            )
            for payload, audio in zip(payloads, audios, strict=True)
        ]
