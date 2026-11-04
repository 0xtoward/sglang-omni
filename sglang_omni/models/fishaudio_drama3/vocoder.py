# SPDX-License-Identifier: Apache-2.0
"""Drama 3 vocoder: Fish codec decoding of sampled frames, streamed in fixed windows."""

from __future__ import annotations

import bisect
import collections
import itertools
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
        max_padded_frames_per_decode: int | None = None,
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
            if max_padded_frames_per_decode is None
            or batch_size * window_length <= max_padded_frames_per_decode
        ]
        self.static_indices: dict[tuple[int, int], torch.Tensor] = {}
        self.static_audio: dict[tuple[int, int], torch.Tensor] = {}
        self.graphs: dict[tuple[int, int], ReplayableGraph] = {}
        if not shapes:
            return
        else:
            pass
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

    def graph_shape(self, row_count: int, frame_count: int) -> tuple[int, int] | None:
        if (
            not self.graphs
            or row_count > self.batch_sizes[-1]
            or frame_count > self.window_lengths[-1]
        ):
            return None
        else:
            shape = (
                self.batch_sizes[bisect.bisect_left(self.batch_sizes, row_count)],
                self.window_lengths[
                    bisect.bisect_left(self.window_lengths, frame_count)
                ],
            )
            return shape if shape in self.graphs else None

    def __call__(self, indices: torch.Tensor) -> torch.Tensor:
        """Decode causal codec windows, trimming the graph padding from the output."""
        row_count = int(indices.shape[0])
        frame_count = int(indices.shape[-1])
        shape = self.graph_shape(row_count, frame_count)
        if shape is None:
            return self.decode(indices)
        else:
            pass
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
    ready_times: collections.deque[tuple[float, int]] = field(
        default_factory=collections.deque
    )
    done_time: tuple[float, int] | None = None


class Drama3VocoderScheduler(StreamingVocoderScheduler):
    """Decode consecutive equal-length ready windows in FIFO order."""

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
        stream_window_batching: bool = False,
    ) -> None:
        if not all(stride > 0 for stride in stream_followup_stride_ramp):
            raise ValueError(
                f"stream_followup_stride_ramp entries must be > 0; got "
                f"{stream_followup_stride_ramp}"
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
        self.stream_window_batching = stream_window_batching
        self.pending_frames: dict[str, PendingStreamFrames] = {}
        self.ready_sequence = itertools.count()
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
        ready_time = (time.monotonic(), next(self.ready_sequence))
        while received_frame_count >= self.window_end_frame(window_count):
            pending.ready_times.append(ready_time)
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

    def ready_order(self, request_id: str) -> tuple[float, int]:
        pending = self.pending_frames[request_id]
        ready_time = (
            pending.ready_times[0] if pending.ready_times else pending.done_time
        )
        assert ready_time is not None
        return ready_time

    def select_ready_batch(
        self, ranked_request_ids: list[str], *, is_final: bool
    ) -> list[tuple[str, int]]:
        selected: list[tuple[str, int]] = []
        window_frame_count = 0
        limit = 8 if self.stream_window_batching and not is_final else 1
        for request_id in ranked_request_ids[:limit]:
            pending = self.pending_frames[request_id]
            if is_final == bool(pending.ready_times):
                break
            else:
                pass
            state = self.stream_states.get(request_id)
            total_tokens = 0 if state is None else state.total_tokens
            if is_final:
                new_frame_count = pending.frame_count
                if new_frame_count == 0:
                    return selected or [(request_id, 0)]
                else:
                    pass
            else:
                window_end = (
                    self.stream_stride
                    if state is None
                    else state.next_vocode_tokens or self.stream_stride
                )
                new_frame_count = window_end - total_tokens
            overlap_frame_count = (
                0
                if state is None
                else min(
                    self.stream_overlap_tokens,
                    state.last_vocode_tokens - state.code_start_token,
                )
            )
            candidate_frame_count = new_frame_count + overlap_frame_count
            if selected and candidate_frame_count != window_frame_count:
                break
            else:
                pass
            row_count = len(selected) + 1
            graph_shape = (
                self.decode_window.graph_shape(row_count, candidate_frame_count)
                if isinstance(self.decode_window, WindowGraphDecoder)
                else None
            )
            padded_frame_count = (
                row_count * candidate_frame_count
                if graph_shape is None
                else graph_shape[0] * graph_shape[1]
            )
            budget = self.max_padded_frames_per_decode
            if budget is not None and padded_frame_count > budget:
                if not selected and not self.is_aborted(request_id):
                    self.emit_error(
                        request_id,
                        ValueError(
                            f"Streaming window requires {padded_frame_count} padded frames "
                            f"but the vocoder budget is {budget}"
                        ),
                    )
                    self.abort(request_id)
                else:
                    pass
                break
            else:
                pass
            selected.append((request_id, new_frame_count))
            window_frame_count = candidate_frame_count
        return selected

    def run_ready_step(self) -> None:
        with self.state_lock:
            ranked_request_ids = sorted(
                (
                    request_id
                    for request_id in self.ready_request_ids
                    if not self.is_aborted(request_id)
                ),
                key=self.ready_order,
            )
            if not ranked_request_ids:
                return
            else:
                pass
            is_final = not self.pending_frames[ranked_request_ids[0]].ready_times
            selected = self.select_ready_batch(ranked_request_ids, is_final=is_final)
            if not selected:
                return
            else:
                pass
            try:
                windows = []
                for request_id, frame_count in selected:
                    pending = self.pending_frames[request_id]
                    if not is_final:
                        pending.ready_times.popleft()
                        if not pending.ready_times and pending.done_time is None:
                            self.ready_request_ids.discard(request_id)
                        else:
                            pass
                    else:
                        pass
                    if frame_count > 0:
                        windows.append(
                            (request_id, self.take_pending_frames(pending, frame_count))
                        )
                    else:
                        pass

                if is_final:
                    request_id = selected[0][0]
                    messages = self.stream_frames(*windows[0]) if windows else []
                    messages.extend(super().on_stream_done(request_id))
                    self.complete_stream_request(request_id, messages)
                    return
                elif len(windows) > 1:
                    messages = self.decode_windows_together(windows)
                elif windows:
                    messages = self.stream_frames(*windows[0])
                else:
                    messages = []

            except Exception as exc:
                for request_id, _ in selected:
                    if not self.is_aborted(request_id):
                        self.emit_error(request_id, exc)
                        self.abort(request_id)
                    else:
                        pass
                return

            for message in messages:
                if not self.is_aborted(message.request_id):
                    self.outbox.put(message)
                else:
                    pass

    def decode_windows_together(
        self, windows: list[tuple[str, torch.Tensor]]
    ) -> list[OutgoingMessage]:
        """Decode one equal-length complete window per stream in one codec call."""
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
        batch = torch.stack([codes for _, _, codes, _ in prepared])
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
                audio[row, 0].float(),
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
        elif frame_codes.ndim != 2:
            raise ValueError(
                f"Drama 3 stream chunk for {request_id!r} must carry a 2D frame tensor"
            )
        else:
            if self.stream_window_batching:
                self.queue_frames(request_id, frame_codes)
                return []
            else:
                return self.stream_frames(request_id, frame_codes)

    def on_stream_done(self, request_id: str) -> list[OutgoingMessage] | None:
        """Queue any frames missing from stream chunks and mark the stream complete."""
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
        else:
            pass
        if not self.stream_window_batching:
            return self.stream_frames(
                request_id, frames[streamed_frame_count:].T
            ) + super().on_stream_done(request_id)
        elif streamed_frame_count < frames.shape[0]:
            self.queue_frames(request_id, frames[streamed_frame_count:].T)
        else:
            pass
        self.pending_frames.setdefault(request_id, PendingStreamFrames()).done_time = (
            time.monotonic(),
            next(self.ready_sequence),
        )
        self.ready_request_ids.add(request_id)

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
