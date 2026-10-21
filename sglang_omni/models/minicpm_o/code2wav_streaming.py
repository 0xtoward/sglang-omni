# SPDX-License-Identifier: Apache-2.0
"""Vocode MiniCPM-o speech while the talker is still decoding it.

The offline speech pipeline handed code2wav a request's codec tokens only once the
talker finished, so a single request paid talker time plus vocoder time. With codec
tokens streamed per step, this scheduler vocodes each 25-token chunk as soon as its
lookahead arrives, on the duplex vocoder runtime's per-session streaming state, and
only the last chunk waits for the talker to finish.
"""

from __future__ import annotations

import logging
import os
import queue
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from sglang_omni.models.minicpm_o.components.code2wav import MiniCPMOCode2Wav
from sglang_omni.models.minicpm_o.components.tts_runtime import (
    CODEC_CHUNK_SIZE,
    MiniCPMOVocoderRuntime,
    SynthesisRequest,
)
from sglang_omni.models.minicpm_o.payload_types import MiniCPMOPipelineState
from sglang_omni.models.minicpm_o.routing import (
    CODE2WAV_STAGE,
    TALKER_STAGE,
    code2wav_reference_audio,
)
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.message import IncomingMessage, OutgoingMessage
from sglang_omni.utils.audio_payload import audio_waveform_payload

logger = logging.getLogger(__name__)

# Stream-done notices that arrive before their request's payload are remembered this long.
DONE_SEEN_MAX = 10000
DONE_SEEN_EVICT_TO = 5000


@dataclass(kw_only=True)
class StreamState:
    """One request's codec tokens, how many the vocoder has taken, and its PCM so far."""

    tokens: list[int] = field(default_factory=list)
    submitted: int = 0
    reference: bytes | None = None
    session_open: bool = False
    pcm: list[np.ndarray] = field(default_factory=list)
    leading_silence: int = 0
    payload: StagePayload | None = None
    done: bool = False


class MiniCPMOStreamingCode2WavScheduler:
    """Stream-aware code2wav stage; falls back to dense vocoding when no stream arrived."""

    def __init__(
        self,
        model: MiniCPMOCode2Wav,
        *,
        dense_vocode,
        max_open_sessions: int,
        max_state_bytes_per_session: int,
    ) -> None:
        self.inbox: queue.Queue[IncomingMessage] = queue.Queue()
        self.outbox: queue.Queue[OutgoingMessage] = queue.Queue()
        self.model = model
        self.dense_vocode = dense_vocode
        self.runtime = MiniCPMOVocoderRuntime(
            model,
            max_state_bytes_per_session=max_state_bytes_per_session,
            max_open_sessions=max_open_sessions,
        )
        self.default_reference = Path(model.default_prompt_wav).read_bytes()
        # note (0xtoward): the offline warm-up builds only the dense flow; chunk decode needs its own graphs.
        self.runtime.warm_up(self.default_reference)
        self.states: dict[str, StreamState] = {}
        self.done_seen: OrderedDict[str, None] = OrderedDict()
        self.is_running = False

    def warm_up_serving_thread(self) -> None:
        self.runtime.warm_up_vocoder()

    def start(self) -> None:
        self.is_running = True
        device_module = torch.get_device_module(self.runtime.token2wav.device)
        # note (0xtoward): the talker shares this process; vocoding on its own stream keeps the two from serializing.
        with device_module.stream(self.model.decode_stream):
            while self.is_running:
                busy = any(
                    state.session_open
                    and len(state.tokens) - state.submitted >= CODEC_CHUNK_SIZE
                    for state in self.states.values()
                )
                try:
                    msg = self.inbox.get(timeout=0.0 if busy else 0.05)
                except queue.Empty:
                    msg = None
                while msg is not None:
                    self.handle(msg)
                    try:
                        msg = self.inbox.get_nowait()
                    except queue.Empty:
                        msg = None
                self.vocode_ready_chunks()

    def stop(self) -> None:
        self.is_running = False

    def handle(self, msg: IncomingMessage) -> None:
        try:
            if msg.type == "new_request":
                self.on_new_request(msg.request_id, msg.data)
            elif msg.type == "stream_chunk":
                self.on_stream_chunk(msg.request_id, msg.data)
            elif msg.type == "stream_done":
                self.on_stream_done(msg.request_id)
            else:
                pass
        except Exception as exc:
            # One request fails; escaping start() would take the stage down.
            logger.exception(f"streaming code2wav failed request {msg.request_id}")
            self.abort(msg.request_id)
            self.outbox.put(OutgoingMessage(request_id=msg.request_id, type="error", data=exc))

    def abort(self, request_id: str) -> None:
        state = self.states.pop(request_id, None)
        self.done_seen.pop(request_id, None)
        if state is not None and state.session_open:
            self.runtime.close_session(request_id)
        else:
            pass

    def state(self, request_id: str) -> StreamState:
        state = self.states.get(request_id)
        if state is None:
            state = StreamState()
            self.states[request_id] = state
        else:
            pass
        return state

    def on_stream_chunk(self, request_id: str, chunk) -> None:
        state = self.state(request_id)
        metadata = chunk.metadata or {}
        if not state.session_open:
            reference = metadata.get("reference_audio") or self.default_reference
            self.runtime.open_session(request_id, reference_audio=reference)
            # The offline dense path starts without the duplex turn's silence prefix.
            self.runtime.sessions[request_id].pending_codec_token_ids = []
            state.session_open = True
        else:
            pass
        data = chunk.data
        state.tokens.append(int(data) if isinstance(data, int) else int(data.reshape(-1)[0].item()))

    def on_stream_done(self, request_id: str) -> None:
        state = self.states.get(request_id)
        if state is None:
            self.done_seen[request_id] = None
            if len(self.done_seen) > DONE_SEEN_MAX:
                for _ in range(len(self.done_seen) - DONE_SEEN_EVICT_TO):
                    self.done_seen.popitem(last=False)
            else:
                pass
            return
        else:
            pass
        state.done = True
        if state.payload is not None:
            self.finalize(request_id)
        else:
            pass

    def on_new_request(self, request_id: str, payload: StagePayload) -> None:
        state = self.state(request_id)
        state.payload = payload
        if request_id in self.done_seen:
            state.done = True
            self.done_seen.pop(request_id, None)
        else:
            pass
        # The payload carries every codec token, so it can finish the request even if
        # the stream-done notice is still in flight.
        self.finalize(request_id)

    def vocode_ready_chunks(self) -> None:
        """Hand the vocoder every request's whole new chunks, in one batch."""
        if os.environ.get("SGLANG_OMNI_MINICPMO_STREAM_DEFER") == "1":
            return
        else:
            pass
        ready = [
            (request_id, state)
            for request_id, state in self.states.items()
            if state.session_open
            and state.payload is None
            and len(state.tokens) - state.submitted >= CODEC_CHUNK_SIZE
        ]
        if not ready:
            return
        else:
            pass
        requests = []
        for request_id, state in ready:
            whole = (len(state.tokens) - state.submitted) // CODEC_CHUNK_SIZE * CODEC_CHUNK_SIZE
            requests.append(
                SynthesisRequest(
                    session_id=request_id,
                    codec_token_ids=state.tokens[state.submitted : state.submitted + whole],
                    is_turn_start=False,
                    end_of_turn=False,
                )
            )
            state.submitted += whole
        for index, waveform in self.runtime.synthesize_batch(requests):
            state = ready[index][1]
            if waveform is not None:
                if not state.pcm:
                    # note (0xtoward): a stream's first steady chunk starts with the HiFT overlap as silence.
                    state.leading_silence = self.runtime.token2wav.source_cache_len
                else:
                    pass
                state.pcm.append(waveform)
            else:
                pass

    def finalize(self, request_id: str) -> None:
        state = self.states.get(request_id)
        if state is None or state.payload is None:
            return
        else:
            pass
        payload = state.payload
        final = (
            MiniCPMOPipelineState.from_dict(payload.data)
            .engine_outputs[TALKER_STAGE]["codec_tokens"]
            .reshape(-1)
            .tolist()
        )
        if not state.session_open or final[: state.submitted] != state.tokens[: state.submitted]:
            if state.session_open:
                logger.warning(
                    f"streamed codec tokens of {request_id} disagree with the talker result; "
                    "vocoding it dense"
                )
            else:
                pass
            self.states.pop(request_id, None)
            if state.session_open:
                self.runtime.close_session(request_id)
            else:
                pass
            self.outbox.put(
                OutgoingMessage(request_id=request_id, type="result", data=self.dense_vocode([payload])[0])
            )
            return
        else:
            pass
        rest = final[state.submitted :]
        for _, waveform in self.runtime.synthesize_batch(
            [SynthesisRequest(session_id=request_id, codec_token_ids=rest, is_turn_start=False, end_of_turn=True)]
        ):
            if waveform is not None:
                state.pcm.append(waveform)
            else:
                pass
        self.states.pop(request_id, None)
        self.done_seen.pop(request_id, None)
        self.runtime.close_session(request_id)
        waveform = np.concatenate(state.pcm) if state.pcm else np.zeros(0, dtype=np.float32)
        waveform = waveform[state.leading_silence :]
        payload.data = dict(
            audio_waveform_payload(
                waveform,
                sample_rate=self.model.sample_rate,
                modality="audio",
                source_hint="MiniCPM-o",
            )
        )
        self.outbox.put(OutgoingMessage(request_id=request_id, type="result", data=payload))


def make_talker_codec_stream_builder(codec_eos_id: int):
    """Stream each sampled codec token to code2wav; the first carries the speaker reference."""
    started: OrderedDict[str, None] = OrderedDict()

    def build(request_id: str, req_data, req_output) -> list[OutgoingMessage]:
        if req_output.data is None:
            return []
        else:
            pass
        token_id = int(req_output.data)
        if token_id == int(codec_eos_id):
            started.pop(request_id, None)
            return []
        else:
            pass
        metadata: dict[str, object] = {"token_id": token_id}
        if request_id not in started:
            started[request_id] = None
            if len(started) > DONE_SEEN_MAX:
                for _ in range(len(started) - DONE_SEEN_EVICT_TO):
                    started.popitem(last=False)
            else:
                pass
            metadata["reference_audio"] = code2wav_reference_audio(req_data.stage_payload)
        else:
            pass
        return [
            OutgoingMessage(
                request_id=request_id,
                type="stream",
                # Stream transport carries tensors, not scalar ids.
                data=torch.tensor([token_id], dtype=torch.long),
                target=CODE2WAV_STAGE,
                metadata=metadata,
            )
        ]

    return build
