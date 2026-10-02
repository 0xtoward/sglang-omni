# SPDX-License-Identifier: Apache-2.0
"""Incremental decoding for the causal AudioVAE decoder: per-slot stage contexts, not a fixed window."""

from __future__ import annotations

import itertools
import logging
import math
import operator
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import torch
import torch.nn.functional as F

from sglang_omni.utils.cuda_staging import indices_to_device

if TYPE_CHECKING:
    from dots_tts.modules.vocoder.vocoder_inference import VocoderInference
else:
    pass

logger = logging.getLogger(__name__)

DecodeKey = tuple[Literal["stream", "window"], int, int]


@dataclass(kw_only=True)
class CapturedDecode:
    graph: torch.cuda.CUDAGraph
    inputs: tuple[torch.Tensor, ...]
    output: torch.Tensor


class StreamingDecoder:
    """Decode only the new frames once a slot holds every stage's receptive field of history.

    Per slot it keeps the last decoder-input frames of the centred conv_pre, the last input sample
    of each causal transposed conv and the last context samples entering each upsampling stage.
    Re-running a stage on [context | new] gives the full-sequence output for the new samples, since
    every op after the stage input is causal within that context. Slots with less history decode
    their left-aligned window as before and record the contexts from it.
    """

    def __init__(
        self,
        inference: VocoderInference,
        *,
        num_slots: int,
        max_batch_size: int,
        stream_frames: list[int],
        window_frames: list[int],
        cudnn_benchmark: bool = True,
        capture_graphs: bool = True,
    ) -> None:
        decoder = inference.vocoder.decoder
        self.decoder = decoder
        self.lookahead = int(
            inference._decoder_stream_lookahead()
        )  # noqa: leading-underscore  # upstream spelling
        self.pre_context = int(decoder.conv_pre.kernel_size[0]) - 1
        if self.pre_context != 2 * self.lookahead:
            raise ValueError(
                f"conv_pre kernel {self.pre_context + 1} is not centred on lookahead {self.lookahead}"
            )
        else:
            pass
        self.num_kernels = int(decoder.num_kernels)
        if any(type(block).__name__ != "AMPBlock1" for block in decoder.resblocks):
            raise ValueError("Streaming decoder expects AMPBlock1 resblocks")
        else:
            pass
        # note (0xtoward): dots' causal ConvTranspose1d overwrites stride with a plain int.
        self.strides = [
            int(
                stage[0].stride[0]
                if isinstance(stage[0].stride, tuple)
                else stage[0].stride
            )
            for stage in decoder.ups
        ]
        self.factors = list(itertools.accumulate(self.strides, operator.mul))
        self.contexts = []
        last_stage = len(self.strides) - 1
        for stage in range(len(self.strides)):
            blocks = decoder.resblocks[
                stage * self.num_kernels : (stage + 1) * self.num_kernels
            ]
            context = max(
                int(inference._ampblock_left_context(block)) for block in blocks
            )  # noqa: leading-underscore  # upstream spelling
            if stage == last_stage:
                context += int(
                    inference._activation_left_context(decoder.activation_post)
                )  # noqa: leading-underscore  # upstream spelling
                context += int(
                    inference._conv1d_left_context(decoder.conv_post)
                )  # noqa: leading-underscore  # upstream spelling
            else:
                pass
            self.contexts.append(context)
        self.warm_frames = max(
            [1]
            + [
                math.ceil(context / factor)
                for context, factor in zip(self.contexts, self.factors)
            ]
        )
        self.use_tanh = bool(decoder.h.get("use_tanh_at_final", True))
        weight = decoder.conv_pre.weight
        self.device = weight.device
        self.dtype = weight.dtype
        self.latent_channels = int(decoder.conv_pre.in_channels)
        with torch.no_grad():
            self.pre_state = weight.new_zeros(
                num_slots, self.latent_channels, self.pre_context
            )
            self.up_states = [
                weight.new_zeros(num_slots, int(stage[0].in_channels), 1)
                for stage in decoder.ups
            ]
            self.stage_states = [
                weight.new_zeros(num_slots, int(stage[0].out_channels), context)
                for stage, context in zip(decoder.ups, self.contexts)
            ]
        self.window_frames = sorted(window_frames)
        self.graphs: dict[DecodeKey, CapturedDecode] = {}
        self.replay_calls = 0
        self.eager_calls = 0
        if capture_graphs and self.device.type == "cuda":
            previous_cudnn_benchmark = torch.backends.cudnn.benchmark
            torch.backends.cudnn.benchmark = cudnn_benchmark
            try:
                self.capture(
                    max_batch_size, sorted(set(stream_frames) | {self.lookahead})
                )
            finally:
                torch.backends.cudnn.benchmark = previous_cudnn_benchmark
        else:
            pass
        logger.info(
            f"Streaming decoder ready: contexts={self.contexts} factors={self.factors} "
            f"warm_frames={self.warm_frames} graphs={len(self.graphs)}"
        )

    def run_stage(self, index: int, value: torch.Tensor) -> torch.Tensor:
        total = None
        for block in self.decoder.resblocks[
            index * self.num_kernels : (index + 1) * self.num_kernels
        ]:
            output = run_block(block, value)
            total = output if total is None else total + output
        value = total / self.num_kernels
        if index == len(self.strides) - 1:
            value = causal_conv(
                self.decoder.conv_post, self.decoder.activation_post(value)
            )
            value = (
                torch.tanh(value)
                if self.use_tanh
                else torch.clamp(value, min=-1.0, max=1.0)
            )
        else:
            pass
        return value

    @staticmethod
    def record(
        buffer: torch.Tensor,
        value: torch.Tensor,
        slot_index: torch.Tensor,
        end: torch.Tensor,
    ) -> None:
        """Store value[..., end - width:end] per row; positions before the stream start are zeros."""
        width = buffer.shape[-1]
        positions = end.unsqueeze(1) - width + torch.arange(width, device=value.device)
        picked = value.gather(
            -1, positions.clamp(min=0).unsqueeze(1).expand(-1, value.shape[1], -1)
        )
        buffer[slot_index] = picked * (positions >= 0).unsqueeze(1).to(picked.dtype)

    def window_forward(
        self,
        window: torch.Tensor,
        slot_index: torch.Tensor,
        stable: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        """Decode left-aligned windows [B, C, L] and record contexts ending at the stable frames."""
        self.record(self.pre_state, window, slot_index, valid)
        value = self.decoder.conv_pre(window)
        previous_factor = 1
        for index, upsample in enumerate(self.decoder.ups):
            self.record(
                self.up_states[index], value, slot_index, stable * previous_factor
            )
            value = upsample[0](value)
            self.record(
                self.stage_states[index],
                value,
                slot_index,
                stable * self.factors[index],
            )
            value = self.run_stage(index, value)
            previous_factor = self.factors[index]
        return value

    def stream_forward(
        self, frames: torch.Tensor, slot_index: torch.Tensor
    ) -> torch.Tensor:
        """Decode [B, C, n] new decoder-input frames into n * hop samples ending lookahead frames back."""
        joined = torch.cat([self.pre_state.index_select(0, slot_index), frames], dim=-1)
        self.pre_state[slot_index] = joined[..., -self.pre_context :]
        value = F.conv1d(
            joined, self.decoder.conv_pre.weight, self.decoder.conv_pre.bias
        )
        for index, upsample in enumerate(self.decoder.ups):
            joined = torch.cat(
                [self.up_states[index].index_select(0, slot_index), value], dim=-1
            )
            self.up_states[index][slot_index] = joined[..., -1:]
            value = upsample[0](joined)[..., self.strides[index] :]
            joined = torch.cat(
                [self.stage_states[index].index_select(0, slot_index), value], dim=-1
            )
            self.stage_states[index][slot_index] = joined[..., -self.contexts[index] :]
            value = self.run_stage(index, joined)[..., -value.shape[-1] :]
        return value

    def window_bucket(self, frames: int) -> int | None:
        for bucket in self.window_frames:
            if bucket >= frames:
                return bucket
            else:
                pass
        return None

    @torch.no_grad()
    def capture(self, max_batch_size: int, stream_frames: list[int]) -> None:
        started_seconds = time.perf_counter()
        generator = torch.Generator(device=self.device).manual_seed(20261003)
        capture_stream = torch.cuda.Stream()
        pool = torch.cuda.graph_pool_handle()
        plans = []
        for batch_size in range(1, max_batch_size + 1):
            slot_index = torch.arange(batch_size, device=self.device)
            for frames in stream_frames:
                inputs = (
                    torch.randn(
                        batch_size,
                        self.latent_channels,
                        frames,
                        device=self.device,
                        generator=generator,
                    )
                    * 0.01,
                    slot_index.clone(),
                )
                plans.append((("stream", batch_size, frames), inputs))
            for frames in self.window_frames:
                inputs = (
                    torch.randn(
                        batch_size,
                        self.latent_channels,
                        frames,
                        device=self.device,
                        generator=generator,
                    )
                    * 0.01,
                    slot_index.clone(),
                    torch.full(
                        (batch_size,),
                        frames - self.lookahead,
                        device=self.device,
                        dtype=torch.long,
                    ),
                    torch.full(
                        (batch_size,), frames, device=self.device, dtype=torch.long
                    ),
                )
                plans.append((("window", batch_size, frames), inputs))
        buffers = [self.pre_state, *self.up_states, *self.stage_states]
        for key, inputs in plans:
            capture_stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(capture_stream):
                for _ in range(2):
                    self.forward(key, inputs)
                snapshot = [buffer.clone() for buffer in buffers]
                expected = self.forward(key, inputs)
            torch.cuda.current_stream().wait_stream(capture_stream)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=pool, stream=capture_stream):
                output = self.forward(key, inputs)
            for buffer, saved in zip(buffers, snapshot):
                buffer.copy_(saved)
            graph.replay()
            torch.cuda.synchronize()
            relative_error = (
                (output - expected).norm() / expected.norm().clamp_min(1e-12)
            ).item()
            if not math.isfinite(relative_error) or relative_error > 1e-3:
                raise RuntimeError(
                    f"Streaming decoder graph {key} replay differs from eager: {relative_error}"
                )
            else:
                pass
            self.graphs[key] = CapturedDecode(graph=graph, inputs=inputs, output=output)
        for buffer in buffers:
            buffer.zero_()
        logger.info(
            f"Streaming decoder captured {len(self.graphs)} graphs in "
            f"{time.perf_counter() - started_seconds:.1f}s"
        )

    def forward(self, key: DecodeKey, inputs: tuple[torch.Tensor, ...]) -> torch.Tensor:
        if key[0] == "stream":
            return self.stream_forward(*inputs)
        else:
            return self.window_forward(*inputs)

    def replay(self, key: DecodeKey, inputs: tuple[torch.Tensor, ...]) -> torch.Tensor:
        captured = self.graphs.get(key)
        if captured is None:
            self.eager_calls += 1
            if self.eager_calls == 1 and self.graphs:
                logger.warning(f"Streaming decoder eager fallback for {key}")
            else:
                pass
            return self.forward(key, inputs)
        else:
            for static, value in zip(captured.inputs, inputs):
                static.copy_(value, non_blocking=True)
            captured.graph.replay()
            self.replay_calls += 1
            return captured.output.clone()

    @torch.no_grad()
    def decode_stream(self, frames: torch.Tensor, slots: list[int]) -> torch.Tensor:
        slot_index = indices_to_device(slots, self.device)
        key = ("stream", int(frames.shape[0]), int(frames.shape[-1]))
        return self.replay(key, (frames.to(self.dtype), slot_index))

    @torch.no_grad()
    def decode_window(
        self,
        window: torch.Tensor,
        slots: list[int],
        stable: list[int],
        valid: list[int],
    ) -> torch.Tensor:
        frames = self.window_bucket(max(valid))
        if frames is None:
            frames = int(window.shape[-1])
        else:
            pass
        inputs = (
            window[..., :frames].to(self.dtype),
            indices_to_device(slots, self.device),
            indices_to_device(stable, self.device),
            indices_to_device(valid, self.device),
        )
        return self.replay(("window", int(window.shape[0]), frames), inputs)

    @torch.no_grad()
    def flush(self, slot: int) -> torch.Tensor:
        frames = torch.zeros(
            1,
            self.latent_channels,
            self.lookahead,
            device=self.device,
            dtype=self.dtype,
        )
        return self.decode_stream(frames, [slot])


def causal_conv(conv: torch.nn.Module, value: torch.Tensor) -> torch.Tensor:
    """Causal Conv1d through cuDNN padding plus a view, instead of a padded copy of the input."""
    output = F.conv1d(
        value, conv.weight, conv.bias, padding=conv.left_padding, dilation=conv.dilation
    )
    return output[..., : value.shape[-1]]


def run_block(block: torch.nn.Module, value: torch.Tensor) -> torch.Tensor:
    """AMPBlock1.forward with causal_conv in place of the padded-copy convolutions."""
    activations = block.activations
    for first, second, first_activation, second_activation in zip(
        block.convs1, block.convs2, activations[::2], activations[1::2]
    ):
        hidden = causal_conv(first, first_activation(value))
        value = causal_conv(second, second_activation(hidden)) + value
    return value


__all__ = ["StreamingDecoder"]
