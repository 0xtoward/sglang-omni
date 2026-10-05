# SPDX-License-Identifier: Apache-2.0
"""Replay the streaming flow estimator's Euler loop as one CUDA graph per cache length."""

from __future__ import annotations

import gc
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass

import torch

ChunkOutputs = tuple[torch.Tensor, torch.Tensor, torch.Tensor]
BUFFER_FRAME_BUCKET = 128


@dataclass(frozen=True, kw_only=True)
class CapturedChunkGraph:
    graph: torch.cuda.CUDAGraph
    static_inputs: tuple[torch.Tensor, ...]
    static_outputs: ChunkOutputs


class ChunkCudaGraphRunner:
    """Stand in for ``CausalConditionalCFM.forward_chunk`` on full chunks that carry a cache.

    Graphs are keyed by cache length and captured only inside ``capturing()``, which the
    speech warm-up enters while it walks one voice through every length a turn can visit.
    Any other call replays a captured graph when its length has one and otherwise runs
    eagerly, so serving never captures. All graphs read their inputs from, and write their
    outputs to, buffers shared per capacity bucket, and their intermediates share one pool,
    so the memory does not grow with the number of lengths.
    """

    def __init__(
        self, forward_chunk: Callable[..., ChunkOutputs], chunk_frames: int
    ) -> None:
        self.forward_chunk = forward_chunk
        self.chunk_frames = chunk_frames
        self.graphs: dict[tuple[int, int, float], CapturedChunkGraph] = {}
        self.buffers: dict[int, tuple[torch.Tensor, ...]] = {}
        self.pool = torch.cuda.graph_pool_handle()
        # note: one capture stream for every graph; the allocator only hands a pool block
        # back to the stream that freed it.
        self.stream = torch.cuda.Stream()
        self.capture_misses = False

    @contextmanager
    def capturing(self) -> Iterator[None]:
        self.capture_misses = True
        try:
            yield
        finally:
            self.capture_misses = False

    def __call__(
        self,
        mu: torch.Tensor,
        speaker_embeddings: torch.Tensor,
        mel_conditioning: torch.Tensor,
        n_timesteps: int = 10,
        temperature: float = 1.0,
        convolution_cache: torch.Tensor | None = None,
        attention_cache: torch.Tensor | None = None,
    ) -> ChunkOutputs:
        inputs = (
            mu,
            speaker_embeddings,
            mel_conditioning,
            convolution_cache,
            attention_cache,
        )
        captured = None
        if attention_cache is not None and mu.shape[2] == self.chunk_frames:
            key = (attention_cache.shape[4], n_timesteps, temperature)
            captured = self.graphs.get(key)
            if captured is None and self.capture_misses:
                captured = self.capture(inputs, n_timesteps, temperature)
                self.graphs[key] = captured
            else:
                pass
        else:
            pass
        if captured is None:
            return self.forward_chunk(
                mu,
                speaker_embeddings,
                mel_conditioning,
                n_timesteps,
                temperature,
                convolution_cache,
                attention_cache,
            )
        else:
            for static, value in zip(captured.static_inputs, inputs, strict=True):
                static.copy_(value)
            captured.graph.replay()
            return tuple(output.clone() for output in captured.static_outputs)

    def shared_buffers(
        self, inputs: tuple[torch.Tensor, ...]
    ) -> tuple[tuple[torch.Tensor, ...], ChunkOutputs]:
        """Views of the bucket's buffers shaped like this call's inputs and outputs.

        A bucket holds the five inputs (mu, speaker, conditioning, convolution cache,
        attention cache) followed by the three outputs (mel, convolution, attention).
        """
        mu, speaker_embeddings, mel_conditioning, convolution_cache, attention_cache = (
            inputs
        )
        frames = attention_cache.shape[4] + mu.shape[2]
        bucket = -(-frames // BUFFER_FRAME_BUCKET) * BUFFER_FRAME_BUCKET
        if bucket not in self.buffers:
            attention_shape = list(attention_cache.shape)
            attention_shape[4] = bucket
            self.buffers[bucket] = (
                torch.empty_like(mu),
                torch.empty_like(speaker_embeddings),
                torch.empty_like(mel_conditioning),
                torch.empty_like(convolution_cache),
                attention_cache.new_empty(attention_shape),
                torch.empty_like(mu),
                torch.empty_like(convolution_cache),
                attention_cache.new_empty(attention_shape),
            )
        else:
            pass
        buffers = self.buffers[bucket]
        static_inputs = (
            *buffers[:4],
            buffers[4][:, :, :, :, : attention_cache.shape[4]],
        )
        static_outputs = (buffers[5], buffers[6], buffers[7][:, :, :, :, :frames])
        return static_inputs, static_outputs

    def capture(
        self, inputs: tuple[torch.Tensor, ...], n_timesteps: int, temperature: float
    ) -> CapturedChunkGraph:
        static_inputs, static_outputs = self.shared_buffers(inputs)
        for static, value in zip(static_inputs, inputs, strict=True):
            static.copy_(value)

        def run() -> None:
            outputs = self.forward_chunk(
                *static_inputs[:3], n_timesteps, temperature, *static_inputs[3:]
            )
            for static, value in zip(static_outputs, outputs, strict=True):
                static.copy_(value)

        # note: the warm-up pass settles cuBLAS/cuDNN workspaces; the collection keeps
        # the onnxruntime session cycle from being freed inside the capture.
        gc.collect()
        current_stream = torch.cuda.current_stream()
        stream = self.stream
        stream.wait_stream(current_stream)
        with torch.cuda.stream(stream):
            run()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(
                graph, pool=self.pool, stream=stream, capture_error_mode="thread_local"
            ):
                run()
        current_stream.wait_stream(stream)
        torch.cuda.empty_cache()
        return CapturedChunkGraph(
            graph=graph, static_inputs=static_inputs, static_outputs=static_outputs
        )
