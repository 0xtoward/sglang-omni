"""CUDA graph runner for the Higgs vocoder (codec) decode.

The B=1 codec decode is launch-bound (~420 kernel launches per call); capturing
it into a CUDA graph and replaying collapses them into one, ~2.5x faster. Graphs
are captured per exact (batch, frames) shape -- the wide-receptive-field decoder
makes bucket padding lossy -- and replay is bit-exact vs eager. All capture
happens in warmup() at startup; the runner then seals and serving only replays
or eager-falls-back, since live capture would race the AR stage's shared mempool.
"""
from __future__ import annotations

import logging
from collections.abc import Iterable

import torch

logger = logging.getLogger(__name__)


class VocoderCudaGraphRunner:
    """Warmup-captured, sealed replay of exact-shape CUDA graphs for codec decode."""

    def __init__(self, model, *, warmup_iters: int = 3) -> None:
        self._model = model
        self._warmup_iters = warmup_iters
        self._graphs: dict[tuple[int, int], tuple] = {}
        self._pool = None
        self._sealed = False

    def _num_codebooks(self) -> int:
        qs = getattr(getattr(self._model, "quantizer", None), "quantizers", None)
        return len(qs) if qs is not None else 8

    @torch.no_grad()
    def _capture_shape(self, batch: int, frames: int) -> None:
        n = self._num_codebooks()
        device = next(self._model.parameters()).device
        static_in = torch.zeros((batch, n, frames), dtype=torch.long, device=device)
        # Warm up on a side stream so lazy conv-algo/workspace allocs settle before capture.
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(self._warmup_iters):
                self._model.decode(static_in).audio_values
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        if self._pool is None:
            self._pool = torch.cuda.graph_pool_handle()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=self._pool):
            static_out = self._model.decode(static_in).audio_values
        self._graphs[(batch, frames)] = (graph, static_in, static_out)
        logger.info(
            "Captured Higgs vocoder CUDA graph batch=%d frames=%d -> %s (%d cached)",
            batch, frames, tuple(static_out.shape), len(self._graphs),
        )

    @torch.no_grad()
    def warmup(self, shapes: Iterable[tuple[int, int]]) -> None:
        """Capture one graph per (batch, frames) shape at startup, then seal."""
        if self._sealed:
            return
        for batch, frames in dict.fromkeys((int(b), int(t)) for b, t in shapes):
            if (batch, frames) in self._graphs:
                continue
            try:
                self._capture_shape(batch, frames)
            except Exception as exc:
                self._graphs.pop((batch, frames), None)
                logger.warning("vocoder CG capture failed for (%d,%d): %s; using eager", batch, frames, exc)
        self._sealed = True
        logger.info("Higgs vocoder CUDA graphs sealed: %s", sorted(self._graphs))

    @torch.no_grad()
    def decode(self, codes_BNT: torch.Tensor):
        """Replay the captured graph for an exact [B, N, T], else None (eager fallback).

        One static buffer per shape, so replays must be serialized -- the Higgs
        streaming scheduler runs a single serial decode loop.
        """
        if not codes_BNT.is_cuda:
            return None
        batch, n, frames = codes_BNT.shape
        entry = self._graphs.get((batch, frames))
        if entry is None or entry[1].shape[1] != n:
            return None
        graph, static_in, static_out = entry
        static_in.copy_(codes_BNT)
        graph.replay()
        return static_out.clone()
