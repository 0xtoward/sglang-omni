"""CUDA graph runner for the Higgs vocoder (codec) decode.

The Higgs vocoder decode (RVQ dequant -> fc2 -> DAC ConvTranspose stack) is
tiny on modern GPUs and dominated by kernel-launch overhead: standalone nsys on
an A800 (bf16, B=1, T=64) measures ~420 host-side ``cudaLaunchKernel`` calls of
only ~27 distinct kernels per decode -- the ~36 Snake1d activations each fan out
to ~4 elementwise kernels (sin/pow/reciprocal/add) plus NCHW<->NHWC layout casts,
while conv/GEMM is <10% -- and the decode costs ~5 ms regardless of length.
Capturing the decode into a CUDA graph and replaying it collapses those launches
into a single replay, cutting per-call latency ~2.5x (5.4 -> 2.17 ms) on the B=1
streaming path.

Capture happens ONCE at startup via :meth:`warmup` (while the AR stage is
quiescent), then the runner is *sealed*: serving only ever replays cached
shapes and falls back to eager for anything else -- it NEVER captures a graph
during serving. Capturing live while the co-located AR stage replays its own
CUDA graphs corrupts the shared mempool ("operation not permitted when stream
is capturing" / "Offset increment outside graph"); warmup-only capture avoids
that entirely.

Graphs are captured per *exact* (batch, frames) shape. Zero-padding shorter
inputs up to a bucket is NOT viable: the stacked ConvTranspose decoder has a
large receptive field, so padding frames contaminate the whole waveform
(measured >40% relative error). Exact-shape capture keeps replay bit-exact vs
eager. The streaming scheduler emits B=1 windows whose frame count T forms a
dense set spanning ``[1, max(stride, followup+holdback+overlap)]`` (~[1, 87] at
default params): a first window near ``stream_stride``, a steady mode near
``followup+overlap``, and held-back / final-flush tails down to 1.
``create_vocoder_executor`` derives the captured range from those params (not a
hardcoded span), so it auto-scales when they change; larger windows and the B>1
bulk path fall back to eager. Device-to-host copies stay OUTSIDE replay (caller
receives a GPU tensor; replay output is cloned so the static buffer can be
reused).
"""
from __future__ import annotations

import logging
from collections.abc import Iterable

import torch

logger = logging.getLogger(__name__)


class VocoderCudaGraphRunner:
    """Warmup-captured, sealed replay of exact-shape CUDA graphs for codec decode."""

    def __init__(
        self,
        model,
        *,
        max_batch: int = 16,
        max_frames: int = 512,
        max_graphs: int = 256,
        warmup_iters: int = 3,
    ) -> None:
        self._model = model
        self._max_batch = max_batch
        self._max_frames = max_frames
        self._max_graphs = max_graphs
        self._warmup_iters = warmup_iters
        # (batch, frames) -> (graph, static_in, static_out)
        self._graphs: dict[tuple[int, int], tuple] = {}
        self._pool = None  # shared CUDA graph mempool (bounds memory across many graphs)
        self._sealed = False

    def _eligible(self, batch: int, frames: int) -> bool:
        return 1 <= batch <= self._max_batch and 1 <= frames <= self._max_frames

    def _num_codebooks(self) -> int:
        quant = getattr(self._model, "quantizer", None)
        qs = getattr(quant, "quantizers", None)
        if qs is not None:
            return len(qs)
        return 8  # Higgs codec default

    @torch.no_grad()
    def _capture_shape(self, batch: int, frames: int) -> None:
        n = self._num_codebooks()
        device = next(self._model.parameters()).device
        static_in = torch.zeros((batch, n, frames), dtype=torch.long, device=device)
        # Eager warmup on a side stream forces lazy allocations (conv algo
        # selection / workspaces) to happen BEFORE capture, so they don't get
        # pulled into the captured graph (which would corrupt replay).
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
        """Capture one graph per (batch, frames) shape, once, then seal.

        Must be called at startup, before serving, while the GPU is quiescent.
        After this returns, the runner only replays / eager-falls-back.
        """
        if self._sealed:
            logger.warning("VocoderCudaGraphRunner.warmup called after seal; ignoring")
            return
        wanted = list(dict.fromkeys((int(b), int(t)) for b, t in shapes))
        for batch, frames in wanted:
            if (batch, frames) in self._graphs:
                continue
            if not self._eligible(batch, frames):
                logger.warning("skip vocoder CG shape (%d,%d): outside caps", batch, frames)
                continue
            if len(self._graphs) >= self._max_graphs:
                logger.warning("vocoder CG graph cap %d reached; skipping rest", self._max_graphs)
                break
            try:
                self._capture_shape(batch, frames)
            except Exception as exc:  # capture is best-effort (let KeyboardInterrupt out)
                self._graphs.pop((batch, frames), None)
                logger.warning(
                    "vocoder CG capture failed for (%d,%d): %s; will use eager",
                    batch, frames, exc,
                )
        self._sealed = True
        logger.info(
            "Higgs vocoder CUDA graphs sealed: %d shape(s) captured %s",
            len(self._graphs), sorted(self._graphs.keys()),
        )

    @torch.no_grad()
    def decode(self, codes_BNT: torch.Tensor):
        """Replay a captured graph for an exact [B, N, T] shape, else return None.

        Never captures here (sealed after warmup) -- a cache miss (uncaptured
        (B, T), a mismatched codebook count N, or CPU input) returns None so the
        caller uses the eager path. The returned tensor is a fresh clone (the
        static buffer is overwritten on the next replay).

        NOT re-entrant: the single static_in/static_out buffer pair per (B, T) is
        shared across calls, so copy_ -> replay -> clone must not interleave. The
        Higgs vocoder stage satisfies this (StreamingSimpleScheduler drains
        decodes on one serial loop); a future multi-threaded caller would have to
        serialize replays per shape.
        """
        if not codes_BNT.is_cuda:
            return None
        batch, n, frames = codes_BNT.shape
        entry = self._graphs.get((batch, frames))
        if entry is None:
            return None
        graph, static_in, static_out = entry
        if static_in.shape[1] != n:  # codebook count differs -> honor miss -> eager
            return None
        static_in.copy_(codes_BNT)
        graph.replay()
        return static_out.clone()
