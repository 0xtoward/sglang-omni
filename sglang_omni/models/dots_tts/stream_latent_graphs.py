"""Explicit CUDA graphs for the streaming latent front end (post_proj + per-frame LSTM)."""

from __future__ import annotations

import functools
import logging
import math
import time
from dataclasses import dataclass

import torch

logger = logging.getLogger(__name__)

PARITY_RELATIVE_L2 = 1e-4


def relative_errors(
    actual: tuple[torch.Tensor, ...], expected: tuple[torch.Tensor, ...]
) -> list[float]:
    """Relative L2 of each tensor pair; a non-finite value on either side is an infinite error."""
    errors: list[float] = []
    for value, reference in zip(actual, expected, strict=True):
        value = value.float()
        reference = reference.float()
        if bool(torch.isfinite(value).all()) and bool(torch.isfinite(reference).all()):
            errors.append(
                ((value - reference).norm() / reference.norm().clamp_min(1e-12)).item()
            )
        else:
            errors.append(math.inf)
    return errors


def cudnn_stream_latents(
    vocoder: torch.nn.Module,
    latents: torch.Tensor,
    hidden: tuple[torch.Tensor, torch.Tensor],
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    """post_proj, the SLSTM as one stateful cuDNN call, and the output projection."""
    value = vocoder.post_proj(latents.float()).permute(0, 2, 1)
    value = vocoder.dec_mi_layer[0](value)
    recurrent = vocoder.dec_mi_layer[1]
    output, next_hidden = recurrent.lstm(value, (hidden[0], hidden[1]))
    if recurrent.skip:
        output = output + value
    else:
        pass
    value = vocoder.dec_mi_layer[2](output)
    decoder_dtype = next(vocoder.decoder.conv_pre.parameters()).dtype
    return value.permute(0, 2, 1).to(dtype=decoder_dtype), next_hidden


@dataclass(kw_only=True)
class StreamLatentGraph:
    graph: torch.cuda.CUDAGraph
    latents: torch.Tensor
    hidden_h: torch.Tensor
    hidden_c: torch.Tensor
    decoder_input: torch.Tensor
    next_hidden_h: torch.Tensor
    next_hidden_c: torch.Tensor


class StreamLatentGraphs:
    """Replays one captured graph per (batch, frames) instead of ~12 launches per frame per layer.

    The captured work is the unchanged eager front end, so replays use the same kernels; shapes
    outside the captured set run eagerly.
    """

    def __init__(
        self,
        inference: torch.nn.Module,
        *,
        max_batch_size: int,
        frame_counts: list[int],
        cudnn_lstm: bool = False,
    ) -> None:
        self.eager_decode = (
            inference._decode_stream_latents
        )  # noqa: leading-underscore  # upstream spelling
        self.capture_decode = self.eager_decode
        if cudnn_lstm:
            # note (0xtoward): one stateful cuDNN LSTM call per chunk instead of the
            # per-frame, per-layer gate loop (~780 kernels per replay); TF32-level differences.
            vocoder = inference.vocoder
            vocoder.dec_mi_layer[1].lstm.flatten_parameters()
            self.capture_decode = functools.partial(cudnn_stream_latents, vocoder)
        else:
            pass
        self.graphs: dict[tuple[int, int], StreamLatentGraph] = {}
        self.replay_calls: int = 0
        self.fallback_calls: int = 0
        self.exact: bool = True
        if (
            int(inference._lstm_num_layers) == 0
        ):  # noqa: leading-underscore  # upstream spelling
            inference._prepare_lstm_stream_params()  # noqa: leading-underscore  # upstream spelling
        else:
            pass
        latent_dim = int(inference.vocoder.h.latent_dim)
        num_layers = int(
            inference._lstm_num_layers
        )  # noqa: leading-underscore  # upstream spelling
        hidden_size = int(
            inference._lstm_hidden_size
        )  # noqa: leading-underscore  # upstream spelling
        device = next(inference.vocoder.parameters()).device
        started_seconds = time.perf_counter()
        generator = torch.Generator(device=device).manual_seed(20261003)
        capture_stream = torch.cuda.Stream()
        with torch.no_grad():
            for batch_size in range(1, max_batch_size + 1):
                for frames in frame_counts:
                    latents = torch.randn(
                        batch_size,
                        latent_dim,
                        frames,
                        device=device,
                        generator=generator,
                    )
                    hidden_h = (
                        torch.randn(
                            num_layers,
                            batch_size,
                            hidden_size,
                            device=device,
                            generator=generator,
                        )
                        * 0.1
                    )
                    hidden_c = (
                        torch.randn(
                            num_layers,
                            batch_size,
                            hidden_size,
                            device=device,
                            generator=generator,
                        )
                        * 0.1
                    )
                    expected_input, (expected_h, expected_c) = self.eager_decode(
                        latents, (hidden_h, hidden_c)
                    )
                    capture_stream.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(capture_stream):
                        for _ in range(2):
                            self.capture_decode(latents, (hidden_h, hidden_c))
                    torch.cuda.current_stream().wait_stream(capture_stream)
                    torch.cuda.synchronize()
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, stream=capture_stream):
                        decoder_input, (next_hidden_h, next_hidden_c) = (
                            self.capture_decode(latents, (hidden_h, hidden_c))
                        )
                    graph.replay()
                    torch.cuda.synchronize()
                    observed = [
                        (
                            decoder_input.clone(),
                            next_hidden_h.clone(),
                            next_hidden_c.clone(),
                        )
                    ]
                    expected = [(expected_input, expected_h, expected_c)]
                    # note (0xtoward): replay again from the graph's own h/c, so a state that
                    # is wrong while the first output still matches fails the gate too.
                    next_latents = torch.randn(
                        batch_size,
                        latent_dim,
                        frames,
                        device=device,
                        generator=generator,
                    )
                    latents.copy_(next_latents)
                    hidden_h.copy_(observed[0][1])
                    hidden_c.copy_(observed[0][2])
                    graph.replay()
                    torch.cuda.synchronize()
                    observed.append(
                        (
                            decoder_input.clone(),
                            next_hidden_h.clone(),
                            next_hidden_c.clone(),
                        )
                    )
                    next_expected_input, (next_expected_h, next_expected_c) = (
                        self.eager_decode(next_latents, (expected_h, expected_c))
                    )
                    expected.append(
                        (next_expected_input, next_expected_h, next_expected_c)
                    )
                    exact = all(
                        torch.equal(value, reference)
                        for step_observed, step_expected in zip(
                            observed, expected, strict=True
                        )
                        for value, reference in zip(
                            step_observed, step_expected, strict=True
                        )
                    )
                    if not exact:
                        errors = [
                            relative_errors(step_observed, step_expected)
                            for step_observed, step_expected in zip(
                                observed, expected, strict=True
                            )
                        ]
                        logger.warning(
                            f"Stream latent graph B{batch_size} T{frames} relative_l2 "
                            f"(input, h, c) per replay={errors}"
                        )
                        if max(max(step) for step in errors) > PARITY_RELATIVE_L2:
                            raise RuntimeError(
                                f"Stream latent graph B{batch_size} T{frames} failed parity gate: {errors}"
                            )
                        else:
                            pass
                    else:
                        pass
                    self.exact &= exact
                    self.graphs[(batch_size, frames)] = StreamLatentGraph(
                        graph=graph,
                        latents=latents,
                        hidden_h=hidden_h,
                        hidden_c=hidden_c,
                        decoder_input=decoder_input,
                        next_hidden_h=next_hidden_h,
                        next_hidden_c=next_hidden_c,
                    )
        logger.info(
            f"Stream latent graphs ready shapes={sorted(self.graphs)} exact={self.exact} "
            f"startup_seconds={time.perf_counter() - started_seconds:.3f}"
        )

    def __call__(
        self, latents: torch.Tensor, hidden: tuple[torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        captured = self.graphs.get((int(latents.shape[0]), int(latents.shape[-1])))
        if captured is None or not latents.is_floating_point() or latents.ndim != 3:
            self.fallback_calls += 1
            if self.fallback_calls == 1:
                logger.warning(
                    f"Stream latent graph fallback shape={tuple(latents.shape)} dtype={latents.dtype}"
                )
            else:
                pass
            return self.capture_decode(latents, hidden)
        else:
            captured.latents.copy_(latents)
            captured.hidden_h.copy_(hidden[0])
            captured.hidden_c.copy_(hidden[1])
            captured.graph.replay()
            self.replay_calls += 1
            if self.replay_calls == 1 or self.replay_calls % 500 == 0:
                logger.info(
                    f"Stream latent graph replays={self.replay_calls} fallbacks={self.fallback_calls}"
                )
            else:
                pass
            return captured.decoder_input, (
                captured.next_hidden_h,
                captured.next_hidden_c,
            )
