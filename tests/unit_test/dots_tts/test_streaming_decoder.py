# SPDX-License-Identifier: Apache-2.0
"""StreamingDecoder: per-slot stage contexts reproduce the windowed causal decode."""

from __future__ import annotations

import pytest
import torch

from sglang_omni.models.dots_tts.streaming_decoder import StreamingDecoder
from sglang_omni.models.dots_tts.vocoder_slot_pool import DotsVocoderSlotPool

PATCH = 4
MERGE = 4


def tiny_inference() -> torch.nn.Module:
    try:
        from sglang_omni.models.dots_tts.compat import import_dots_tts

        import_dots_tts()
        from dots_tts.modules.vocoder.bigvgan import AudioVAE
        from dots_tts.modules.vocoder.config import AudioVAEConfig
        from dots_tts.modules.vocoder.vocoder_inference import VocoderInference
    except ImportError as exc:
        pytest.skip(f"dots_tts unavailable: {exc}")
    else:
        pass
    torch.manual_seed(0)
    config = AudioVAEConfig(
        sample_rate=1600,
        upsample_rates=[4, 2],
        upsample_kernel_sizes=[8, 4],
        upsample_initial_channel=32,
        resblock="1",
        resblock_kernel_sizes=[3, 5],
        resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5]],
        downsample_rates=[2, 4],
        downsample_channels=[4, 8, 16],
        latent_dim=8,
        causal=True,
        mi_num_layers=1,
        causal_encoder=True,
        use_bias_at_final=False,
        use_tanh_at_final=False,
    )
    vocoder = AudioVAE(config).eval()
    vocoder.remove_weight_norm()
    return VocoderInference(vocoder)


def schedule(total_patches: int) -> list[int]:
    """Two single-patch steps for first audio, then merged steps, like the coalesced pump."""
    steps: list[int] = []
    taken = 0
    while taken < total_patches:
        size = 1 if len(steps) < 2 else min(MERGE, total_patches - taken)
        steps.append(size)
        taken += size
    return steps


def decode(
    pool: DotsVocoderSlotPool, streams: list[torch.Tensor]
) -> list[torch.Tensor]:
    """Run staggered streams through one pool; rows of different ages share equal-length steps."""
    plans = [schedule(stream.shape[1] // PATCH) for stream in streams]
    slots = [pool.acquire() for _ in streams]
    cursor = [0] * len(streams)
    progress = [0] * len(streams)
    chunks: list[list[torch.Tensor]] = [[] for _ in streams]
    round_index = 0
    while any(cursor[index] < len(plans[index]) for index in range(len(streams))):
        active = [
            index
            for index in range(len(streams))
            if index <= round_index and cursor[index] < len(plans[index])
        ]
        by_size: dict[int, list[int]] = {}
        for index in active:
            by_size.setdefault(plans[index][cursor[index]], []).append(index)
        for size, members in sorted(by_size.items()):
            latents = {}
            for index in members:
                frames = size * PATCH
                latents[slots[index]] = streams[index][
                    :, progress[index] : progress[index] + frames
                ]
                progress[index] += frames
                cursor[index] += 1
            out = pool.step(latents)
            for index in members:
                chunks[index].append(out[slots[index]].reshape(-1))
        round_index += 1
    for index, slot in enumerate(slots):
        chunks[index].append(pool.flush(slot).reshape(-1))
        pool.release(slot)
    return [torch.cat(parts) for parts in chunks]


@torch.no_grad()
def test_streaming_decoder_matches_window_decode() -> None:
    inference = tiny_inference()
    window_pool = DotsVocoderSlotPool(inference, num_slots=2, chunk_size=PATCH * MERGE)
    stream_pool = DotsVocoderSlotPool(inference, num_slots=2, chunk_size=PATCH * MERGE)
    stream_pool.streaming = StreamingDecoder(
        inference,
        num_slots=2,
        max_batch_size=2,
        stream_frames=[PATCH * patches for patches in range(1, MERGE + 1)],
        window_frames=sorted({8, 16, 24, 32, stream_pool.window_size}),
        capture_graphs=False,
    )
    generator = torch.Generator().manual_seed(1)
    latent_dim = int(inference.vocoder.h.latent_dim)
    streams = [
        torch.randn(1, frames, latent_dim, generator=generator) for frames in (96, 60)
    ]

    expected = decode(window_pool, streams)
    observed = decode(stream_pool, streams)

    for reference, candidate in zip(expected, observed, strict=True):
        assert candidate.shape == reference.shape
        error = (candidate - reference).norm() / reference.norm()
        assert error < 1e-5, error
    assert stream_pool.streaming.warm_frames < 96 // 2
