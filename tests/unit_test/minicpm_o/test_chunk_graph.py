# SPDX-License-Identifier: Apache-2.0
"""The streaming flow chunk graph must reproduce the eager Euler loop and its caches."""

from __future__ import annotations

import pytest
import torch

from sglang_omni.models.minicpm_o.components.token2wav.chunk_graph import (
    ChunkCudaGraphRunner,
)
from sglang_omni.models.minicpm_o.components.token2wav.dit import DiT
from sglang_omni.models.minicpm_o.components.token2wav.flow import (
    CausalConditionalCFM,
)

MEL_CHANNELS = 8
PROMPT_FRAMES = 9
CHUNK_FRAMES = 6
STEPS = 4


def build_decoder() -> CausalConditionalCFM:
    torch.manual_seed(0)
    estimator = DiT(
        in_channels=4 * MEL_CHANNELS,
        out_channels=MEL_CHANNELS,
        depth=2,
        num_heads=2,
        head_dim=16,
        hidden_size=32,
    )
    for parameter in estimator.parameters():
        torch.nn.init.normal_(parameter, std=0.2)
    return CausalConditionalCFM(estimator).cuda().eval()


def chunk_inputs(frames: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.randn(1, MEL_CHANNELS, frames, device="cuda"),
        torch.randn(1, MEL_CHANNELS, device="cuda"),
        torch.zeros(1, MEL_CHANNELS, frames, device="cuda"),
    )


def run_stream(forward, chunks) -> tuple[list[torch.Tensor], torch.Tensor]:
    mel, convolution, attention = forward(*chunk_inputs(PROMPT_FRAMES), STEPS, 1.0)
    mels = [mel]
    for mu, speaker, conditioning in chunks:
        mel, convolution, attention = forward(
            mu, speaker, conditioning, STEPS, 1.0, convolution, attention
        )
        mels.append(mel)
    return mels, attention


def assert_stream_matches(actual, expected) -> None:
    for mel, reference in zip(actual[0], expected[0], strict=True):
        torch.testing.assert_close(mel, reference, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(actual[1], expected[1], rtol=1e-5, atol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_chunk_graph_captures_in_warm_up_and_replays_full_chunks() -> None:
    decoder = build_decoder()
    runner = ChunkCudaGraphRunner(decoder.forward_chunk, CHUNK_FRAMES)
    torch.manual_seed(1)
    full = [chunk_inputs(CHUNK_FRAMES) for _ in range(3)]
    # note: the partial chunk arrives at a captured cache length and must still run eagerly.
    mixed = [full[0], chunk_inputs(CHUNK_FRAMES - 2), full[1], full[2]]
    warm_up_keys = {
        (PROMPT_FRAMES + index * CHUNK_FRAMES, STEPS, 1.0) for index in range(3)
    }

    def stream(forward, chunks):
        torch.manual_seed(2)
        return run_stream(forward, chunks)

    with torch.inference_mode():
        expected_full = stream(decoder.forward_chunk, full)
        expected_mixed = stream(decoder.forward_chunk, mixed)
        assert_stream_matches(stream(runner, full), expected_full)
        assert not runner.graphs
        with runner.capturing():
            assert_stream_matches(stream(runner, full), expected_full)
        assert set(runner.graphs) == warm_up_keys
        assert len(runner.buffers) == 1
        assert_stream_matches(stream(runner, mixed), expected_mixed)
        assert_stream_matches(stream(runner, full), expected_full)
        assert set(runner.graphs) == warm_up_keys
