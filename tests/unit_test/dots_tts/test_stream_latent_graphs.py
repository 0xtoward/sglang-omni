# SPDX-License-Identifier: Apache-2.0
"""One stateful LSTM call per chunk matches the per-frame streaming gate loop."""

from __future__ import annotations

import math

import pytest
import torch

from sglang_omni.models.dots_tts.compat import import_dots_tts

import_dots_tts()

from dots_tts.modules.vocoder.vocoder_inference import VocoderInference

from sglang_omni.models.dots_tts import stream_latent_graphs
from sglang_omni.models.dots_tts.stream_latent_graphs import (
    StreamLatentGraphs,
    cudnn_stream_latents,
    relative_errors,
)
from tests.unit_test.dots_tts.test_incremental_codec import tiny_inference


@torch.no_grad()
def test_chunked_lstm_matches_frame_loop() -> None:
    inference = tiny_inference()
    inference._prepare_lstm_stream_params()  # noqa: leading-underscore  # upstream spelling
    generator = torch.Generator().manual_seed(2)
    latent_dim = int(inference.vocoder.h.latent_dim)
    layers = int(
        inference._lstm_num_layers
    )  # noqa: leading-underscore  # upstream spelling
    hidden_size = int(
        inference._lstm_hidden_size
    )  # noqa: leading-underscore  # upstream spelling
    hidden = (
        torch.randn(layers, 2, hidden_size, generator=generator) * 0.1,
        torch.randn(layers, 2, hidden_size, generator=generator) * 0.1,
    )
    for frames in (4, 16):
        latents = torch.randn(2, latent_dim, frames, generator=generator)
        expected, (expected_hidden_state, expected_cell_state) = (
            inference._decode_stream_latents(  # noqa: leading-underscore  # upstream spelling
                latents, hidden
            )
        )
        observed, (observed_hidden_state, observed_cell_state) = cudnn_stream_latents(
            inference.vocoder, latents, hidden
        )
        torch.testing.assert_close(observed, expected, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(
            observed_hidden_state, expected_hidden_state, rtol=1e-5, atol=1e-5
        )
        torch.testing.assert_close(
            observed_cell_state, expected_cell_state, rtol=1e-5, atol=1e-5
        )
        hidden = (observed_hidden_state, observed_cell_state)


def test_relative_errors_count_non_finite_values_as_infinite() -> None:
    reference = torch.ones(4)
    assert relative_errors((reference.clone(),), (reference,)) == [0.0]
    assert relative_errors((torch.full((4,), math.nan),), (reference,)) == [math.inf]
    assert relative_errors((reference,), (torch.full((4,), math.inf),)) == [math.inf]


def cuda_inference(monkeypatch: pytest.MonkeyPatch) -> VocoderInference:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    else:
        pass
    monkeypatch.setattr(torch.backends.cudnn, "allow_tf32", False)
    inference = tiny_inference()
    inference.vocoder.to("cuda")
    return inference


@pytest.mark.accelerator
@pytest.mark.parametrize("cudnn_lstm", [False, True])
@torch.no_grad()
def test_graphs_match_eager_over_consecutive_replays(
    monkeypatch: pytest.MonkeyPatch, cudnn_lstm: bool
) -> None:
    inference = cuda_inference(monkeypatch)
    if cudnn_lstm:
        # note (0xtoward): cuDNN's TF32 default stays on, so the fp32 LSTM call
        # is what keeps the carried state on the per-frame gate loop.
        monkeypatch.setattr(torch.backends.cudnn, "allow_tf32", True)
    else:
        pass
    eager = (
        inference._decode_stream_latents
    )  # noqa: leading-underscore  # upstream spelling
    graphs = StreamLatentGraphs(
        inference, max_batch_size=2, frame_counts=[4, 8], cudnn_lstm=cudnn_lstm
    )
    generator = torch.Generator(device="cuda").manual_seed(4)
    latent_dim = int(inference.vocoder.h.latent_dim)
    layers = int(
        inference._lstm_num_layers
    )  # noqa: leading-underscore  # upstream spelling
    hidden_size = int(
        inference._lstm_hidden_size
    )  # noqa: leading-underscore  # upstream spelling
    state = (
        torch.zeros(layers, 2, hidden_size, device="cuda"),
        torch.zeros(layers, 2, hidden_size, device="cuda"),
    )
    expected_state = state
    for frames in (4, 8, 4):
        latents = torch.randn(2, latent_dim, frames, device="cuda", generator=generator)
        value, state = graphs(latents, state)
        state = (state[0].clone(), state[1].clone())
        expected, expected_state = eager(latents, expected_state)
        torch.testing.assert_close(value, expected, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(state[0], expected_state[0], rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(state[1], expected_state[1], rtol=1e-5, atol=1e-5)
    assert graphs.replay_calls == 3
    assert graphs.fallback_calls == 0


@pytest.mark.accelerator
@pytest.mark.parametrize("fault", ["state", "nan"])
@torch.no_grad()
def test_parity_gate_rejects_a_wrong_state_or_a_non_finite_output(
    monkeypatch: pytest.MonkeyPatch, fault: str
) -> None:
    inference = cuda_inference(monkeypatch)
    correct = stream_latent_graphs.cudnn_stream_latents

    def faulty(
        vocoder: torch.nn.Module,
        latents: torch.Tensor,
        hidden: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        value, (next_hidden_state, next_cell_state) = correct(vocoder, latents, hidden)
        if fault == "state":
            # note (0xtoward): the output stays right; only the carried state is wrong.
            return value, (next_hidden_state + 1.0, next_cell_state)
        else:
            return value * math.nan, (next_hidden_state, next_cell_state)

    monkeypatch.setattr(stream_latent_graphs, "cudnn_stream_latents", faulty)
    with pytest.raises(RuntimeError, match="failed parity gate"):
        StreamLatentGraphs(
            inference, max_batch_size=1, frame_counts=[4], cudnn_lstm=True
        )
