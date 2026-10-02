# SPDX-License-Identifier: Apache-2.0
"""One stateful LSTM call per chunk matches the per-frame streaming gate loop."""

from __future__ import annotations

import torch

from sglang_omni.models.dots_tts.stream_latent_graphs import cudnn_stream_latents
from tests.unit_test.dots_tts.test_streaming_decoder import tiny_inference


@torch.no_grad()
def test_chunked_lstm_matches_frame_loop() -> None:
    inference = tiny_inference()
    inference._prepare_lstm_stream_params()  # noqa: leading-underscore  # upstream spelling
    generator = torch.Generator().manual_seed(2)
    latent_dim = int(inference.vocoder.h.latent_dim)
    layers = int(inference._lstm_num_layers)  # noqa: leading-underscore  # upstream spelling
    hidden_size = int(inference._lstm_hidden_size)  # noqa: leading-underscore  # upstream spelling
    hidden = (
        torch.randn(layers, 2, hidden_size, generator=generator) * 0.1,
        torch.randn(layers, 2, hidden_size, generator=generator) * 0.1,
    )
    for frames in (4, 16):
        latents = torch.randn(2, latent_dim, frames, generator=generator)
        expected, (expected_h, expected_c) = inference._decode_stream_latents(  # noqa: leading-underscore  # upstream spelling
            latents, hidden
        )
        observed, (observed_h, observed_c) = cudnn_stream_latents(inference.vocoder, latents, hidden)
        torch.testing.assert_close(observed, expected, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(observed_h, expected_h, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(observed_c, expected_c, rtol=1e-5, atol=1e-5)
        hidden = (observed_h, observed_c)
