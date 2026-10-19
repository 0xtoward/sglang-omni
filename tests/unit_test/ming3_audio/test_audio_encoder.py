# SPDX-License-Identifier: Apache-2.0
"""Audio encoder contracts checked against fixed reference outputs."""

from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file

from sglang_omni.models.ming3_audio.audio_encoder import (
    AudioEncoderConfig,
    Ming3AudioEncoder,
)


@pytest.mark.parametrize(
    ("mel_frames", "expected_tokens"),
    [(1, 1), (16, 1), (17, 2), (2999, 188), (3000, 188), (3001, 189), (6000, 376)],
)
def test_audio_token_count(mel_frames: int, expected_tokens: int) -> None:
    config = AudioEncoderConfig(
        n_mels=4,
        n_audio_ctx=3000,
        n_audio_state=8,
        n_audio_head=2,
        n_audio_layer=2,
        downsampler_kernel_size_list=[3, 3, 3],
        downsampler_stride_list=[2, 2, 2],
    )
    assert config.output_length(mel_frames) == expected_tokens


def test_encoder_matches_reference() -> None:
    reference = load_file(
        str(Path(__file__).parent / "fixtures/audio_encoder.safetensors")
    )
    config = AudioEncoderConfig(
        n_mels=4,
        n_audio_ctx=32,
        n_audio_state=8,
        n_audio_head=2,
        n_audio_layer=2,
        downsampler_kernel_size_list=[3, 3, 3],
        downsampler_stride_list=[2, 2, 2],
        whisper_chunk_size=32,
    )
    encoder = Ming3AudioEncoder(config, hidden_size=16).eval()
    encoder.load_state_dict(
        {
            name.removeprefix("weights."): value
            for name, value in reference.items()
            if name.startswith("weights.")
        }
    )
    with torch.inference_mode():
        actual = encoder([reference["features.0"], reference["features.1"]])
    torch.testing.assert_close(actual, reference["expected"], rtol=1e-5, atol=1e-6)
