"""Unit tests for the Higgs vocoder CUDA-graph runner.

CUDA-graph capture needs a real GPU, so these are skipped on CPU-only hosts.
A tiny fake codec model stands in for the DAC decoder — we only exercise the
runner's capture / seal / replay / eager-fallback machinery, not the codec.
"""

import pytest
import torch

from sglang_omni.models.higgs_tts.vocoder_cuda_graph import VocoderCudaGraphRunner

cuda_only = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA-graph capture requires a GPU"
)

_N_CODEBOOKS = 8


class _DecodeOutput:
    def __init__(self, audio_values: torch.Tensor) -> None:
        self.audio_values = audio_values


class _FakeQuantizer:
    """Only ``len(quantizers)`` is read by the runner (for the codebook count)."""

    def __init__(self, n: int) -> None:
        self.quantizers = [None] * n


class _FakeCodecModel(torch.nn.Module):
    """Deterministic ``decode([B, N, T] long) -> .audio_values [B, 1, T]``."""

    def __init__(self, n_codebooks: int = _N_CODEBOOKS) -> None:
        super().__init__()
        self.quantizer = _FakeQuantizer(n_codebooks)
        self.proj = torch.nn.Conv1d(n_codebooks, 1, kernel_size=3, padding=1)

    def decode(self, codes_BNT: torch.Tensor) -> _DecodeOutput:
        return _DecodeOutput(self.proj(codes_BNT.to(self.proj.weight.dtype)))


def _model() -> _FakeCodecModel:
    return _FakeCodecModel().to("cuda").eval()


def _codes(frames: int, batch: int = 1) -> torch.Tensor:
    return torch.randint(0, 16, (batch, _N_CODEBOOKS, frames), device="cuda")


def _codes_n(frames: int, n: int, batch: int = 1) -> torch.Tensor:
    return torch.randint(0, 16, (batch, n, frames), device="cuda")


class _FlakyCodecModel(_FakeCodecModel):
    """_FakeCodecModel that raises while decoding one specific frame count, to
    exercise the warmup capture-failure path."""

    def __init__(self, bad_frames: int, n_codebooks: int = _N_CODEBOOKS) -> None:
        super().__init__(n_codebooks)
        self._bad_frames = bad_frames

    def decode(self, codes_BNT: torch.Tensor) -> _DecodeOutput:
        if codes_BNT.shape[-1] == self._bad_frames:
            raise RuntimeError("synthetic capture failure")
        return super().decode(codes_BNT)


@cuda_only
@torch.no_grad()
def test_warmup_replay_is_bit_exact():
    model = _model()
    runner = VocoderCudaGraphRunner(model, warmup_iters=2)
    runner.warmup([(1, 16), (1, 24)])
    for frames in (16, 24):
        codes = _codes(frames)
        replayed = runner.decode(codes)
        eager = model.decode(codes).audio_values
        assert replayed is not None
        # Replay must be bit-exact against eager (the headline correctness claim).
        assert torch.equal(replayed, eager)


@cuda_only
@torch.no_grad()
def test_uncaptured_shape_falls_back_to_eager():
    model = _model()
    runner = VocoderCudaGraphRunner(model, warmup_iters=2)
    runner.warmup([(1, 16)])
    # None signals the caller to take the eager path; never live-captures here.
    assert runner.decode(_codes(99)) is None  # frame count not captured
    assert runner.decode(_codes(16, batch=2)) is None  # batch not captured


@cuda_only
@torch.no_grad()
def test_warmup_is_sealed_after_first_call():
    model = _model()
    runner = VocoderCudaGraphRunner(model, warmup_iters=2)
    runner.warmup([(1, 16)])
    runner.warmup([(1, 24)])  # after seal -> ignored, nothing new captured
    assert runner.decode(_codes(24)) is None


@cuda_only
@torch.no_grad()
def test_cpu_codes_return_none():
    model = _model()
    runner = VocoderCudaGraphRunner(model, warmup_iters=2)
    runner.warmup([(1, 16)])
    cpu_codes = torch.randint(0, 16, (1, _N_CODEBOOKS, 16))
    assert runner.decode(cpu_codes) is None


@cuda_only
@torch.no_grad()
def test_codebook_count_mismatch_falls_back_to_eager():
    model = _model()  # 8 codebooks
    runner = VocoderCudaGraphRunner(model, warmup_iters=2)
    runner.warmup([(1, 16)])
    # (1, 16) is captured at N=8; a mismatched codebook count must miss -> eager
    # (None), not crash in static_in.copy_ on the shape mismatch.
    assert runner.decode(_codes_n(16, n=6)) is None


@cuda_only
@torch.no_grad()
def test_capture_failure_is_swallowed_and_others_still_capture():
    model = _FlakyCodecModel(bad_frames=13).to("cuda").eval()
    runner = VocoderCudaGraphRunner(model, warmup_iters=2)
    runner.warmup([(1, 12), (1, 13), (1, 16)])
    # The shape whose decode raised is skipped (except Exception); the rest
    # capture and replay normally.
    assert (1, 13) not in runner._graphs
    assert runner.decode(_codes(12)) is not None
    assert runner.decode(_codes(16)) is not None
    assert runner.decode(_codes(13)) is None  # uncaptured -> eager fallback


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
