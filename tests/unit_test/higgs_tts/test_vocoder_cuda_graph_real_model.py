"""Real-model validation for the Higgs vocoder CUDA graph (gated, heavy).

``test_vocoder_cuda_graph.py`` uses a fake single-Conv1d stand-in and only
exercises the runner's capture/seal/replay machinery. This file loads the REAL
Higgs codec and pins the headline claim on the real DAC decoder: graph replay is
bit-exact vs eager across the RVQ gather + fc2 GEMM + ConvTranspose stack + 36
Snake1d activations, over the streaming window sizes (including the sub-floor
T=1 and the derived ceiling), and the warmup startup cost stays bounded.

Heavy (needs the ~9 GB checkpoint + a GPU), so it is gated on ``HIGGS_TTS_CKPT``
pointing at a local Higgs TTS checkpoint dir (mirrors how tests/test_model gates
on a local model). Per-decode launch-count reconciliation (~420 cudaLaunchKernel
of ~27 distinct kernels) is measured out-of-band with nsys on the single-process
codec decode, per the sgl-omni-profiling SOP -- it is a profiling artifact, not a
CI assertion.

Run:  HIGGS_TTS_CKPT=/path/to/higgs-audio-v3-tts-4b \
      python -m pytest tests/unit_test/higgs_tts/test_vocoder_cuda_graph_real_model.py -v -s
"""

import os
import time

import pytest
import torch

_CKPT = os.environ.get("HIGGS_TTS_CKPT")

real_model = pytest.mark.skipif(
    not torch.cuda.is_available() or not _CKPT,
    reason="needs a GPU and HIGGS_TTS_CKPT pointing at a local Higgs TTS checkpoint",
)

# Default-config streaming covers T in [1, max(stride, followup+holdback+overlap)]
# = [1, 87]; capture exactly that, matching create_vocoder_executor.
_SHAPES = [(1, f) for f in range(1, 88)]
_WARMUP: dict = {}


@pytest.fixture(scope="module")
def codec():
    from sglang_omni.models.higgs_tts.audio_codec import HiggsAudioCodec

    c = HiggsAudioCodec.from_pretrained(
        _CKPT, device="cuda", dtype=torch.bfloat16, enable_cuda_graph=True
    )
    assert c._cg_runner is not None, "enable_cuda_graph=True did not build the runner"
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    v0 = torch.cuda.memory_allocated()
    t0 = time.time()
    c.warmup_cuda_graph(_SHAPES)
    torch.cuda.synchronize()
    _WARMUP["seconds"] = time.time() - t0
    _WARMUP["gib"] = (torch.cuda.memory_allocated() - v0) / 1e9
    _WARMUP["graphs"] = len(c._cg_runner._graphs)
    return c


def _codes(codec, frames, *, seconds=6):
    full = codec.encode_reference(torch.randn(codec.SAMPLE_RATE * seconds) * 0.05)
    assert full.shape[0] >= frames, f"need >= {frames} frames, got {full.shape[0]}"
    return full[:frames].transpose(0, 1).unsqueeze(0).to("cuda", torch.long)


@real_model
@torch.no_grad()
def test_warmup_startup_cost_is_bounded(codec):
    print(
        f"\nwarmup: {_WARMUP['graphs']} graphs, "
        f"{_WARMUP['seconds']:.1f}s, +{_WARMUP['gib']:.2f} GiB"
    )
    assert _WARMUP["graphs"] == len(_SHAPES)
    assert _WARMUP["seconds"] < 60  # ~6 s on an A800; generous ceiling for slow disks
    assert _WARMUP["gib"] < 2.0  # ~0.3 GiB on an A800


@real_model
@torch.no_grad()
@pytest.mark.parametrize("frames", [1, 7, 16, 64, 83, 87])
def test_real_dac_replay_is_bit_exact(codec, frames):
    codes = _codes(codec, frames)
    replayed = codec._cg_runner.decode(codes)
    eager = codec.model.decode(codes).audio_values
    assert replayed is not None, f"(1, {frames}) should have been captured"
    # The headline claim, on the REAL DAC (not a fake Conv1d): replay == eager.
    assert torch.equal(replayed, eager)
    assert replayed.dtype == torch.bfloat16


@real_model
@torch.no_grad()
def test_out_of_range_and_batch_and_codebook_mismatch_fall_back(codec):
    full = codec.encode_reference(torch.randn(codec.SAMPLE_RATE * 12) * 0.05)
    # T=120 > 87 ceiling -> miss -> eager.
    big = full[:120].transpose(0, 1).unsqueeze(0).to("cuda", torch.long)
    assert codec._cg_runner.decode(big) is None
    # B=2 (only B=1 captured) -> eager.
    batch2 = full[:64].transpose(0, 1).unsqueeze(0).repeat(2, 1, 1).to("cuda", torch.long)
    assert codec._cg_runner.decode(batch2) is None
    # Codebook count N=6 != captured N=8 -> eager (the N-check), not a copy_ crash.
    n6 = full[:64].transpose(0, 1)[:6].unsqueeze(0).to("cuda", torch.long)
    assert codec._cg_runner.decode(n6) is None


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v", "-s"]))
