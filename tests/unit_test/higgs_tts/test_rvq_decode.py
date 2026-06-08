"""Unit tests for the vendored RVQ decode (capture-safe, byte-identical rewrite).

The #581 vocoder CUDA-graph work rewrote
``HiggsAudioV2TokenizerResidualVectorQuantization.decode`` to drop the
``torch.tensor(0.0)`` scalar seed -- an H2D copy that aborts CUDA-graph stream
capture -- by accumulating from the first quantizer instead. The 0-dim fp32
``0.0`` is a *wrapped scalar* that does not promote the bf16 accumulation, so
the rewrite is byte-identical to upstream (the eager / graph-off path is
unchanged). These tests lock that on CPU.
"""

import pytest
import torch

from sglang_omni.models.higgs_tts._vendored.higgs_audio_v2_tokenizer_hf import (
    HiggsAudioV2TokenizerResidualVectorQuantization as RVQ,
)


class _FakeQuantizer:
    """``decode(indices[B, T]) -> [B, D, T]`` via an embedding-table lookup."""

    def __init__(self, table: torch.Tensor) -> None:
        self._table = table  # [vocab, D]

    def decode(self, indices: torch.Tensor) -> torch.Tensor:
        return self._table[indices].permute(0, 2, 1).contiguous()


class _FakeRVQ:
    """Stand-in whose only attribute read by ``RVQ.decode`` is ``.quantizers``."""

    def __init__(self, quantizers: list) -> None:
        self.quantizers = quantizers


def _rvq(k: int, vocab: int, dim: int, dtype: torch.dtype) -> _FakeRVQ:
    torch.manual_seed(0)
    return _FakeRVQ(
        [_FakeQuantizer(torch.randn(vocab, dim, dtype=dtype)) for _ in range(k)]
    )


def _upstream_reference(rvq: _FakeRVQ, codes: torch.Tensor) -> torch.Tensor:
    """The original upstream form: ``torch.tensor(0.0)`` seed + accumulate."""
    out = torch.tensor(0.0)
    for i, indices in enumerate(codes):
        out = out + rvq.quantizers[i].decode(indices)
    return out


def test_decode_byte_identical_to_upstream_bf16():
    # bf16 tables (the serving dtype): the rewrite must be byte-identical to the
    # upstream torch.tensor(0.0) form -- the 0-dim fp32 seed is a wrapped scalar
    # that keeps the accumulation in bf16, so nothing changes for eager.
    rvq = _rvq(k=8, vocab=32, dim=16, dtype=torch.bfloat16)
    codes = torch.randint(0, 32, (8, 1, 7))
    out = RVQ.decode(rvq, codes)
    ref = _upstream_reference(rvq, codes)
    assert out.dtype == torch.bfloat16 and ref.dtype == torch.bfloat16
    assert torch.equal(out, ref)


def test_decode_byte_identical_fp32():
    rvq = _rvq(k=4, vocab=16, dim=8, dtype=torch.float32)
    codes = torch.randint(0, 16, (4, 1, 5))
    assert torch.equal(RVQ.decode(rvq, codes), _upstream_reference(rvq, codes))


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
