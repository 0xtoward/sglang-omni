"""Smoke test: the warmup CUDA-graph frame range must cover every streaming window.

``create_vocoder_executor`` derives the captured range as
``vocoder_cg_max_frames = max(stream_stride, followup + holdback + overlap)``
instead of a hardcoded 8..160. The load-bearing invariant: this MUST cover every
decode window the streaming scheduler actually emits -- at the default params AND
when stride / overlap / followup / holdback change. If someone bumps
``stream_stride``, the steady window (83 = followup+overlap at default) moves, and
the captured range must move with it or the hot path silently eager-falls-back.

Pure CPU: drives the REAL ``_decode_delta`` window arithmetic with a recording
stub in place of the codec decode, so it validates the live scheduler -- not a
re-derivation of the same formula.
"""

import pytest
import torch

from sglang_omni.models.higgs_tts import vocoder_scheduler as vs
from sglang_omni.models.higgs_tts.vocoder_scheduler import (
    HiggsStreamingVocoderScheduler,
    _HiggsStreamState,
)

_N = 8  # codebooks
_SPF = 960  # samples per frame (hop)


def _derived_cap(stride, followup, overlap, holdback):
    """Exactly create_vocoder_executor's vocoder_cg_max_frames."""
    return max(stride, followup + holdback + overlap)


def _windows(stride, followup, overlap, holdback, n_tokens, *, initial_chunk=0):
    """Drive the real _decode_delta for one utterance; return every window's T."""
    sched = HiggsStreamingVocoderScheduler.__new__(HiggsStreamingVocoderScheduler)
    sched._stream_stride = stride
    sched._stream_followup_stride = followup
    sched._stream_overlap_tokens = overlap
    sched._stream_holdback_tokens = holdback
    sched._vocoder_cg_max_frames = _derived_cap(stride, followup, overlap, holdback)
    sched._samples_per_frame = _SPF
    sched._sample_rate = 24000

    seen: list[int] = []

    def _record(rows, *, num_codebooks, codebook_size):
        t = len(rows) - num_codebooks + 1
        seen.append(t)
        return torch.zeros(max(t, 0) * _SPF)

    sched._decode_delayed_rows = _record  # instance patch

    state = _HiggsStreamState()
    state.num_codebooks = _N
    state.codebook_size = 1026
    state.initial_codec_chunk_frames = initial_chunk
    row = torch.zeros(_N, dtype=torch.long)
    for _ in range(n_tokens):
        state.delayed_rows.append(row)
        sched._decode_delta(state, is_final=False)
    sched._decode_delta(state, is_final=True)
    return [t for t in seen if t > 0]


# (stride, followup, overlap, holdback)
CONFIGS = [
    (75, 75, 8, 4),    # default        -> steady 83, cap 87
    (50, 50, 8, 4),    # smaller stride -> steady 58, cap 62
    (150, 150, 8, 4),  # larger stride  -> steady 158, cap 162
    (75, 75, 0, 0),    # no overlap/holdback
    (100, 60, 16, 8),  # stride > followup
    (40, 90, 4, 2),    # followup > stride
]


@pytest.fixture(autouse=True)
def _stub_payload(monkeypatch):
    # We only test window arithmetic; skip the waveform payload encoding.
    monkeypatch.setattr(vs, "audio_waveform_payload", lambda *a, **k: {})


@pytest.mark.parametrize("stride,followup,overlap,holdback", CONFIGS)
def test_derived_range_covers_every_streaming_window(stride, followup, overlap, holdback):
    cap = _derived_cap(stride, followup, overlap, holdback)
    seen: set[int] = set()
    # sweep utterance lengths over a few strides to hit first / steady / repeated /
    # final-flush windows at every residue
    for n in range(_N, 3 * max(stride, followup) + 3 * _N):
        seen.update(_windows(stride, followup, overlap, holdback, n))
    assert seen, "no decode windows emitted"
    assert min(seen) >= 1
    # THE INVARIANT: every real decode window fits inside the derived capture range.
    assert max(seen) <= cap, (
        f"emitted window T={max(seen)} exceeds derived cap {cap} for "
        f"(stride={stride}, followup={followup}, overlap={overlap}, holdback={holdback}); "
        f"the CG range would miss it -> silent eager fallback. Widen vocoder_cg_max_frames."
    )


def test_steady_window_and_cap_track_stride_change():
    # '83' = followup+overlap at default. It MUST move when params change, and the
    # derived cap must move with it -- the whole point of deriving from params.
    default = set(_windows(75, 75, 8, 4, 1500))
    assert 83 in default, f"expected a steady window of 83 at default; tail={sorted(default)[-6:]}"
    assert max(default) <= _derived_cap(75, 75, 8, 4)  # 87

    bumped = set(_windows(150, 75, 8, 4, 3000))
    # stride 75 -> 150: the first window (~stride) now exceeds the OLD 87 cap...
    assert max(bumped) > 87, "bumping stride should push a window past the old cap"
    # ...but the derived cap grew to 162 and still covers everything.
    assert max(bumped) <= _derived_cap(150, 75, 8, 4)  # 162


@pytest.mark.parametrize("stride,followup,overlap,holdback", CONFIGS)
def test_catch_up_after_initial_chunk_stays_capped(
    stride, followup, overlap, holdback
):
    # Fast-first-audio (initial_codec_chunk_frames > 0) emits a tiny first window, so
    # the next decode catches up all buffered frames at once -- the window that used to
    # reach 139 > 87 and miss the CUDA graph. The window cap must keep every catch-up
    # window inside the captured range. Without the cap this fails (e.g. T=139 > 87).
    cap = _derived_cap(stride, followup, overlap, holdback)
    seen: set[int] = set()
    for n in range(_N, 3 * max(stride, followup) + 3 * _N):
        seen.update(_windows(stride, followup, overlap, holdback, n, initial_chunk=1))
    assert seen, "no decode windows emitted"
    assert max(seen) <= cap, (
        f"catch-up window T={max(seen)} exceeds derived cap {cap} for "
        f"(stride={stride}, followup={followup}, overlap={overlap}, holdback={holdback}); "
        f"the CG range would miss it -> silent eager fallback."
    )


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
