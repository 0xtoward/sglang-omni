# SPDX-License-Identifier: Apache-2.0
"""Drama 3 prefill cadence: the interval from arrivals per decode step, the first window from
the decode step time."""
from __future__ import annotations

import pytest

from sglang_omni.models.fishaudio_drama3.prefill_cadence import (
    PrefillCadence,
    PrefillCadenceConfig,
)

STEP_S = 0.016
NOW_S = 1.0
BATCH_SIZE = 44


def make_cadence(arrivals_per_second: float, step_s: float = STEP_S) -> PrefillCadence:
    cadence = PrefillCadence(PrefillCadenceConfig())
    cadence.observe_decode_step(step_s, BATCH_SIZE)
    cadence.observe_admission(
        NOW_S, round(arrivals_per_second * cadence.config.rate_window_s)
    )
    return cadence


@pytest.mark.parametrize(
    ("arrivals_per_second", "interval"), [(14.0, 8), (20.0, 4), (45.0, 2), (3.0, 0)]
)
def test_interval_is_the_shortest_wait_expected_to_gather_the_batch_target(
    arrivals_per_second: float, interval: int
) -> None:
    cadence = make_cadence(arrivals_per_second)

    chosen = cadence.choose_interval(NOW_S)

    assert chosen == interval, f"{arrivals_per_second} arrivals/s at 16 ms per step"


def test_a_wait_ends_once_enough_requests_wait_or_the_oldest_waited_too_long() -> None:
    cadence = PrefillCadence(PrefillCadenceConfig())
    config = cadence.config

    released = [
        cadence.should_release(config.release_waiting, 0.0),
        cadence.should_release(1, config.wait_cap_s),
        cadence.should_release(1, config.wait_cap_s / 2),
    ]

    assert released == [True, True, False], "only the waiting count or the cap releases"


@pytest.mark.parametrize(("step_s", "first_window"), [(0.016, 4), (0.022, 10)])
def test_first_window_is_small_while_its_audio_outlasts_making_the_next_window(
    step_s: float, first_window: int
) -> None:
    cadence = make_cadence(0.0, step_s)

    chosen = cadence.choose_first_window()

    assert chosen == first_window, f"{step_s * 1000} ms per decode step"
