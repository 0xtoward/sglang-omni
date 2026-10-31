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
    ("prefill_s", "arrivals_per_second", "interval"),
    [(0.040, 14.0, 8), (0.012, 14.0, 4), (0.006, 14.0, 2), (0.040, 3.0, 0)],
)
def test_interval_is_the_shortest_wait_that_keeps_prefills_within_the_overhead_target(
    prefill_s: float, arrivals_per_second: float, interval: int
) -> None:
    cadence = make_cadence(arrivals_per_second)
    cadence.observe_prefill(prefill_s)

    chosen = cadence.choose_interval(NOW_S, running_count=48)

    assert (
        chosen == interval
    ), f"{prefill_s * 1000} ms prefills, {arrivals_per_second}/s"


@pytest.mark.parametrize(("running_count", "interval"), [(15, 0), (16, 16), (48, 16)])
def test_gated_interval_holds_prefills_only_for_a_large_running_batch(
    running_count: int, interval: int
) -> None:
    cadence = PrefillCadence(
        PrefillCadenceConfig(is_gated=True, min_batch=16, longest_interval=16)
    )

    assert cadence.choose_interval(NOW_S, running_count) == interval


def test_a_wait_ends_once_the_oldest_request_waited_too_long() -> None:
    cadence = PrefillCadence(PrefillCadenceConfig())
    config = cadence.config

    released = [
        cadence.should_release(64, 0.0),
        cadence.should_release(1, config.wait_cap_s),
        cadence.should_release(1, config.wait_cap_s / 2),
    ]

    assert released == [False, True, False], "only the wait cap releases by default"


@pytest.mark.parametrize(("step_s", "first_window"), [(0.016, 4), (0.022, 10)])
def test_first_window_is_small_while_its_audio_outlasts_making_the_next_window(
    step_s: float, first_window: int
) -> None:
    cadence = PrefillCadence(PrefillCadenceConfig(large_first_window=10))
    cadence.observe_decode_step(step_s, BATCH_SIZE)

    chosen = cadence.choose_first_window()

    assert chosen == first_window, f"{step_s * 1000} ms per decode step"


@pytest.mark.parametrize(
    ("arrivals_per_second", "interval"), [(2.5, 0), (5.0, 0), (10.0, 16), (24.0, 16)]
)
def test_arrival_aware_interval_waits_only_while_prefills_would_take_too_much_time(
    arrivals_per_second: float, interval: int
) -> None:
    cadence = PrefillCadence(
        PrefillCadenceConfig(is_arrival_aware=True, longest_interval=32)
    )
    cadence.observe_decode_step(STEP_S, BATCH_SIZE)
    cadence.observe_admission(
        NOW_S, round(arrivals_per_second * cadence.config.rate_window_s)
    )
    cadence.observe_prefill(0.035)

    assert cadence.choose_interval(NOW_S, running_count=BATCH_SIZE) == interval


def test_ttfa_target_leaves_the_wait_after_prefill_and_first_window() -> None:
    cadence = PrefillCadence(PrefillCadenceConfig(ttfa_target_s=0.4))
    cadence.observe_decode_step(STEP_S, BATCH_SIZE)
    cadence.observe_prefill(0.072)

    assert cadence.wait_cap_s() == pytest.approx(0.4 - 0.072 - 4 * STEP_S - 0.03)
