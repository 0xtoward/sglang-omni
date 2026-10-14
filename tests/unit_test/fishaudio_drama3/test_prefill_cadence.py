# SPDX-License-Identifier: Apache-2.0
"""Drama 3 prefill cadence: the streaming window budget, the load-to-interval table, and the releases."""

from __future__ import annotations

import time
from types import SimpleNamespace

import pytest

from sglang_omni.models.fishaudio_drama3.config import (
    DEFAULT_STREAM_FOLLOWUP_STRIDE,
    DEFAULT_STREAM_FOLLOWUP_STRIDE_RAMP,
    DEFAULT_STREAM_STRIDE,
)
from sglang_omni.models.fishaudio_drama3.scheduler import (
    AUDIO_FRAME_SECONDS,
    Drama3Scheduler,
    PrefillCadence,
    critical_step_fraction,
)

CRITICAL_STEP_S = 0.025


def forward_mode(*, extend: bool) -> SimpleNamespace:
    return SimpleNamespace(is_extend=lambda: extend, is_decode=lambda: not extend)


def launched_batch(
    *, extend: bool, launch_s: float, after_idle_gap: bool = False
) -> SimpleNamespace:
    return SimpleNamespace(
        forward_mode=forward_mode(extend=extend),
        launch_ts=launch_s,
        after_idle_gap=after_idle_gap,
    )


def request(*, finished: bool = False, waited_s: float = 0.0) -> SimpleNamespace:
    return SimpleNamespace(
        finished=lambda: finished, _coalesce_enqueue_t=time.perf_counter() - waited_s
    )


def scheduler(
    *,
    running: list[SimpleNamespace],
    waiting: list[SimpleNamespace] = (),
    decode_step_s: float | None = None,
    cadence: PrefillCadence | None = None,
    prefill_decode_interval: int = 8,
) -> Drama3Scheduler:
    instance = object.__new__(Drama3Scheduler)
    instance.running_batch = SimpleNamespace(reqs=list(running))
    instance.last_batch = None
    instance.waiting_queue = list(waiting)
    if cadence is not None:
        instance.install_cadence(cadence)
    else:
        pass
    instance.decode_step_s = decode_step_s
    instance.prefill_decode_interval = prefill_decode_interval
    instance.require_mlp_sync = False
    instance._prefill_decode_interval_remaining = 0  # noqa: leading-underscore  # upstream spelling, or the public name is already taken
    return instance


def cadence(*, follow_frame_period: bool = False) -> PrefillCadence:
    return PrefillCadence(
        max_interval=8,
        critical_step_s=CRITICAL_STEP_S,
        follow_frame_period=follow_frame_period,
    )


def test_default_windows_starve_a_stream_above_a_25_ms_step() -> None:
    """Window 5 (frames 64 to 128) arrives 118 steps after the first and has 64 frames to cover."""
    fraction = critical_step_fraction(
        DEFAULT_STREAM_STRIDE,
        DEFAULT_STREAM_FOLLOWUP_STRIDE_RAMP,
        DEFAULT_STREAM_FOLLOWUP_STRIDE,
    )
    assert fraction == pytest.approx(64 / 118)
    assert fraction * AUDIO_FRAME_SECONDS == pytest.approx(0.0252, abs=1e-4)


def test_decode_step_is_the_launch_cadence_of_back_to_back_decodes() -> None:
    """A prefill or an idle gap between two decodes is not a decode step."""
    instance = scheduler(running=[request()], cadence=cadence())
    for batch in [
        launched_batch(extend=False, launch_s=1.000),
        launched_batch(extend=False, launch_s=1.020),
        launched_batch(extend=True, launch_s=1.030),
        launched_batch(extend=False, launch_s=1.070),
        launched_batch(extend=False, launch_s=1.200, after_idle_gap=True),
        launched_batch(extend=False, launch_s=1.224),
    ]:
        instance.last_batch = batch
        instance.observe_decode_step()
        instance.observe_decode_step()  # a second look at the same launch adds nothing
    weight = cadence().step_weight
    assert instance.decode_step_count == 2
    assert instance.decode_step_s == pytest.approx(
        weight * 0.024 + (1 - weight) * 0.020
    )
    period_weight = cadence().period_weight
    assert (
        instance.frame_period_count == 3
    ), "the prefill gap counts, the idle gap does not"
    assert instance.frame_period_s == pytest.approx(
        period_weight * 0.024
        + (1 - period_weight) * (period_weight * 0.050 + (1 - period_weight) * 0.020)
    )


def measure(
    instance: Drama3Scheduler, *, period_fraction: float, prefill_share: float
) -> None:
    """A frame period at period_fraction of the target, prefill_share of it prefills."""
    period_s = period_fraction * instance.cadence.period_target * CRITICAL_STEP_S
    instance.frame_period_s = period_s
    instance.slow_decode_step_s = (1 - prefill_share) * period_s
    instance.decode_steps_at_level = instance.cadence.level_hold_steps


def test_frame_period_steps_the_interval_one_rung_per_hold() -> None:
    instance = scheduler(running=[request()], cadence=cadence(follow_frame_period=True))
    measure(instance, period_fraction=1.1, prefill_share=0.4)
    assert instance.choose_interval() == 1
    assert instance.choose_interval() == 1, "no second rung before the hold passes"
    measure(instance, period_fraction=1.1, prefill_share=0.4)
    assert instance.choose_interval() == 2
    measure(instance, period_fraction=0.95, prefill_share=0.4)
    assert instance.choose_interval() == 2, "inside the band the rung stays"
    measure(instance, period_fraction=0.80, prefill_share=0.4)
    assert instance.choose_interval() == 1
    instance.running_batch.reqs = [request(finished=True)]
    assert instance.choose_interval() == 0, "nothing decoding, nothing to defer"
    assert instance.cadence_level == 1, "the rung survives an idle decode"


def test_a_decode_bound_frame_period_does_not_climb() -> None:
    """Batching prefills cannot shorten a frame period that decode steps fill."""
    instance = scheduler(running=[request()], cadence=cadence(follow_frame_period=True))
    measure(instance, period_fraction=1.1, prefill_share=0.15)
    assert instance.choose_interval() == 0, "a small prefill share does not climb"
    measure(instance, period_fraction=1.1, prefill_share=0.4)
    assert instance.choose_interval() == 1
    measure(instance, period_fraction=1.1, prefill_share=0.05)
    assert instance.choose_interval() == 1, "a rung that batched the prefills stays"


@pytest.mark.parametrize(
    ("load", "interval"),
    [(0.30, 0), (0.50, 1), (0.70, 2), (0.85, 4), (0.95, 8)],
)
def test_interval_grows_with_the_decode_step(load: float, interval: int) -> None:
    instance = scheduler(
        running=[request()], decode_step_s=load * CRITICAL_STEP_S, cadence=cadence()
    )
    assert instance.choose_interval() == interval


def test_interval_steps_down_only_past_the_hysteresis_band() -> None:
    instance = scheduler(
        running=[request()], decode_step_s=0.85 * CRITICAL_STEP_S, cadence=cadence()
    )
    assert instance.choose_interval() == 4
    instance.decode_step_s = 0.78 * CRITICAL_STEP_S  # below 0.80, inside the band
    assert instance.choose_interval() == 4, "a small dip keeps the interval"
    instance.decode_step_s = 0.70 * CRITICAL_STEP_S  # below 0.80 - 0.05
    assert instance.choose_interval() == 2, "a clear drop steps down one level"


def test_no_decoding_request_means_no_deferral() -> None:
    instance = scheduler(
        running=[request(finished=True)],
        decode_step_s=0.95 * CRITICAL_STEP_S,
        cadence=cadence(),
    )
    assert instance.choose_interval() == 0
    instance._prefill_decode_interval_remaining = 3  # noqa: leading-underscore  # upstream spelling, or the public name is already taken
    instance.release_interval_if_due()
    assert (
        instance._prefill_decode_interval_remaining == 0
    )  # noqa: leading-underscore  # upstream spelling, or the public name is already taken


def test_a_request_past_the_wait_cap_releases_the_countdown() -> None:
    instance = scheduler(
        running=[request()], waiting=[request(waited_s=0.1)], cadence=cadence()
    )
    instance._prefill_decode_interval_remaining = 3  # noqa: leading-underscore  # upstream spelling, or the public name is already taken
    instance.release_interval_if_due()
    assert (
        instance._prefill_decode_interval_remaining == 3
    )  # noqa: leading-underscore  # upstream spelling, or the public name is already taken
    instance.waiting_queue.append(request(waited_s=0.5))
    instance.release_interval_if_due()
    assert (
        instance._prefill_decode_interval_remaining == 0
    )  # noqa: leading-underscore  # upstream spelling, or the public name is already taken


def test_without_a_cadence_a_prefill_arms_the_fixed_interval() -> None:
    instance = scheduler(running=[request()], prefill_decode_interval=4)
    instance._arm_prefill_decode_interval(  # noqa: leading-underscore  # upstream spelling, or the public name is already taken
        SimpleNamespace(forward_mode=forward_mode(extend=True))
    )
    assert (
        instance._prefill_decode_interval_remaining == 4
    )  # noqa: leading-underscore  # upstream spelling, or the public name is already taken
    instance.release_interval_if_due()
    assert (
        instance._prefill_decode_interval_remaining == 4
    ), "no cadence, no release"  # noqa: leading-underscore  # upstream spelling, or the public name is already taken


def test_with_a_cadence_a_prefill_arms_the_chosen_interval() -> None:
    instance = scheduler(
        running=[request()], decode_step_s=0.70 * CRITICAL_STEP_S, cadence=cadence()
    )
    instance._arm_prefill_decode_interval(  # noqa: leading-underscore  # upstream spelling, or the public name is already taken
        SimpleNamespace(forward_mode=forward_mode(extend=True))
    )
    assert (
        instance._prefill_decode_interval_remaining == 2
    )  # noqa: leading-underscore  # upstream spelling, or the public name is already taken
    assert instance.prefills_by_interval == {2: 1}
    instance._arm_prefill_decode_interval(  # noqa: leading-underscore  # upstream spelling, or the public name is already taken
        SimpleNamespace(forward_mode=forward_mode(extend=False))
    )
    assert (
        instance._prefill_decode_interval_remaining == 2
    ), "a decode batch does not re-arm"  # noqa: leading-underscore  # upstream spelling, or the public name is already taken
