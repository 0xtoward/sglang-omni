# SPDX-License-Identifier: Apache-2.0
"""CPU coverage of Drama's production first-audio deadline hooks."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from sglang_omni.models.fishaudio_drama3.scheduler import Drama3BudgetInterval


class RecordingScheduler:
    def process_batch_result(self, batch: SimpleNamespace, result: None) -> None:
        self.completed_batch = batch

    def is_prefill_expected(self) -> bool:
        return (
            self._prefill_decode_interval_remaining == 0
        )  # noqa: leading-underscore  # production hook


class BudgetScheduler(Drama3BudgetInterval, RecordingScheduler):
    def __init__(self, running: bool = True, first_window_frames: int = 4) -> None:
        self.configure_interval(1.0, 0.125, 24, first_window_frames)
        self._prefill_decode_interval_remaining = (
            0  # noqa: leading-underscore  # production hook
        )
        self.running_batch = SimpleNamespace(is_empty=lambda: not running)
        self.waiting_queue = [SimpleNamespace(rid="waiting", _coalesce_enqueue_t=0.0)]


def make_batch(prefill: bool, launch_seconds: float = 0.0) -> SimpleNamespace:
    return SimpleNamespace(
        forward_mode=SimpleNamespace(
            is_extend=lambda: prefill, is_decode=lambda: not prefill
        ),
        launch_ts=launch_seconds,
    )


def complete_warmup(scheduler: BudgetScheduler) -> None:
    prefill = make_batch(True)
    decode = make_batch(False)
    completed_seconds = [0.125] * 4 + [1.0, 1.03125, 1.0625, 1.09375, 1.125]
    with patch(
        "sglang_omni.models.fishaudio_drama3.scheduler.time.monotonic",
        side_effect=completed_seconds,
    ):
        scheduler.process_batch_result(prefill, None)
        scheduler.process_batch_result(prefill, None)
        scheduler.process_batch_result(prefill, None)
        scheduler.process_batch_result(prefill, None)
        scheduler.process_batch_result(decode, None)
        scheduler.process_batch_result(decode, None)
        scheduler.process_batch_result(decode, None)
        scheduler.process_batch_result(decode, None)
        scheduler.process_batch_result(decode, None)


@pytest.mark.parametrize("first_window_frames,expected_steps", [(4, 17), (10, 11)])
def test_deadline_subtracts_completed_costs_and_actual_first_window(
    first_window_frames: int, expected_steps: int
) -> None:
    scheduler = BudgetScheduler(first_window_frames=first_window_frames)
    complete_warmup(scheduler)

    with patch(
        "sglang_omni.models.fishaudio_drama3.scheduler.time.perf_counter",
        return_value=0.125,
    ):
        scheduler.is_prefill_expected()

    assert (
        scheduler._prefill_decode_interval_remaining == expected_steps
    )  # noqa: leading-underscore  # production hook
    assert scheduler.interval_reason == "truebudget"
    assert scheduler.prefill_seconds == 0.125
    assert scheduler.decode_seconds == 0.03125


def test_deadline_tightens_without_rearming_the_same_request() -> None:
    scheduler = BudgetScheduler()
    complete_warmup(scheduler)

    with patch(
        "sglang_omni.models.fishaudio_drama3.scheduler.time.perf_counter",
        return_value=0.5625,
    ):
        scheduler.is_prefill_expected()
        initial_steps = (
            scheduler._prefill_decode_interval_remaining
        )  # noqa: leading-underscore  # production hook
        scheduler._prefill_decode_interval_remaining -= (
            1  # noqa: leading-underscore  # upstream consumes one scheduling pass
        )
        scheduler.is_prefill_expected()
        remaining_steps = (
            scheduler._prefill_decode_interval_remaining
        )  # noqa: leading-underscore  # production hook
    with patch(
        "sglang_omni.models.fishaudio_drama3.scheduler.time.perf_counter",
        return_value=0.7,
    ):
        released = scheduler.is_prefill_expected()

    assert initial_steps == 3
    assert remaining_steps == 2
    assert released


def test_idle_queue_has_no_budget_and_new_arrival_is_evaluated() -> None:
    scheduler = BudgetScheduler()
    complete_warmup(scheduler)
    waiting = scheduler.waiting_queue
    scheduler.waiting_queue = []

    scheduler.is_prefill_expected()
    idle_steps = (
        scheduler._prefill_decode_interval_remaining
    )  # noqa: leading-underscore  # production hook
    scheduler.waiting_queue = waiting
    with patch(
        "sglang_omni.models.fishaudio_drama3.scheduler.time.perf_counter",
        return_value=0.7,
    ):
        released = scheduler.is_prefill_expected()

    assert idle_steps == 0
    assert released
    assert scheduler.interval_reason == "truebudget"


@pytest.mark.parametrize(
    "running,waited_seconds,expected_steps",
    [(True, 0.0, 4), (True, 1.0, 0), (False, 0.0, 0)],
)
def test_warmup_uses_four_passes_but_releases_expired_or_idle_requests(
    running: bool, waited_seconds: float, expected_steps: int
) -> None:
    scheduler = BudgetScheduler(running=running)

    with patch(
        "sglang_omni.models.fishaudio_drama3.scheduler.time.perf_counter",
        return_value=waited_seconds,
    ):
        scheduler.is_prefill_expected()

    assert (
        scheduler._prefill_decode_interval_remaining == expected_steps
    )  # noqa: leading-underscore  # production hook


@pytest.mark.parametrize("maximum_steps", [8, 24])
def test_deadline_respects_configured_cap(maximum_steps: int) -> None:
    scheduler = BudgetScheduler()
    scheduler.configure_interval(4.0, 0.125, maximum_steps, 4)
    complete_warmup(scheduler)

    with patch(
        "sglang_omni.models.fishaudio_drama3.scheduler.time.perf_counter",
        return_value=0.0,
    ):
        scheduler.is_prefill_expected()

    assert (
        scheduler._prefill_decode_interval_remaining == maximum_steps
    )  # noqa: leading-underscore  # production hook
