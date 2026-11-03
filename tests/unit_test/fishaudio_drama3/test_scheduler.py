# SPDX-License-Identifier: Apache-2.0
"""CPU coverage of Drama's production interval hooks and completion timings."""

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
    def __init__(self, running: bool, enqueue_seconds: float) -> None:
        self.configure_interval(0.1, 8)
        self.prefill_decode_interval = 4
        self._prefill_decode_interval_remaining = (
            8  # noqa: leading-underscore  # production hook
        )
        self.running_batch = SimpleNamespace(is_empty=lambda: not running)
        self.waiting_queue = [SimpleNamespace(_coalesce_enqueue_t=enqueue_seconds)]


def make_batch(prefill: bool, launch_seconds: float = 0.0) -> SimpleNamespace:
    return SimpleNamespace(
        forward_mode=SimpleNamespace(
            is_extend=lambda: prefill,
            is_decode=lambda: not prefill,
        ),
        launch_ts=launch_seconds,
    )


@pytest.mark.parametrize(
    "running,now_seconds,expected",
    [(True, 0.099, False), (True, 0.101, True), (False, 0.0, True)],
)
def test_budget_releases_prefill_prediction(
    running: bool, now_seconds: float, expected: bool
) -> None:
    scheduler = BudgetScheduler(running, 0.0)

    with patch(
        "sglang_omni.models.fishaudio_drama3.scheduler.time.perf_counter",
        return_value=now_seconds,
    ):
        actual = scheduler.is_prefill_expected()

    assert actual is expected


def test_low_sample_count_keeps_configured_interval() -> None:
    scheduler = BudgetScheduler(True, 0.0)
    batch = make_batch(True)

    scheduler._arm_prefill_decode_interval(
        batch
    )  # noqa: leading-underscore  # production hook

    assert (
        scheduler._prefill_decode_interval_remaining == 4
    )  # noqa: leading-underscore  # production hook
    assert scheduler.interval_reason == "warmup"


@pytest.mark.parametrize("maximum_steps", [8, 24])
def test_completed_cost_ratio_is_bounded_and_slow_decode_reduces_interval(
    maximum_steps: int,
) -> None:
    scheduler = BudgetScheduler(True, 0.0)
    prefill = make_batch(True)
    decode = make_batch(False)
    scheduler.configure_interval(0.1, maximum_steps)
    completion_seconds = [0.5] * 4 + [1.0, 1.01, 1.02, 1.03, 1.04]

    with patch(
        "sglang_omni.models.fishaudio_drama3.scheduler.time.monotonic",
        side_effect=completion_seconds,
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
    scheduler._arm_prefill_decode_interval(
        prefill
    )  # noqa: leading-underscore  # production hook
    fast_interval = (
        scheduler._prefill_decode_interval_remaining
    )  # noqa: leading-underscore  # production hook
    with patch(
        "sglang_omni.models.fishaudio_drama3.scheduler.time.monotonic",
        return_value=1.54,
    ):
        scheduler.process_batch_result(decode, None)
    scheduler._arm_prefill_decode_interval(
        prefill
    )  # noqa: leading-underscore  # production hook

    assert fast_interval == maximum_steps
    assert (
        scheduler._prefill_decode_interval_remaining == 5
    )  # noqa: leading-underscore  # production hook
    assert scheduler.interval_reason == "cost_ratio"
    assert scheduler.prefill_seconds == pytest.approx(0.5)
