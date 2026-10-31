# SPDX-License-Identifier: Apache-2.0
"""Load-adaptive prefill interval and first streaming window for Drama 3.

A prefill stops every running stream for one forward, and most of its cost does not grow
with the requests it admits (every active expert is read, the tower runs over the prompt
rows). Waiting a few decode steps lets more arrivals share one prefill, but the wait adds to
their time to first audio. The interval is therefore the shortest wait that keeps prefills
within overhead_target of the time, or no wait when even the longest wait would gather
fewer than min_arrivals requests.

The first window is the small one while decode makes frames fast enough for the small
window's audio to cover the next window, and the large one otherwise.
"""
from __future__ import annotations

import collections
from dataclasses import dataclass, field

AUDIO_FRAME_SECONDS = 2048 / 44100
INTERVAL_LEVELS = (2, 4, 8)
BATCH_BUCKET = 8


@dataclass(frozen=True, kw_only=True)
class PrefillCadenceConfig:
    overhead_target: float = 0.2
    min_arrivals: float = 1.0
    release_waiting: int = 0
    wait_cap_s: float = 0.25
    rate_window_s: float = 2.0
    step_smoothing: float = 0.1
    small_first_window: int = 4
    large_first_window: int = 10
    next_window: int = 10


@dataclass
class PrefillCadence:
    """Online signals: decode step time, admitted requests per second."""

    config: PrefillCadenceConfig
    step_s: float | None = None
    prefill_s: float | None = None
    admissions: collections.deque[tuple[float, int]] = field(
        default_factory=collections.deque
    )
    chosen_intervals: collections.Counter[int] = field(
        default_factory=collections.Counter
    )
    chosen_first_windows: collections.Counter[int] = field(
        default_factory=collections.Counter
    )
    releases: collections.Counter[str] = field(default_factory=collections.Counter)
    step_by_batch: dict[int, list[float]] = field(default_factory=dict)

    def observe_decode_step(self, step_s: float, batch_size: int) -> None:
        bucket = self.step_by_batch.setdefault(batch_size // BATCH_BUCKET * BATCH_BUCKET, [0, 0.0])
        bucket[0] += 1
        bucket[1] += step_s
        if self.step_s is None:
            self.step_s = step_s
        else:
            weight = self.config.step_smoothing
            self.step_s = (1 - weight) * self.step_s + weight * step_s

    def observe_prefill(self, prefill_s: float) -> None:
        if self.prefill_s is None:
            self.prefill_s = prefill_s
        else:
            weight = self.config.step_smoothing
            self.prefill_s = (1 - weight) * self.prefill_s + weight * prefill_s

    def observe_admission(self, now_s: float, request_count: int) -> None:
        self.admissions.append((now_s, request_count))
        while self.admissions and self.admissions[0][0] < now_s - self.config.rate_window_s:
            self.admissions.popleft()

    def arrivals_per_step(self, now_s: float) -> float:
        if self.step_s is None:
            return 0.0
        else:
            recent = sum(
                count
                for time_s, count in self.admissions
                if time_s >= now_s - self.config.rate_window_s
            )
            return recent / self.config.rate_window_s * self.step_s

    def choose_interval(self, now_s: float) -> int:
        step_s, prefill_s = self.step_s, self.prefill_s
        longest = INTERVAL_LEVELS[-1]
        if (
            step_s is None
            or prefill_s is None
            or self.arrivals_per_step(now_s) * longest < self.config.min_arrivals
        ):
            interval = 0
        else:
            interval = next(
                (
                    level
                    for level in INTERVAL_LEVELS
                    if prefill_s / (level * step_s + prefill_s)
                    <= self.config.overhead_target
                ),
                longest,
            )
        self.chosen_intervals[interval] += 1
        return interval

    def should_release(self, waiting_count: int, oldest_wait_s: float) -> bool:
        if self.config.release_waiting and waiting_count >= self.config.release_waiting:
            self.releases["waiting"] += 1
            return True
        elif oldest_wait_s >= self.config.wait_cap_s:
            self.releases["wait_cap"] += 1
            return True
        else:
            return False

    def choose_first_window(self) -> int:
        """Small while the small window's audio lasts longer than making the next window."""
        config = self.config
        if self.step_s is None or (
            config.next_window * self.step_s
            <= config.small_first_window * AUDIO_FRAME_SECONDS
        ):
            first_window = config.small_first_window
        else:
            first_window = config.large_first_window
        self.chosen_first_windows[first_window] += 1
        return first_window

    def report(self, now_s: float) -> str:
        step_ms = 1000 * self.step_s if self.step_s is not None else 0.0
        prefill_ms = 1000 * self.prefill_s if self.prefill_s is not None else 0.0
        text = (
            f"Drama cadence: step {step_ms:.1f} ms, prefill {prefill_ms:.1f} ms, arrivals/step "
            f"{self.arrivals_per_step(now_s):.2f}, intervals "
            f"{dict(sorted(self.chosen_intervals.items()))}, releases "
            f"{dict(sorted(self.releases.items()))}, first windows "
            f"{dict(sorted(self.chosen_first_windows.items()))}, decode ms by batch "
            f"{ {batch: round(1000 * total / count, 1) for batch, (count, total) in sorted(self.step_by_batch.items())} }"
        )
        self.step_by_batch.clear()
        self.chosen_intervals.clear()
        self.chosen_first_windows.clear()
        self.releases.clear()
        return text

