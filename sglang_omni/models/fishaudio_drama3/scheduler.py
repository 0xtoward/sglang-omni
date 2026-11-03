# SPDX-License-Identifier: Apache-2.0
"""Bounded prefill cadence using completed host timings for Drama 3."""

from __future__ import annotations

import logging
import math
import time
from collections import Counter
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import ScheduleBatch
    from sglang.srt.managers.scheduler import GenerationBatchResult
else:
    pass

logger = logging.getLogger(__name__)
EWMA_WEIGHT = 0.2
MINIMUM_SAMPLES = 4
REPORT_INTERVAL_SECONDS = 30.0


class Drama3BudgetInterval:
    """Model-local overrides of the scheduler's existing interval hooks."""

    def configure_interval(
        self, queue_budget_seconds: float, maximum_steps: int
    ) -> None:
        self.queue_budget_seconds: float = queue_budget_seconds
        self.maximum_steps: int = maximum_steps
        self.prefill_seconds: float = 0.0
        self.decode_seconds: float = 0.0
        self.prefill_samples: int = 0
        self.decode_samples: int = 0
        self.last_decode_completed_seconds: float | None = None
        self.interval_counts: Counter[int] = Counter()
        self.release_counts: Counter[str] = Counter()
        self.reported_seconds: float = time.monotonic()
        self.interval_reason: Literal[
            "warmup", "cost_ratio", "no_running", "queue_budget"
        ] = "warmup"

    def process_batch_result(
        self, batch: ScheduleBatch, result: GenerationBatchResult
    ) -> None:
        completed_seconds = time.monotonic()
        if batch.forward_mode.is_extend():
            elapsed_seconds = completed_seconds - batch.launch_ts
            self.prefill_seconds = (
                elapsed_seconds
                if self.prefill_samples == 0
                else self.prefill_seconds
                + EWMA_WEIGHT * (elapsed_seconds - self.prefill_seconds)
            )
            self.prefill_samples += 1
            self.last_decode_completed_seconds = None
        elif batch.forward_mode.is_decode():
            if self.last_decode_completed_seconds is not None:
                elapsed_seconds = completed_seconds - self.last_decode_completed_seconds
                self.decode_seconds = (
                    elapsed_seconds
                    if self.decode_samples == 0
                    else self.decode_seconds
                    + EWMA_WEIGHT * (elapsed_seconds - self.decode_seconds)
                )
                self.decode_samples += 1
            else:
                pass
            self.last_decode_completed_seconds = completed_seconds
        else:
            self.last_decode_completed_seconds = None
        super().process_batch_result(batch, result)

    def get_next_batch_to_run(self) -> ScheduleBatch | None:
        self.release_interval_if_due()
        batch = super().get_next_batch_to_run()
        if batch is None:
            self.last_decode_completed_seconds = None
        else:
            pass
        now_seconds = time.monotonic()
        if now_seconds - self.reported_seconds >= REPORT_INTERVAL_SECONDS:
            logger.info(
                f"Drama interval: prefill_ms={1000 * self.prefill_seconds:.2f} "
                f"decode_ms={1000 * self.decode_seconds:.2f} "
                f"samples={self.prefill_samples}/{self.decode_samples} "
                f"intervals={dict(self.interval_counts)} releases={dict(self.release_counts)} "
                f"reason={self.interval_reason}"
            )
            self.reported_seconds = now_seconds
        else:
            pass
        return batch

    def is_prefill_expected(self) -> bool:
        self.release_interval_if_due()
        return super().is_prefill_expected()

    def release_interval_if_due(self) -> None:
        if self.running_batch.is_empty():
            reason = "no_running"
        elif self.waiting_queue:
            now_seconds = time.perf_counter()
            oldest_wait_seconds = max(
                now_seconds
                - request._coalesce_enqueue_t  # noqa: leading-underscore  # existing scheduler timestamp
                for request in self.waiting_queue
            )
            if oldest_wait_seconds >= self.queue_budget_seconds:
                reason = "queue_budget"
            else:
                return
        else:
            return
        if (
            self._prefill_decode_interval_remaining > 0
        ):  # noqa: leading-underscore  # upstream hook
            self._prefill_decode_interval_remaining = (
                0  # noqa: leading-underscore  # upstream hook
            )
            self.release_counts[reason] += 1
            self.interval_reason = reason
        else:
            pass

    def _arm_prefill_decode_interval(
        self, batch: ScheduleBatch | None
    ) -> None:  # noqa: leading-underscore  # upstream hook
        if batch is None or not batch.forward_mode.is_extend():
            return
        elif min(self.prefill_samples, self.decode_samples) < MINIMUM_SAMPLES:
            interval = min(self.maximum_steps, self.prefill_decode_interval)
            self.interval_reason = "warmup"
        else:
            # note (0xtoward): Equal P and aggregate D time is a heuristic amortization target.
            interval = min(
                self.maximum_steps,
                max(
                    1, math.ceil(self.prefill_seconds / max(self.decode_seconds, 1e-6))
                ),
            )
            self.interval_reason = "cost_ratio"
        self._prefill_decode_interval_remaining = (
            interval  # noqa: leading-underscore  # upstream hook
        )
        self.interval_counts[interval] += 1
