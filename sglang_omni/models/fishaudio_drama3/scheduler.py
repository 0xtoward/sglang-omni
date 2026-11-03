# SPDX-License-Identifier: Apache-2.0
"""Prefill deferral from an engine-to-first-audio budget for Drama 3."""

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
        self,
        target_seconds: float,
        reserve_seconds: float,
        maximum_steps: int,
        first_window_frames: int,
    ) -> None:
        self.target_seconds: float = target_seconds
        self.reserve_seconds: float = reserve_seconds
        self.first_window_decode_steps: int = max(0, first_window_frames - 1)
        self.budget_request_id: str | None = None
        self.remaining_budget_seconds: float = 0.0
        self.allowed_steps: int = 0
        self.maximum_steps: int = maximum_steps
        self.prefill_seconds: float = 0.0
        self.decode_seconds: float = 0.0
        self.prefill_samples: int = 0
        self.decode_samples: int = 0
        self.last_decode_completed_seconds: float | None = None
        self.interval_counts: Counter[int] = Counter()
        self.decision_counts: Counter[str] = Counter()
        self.reported_seconds: float = time.monotonic()
        self.interval_reason: Literal[
            "warmup", "truebudget", "no_running", "no_waiting", "budget_expired"
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
                f"Drama deadline: prefill_ms={1000 * self.prefill_seconds:.2f} "
                f"decode_ms={1000 * self.decode_seconds:.2f} "
                f"samples={self.prefill_samples}/{self.decode_samples} "
                f"intervals={dict(self.interval_counts)} decisions={dict(self.decision_counts)} "
                f"reason={self.interval_reason} allowed={self.allowed_steps} "
                f"remaining_ms={1000 * self.remaining_budget_seconds:.2f} "
                f"target_ms={1000 * self.target_seconds:.2f} "
                f"reserve_ms={1000 * self.reserve_seconds:.2f} "
                f"first_window_decode_steps={self.first_window_decode_steps}"
            )
            self.reported_seconds = now_seconds
        else:
            pass
        return batch

    def is_prefill_expected(self) -> bool:
        self.release_interval_if_due()
        return super().is_prefill_expected()

    def release_interval_if_due(self) -> None:
        if self.running_batch.is_empty() or not self.waiting_queue:
            self.interval_reason = (
                "no_running" if self.running_batch.is_empty() else "no_waiting"
            )
            self._prefill_decode_interval_remaining = (
                0  # noqa: leading-underscore  # upstream hook
            )
            self.budget_request_id = None
            self.allowed_steps = 0
            self.remaining_budget_seconds = 0.0
            self.decision_counts[self.interval_reason] += 1
            return
        else:
            pass
        oldest_request = min(
            self.waiting_queue,
            key=lambda request: request._coalesce_enqueue_t,  # noqa: leading-underscore  # scheduler timestamp
        )
        waited_seconds = max(
            0.0, time.perf_counter() - oldest_request._coalesce_enqueue_t
        )  # noqa: leading-underscore  # scheduler timestamp
        self.remaining_budget_seconds = self.target_seconds - waited_seconds
        if self.remaining_budget_seconds <= 0:
            self.allowed_steps = 0
            self.interval_reason = "budget_expired"
        elif min(self.prefill_samples, self.decode_samples) < MINIMUM_SAMPLES:
            self.allowed_steps = min(4, self.maximum_steps)
            self.interval_reason = "warmup"
        else:
            self.remaining_budget_seconds -= (
                self.prefill_seconds
                + self.first_window_decode_steps * self.decode_seconds
                + self.reserve_seconds
            )
            self.allowed_steps = min(
                self.maximum_steps,
                math.floor(
                    max(0.0, self.remaining_budget_seconds)
                    / max(self.decode_seconds, 1e-6)
                ),
            )
            self.interval_reason = "truebudget"
        if self.budget_request_id != oldest_request.rid:
            self._prefill_decode_interval_remaining = (
                self.allowed_steps
            )  # noqa: leading-underscore  # upstream hook
            self.budget_request_id = oldest_request.rid
            self.interval_counts[self.allowed_steps] += 1
        else:
            self._prefill_decode_interval_remaining = (
                min(  # noqa: leading-underscore  # upstream hook
                    self._prefill_decode_interval_remaining,
                    self.allowed_steps,  # noqa: leading-underscore  # upstream hook
                )
            )
        self.decision_counts[self.interval_reason] += 1

    def _arm_prefill_decode_interval(
        self, batch: ScheduleBatch | None
    ) -> None:  # noqa: leading-underscore  # upstream hook
        if batch is not None and batch.forward_mode.is_extend():
            # note (0xtoward): Only an actual waiting request owns a deadline.
            self.budget_request_id = None
            self._prefill_decode_interval_remaining = (
                0  # noqa: leading-underscore  # upstream hook
            )
        else:
            pass
