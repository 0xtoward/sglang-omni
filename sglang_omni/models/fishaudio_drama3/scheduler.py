# SPDX-License-Identifier: Apache-2.0
"""Drama 3 engine scheduler that picks each prefill's interval and first window online."""
from __future__ import annotations

import logging
import time

from sglang.srt.managers.schedule_batch import ScheduleBatch

from sglang_omni.models.fishaudio_drama3.prefill_cadence import PrefillCadence
from sglang_omni.models.fishaudio_drama3.request_builders import Drama3RequestData
from sglang_omni.scheduling.omni_scheduler import OmniScheduler

logger = logging.getLogger(__name__)

REPORT_INTERVAL_S = 30.0


class Drama3Scheduler(OmniScheduler[Drama3RequestData]):
    """Upstream's interval countdown, armed with an interval the cadence picks per prefill."""

    def install_cadence(self, cadence: PrefillCadence) -> None:
        self.cadence = cadence
        self.round_started_s: float | None = None
        self.last_batch_was_decode = False
        self.last_decode_batch_size = 0
        self.reported_s = time.perf_counter()

    def get_next_batch_to_run(self):
        cadence = self.cadence
        now_s = time.perf_counter()
        if self.last_batch_was_decode and self.round_started_s is not None:
            cadence.observe_decode_step(
                now_s - self.round_started_s, self.last_decode_batch_size
            )
        else:
            pass
        if self._prefill_decode_interval_remaining > 0 and self.waiting_queue:  # noqa: leading-underscore  # upstream spelling, or the public name is already taken
            oldest_wait_s = max(
                now_s - request._coalesce_enqueue_t  # noqa: leading-underscore  # upstream spelling, or the public name is already taken
                for request in self.waiting_queue
            )
            if cadence.should_release(len(self.waiting_queue), oldest_wait_s):
                self._prefill_decode_interval_remaining = 0  # noqa: leading-underscore  # upstream spelling, or the public name is already taken
            else:
                pass
        else:
            pass
        if now_s - self.reported_s >= REPORT_INTERVAL_S:
            logger.info(cadence.report(now_s))
            self.reported_s = now_s
        else:
            pass
        self.round_started_s = now_s
        batch = super().get_next_batch_to_run()
        self.last_batch_was_decode = batch is not None and batch.forward_mode.is_decode()
        self.last_decode_batch_size = len(batch.reqs) if self.last_batch_was_decode else 0
        return batch

    def _arm_prefill_decode_interval(  # noqa: leading-underscore  # upstream spelling, or the public name is already taken
        self, batch: ScheduleBatch | None
    ) -> None:
        if batch is not None and batch.forward_mode.is_extend():
            now_s = time.perf_counter()
            self.cadence.observe_admission(now_s, len(batch.reqs))
            for request in batch.reqs:
                request_data = request.omni_data
                if request_data.first_window == 0:
                    request_data.first_window = self.cadence.choose_first_window()
                else:
                    pass
            self._prefill_decode_interval_remaining = self.cadence.choose_interval(now_s)  # noqa: leading-underscore  # upstream spelling, or the public name is already taken
        else:
            pass
