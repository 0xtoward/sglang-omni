# SPDX-License-Identifier: Apache-2.0
"""OmniScheduler specialization that sets Drama 3's prefill/decode cadence from load."""

from __future__ import annotations

import time
from dataclasses import dataclass

from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.managers.scheduler import Scheduler as UpstreamScheduler

from sglang_omni.models.fishaudio_drama3.request_builders import Drama3RequestData
from sglang_omni.scheduling.omni_scheduler import OmniScheduler

# note (0xtoward): The S2-Pro codec decodes one frame from 2048 samples at 44.1 kHz.
AUDIO_FRAME_SECONDS = 2048 / 44100


def critical_step_fraction(
    first_stride: int, stride_ramp: tuple[int, ...], followup_stride: int
) -> float:
    """The largest steady decode step, in frames, that never starves a stream.

    Playback starts with the first window; window k arrives (W_k - W_1) steps later,
    when W_(k-1) frames have played.
    """
    window_ends = []
    total = 0
    for stride in (first_stride, *stride_ramp, *([followup_stride] * 40)):
        total += stride
        window_ends.append(total)
    return min(
        window_ends[k - 1] / (window_ends[k] - window_ends[0])
        for k in range(1, len(window_ends))
    )


@dataclass(frozen=True, kw_only=True)
class PrefillCadence:
    """Decode steps between prefills by how close the decode step is to the critical one.

    Above load_bounds[i] the interval is load_intervals[i + 1]; above the last bound it
    is max_interval. A waiting request older than wait_cap_s, or an idle decode,
    releases the countdown.
    """

    max_interval: int
    critical_step_s: float
    wait_cap_s: float = 0.30
    load_bounds: tuple[float, ...] = (0.45, 0.65, 0.80, 0.92)
    load_intervals: tuple[int, ...] = (0, 1, 2, 4)
    hysteresis: float = 0.05
    step_weight: float = 0.25


class Drama3Scheduler(OmniScheduler[Drama3RequestData]):
    # note (0xtoward): Drama3EngineBuilder.post_scheduler_setup installs a cadence; without one the scheduler keeps upstream's fixed countdown.
    cadence: PrefillCadence | None = None
    decode_step_s: float | None = None
    cadence_level: int = 0

    def decoding_requests(self) -> list[Req]:
        """The requests the next decode advances.

        Upstream swaps an empty running batch for the last prefill batch before
        self.running_batch is updated, so both are read.
        """
        requests = self.running_batch.reqs
        last_batch = self.last_batch
        if (
            not requests
            and last_batch is not None
            and last_batch.forward_mode.is_extend()
        ):
            requests = last_batch.reqs
        else:
            pass
        return [request for request in requests if not request.finished()]

    def release_interval_if_due(self) -> None:
        if (
            self.cadence is None or self._prefill_decode_interval_remaining == 0
        ):  # noqa: leading-underscore  # upstream spelling, or the public name is already taken
            return
        else:
            pass
        now_s = time.perf_counter()
        oldest_wait_s = max(
            (
                now_s - request._coalesce_enqueue_t for request in self.waiting_queue
            ),  # noqa: leading-underscore  # upstream spelling, or the public name is already taken
            default=0.0,
        )
        if not self.decoding_requests() or oldest_wait_s >= self.cadence.wait_cap_s:
            self._prefill_decode_interval_remaining = 0  # noqa: leading-underscore  # upstream spelling, or the public name is already taken
        else:
            pass

    def observe_decode_step(self) -> None:
        """Fold the wall time of the step that just finished into the decode-step EWMA."""
        last_batch = self.last_batch
        if (
            self.cadence is None
            or last_batch is None
            or not last_batch.forward_mode.is_decode()
            or getattr(last_batch, "launch_ts", None) is None
        ):
            return
        else:
            pass
        step_s = time.monotonic() - last_batch.launch_ts
        previous_s = self.decode_step_s
        weight = self.cadence.step_weight
        self.decode_step_s = (
            step_s
            if previous_s is None
            else weight * step_s + (1 - weight) * previous_s
        )

    def get_next_batch_to_run(self):
        self.observe_decode_step()
        self.release_interval_if_due()
        return super().get_next_batch_to_run()

    def is_prefill_expected(self) -> bool:
        # note (0xtoward): The async decode loop predicts this round's prefill; releasing here keeps the prediction equal to the decision.
        self.release_interval_if_due()
        return super().is_prefill_expected()

    def _arm_prefill_decode_interval(  # noqa: leading-underscore  # upstream spelling, or the public name is already taken
        self, batch: ScheduleBatch | None
    ) -> None:
        if self.cadence is None:
            UpstreamScheduler._arm_prefill_decode_interval(  # noqa: leading-underscore  # upstream spelling, or the public name is already taken
                self, batch
            )
        elif batch is not None and (
            batch.is_extend_in_batch
            if self.require_mlp_sync
            else batch.forward_mode.is_extend()
        ):
            self._prefill_decode_interval_remaining = (  # noqa: leading-underscore  # upstream spelling, or the public name is already taken
                self.choose_interval()
            )
        else:
            pass

    def choose_interval(self) -> int:
        cadence = self.cadence
        assert cadence is not None
        if not self.decoding_requests() or self.decode_step_s is None:
            self.cadence_level = 0
            return 0
        else:
            pass
        load = self.decode_step_s / cadence.critical_step_s
        level = sum(load >= bound for bound in cadence.load_bounds)
        # note (0xtoward): Stepping down waits for the load to clear the band below the current bound, so the interval does not flap.
        if level < self.cadence_level:
            lower_bound = (
                cadence.load_bounds[self.cadence_level - 1] - cadence.hysteresis
            )
            level = self.cadence_level - 1 if load < lower_bound else self.cadence_level
        else:
            pass
        self.cadence_level = level
        intervals = (*cadence.load_intervals, cadence.max_interval)
        return min(intervals[level], cadence.max_interval)
