# SPDX-License-Identifier: Apache-2.0
"""OmniScheduler specialization that sets Drama 3's prefill/decode cadence from load."""

from __future__ import annotations

import logging
import time
from collections import Counter
from dataclasses import dataclass

from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.managers.scheduler import Scheduler as UpstreamScheduler

from sglang_omni.models.fishaudio_drama3.request_builders import Drama3RequestData
from sglang_omni.scheduling.omni_scheduler import OmniScheduler

logger = logging.getLogger(__name__)

# note (0xtoward): The S2-Pro codec decodes one frame from 2048 samples at 44.1 kHz.
AUDIO_FRAME_SECONDS = 2048 / 44100


def stream_window_ends(
    first_stride: int, stride_ramp: tuple[int, ...], followup_stride: int
) -> tuple[int, ...]:
    """Cumulative frame counts at which the stream's windows complete, far enough for any request."""
    window_ends = []
    total = 0
    for stride in (first_stride, *stride_ramp, *([followup_stride] * 400)):
        total += stride
        window_ends.append(total)
    return tuple(window_ends)


def critical_step_fraction(
    first_stride: int, stride_ramp: tuple[int, ...], followup_stride: int
) -> float:
    """The largest steady decode step, in frames, that never starves a stream.

    Playback starts with the first window; window k arrives (W_k - W_1) steps later,
    when W_(k-1) frames have played.
    """
    window_ends = stream_window_ends(first_stride, stride_ramp, followup_stride)[:48]
    return min(
        window_ends[k - 1] / (window_ends[k] - window_ends[0])
        for k in range(1, len(window_ends))
    )


def critical_window_end(
    first_stride: int, stride_ramp: tuple[int, ...], followup_stride: int
) -> int:
    """The frame count that ends the window with the tightest step budget; a stream before it is still starting."""
    window_ends = stream_window_ends(first_stride, stride_ramp, followup_stride)[:48]
    critical = min(
        range(1, len(window_ends)),
        key=lambda k: window_ends[k - 1] / (window_ends[k] - window_ends[0]),
    )
    return window_ends[critical]


@dataclass(frozen=True, kw_only=True)
class PrefillCadence:
    """Decode steps between prefills, chosen from the measured step against the critical one.

    With follow_frame_period off, the decode-only step picks the interval from a table:
    above load_bounds[i] it is load_intervals[i + 1], above the last bound max_interval.
    With it on, the frame period (decode steps plus the prefills between them) moves the
    interval one rung up when it passes period_target of the critical step while prefills
    take at least climb_prefill_share of it, and one rung down when it falls period_band
    below that, at most once per level_hold_steps steps. A waiting request older than
    wait_cap_s, or an idle decode, releases the countdown.
    """

    max_interval: int
    critical_step_s: float
    follow_frame_period: bool = False
    wait_cap_s: float = 0.30
    load_bounds: tuple[float, ...] = (0.45, 0.65, 0.80, 0.92)
    load_intervals: tuple[int, ...] = (0, 1, 2, 4)
    hysteresis: float = 0.05
    step_weight: float = 0.25
    period_target: float = 0.90
    period_band: float = 0.15
    period_weight: float = 0.05
    climb_prefill_share: float = 0.15
    level_hold_steps: int = 32
    report_period_s: float = 30.0
    # test only (laxity): admit a prefill only while its predicted time fits every playing stream's laxity
    follow_laxity: bool = False
    window_ends: tuple[int, ...] = ()
    startup_frames: int = 0
    startup_cap: int = 0
    vocoder_margin_s: float = 0.015
    laxity_margin_s: float = 0.020
    prefill_weight: float = 0.25


class Drama3Scheduler(OmniScheduler[Drama3RequestData]):
    # note (0xtoward): Drama3EngineBuilder.post_scheduler_setup installs a cadence; without one the scheduler keeps upstream's fixed countdown.
    cadence: PrefillCadence | None = None
    decode_step_s: float | None = None
    slow_decode_step_s: float | None = None
    frame_period_s: float | None = None
    observed_launch_s: float | None = None
    decode_launch_s: float | None = None
    frame_launch_s: float | None = None
    cadence_level: int = 0
    decode_steps_at_level: int = 0
    prefill_s: float | None = None
    prefill_launch_s: float | None = None

    def install_cadence(self, cadence: PrefillCadence) -> None:
        self.cadence = cadence
        self.first_window_s: dict[str, float] = {}
        self.gate_states: Counter[str] = Counter()
        self.min_laxity_samples: list[float] = []
        self.prefills_by_interval: Counter[int] = Counter()
        self.decode_step_total_s = 0.0
        self.decode_step_count = 0
        self.frame_period_total_s = 0.0
        self.frame_period_count = 0
        self.cadence_reported_s = time.monotonic()

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

    def gate_prefill_by_laxity(self) -> None:
        """test only (laxity): open the prefill gate while the predicted prefill fits every playing stream.

        A stream's laxity is how long a prefill may hold the decode before its next window misses
        playback: playback started when its first window went out, it has the delivered windows'
        audio, and its next window still needs (next end - sampled) decode steps plus the vocoder.
        Streams before startup_frames count as starting; startup_cap of them closes the gate.
        A waiting request older than wait_cap_s, or no decode to protect, opens it.
        """
        cadence = self.cadence
        assert cadence is not None
        now_s = time.perf_counter()
        decoding = self.decoding_requests()
        step_s = self.decode_step_s or 0.0
        min_laxity_s = float("inf")
        starting_count = 0
        live_request_ids: set[str] = set()
        for request in decoding:
            data = request.omni_data
            live_request_ids.add(request.rid)
            streamed = data.streamed_frame_count
            if streamed < cadence.startup_frames:
                starting_count += 1
            else:
                pass
            if streamed < cadence.window_ends[0]:
                continue
            else:
                pass
            first_window_s = self.first_window_s.setdefault(request.rid, now_s)
            sampled = len(data.frame_rows) - data.prompt_row_count
            next_end = next(end for end in cadence.window_ends if end > streamed)
            laxity_s = (
                first_window_s
                + streamed * AUDIO_FRAME_SECONDS
                - now_s
                - max(0, next_end - sampled) * step_s
                - cadence.vocoder_margin_s
            )
            min_laxity_s = min(min_laxity_s, laxity_s)
        for request_id in [
            request_id
            for request_id in self.first_window_s
            if request_id not in live_request_ids
        ]:
            self.first_window_s.pop(request_id)
        oldest_wait_s = max(
            (
                now_s - request._coalesce_enqueue_t for request in self.waiting_queue
            ),  # noqa: leading-underscore  # upstream spelling, or the public name is already taken
            default=0.0,
        )
        if not self.waiting_queue:
            state = "idle"
        elif not decoding:
            state = "no_decode"
        elif oldest_wait_s >= cadence.wait_cap_s:
            state = "wait_cap"
        elif cadence.startup_cap and starting_count >= cadence.startup_cap:
            state = "startup_cap"
        elif (
            self.prefill_s is None
            or min_laxity_s - cadence.laxity_margin_s >= self.prefill_s
        ):
            state = "fits"
        else:
            state = "laxity"
        if state in ("startup_cap", "laxity"):
            self._prefill_decode_interval_remaining = max(  # noqa: leading-underscore  # upstream spelling, or the public name is already taken
                1,
                self._prefill_decode_interval_remaining,  # noqa: leading-underscore  # upstream spelling, or the public name is already taken
            )
        else:
            self._prefill_decode_interval_remaining = 0  # noqa: leading-underscore  # upstream spelling, or the public name is already taken
        if self.waiting_queue and min_laxity_s != float("inf"):
            self.min_laxity_samples.append(min_laxity_s)
        else:
            pass
        self.gate_states[state] += 1

    def release_interval_if_due(self) -> None:
        if self.cadence is not None and self.cadence.follow_laxity:
            self.gate_prefill_by_laxity()
            return
        elif (
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
        """Fold the batch that just ran into the decode-step and frame-period EWMAs.

        The decode step is the launch-to-launch time of back-to-back decodes; the frame
        period also spans the prefills between two decodes. An idle gap starts both over.
        """
        last_batch = self.last_batch
        cadence = self.cadence
        if (
            cadence is None
            or last_batch is None
            or last_batch.launch_ts == self.observed_launch_s
        ):
            return
        else:
            pass
        self.observed_launch_s = last_batch.launch_ts
        if self.prefill_launch_s is not None and not last_batch.after_idle_gap:
            prefill_s = last_batch.launch_ts - self.prefill_launch_s
            self.prefill_s = (
                prefill_s
                if self.prefill_s is None
                else cadence.prefill_weight * prefill_s
                + (1 - cadence.prefill_weight) * self.prefill_s
            )
        else:
            pass
        self.prefill_launch_s = (
            last_batch.launch_ts if last_batch.forward_mode.is_extend() else None
        )
        if last_batch.after_idle_gap:
            self.decode_launch_s = None
            self.frame_launch_s = None
        else:
            pass
        if not last_batch.forward_mode.is_decode():
            self.decode_launch_s = None
            return
        else:
            pass
        launch_s = last_batch.launch_ts
        if self.decode_launch_s is not None:
            step_s = launch_s - self.decode_launch_s
            self.decode_step_s = (
                step_s
                if self.decode_step_s is None
                else cadence.step_weight * step_s
                + (1 - cadence.step_weight) * self.decode_step_s
            )
            self.slow_decode_step_s = (
                step_s
                if self.slow_decode_step_s is None
                else cadence.period_weight * step_s
                + (1 - cadence.period_weight) * self.slow_decode_step_s
            )
            self.decode_step_total_s += step_s
            self.decode_step_count += 1
        else:
            pass
        if self.frame_launch_s is not None:
            period_s = launch_s - self.frame_launch_s
            self.frame_period_s = (
                period_s
                if self.frame_period_s is None
                else cadence.period_weight * period_s
                + (1 - cadence.period_weight) * self.frame_period_s
            )
            self.frame_period_total_s += period_s
            self.frame_period_count += 1
        else:
            pass
        self.decode_launch_s = launch_s
        self.frame_launch_s = launch_s
        self.decode_steps_at_level += 1

    def report_cadence(self) -> None:
        cadence = self.cadence
        now_s = time.monotonic()
        if (
            cadence is None
            or now_s - self.cadence_reported_s < cadence.report_period_s
            or not self.prefills_by_interval
        ):
            return
        else:
            pass
        mean_step_ms = (
            1000 * self.decode_step_total_s / self.decode_step_count
            if self.decode_step_count
            else float("nan")
        )
        mean_period_ms = (
            1000 * self.frame_period_total_s / self.frame_period_count
            if self.frame_period_count
            else float("nan")
        )
        laxity_samples = sorted(self.min_laxity_samples)
        laxity_text = (
            f"; prefill {1000 * (self.prefill_s or 0.0):.1f} ms, gate "
            f"{dict(sorted(self.gate_states.items()))}, min laxity p10/p50 "
            f"{1000 * laxity_samples[len(laxity_samples) // 10]:.0f}/"
            f"{1000 * laxity_samples[len(laxity_samples) // 2]:.0f} ms"
            if cadence.follow_laxity and laxity_samples
            else ""
        )
        logger.info(
            f"Drama 3 prefill cadence: {self.decode_step_count} decode steps, mean "
            f"{mean_step_ms:.1f} ms, frame period {mean_period_ms:.1f} ms against "
            f"{cadence.critical_step_s * 1000:.1f} ms; prefills by interval "
            f"{dict(sorted(self.prefills_by_interval.items()))}{laxity_text}"
        )
        self.gate_states.clear()
        self.min_laxity_samples.clear()
        self.prefills_by_interval.clear()
        self.decode_step_total_s = 0.0
        self.decode_step_count = 0
        self.frame_period_total_s = 0.0
        self.frame_period_count = 0
        self.cadence_reported_s = now_s

    def get_next_batch_to_run(self):
        self.observe_decode_step()
        self.release_interval_if_due()
        self.report_cadence()
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
        elif self.cadence.follow_laxity:
            if batch is not None and batch.forward_mode.is_extend():
                self.prefills_by_interval[0] += 1
            else:
                pass
            self._prefill_decode_interval_remaining = 0  # noqa: leading-underscore  # upstream spelling, or the public name is already taken
        elif batch is not None and (
            batch.is_extend_in_batch
            if self.require_mlp_sync
            else batch.forward_mode.is_extend()
        ):
            interval = self.choose_interval()
            self.prefills_by_interval[interval] += 1
            self._prefill_decode_interval_remaining = interval  # noqa: leading-underscore  # upstream spelling, or the public name is already taken
        else:
            pass

    def choose_interval(self) -> int:
        cadence = self.cadence
        assert cadence is not None
        decoding = bool(self.decoding_requests())
        target_s = cadence.period_target * cadence.critical_step_s
        if cadence.follow_frame_period:
            # note (0xtoward): A rung change waits level_hold_steps decode steps, so the frame-period EWMA shows one change before the next.
            if (
                self.frame_period_s is None
                or self.slow_decode_step_s is None
                or self.decode_steps_at_level < cadence.level_hold_steps
            ):
                pass
            else:
                # note (0xtoward): A longer interval only batches prefills; when they are a small share of the frame period it adds TTFA and buys no continuity. Stepping down still follows the period alone, since a high rung shrinks the prefill share it is judged by.
                prefill_share = 1 - self.slow_decode_step_s / self.frame_period_s
                if (
                    self.frame_period_s > target_s
                    and prefill_share >= cadence.climb_prefill_share
                    and self.cadence_level < len(cadence.load_intervals)
                ):
                    self.cadence_level += 1
                    self.decode_steps_at_level = 0
                elif (
                    self.frame_period_s < target_s * (1 - cadence.period_band)
                    and self.cadence_level > 0
                ):
                    self.cadence_level -= 1
                    self.decode_steps_at_level = 0
                else:
                    pass
        elif not decoding or self.decode_step_s is None:
            self.cadence_level = 0
        else:
            load = self.decode_step_s / cadence.critical_step_s
            level = sum(load >= bound for bound in cadence.load_bounds)
            # note (0xtoward): Stepping down waits for the load to clear the band below the current bound, so the interval does not flap.
            if level < self.cadence_level:
                lower_bound = (
                    cadence.load_bounds[self.cadence_level - 1] - cadence.hysteresis
                )
                level = (
                    self.cadence_level - 1 if load < lower_bound else self.cadence_level
                )
            else:
                pass
            self.cadence_level = level
        intervals = (*cadence.load_intervals, cadence.max_interval)
        return (
            min(intervals[self.cadence_level], cadence.max_interval) if decoding else 0
        )
