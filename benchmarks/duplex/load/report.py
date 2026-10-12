# SPDX-License-Identifier: Apache-2.0
"""Unit lag and per-stage split of full-duplex load cells.

lag: from client.py records, every unit's lag (its sglang.unit.done minus the time that input second was
sent), split into listening and speaking units, with the units more than a second late.
stages: from the event_dir of /start_profile, every unit (one request id) split into perception, thinker,
talker and speech segments.
"""

from __future__ import annotations

import argparse
import collections
import json
import statistics
from dataclasses import dataclass
from pathlib import Path

LATE_UNIT_S = 1.0
STAGE_SEGMENTS = (
    (
        "admit -> perception",
        ("coordinator", "request_admission"),
        ("perception", "stage_input_received"),
    ),
    (
        "perception queue",
        ("perception", "stage_input_received"),
        ("perception", "stage_dispatch"),
    ),
    (
        "perception compute",
        ("perception", "stage_dispatch"),
        ("perception", "stage_complete"),
    ),
    (
        "hop -> thinker",
        ("perception", "stage_complete"),
        ("thinker", "stage_input_received"),
    ),
    (
        "thinker wait",
        ("thinker", "stage_input_received"),
        ("thinker", "scheduler_queue_enter"),
    ),
    (
        "thinker queue",
        ("thinker", "scheduler_queue_enter"),
        ("thinker", "scheduler_prefill_start"),
    ),
    (
        "thinker prefill",
        ("thinker", "scheduler_prefill_start"),
        ("thinker", "scheduler_prefill_end"),
    ),
    (
        "thinker decode",
        ("thinker", "scheduler_prefill_end"),
        ("thinker", "model_path_end"),
    ),
    ("thinker post", ("thinker", "model_path_end"), ("thinker", "stage_complete")),
    (
        "hop -> talker",
        ("thinker", "stage_complete"),
        ("talker", "stage_input_received"),
    ),
    ("talker wait", ("talker", "stage_input_received"), ("talker", "stage_dispatch")),
    ("talker compute", ("talker", "stage_dispatch"), ("talker", "stage_complete")),
    ("hop -> speech", ("talker", "stage_complete"), ("speech", "stage_input_received")),
    ("speech wait", ("speech", "stage_input_received"), ("speech", "stage_dispatch")),
    (
        "speech first chunk",
        ("speech", "stage_dispatch"),
        ("speech", "stage_first_stream_chunk_sent"),
    ),
    (
        "speech -> response",
        ("speech", "stage_first_stream_chunk_sent"),
        ("coordinator", "terminal_response"),
    ),
)

JsonValue = str | int | float | bool | None | list["JsonValue"] | dict[str, "JsonValue"]
StageEvent = tuple[str, str]


@dataclass(kw_only=True, frozen=True)
class UnitLag:
    lag_s: float
    is_speaking: bool


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[int(fraction * (len(ordered) - 1))]


def session_unit_lags(session_path: Path) -> tuple[dict[str, JsonValue], list[UnitLag]]:
    """A unit speaks when output audio arrived since the previous sglang.unit.done."""
    record = json.loads(session_path.read_text(encoding="utf-8"))
    second_sent_s: dict[int, float] = {}
    for send in record["sends"]:
        second = int(send["input_s"] + 1e-6)
        if send["input_s"] - second > -1e-6 and second not in second_sent_s:
            second_sent_s[second] = send["t"]
        else:
            pass
    unit_lags = []
    previous_done_s = 0.0
    audio_events = [
        event
        for event in record["events"]
        if event["type"] == "response.output_audio.delta"
    ]
    unit_done_events = [
        event for event in record["events"] if event["type"] == "sglang.unit.done"
    ]
    for unit_number, done in enumerate(unit_done_events, start=1):
        audio_s = sum(
            event["audio_s"]
            for event in audio_events
            if previous_done_s < event["t"] <= done["t"]
        )
        if unit_number in second_sent_s:
            unit_lags.append(
                UnitLag(
                    lag_s=done["t"] - second_sent_s[unit_number],
                    is_speaking=audio_s > 0,
                )
            )
        else:
            pass
        previous_done_s = done["t"]
    return record["summary"], unit_lags


def report_lag(cell_dirs: list[Path]) -> None:
    print(
        f"{'cell':32} {'sess':>4} {'fail':>4} {'listen med':>10} {'speak med':>9} {'p90':>6} "
        f"{'max':>6} {'late':>11} {'first audio':>11}"
    )
    for cell_dir in cell_dirs:
        listening, speaking, failed, first_audio = [], [], 0, []
        session_paths = sorted(cell_dir.glob("s[0-9]*.json"))
        for session_path in session_paths:
            summary, unit_lags = session_unit_lags(session_path)
            failed += summary["error"] is not None
            if summary["first_audio_t"] is not None:
                first_audio.append(summary["first_audio_t"])
            else:
                pass
            for unit in unit_lags:
                (speaking if unit.is_speaking else listening).append(unit.lag_s)
        all_lags = listening + speaking
        late = sum(lag_s > LATE_UNIT_S for lag_s in all_lags)
        late_text = f"{late}/{len(all_lags)} {late / max(len(all_lags), 1) * 100:.1f}%"
        print(
            f"{str(cell_dir)[-32:]:32} {len(session_paths):>4} {failed:>4} "
            f"{statistics.median(listening) if listening else float('nan'):>10.3f} "
            f"{statistics.median(speaking) if speaking else float('nan'):>9.3f} "
            f"{percentile(speaking, 0.9) if speaking else float('nan'):>6.3f} "
            f"{max(speaking) if speaking else float('nan'):>6.3f} {late_text:>11} "
            f"{statistics.median(first_audio) if first_audio else float('nan'):>11.3f}"
        )


def load_stage_events(event_dir: Path) -> dict[str, dict[StageEvent, float]]:
    """First time (ms) of every (stage, event name) per request id; a stage may stream several chunks."""
    units: dict[str, dict[StageEvent, float]] = collections.defaultdict(dict)
    for path in sorted(event_dir.glob("*.jsonl")):
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                event = json.loads(line)
                units[event["request_id"]].setdefault(
                    (event["stage"], event["event_name"]), event["timestamp_ns"] / 1e6
                )
    return units


def report_stages(run_dir: Path) -> None:
    units = load_stage_events(run_dir / "events")
    admission, response = ("coordinator", "request_admission"), (
        "coordinator",
        "terminal_response",
    )
    complete = [
        unit for unit in units.values() if admission in unit and response in unit
    ]
    print(f"units with admission and response: {len(complete)} of {len(units)}")
    for kind in ("speaking", "listening"):
        group = [
            unit
            for unit in complete
            if (("talker", "model_path_start") in unit) == (kind == "speaking")
        ]
        if not group:
            continue
        else:
            pass
        totals_ms = [unit[response] - unit[admission] for unit in group]
        median_total_ms = statistics.median(totals_ms)
        print(
            f"\n{kind} units: {len(group)}; admission -> response ms median {median_total_ms:.0f} "
            f"p90 {percentile(totals_ms, 0.9):.0f} max {max(totals_ms):.0f}"
        )
        print(
            f"  {'segment':22} {'n':>4} {'median':>7} {'p90':>7} {'max':>7}  share of median"
        )
        for name, start, end in STAGE_SEGMENTS:
            values_ms = [
                unit[end] - unit[start]
                for unit in group
                if start in unit and end in unit
            ]
            if values_ms:
                median_ms = statistics.median(values_ms)
                print(
                    f"  {name:22} {len(values_ms):>4} {median_ms:7.1f} {percentile(values_ms, 0.9):7.1f} "
                    f"{max(values_ms):7.1f}  {median_ms / median_total_ms * 100:5.1f}%"
                )
            else:
                pass


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    lag = commands.add_parser("lag", help="unit lag per cell from client records")
    lag.add_argument("cell_dirs", type=Path, nargs="+")
    stages = commands.add_parser(
        "stages", help="per-unit stage split from a profile event_dir"
    )
    stages.add_argument("run_dir", type=Path, help="directory holding events/*.jsonl")
    args = parser.parse_args()

    if args.command == "lag":
        report_lag(args.cell_dirs)
    else:
        report_stages(args.run_dir)


if __name__ == "__main__":
    main()
