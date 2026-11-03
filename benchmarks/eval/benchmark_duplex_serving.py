# SPDX-License-Identifier: Apache-2.0
"""Record one duplex system at several concurrency levels and report its serving timing."""

from __future__ import annotations

import argparse
import contextlib
import logging
from importlib import import_module
from pathlib import Path

from pydantic import JsonValue

from benchmarks.duplex.serving import serving_report
from benchmarks.duplex.v15_audio import write_json
from benchmarks.metrics.server_probes import (
    format_server_tables,
    gpu_memory_samples,
    level_event_dir,
    level_memory_path,
    level_trace_path,
    nsys_collection,
    request_events,
)

logger = logging.getLogger(__name__)
RECORDERS = {
    "v1.0": "benchmarks.eval.benchmark_duplex_v10",
    "v1.5": "benchmarks.eval.benchmark_duplex_v15",
    "sessions": "benchmarks.eval.benchmark_duplex_sessions",
}
# Options the sweep sets on every recorder run itself.
SWEEP_OPTIONS = ("--concurrency", "--output")

# note (jeffro): this is a small default startup buffer, and other values can be passed to trace the underrun curve.
DEFAULT_PLAYBACK_RESERVE_MS = 80.0
# note (0xtoward): players hold 200-400 ms of a reply's first audio; 0 shows what an unbuffered one hears.
DEFAULT_PREBUFFERS_MS = "0,200,300"


def milliseconds(value_s: JsonValue) -> str:
    return "-" if value_s is None else f"{value_s * 1000:.0f}"


def ratio(value: JsonValue, form: str) -> str:
    return "-" if value is None else format(value, form)


def format_table(report: dict[str, JsonValue]) -> str:
    prebuffers_ms = [
        f"{entry['prebuffer_s'] * 1000:g}"
        for entry in report["levels"][0]["continuity"]
    ]
    rows = [
        (
            "| N | mean open | sessions failed | 503 | pacing violations | unit miss "
            "| unit lag p50 / p90 / p99 (ms) | first audio p50 / p90 (ms) "
            "| buffer needed p50 / p90 (ms) | underrun replies | RTF p90 "
            f"| gap-free replies at {' / '.join(prebuffers_ms)} ms "
            "| speaking unit lag p50 / p90 (ms) | meets targets |"
        ),
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for level in report["levels"]:
        real_time_factor = level["real_time_factor"]
        unit_lag = level["unit_lag_s"]
        first_audio = level["first_audio_latency_s"]
        buffer_needed = level["startup_buffer_needed_s"]
        speaking_lag = level["speaking_unit_lag_s"]
        gap_free = " / ".join(
            ratio(
                (
                    entry["gap_free_replies"] / level["replies"]
                    if level["replies"]
                    else None
                ),
                ".0%",
            )
            for entry in level["continuity"]
        )
        rows.append(
            f"| {level['concurrency']} | {level['mean_open_sessions']:.1f} "
            f"| {level['sessions_failed']}/{level['sessions']} "
            f"| {level['admission_denials']} | {level['input_pacing_violations']} "
            f"| {ratio(level['unit_miss_rate'], '.1%')} "
            f"| {milliseconds(unit_lag['p50'])} / {milliseconds(unit_lag['p90'])} / "
            f"{milliseconds(unit_lag['p99'])} "
            f"| {milliseconds(first_audio['p50'])} / {milliseconds(first_audio['p90'])} "
            f"| {milliseconds(buffer_needed['p50'])} / "
            f"{milliseconds(buffer_needed['p90'])} "
            f"| {level['replies_with_underrun']}/{level['replies']} "
            f"| {ratio(None if real_time_factor is None else real_time_factor['p90'], '.2f')} "
            f"| {gap_free} "
            f"| {milliseconds(speaking_lag['p50'])} / {milliseconds(speaking_lag['p90'])} "
            f"| {'yes' if not level['unmet_targets'] else '; '.join(level['unmet_targets'])} |"
        )
    capacity = report["max_concurrency_meeting_targets"]
    if capacity is None:
        verdict = "No recorded concurrency meets the targets."
    elif report["capacity_bounded"]:
        verdict = f"Highest concurrency meeting the targets: {capacity}."
    else:
        verdict = (
            f"Every recorded concurrency meets the targets; capacity is at least "
            f"{capacity}."
        )
    return (
        "\n".join(rows)
        + f"\n\n{verdict}\n"
        + format_server_tables(
            {f"N={level['concurrency']}": level for level in report["levels"]}
        )
    )


def concurrency_levels(text: str) -> list[int]:
    """Distinct positive levels separated by commas, such as 1,2,4,8."""
    levels = [int(part) for part in text.split(",")]
    if min(levels) < 1 or len(set(levels)) != len(levels):
        raise argparse.ArgumentTypeError(
            f"expected distinct positive levels such as 1,2,4,8, got {text!r}"
        )
    else:
        return sorted(levels)


def prebuffer_list(text: str) -> list[float]:
    """Nonnegative prebuffers in ms separated by commas, such as 0,200,300; returned in seconds."""
    prebuffers_ms = [float(part) for part in text.split(",")]
    if min(prebuffers_ms) < 0:
        raise argparse.ArgumentTypeError(
            f"prebuffers must be nonnegative, got {text!r}"
        )
    else:
        return [prebuffer_ms / 1000 for prebuffer_ms in prebuffers_ms]


def report(args: argparse.Namespace) -> int:
    serving = serving_report(
        args.run, args.playback_reserve_ms / 1000, args.prebuffer_ms
    )
    write_json(args.output, serving)
    print(format_table(serving))
    return 0


def sweep(args: argparse.Namespace) -> int:
    recorder = import_module(RECORDERS[args.dataset]).main
    args.output.mkdir(parents=True, exist_ok=False)
    run_dirs = []
    for concurrency in args.concurrency:
        run_dir = args.output / f"concurrency-{concurrency}"
        logger.info(
            f"Recording {args.dataset} at concurrency {concurrency} -> {run_dir}"
        )
        with contextlib.ExitStack() as probes:
            if args.stage_events:
                probes.enter_context(
                    request_events(args.server_url, level_event_dir(run_dir))
                )
            else:
                pass
            if args.gpu_memory:
                probes.enter_context(gpu_memory_samples(level_memory_path(run_dir)))
            else:
                pass
            if args.nsys_session:
                probes.enter_context(
                    nsys_collection(args.nsys_session, level_trace_path(run_dir))
                )
            else:
                pass
            # note (jeffro): the recorder exits nonzero when a session fails, which under
            # load is a result to report, not a reason to stop the sweep.
            recorder(
                [
                    "record",
                    *args.recorder_args,
                    "--concurrency",
                    str(concurrency),
                    "--output",
                    str(run_dir),
                ]
            )
        run_dirs.append(run_dir)
    serving = serving_report(
        run_dirs, args.playback_reserve_ms / 1000, args.prebuffer_ms
    )
    write_json(args.output / "serving.json", serving)
    table = format_table(serving)
    (args.output / "SERVING.md").write_text(table)
    print(table)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    playback = argparse.ArgumentParser(add_help=False)
    playback.add_argument(
        "--playback-reserve-ms",
        type=float,
        default=DEFAULT_PLAYBACK_RESERVE_MS,
        help="Simulated player startup buffer (default: %(default)s)",
    )
    playback.add_argument(
        "--prebuffer-ms",
        type=prebuffer_list,
        default=prebuffer_list(DEFAULT_PREBUFFERS_MS),
        help="Player prebuffers, in ms separated by commas, at which to count "
        f"gap-free replies (default: {DEFAULT_PREBUFFERS_MS})",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    report_parser = commands.add_parser(
        "report", parents=[playback], help="Report recorded runs"
    )
    report_parser.add_argument(
        "--run",
        type=Path,
        action="append",
        required=True,
        help="Recorded run directory, one per concurrency level",
    )
    report_parser.add_argument(
        "--output", type=Path, required=True, help="serving.json path"
    )
    report_parser.set_defaults(handler=report)

    sweep_parser = commands.add_parser(
        "sweep",
        parents=[playback],
        help="Record every level against one running server, then report",
        description="Options after -- go to the recorder's record command unchanged "
        "(--url, --profile, --dataset-root, server identity, sample selection); the "
        "sweep sets --concurrency and --output per level.",
    )
    sweep_parser.add_argument("--dataset", choices=RECORDERS, required=True)
    sweep_parser.add_argument(
        "--concurrency",
        type=concurrency_levels,
        required=True,
        help="Levels to record, such as 1,2,4,8",
    )
    sweep_parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="New directory for the level runs, serving.json and SERVING.md",
    )
    sweep_parser.add_argument(
        "--server-url",
        help="HTTP base URL of the server, such as http://127.0.0.1:8000, for the "
        "server-side probes; the server must run on this host",
    )
    sweep_parser.add_argument(
        "--stage-events",
        action="store_true",
        help="Record the profiler's request events at every level and split requests by stage",
    )
    sweep_parser.add_argument(
        "--gpu-memory",
        action="store_true",
        help="Sample GPU memory and utilization at every level; with --stage-events, "
        "processes are named by stage",
    )
    sweep_parser.add_argument(
        "--nsys-session",
        help="Nsight Systems session the server runs under (nsys launch --session-new=NAME "
        "--trace=python-gil,nvtx --trace-fork-before-exec=true ...); collects it at every "
        "level and reports GIL holding and waiting per process, named by stage with "
        "--stage-events. Costs about 2%% throughput; does not work under MPS",
    )
    sweep_parser.add_argument("recorder_args", nargs=argparse.REMAINDER)
    sweep_parser.set_defaults(handler=sweep)

    args = parser.parse_args(argv)
    if args.playback_reserve_ms < 0:
        parser.error("--playback-reserve-ms must be nonnegative")
    elif args.command == "sweep" and args.stage_events and args.server_url is None:
        parser.error("--stage-events needs --server-url")
    elif args.command == "sweep":
        args.recorder_args = [arg for arg in args.recorder_args if arg != "--"]
        if any(arg.split("=")[0] in SWEEP_OPTIONS for arg in args.recorder_args):
            parser.error(f"the sweep sets {' and '.join(SWEEP_OPTIONS)} itself")
        else:
            pass
    else:
        pass
    logging.basicConfig(level=logging.INFO)
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
else:
    pass
