# SPDX-License-Identifier: Apache-2.0
"""Read the per-stage torch traces that /start_profile writes for a full-duplex server.

summary: GPU busy time per stage and over all stages (they share one GPU), kernel classes, CUDA syncs and
launches, and the Python frames with the most self time.
slice: cut one window out of every stage trace and lay the stages on one timeline for Perfetto.
attribute: charge every GPU kernel to the Python frame that launched it, through the launch's correlation id.
"""

from __future__ import annotations

import argparse
import bisect
import collections
import gzip
import json
import multiprocessing
import re
from dataclasses import dataclass
from pathlib import Path

STAGES = ("perception", "thinker", "talker", "speech")
GPU_EVENT_CATEGORIES = ("kernel", "gpu_memcpy", "gpu_memset")
SYNC_CALLS = (
    "cudaStreamSynchronize",
    "cudaDeviceSynchronize",
    "cudaEventSynchronize",
    "cudaMemcpyAsync",
)
LAUNCH_CALLS = (
    "cudaLaunchKernel",
    "cuLaunchKernel",
    "cudaLaunchKernelExC",
    "cuLaunchKernelEx",
)
KERNEL_CLASSES = (
    ("gemm", r"gemm|cutlass|xmma|nvjet|cublas|matmul|splitK"),
    ("attention", r"flash|fmha|attn|attention|paged"),
    ("conv", r"conv|cudnn|implicit|winograd"),
    ("norm", r"norm"),
    ("sort/topk/sample", r"sort|topk|radix|multinomial|sampling|cumsum|scan"),
    ("softmax/reduce", r"softmax|reduce|argmax"),
    (
        "copy/cat/index",
        r"copy|cat|index|gather|scatter|fill|where|masked|Memcpy|Memset",
    ),
    ("elementwise", r"elementwise|vectorized|unrolled|pointwise"),
    ("triton", r"^triton"),
)
TOP_ROWS = 8
SLICE_GPU_PID_BASE = 9000

JsonValue = str | int | float | bool | None | list["JsonValue"] | dict[str, "JsonValue"]
TraceEvent = dict[str, JsonValue]


@dataclass(kw_only=True)
class StageSummary:
    busy_intervals_us: list[tuple[float, float]]
    kernel_classes_ms: collections.Counter[str]
    top_kernels_ms: list[tuple[str, float]]
    runtime_calls: collections.Counter[str]
    runtime_call_ms: collections.Counter[str]
    python_self_ms: list[tuple[str, float]]
    unclosed_frames: int


def load_trace_events(trace_path: Path) -> list[TraceEvent]:
    with (
        gzip.open(trace_path, "rt") if trace_path.suffix == ".gz" else open(trace_path)
    ) as handle:
        return json.load(handle)["traceEvents"]


def stage_trace_paths(trace_dir: Path) -> dict[str, Path]:
    paths = {}
    for stage in STAGES:
        found = sorted((trace_dir / stage).glob("*.trace.json*"))
        if found:
            paths[stage] = found[0]
        else:
            pass
    return paths


def busy_ms(
    intervals_us: list[tuple[float, float]],
    window_start_us: float,
    window_end_us: float,
) -> float:
    total_us, cursor_us = 0.0, window_start_us
    for start_us, end_us in sorted(intervals_us):
        start_us, end_us = max(start_us, cursor_us), min(end_us, window_end_us)
        if end_us > start_us:
            total_us += end_us - start_us
            cursor_us = end_us
        else:
            pass
    return total_us / 1000


def python_self_time_ms(
    events: list[TraceEvent],
) -> tuple[collections.Counter[str], int]:
    """Self time per Python frame name, and the count of frames dropped for outliving their caller.

    The Python tracer leaves some builtin frames open until the trace ends; they nest in nothing and would
    swamp every real frame, so a frame that does not end inside its caller is dropped.
    """
    frames = sorted(
        (event for event in events if event.get("cat") == "python_function"),
        key=lambda event: (str(event["tid"]), event["ts"], -event.get("dur", 0)),
    )
    self_time_us: collections.Counter[str] = collections.Counter()
    unclosed = 0
    stack: list[tuple[str, float]] = []
    current_thread = None
    for frame in frames:
        start_us, end_us = frame["ts"], frame["ts"] + frame.get("dur", 0)
        if frame["tid"] != current_thread:
            stack, current_thread = [], frame["tid"]
        else:
            pass
        while stack and stack[-1][1] <= start_us:
            stack.pop()
        if stack and end_us > stack[-1][1]:
            unclosed += 1
        else:
            if stack:
                self_time_us[stack[-1][0]] -= end_us - start_us
            else:
                pass
            self_time_us[frame["name"]] += end_us - start_us
            stack.append((frame["name"], end_us))
    return (
        collections.Counter({name: us / 1000 for name, us in self_time_us.items()}),
        unclosed,
    )


def summarize_stage(trace_path: Path) -> StageSummary:
    events = load_trace_events(trace_path)
    gpu_events = [event for event in events if event.get("cat") in GPU_EVENT_CATEGORIES]
    kernel_classes_ms: collections.Counter[str] = collections.Counter()
    kernels_ms: collections.Counter[str] = collections.Counter()
    for event in gpu_events:
        kernel_class = next(
            (
                label
                for label, pattern in KERNEL_CLASSES
                if re.search(pattern, event["name"], re.IGNORECASE)
            ),
            "other",
        )
        kernel_classes_ms[kernel_class] += event["dur"] / 1000
        kernels_ms[event["name"][:80]] += event["dur"] / 1000
    runtime_calls: collections.Counter[str] = collections.Counter()
    runtime_call_ms: collections.Counter[str] = collections.Counter()
    for event in events:
        if event.get("cat") == "cuda_runtime":
            runtime_calls[event["name"]] += 1
            runtime_call_ms[event["name"]] += event.get("dur", 0) / 1000
        else:
            pass
    python_self_ms, unclosed = python_self_time_ms(events)
    return StageSummary(
        busy_intervals_us=[
            (event["ts"], event["ts"] + event["dur"]) for event in gpu_events
        ],
        kernel_classes_ms=kernel_classes_ms,
        top_kernels_ms=kernels_ms.most_common(TOP_ROWS),
        runtime_calls=runtime_calls,
        runtime_call_ms=runtime_call_ms,
        python_self_ms=python_self_ms.most_common(TOP_ROWS * 2),
        unclosed_frames=unclosed,
    )


def report_summary(trace_dir: Path) -> None:
    paths = stage_trace_paths(trace_dir)
    with multiprocessing.Pool(len(paths)) as pool:
        summaries = dict(zip(paths, pool.map(summarize_stage, paths.values())))
    window_start_us = max(
        min(start for start, _ in summary.busy_intervals_us)
        for summary in summaries.values()
    )
    window_end_us = min(
        max(end for _, end in summary.busy_intervals_us)
        for summary in summaries.values()
    )
    window_ms = (window_end_us - window_start_us) / 1000
    stage_busy_ms = {
        stage: busy_ms(summary.busy_intervals_us, window_start_us, window_end_us)
        for stage, summary in summaries.items()
    }
    union_ms = busy_ms(
        [
            interval
            for summary in summaries.values()
            for interval in summary.busy_intervals_us
        ],
        window_start_us,
        window_end_us,
    )
    print(
        f"common window {window_ms:.0f} ms; GPU busy by any stage {union_ms:.0f} ms "
        f"({union_ms / window_ms * 100:.0f}%); sum over stages {sum(stage_busy_ms.values()):.0f} ms"
    )
    for stage, summary in summaries.items():
        launches = sum(summary.runtime_calls[name] for name in LAUNCH_CALLS)
        syncs = ", ".join(
            f"{name} {summary.runtime_calls[name]}/{summary.runtime_call_ms[name]:.0f}"
            for name in SYNC_CALLS
            if summary.runtime_calls[name]
        )
        print(
            f"\n[{stage}] GPU busy {stage_busy_ms[stage]:.0f} ms ({stage_busy_ms[stage] / window_ms * 100:.0f}%); "
            f"kernel launches {launches}, graph launches {summary.runtime_calls['cudaGraphLaunch']}"
        )
        print(
            "  kernel classes ms:",
            ", ".join(
                f"{name} {ms:.0f}"
                for name, ms in summary.kernel_classes_ms.most_common()
            ),
        )
        print(f"  syncs (calls/ms): {syncs}")
        print("  top kernels ms:")
        for name, ms in summary.top_kernels_ms:
            print(f"    {ms:8.1f}  {name}")
        print(
            f"  python self time ms ({summary.unclosed_frames} unclosed frames dropped):"
        )
        for name, ms in summary.python_self_ms:
            print(f"    {ms:8.1f}  {name[:110]}")


def write_slice(
    trace_dir: Path, start_s: float, length_s: float, output_path: Path
) -> None:
    """Every stage keeps its CPU process row and gets its own GPU row, so the stages stack in Perfetto."""
    stage_events = {
        stage: load_trace_events(path)
        for stage, path in stage_trace_paths(trace_dir).items()
    }
    origin_us = max(
        min(event["ts"] for event in events if event.get("cat") == "kernel")
        for events in stage_events.values()
    )
    window_start_us = origin_us + start_s * 1e6
    window_end_us = window_start_us + length_s * 1e6
    merged: list[TraceEvent] = []
    for index, (stage, events) in enumerate(stage_events.items()):
        gpu_pid = SLICE_GPU_PID_BASE + index
        cpu_pids = {
            event["pid"]
            for event in events
            if event.get("cat") in ("cpu_op", "python_function")
        }
        for event in events:
            if event.get("pid") in ("Spans", "Traces", ""):
                continue
            elif event.get("ph") == "M":
                if event["name"] not in (
                    "process_name",
                    "process_labels",
                    "process_sort_index",
                ):
                    merged.append(
                        {
                            **event,
                            "pid": (
                                event["pid"] if event["pid"] in cpu_pids else gpu_pid
                            ),
                        }
                    )
                else:
                    pass
            elif (
                "ts" in event
                and window_start_us <= event["ts"] + event.get("dur", 0)
                and event["ts"] <= window_end_us
            ):
                moved = {**event, "ts": event["ts"] - window_start_us}
                if event["pid"] not in cpu_pids:
                    moved["pid"] = gpu_pid
                else:
                    pass
                if "id" in event:
                    moved["id"] = f"{stage}-{event['id']}"
                else:
                    pass
                merged.append(moved)
            else:
                pass
        for pid, name, order in (
            (min(cpu_pids), f"{stage} CPU", 2 * index),
            (gpu_pid, f"{stage} GPU", 2 * index + 1),
        ):
            merged.append(
                {
                    "ph": "M",
                    "name": "process_name",
                    "pid": pid,
                    "tid": 0,
                    "args": {"name": name},
                }
            )
            merged.append(
                {
                    "ph": "M",
                    "name": "process_sort_index",
                    "pid": pid,
                    "tid": 0,
                    "args": {"sort_index": order},
                }
            )
    with gzip.open(output_path, "wt") as handle:
        json.dump({"traceEvents": merged, "displayTimeUnit": "ms"}, handle)
    print(f"{output_path}: {len(merged)} events")


def report_attribution(
    trace_path: Path, frame_substring: str, stage: str | None
) -> None:
    """Split every call of the matching frame by its direct children: CPU wall versus the GPU time they launched."""
    events = load_trace_events(trace_path)
    if stage is not None:
        process_names = {
            event["pid"]: event["args"]["name"]
            for event in events
            if event.get("ph") == "M" and event["name"] == "process_name"
        }
        events = [
            event
            for event in events
            if process_names.get(event.get("pid"), "").startswith(f"{stage} ")
        ]
    else:
        pass
    gpu_ms_by_correlation: collections.Counter[int] = collections.Counter()
    kernels_by_correlation: dict[int, collections.Counter[str]] = (
        collections.defaultdict(collections.Counter)
    )
    for event in events:
        if event.get("cat") in GPU_EVENT_CATEGORIES:
            correlation = event["args"].get("correlation")
            gpu_ms_by_correlation[correlation] += event["dur"] / 1000
            kernels_by_correlation[correlation][event["name"][:60]] += (
                event["dur"] / 1000
            )
        else:
            pass
    frames_by_thread: dict[JsonValue, list[TraceEvent]] = collections.defaultdict(list)
    for frame in sorted(
        (event for event in events if event.get("cat") == "python_function"),
        key=lambda frame: (frame["ts"], -frame["dur"]),
    ):
        frames_by_thread[frame["tid"]].append(frame)
    frame_starts_by_thread = {
        thread: [frame["ts"] for frame in thread_frames]
        for thread, thread_frames in frames_by_thread.items()
    }
    launches_by_thread: dict[JsonValue, list[TraceEvent]] = collections.defaultdict(
        list
    )
    for event in events:
        if (
            event.get("cat") == "cuda_runtime"
            and event["args"].get("correlation") in gpu_ms_by_correlation
        ):
            launches_by_thread[event["tid"]].append(event)
        else:
            pass
    parents = [
        frame
        for thread_frames in frames_by_thread.values()
        for frame in thread_frames
        if frame_substring in frame["name"]
    ]
    if not parents:
        raise ValueError(f"no Python frame contains {frame_substring!r}")
    else:
        pass
    wall_ms: collections.Counter[str] = collections.Counter()
    gpu_ms: collections.Counter[str] = collections.Counter()
    kernels_ms: dict[str, collections.Counter[str]] = collections.defaultdict(
        collections.Counter
    )
    for parent in parents:
        parent_end = parent["ts"] + parent["dur"]
        thread_frames = frames_by_thread[parent["tid"]]
        first = bisect.bisect_left(frame_starts_by_thread[parent["tid"]], parent["ts"])
        children, cursor = [], parent["ts"]
        for frame in thread_frames[first:]:
            if frame["ts"] > parent_end:
                break
            elif (
                frame is not parent
                and cursor <= frame["ts"]
                and frame["ts"] + frame["dur"] <= parent_end
            ):
                children.append(frame)
                cursor = frame["ts"] + frame["dur"]
            else:
                pass
        for child in children:
            wall_ms[child["name"][:80]] += child["dur"] / 1000
        for launch in launches_by_thread[parent["tid"]]:
            if parent["ts"] <= launch["ts"] <= parent_end:
                owner = next(
                    (
                        child["name"][:80]
                        for child in children
                        if child["ts"] <= launch["ts"] <= child["ts"] + child["dur"]
                    ),
                    "(launched by the frame itself)",
                )
                correlation = launch["args"]["correlation"]
                gpu_ms[owner] += gpu_ms_by_correlation[correlation]
                kernels_ms[owner].update(kernels_by_correlation[correlation])
            else:
                pass
    calls = len(parents)
    print(f"{frame_substring}: {calls} calls, per call:")
    for owner in sorted(
        set(wall_ms) | set(gpu_ms),
        key=lambda owner: -max(wall_ms[owner], gpu_ms[owner]),
    )[:TOP_ROWS]:
        print(
            f"  CPU wall {wall_ms[owner] / calls:7.1f} ms  GPU launched {gpu_ms[owner] / calls:7.1f} ms  {owner}"
        )
        for name, ms in kernels_ms[owner].most_common(3):
            print(f"      {ms / calls:7.2f}  {name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    summary = commands.add_parser(
        "summary", help="GPU and host summary of every stage trace"
    )
    summary.add_argument(
        "trace_dir",
        type=Path,
        help="the trace-<run id> directory with one folder per stage",
    )
    cut = commands.add_parser("slice", help="all stages of one window on one timeline")
    cut.add_argument("trace_dir", type=Path)
    cut.add_argument(
        "--start-s",
        type=float,
        default=1.5,
        help="from the first moment every stage ran a kernel",
    )
    cut.add_argument("--length-s", type=float, default=1.5)
    cut.add_argument("--output", type=Path, required=True)
    attribute = commands.add_parser(
        "attribute", help="GPU time launched under each child of a Python frame"
    )
    attribute.add_argument(
        "trace", type=Path, help="one stage trace, or a slice with --stage"
    )
    attribute.add_argument(
        "frame",
        help="substring of the Python frame name, e.g. vocoder.py(372): stream_batch",
    )
    attribute.add_argument("--stage", choices=STAGES, default=None)
    args = parser.parse_args()

    if args.command == "summary":
        report_summary(args.trace_dir)
    elif args.command == "slice":
        write_slice(args.trace_dir, args.start_s, args.length_s, args.output)
    else:
        report_attribution(args.trace, args.frame, args.stage)


if __name__ == "__main__":
    main()
