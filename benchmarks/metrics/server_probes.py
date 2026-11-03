# SPDX-License-Identifier: Apache-2.0
"""Server-side views of one serving level, for any pipeline: each request's time split by stage and each scheduler's real batches from the profiler's request events, GPU memory per stage, and GIL holding per process from Nsight Systems. Run as a module to record them around any benchmark command."""

from __future__ import annotations

import argparse
import contextlib
import json
import re
import sqlite3
import subprocess
import threading
import time
import urllib.request
from collections import defaultdict
from collections.abc import Iterator
from pathlib import Path

import numpy as np
from pydantic import JsonValue

from sglang_omni.http.admin_auth import admin_auth_headers

EVENT_FILE_NAME = re.compile(r"events_(?P<stage>[A-Za-z0-9_]+)_(?P<pid>\d+)\.jsonl")
RUN_STATUS_FILE_NAME = re.compile(r"run_status_(?P<pid>\d+)\.json")
# note (0xtoward): each process writes its events and then its run status only after it gets the stop, so the files are final once every events file has its status and no new file shows up for a while.
EVENT_FILES_SETTLE_S = 2.0
EVENT_FILES_POLL_S = 0.1
EVENT_FILES_TIMEOUT_S = 120.0
MEMORY_SAMPLE_PERIOD_S = 1.0
PROFILE_REQUEST_TIMEOUT_S = 30.0
# note (0xtoward): a scheduler emits one batch's prefill events in one loop, well under a millisecond apart; two batches are a model step apart.
PREFILL_BATCH_GAP_NS = 1_000_000
LONG_GIL_HOLD_NS = 20_000_000
BUSIEST_THREADS = 3
ADMISSION = ("coordinator", "request_admission")
RESPONSE = ("coordinator", "terminal_response")
# note (0xtoward): segments within one stage, between events that stage records when it has them.
STAGE_SEGMENTS = (
    ("queue", "stage_input_received", "stage_dispatch"),
    ("prefill wait", "stage_dispatch", "scheduler_prefill_start"),
    ("prefill", "scheduler_prefill_start", "scheduler_prefill_end"),
    ("decode", "scheduler_prefill_end", "model_path_end"),
    ("first chunk", "stage_dispatch", "stage_first_stream_chunk_sent"),
    ("compute", "stage_dispatch", "stage_complete"),
)


def level_event_dir(run_dir: Path) -> Path:
    return run_dir.with_name(f"{run_dir.name}.events")


def level_memory_path(run_dir: Path) -> Path:
    return run_dir.with_name(f"{run_dir.name}.gpu-memory.jsonl")


def level_trace_path(run_dir: Path) -> Path:
    return run_dir.with_name(f"{run_dir.name}.nsys-rep")


def post_profile_command(url: str, body: dict[str, JsonValue]) -> None:
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", **admin_auth_headers()},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=PROFILE_REQUEST_TIMEOUT_S):
        pass


def wait_for_event_files(event_dir: Path) -> None:
    """Wait until every process that wrote events has written its run status and no new file came for EVENT_FILES_SETTLE_S."""
    deadline_s = time.monotonic() + EVENT_FILES_TIMEOUT_S
    seen_names: set[str] = set()
    unchanged_since_s = time.monotonic()
    while time.monotonic() < deadline_s:
        names = {path.name for path in event_dir.glob("*")}
        event_pids = {
            match["pid"] for match in map(EVENT_FILE_NAME.fullmatch, names) if match
        }
        status_pids = {
            match["pid"]
            for match in map(RUN_STATUS_FILE_NAME.fullmatch, names)
            if match
        }
        if names != seen_names:
            seen_names = names
            unchanged_since_s = time.monotonic()
        elif (
            status_pids
            and event_pids <= status_pids
            and time.monotonic() - unchanged_since_s >= EVENT_FILES_SETTLE_S
        ):
            return
        else:
            pass
        time.sleep(EVENT_FILES_POLL_S)
    raise TimeoutError(
        f"request event files in {event_dir} did not settle within {EVENT_FILES_TIMEOUT_S:g} s"
    )


@contextlib.contextmanager
def request_events(server_url: str, event_dir: Path) -> Iterator[None]:
    """Record every stage's request events into event_dir while the block runs, and return once every process has written them; the server must share this filesystem."""
    run_id = event_dir.name
    post_profile_command(
        f"{server_url}/start_request_profile",
        {"run_id": run_id, "event_dir": str(event_dir.resolve())},
    )
    try:
        yield
    finally:
        post_profile_command(f"{server_url}/stop_request_profile", {"run_id": run_id})
        wait_for_event_files(event_dir)


def nvidia_smi_numbers(query: list[str]) -> list[list[int]]:
    output = subprocess.run(
        ["nvidia-smi", *query, "--format=csv,noheader,nounits"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return [
        [int(field) for field in line.split(",")]
        for line in output.splitlines()
        if line.strip()
    ]


@contextlib.contextmanager
def gpu_memory_samples(samples_path: Path) -> Iterator[None]:
    """Sample every visible GPU's used memory and utilization and every compute process's memory once a second while the block runs."""
    is_stopping = threading.Event()

    def sample() -> None:
        with samples_path.open("x", encoding="utf-8") as samples_file:
            while not is_stopping.is_set():
                cards = nvidia_smi_numbers(["--query-gpu=memory.used,utilization.gpu"])
                process_used_mib = {
                    str(pid): used
                    for pid, used in nvidia_smi_numbers(
                        ["--query-compute-apps=pid,used_memory"]
                    )
                }
                samples_file.write(
                    json.dumps(
                        {
                            "time_s": time.time(),
                            "card_used_mib": [used for used, _ in cards],
                            "card_busy_percent": [busy for _, busy in cards],
                            "process_used_mib": process_used_mib,
                        }
                    )
                    + "\n"
                )
                is_stopping.wait(MEMORY_SAMPLE_PERIOD_S)

    sampler = threading.Thread(target=sample, daemon=True)
    sampler.start()
    try:
        yield
    finally:
        is_stopping.set()
        sampler.join()


@contextlib.contextmanager
def nsys_collection(session: str, trace_path: Path) -> Iterator[None]:
    """Collect the server's Nsight Systems session into trace_path while the block runs, then export it to SQLite next to it.

    The server must run under ``nsys launch --session-new=<session> --trace=python-gil,nvtx --trace-fork-before-exec=true``.
    """
    subprocess.run(
        ["nsys", "start", f"--session={session}", "--sample=none", "--cpuctxsw=none"]
        + ["--force-overwrite=true", "-o", str(trace_path)],
        check=True,
        capture_output=True,
    )
    try:
        yield
    finally:
        subprocess.run(
            ["nsys", "stop", f"--session={session}"], check=True, capture_output=True
        )
        # note (0xtoward): export before the next level starts, so it never runs inside a measured window.
        subprocess.run(
            ["nsys", "export", "--type=sqlite", "--force-overwrite=true"]
            + ["-o", str(trace_path.with_suffix(".sqlite")), str(trace_path)],
            check=True,
            capture_output=True,
        )


def stage_pids(event_dir: Path) -> dict[str, str]:
    """Stage of every server process that wrote request events, by process ID."""
    return {
        match["pid"]: match["stage"]
        for match in (
            EVENT_FILE_NAME.fullmatch(path.name) for path in event_dir.glob("*.jsonl")
        )
        if match is not None
    }


def quantiles_ms(values_ms: list[float]) -> dict[str, JsonValue]:
    return {
        "count": len(values_ms),
        "p50": float(np.percentile(values_ms, 50)) if values_ms else None,
        "p90": float(np.percentile(values_ms, 90)) if values_ms else None,
    }


def request_segments_ms(
    events: dict[tuple[str, str], float]
) -> tuple[str, dict[str, float]]:
    """The stages a request went through, in order, and the time of every segment between two of its events."""
    path = sorted(
        (stage for stage, event_name in events if event_name == "stage_input_received"),
        key=lambda stage: events[stage, "stage_input_received"],
    )
    segments_ms: dict[str, float] = {}
    previous_end = ADMISSION
    previous_name = "admission"
    for stage in path:
        segments_ms[f"{previous_name} to {stage}"] = (
            events[stage, "stage_input_received"] - events[previous_end]
        )
        for name, start, end in STAGE_SEGMENTS:
            if (stage, start) in events and (stage, end) in events:
                segments_ms[f"{stage} {name}"] = (
                    events[stage, end] - events[stage, start]
                )
            else:
                pass
        # note (0xtoward): a streaming stage hands on its first chunk; the response does not wait for it to complete.
        previous_end = (
            (stage, "stage_first_stream_chunk_sent")
            if (stage, "stage_first_stream_chunk_sent") in events
            else (stage, "stage_complete")
        )
        previous_name = stage
    segments_ms[f"{previous_name} to response"] = (
        events[RESPONSE] - events[previous_end]
    )
    return " > ".join(path), segments_ms


def level_events(event_dir: Path) -> Iterator[dict[str, JsonValue]]:
    for path in sorted(event_dir.glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            yield json.loads(line)


def stage_split(event_dir: Path) -> dict[str, JsonValue]:
    """Requests grouped by the stages they went through, each group with its admission-to-response time and per-segment times."""
    requests: dict[str, dict[tuple[str, str], float]] = defaultdict(dict)
    for event in level_events(event_dir):
        # note (0xtoward): a streaming stage repeats its events per chunk; the first one marks the request.
        requests[event["request_id"]].setdefault(
            (event["stage"], event["event_name"]), event["timestamp_ns"] / 1e6
        )
    groups: dict[str, list[dict[str, float]]] = defaultdict(list)
    totals_ms: dict[str, list[float]] = defaultdict(list)
    for events in requests.values():
        if ADMISSION in events and RESPONSE in events:
            stage_path, segments_ms = request_segments_ms(events)
            groups[stage_path].append(segments_ms)
            totals_ms[stage_path].append(events[RESPONSE] - events[ADMISSION])
        else:
            pass
    return {
        stage_path: {
            "admission_to_response_ms": quantiles_ms(totals_ms[stage_path]),
            "segments_ms": {
                name: quantiles_ms(
                    [segments[name] for segments in group if name in segments]
                )
                for name in dict.fromkeys(
                    name for segments in group for name in segments
                )
            },
        }
        for stage_path, group in sorted(groups.items(), key=lambda item: -len(item[1]))
    }


def stage_memory(samples_path: Path, event_dir: Path) -> dict[str, JsonValue]:
    """Peak used memory of the cards and of every process, named by its stage when it wrote request events, and the cards' busy share."""
    samples = [
        json.loads(line)
        for line in samples_path.read_text(encoding="utf-8").splitlines()
    ]
    names = stage_pids(event_dir) if event_dir.is_dir() else {}
    peak_process_mib: dict[str, int] = defaultdict(int)
    for sample in samples:
        for pid, used_mib in sample["process_used_mib"].items():
            name = names.get(pid, f"pid {pid}")
            peak_process_mib[name] = max(peak_process_mib[name], used_mib)
    busy_percent = [
        sum(sample["card_busy_percent"]) / len(sample["card_busy_percent"])
        for sample in samples
    ]
    return {
        "samples": len(samples),
        "card_peak_mib": max(
            (sum(sample["card_used_mib"]) for sample in samples), default=None
        ),
        "card_busy_percent_mean": (
            float(np.mean(busy_percent)) if busy_percent else None
        ),
        "process_peak_mib": dict(sorted(peak_process_mib.items())),
    }


def time_share_quantile(
    seconds_by_count: dict[int, float], fraction: float
) -> int | None:
    """The smallest count at or below which the scheduler spent at least fraction of its busy time."""
    covered_s = 0.0
    for count in sorted(seconds_by_count):
        covered_s += seconds_by_count[count]
        if covered_s >= fraction * sum(seconds_by_count.values()):
            return count
        else:
            pass
    return None


def batch_summary(
    first_forward_ns: dict[str, int], end_ns: dict[str, int]
) -> dict[str, JsonValue]:
    last_ns = max([*first_forward_ns.values(), *end_ns.values()])
    # note (0xtoward): a request still running when the level stops counts until the level's last event.
    changes = sorted(
        [(start_ns, 1) for start_ns in first_forward_ns.values()]
        + [(end_ns.get(request_id, last_ns), -1) for request_id in first_forward_ns]
    )
    seconds_by_running: dict[int, float] = defaultdict(float)
    running = 0
    previous_ns = changes[0][0]
    for time_ns, change in changes:
        if running > 0:
            seconds_by_running[running] += (time_ns - previous_ns) / 1e9
        else:
            pass
        running += change
        previous_ns = time_ns
    prefill_ns = sorted(first_forward_ns.values())
    requests_per_prefill = [1]
    for previous, current in zip(prefill_ns, prefill_ns[1:]):
        if current - previous < PREFILL_BATCH_GAP_NS:
            requests_per_prefill[-1] += 1
        else:
            requests_per_prefill.append(1)
    busy_s = sum(seconds_by_running.values())
    window_s = (last_ns - prefill_ns[0]) / 1e9
    return {
        "busy_s": busy_s,
        "running_mean": (
            sum(count * seconds for count, seconds in seconds_by_running.items())
            / busy_s
            if busy_s > 0
            else None
        ),
        "running_p50": time_share_quantile(seconds_by_running, 0.5),
        "running_p90": time_share_quantile(seconds_by_running, 0.9),
        "running_max": max(seconds_by_running, default=None),
        "prefills": len(requests_per_prefill),
        "prefills_per_s": (
            len(requests_per_prefill) / window_s if window_s > 0 else None
        ),
        "requests_per_prefill_mean": float(np.mean(requests_per_prefill)),
        "requests_per_prefill_p90": float(np.percentile(requests_per_prefill, 90)),
    }


def scheduler_batches(event_dir: Path) -> dict[str, JsonValue]:
    """Per stage scheduler, how many requests ran together over time and how many each prefill took in.

    A request runs from its first forward (``scheduler_prefill_end``) to ``model_path_end``; for a scheduler that
    steps every running request, that count is its decode batch. Replicas record under their own stage names.
    """
    first_forward_ns: dict[str, dict[str, int]] = defaultdict(dict)
    end_ns: dict[str, dict[str, int]] = defaultdict(dict)
    for event in level_events(event_dir):
        if event["event_name"] == "scheduler_prefill_end":
            first_forward_ns[event["stage"]].setdefault(
                event["request_id"], event["timestamp_ns"]
            )
        elif event["event_name"] == "model_path_end":
            end_ns[event["stage"]].setdefault(
                event["request_id"], event["timestamp_ns"]
            )
        else:
            pass
    return {
        stage: batch_summary(starts, end_ns[stage])
        for stage, starts in sorted(first_forward_ns.items())
    }


def simple_stage_batches(event_dir: Path) -> dict[str, JsonValue]:
    """Per stage on a SimpleScheduler or SessionScheduler, its batch sizes, how long it held each batch open for more requests, each request's wait from dispatch to its batch and each batch's compute time.

    The requests of one batch share its scheduler_batch_start timestamp; busy is the share of the stage's window spent computing.
    """
    dispatch_ns: dict[str, dict[str, int]] = defaultdict(dict)
    batch_request_ids: dict[str, dict[int, list[str]]] = defaultdict(
        lambda: defaultdict(list)
    )
    gather_ms: dict[str, dict[int, float]] = defaultdict(dict)
    end_ns: dict[str, dict[str, int]] = defaultdict(dict)
    for event in level_events(event_dir):
        stage, request_id = event["stage"], event["request_id"]
        if event["event_name"] == "stage_dispatch":
            dispatch_ns[stage].setdefault(request_id, event["timestamp_ns"])
        elif event["event_name"] == "scheduler_batch_start":
            batch_request_ids[stage][event["timestamp_ns"]].append(request_id)
            gather_ms[stage][event["timestamp_ns"]] = event["metadata"]["gather_ms"]
        elif event["event_name"] == "scheduler_batch_end":
            end_ns[stage].setdefault(request_id, event["timestamp_ns"])
        else:
            pass
    summaries: dict[str, JsonValue] = {}
    for stage, batches in sorted(batch_request_ids.items()):
        sizes = [len(request_ids) for request_ids in batches.values()]
        compute_ms = [
            (end_ns[stage][request_ids[0]] - start_ns) / 1e6
            for start_ns, request_ids in batches.items()
            if request_ids[0] in end_ns[stage]
        ]
        window_ms = (max([*batches, *end_ns[stage].values()]) - min(batches)) / 1e6
        summaries[stage] = {
            "batches": len(sizes),
            "requests_per_batch_mean": float(np.mean(sizes)),
            "requests_per_batch_max": max(sizes),
            "requests_alone_percent": 100 * sizes.count(1) / sum(sizes),
            "gather_ms": quantiles_ms(list(gather_ms[stage].values())),
            "wait_ms": quantiles_ms(
                [
                    (start_ns - dispatch_ns[stage][request_id]) / 1e6
                    for start_ns, request_ids in batches.items()
                    for request_id in request_ids
                    if request_id in dispatch_ns[stage]
                ]
            ),
            "compute_ms": quantiles_ms(compute_ms),
            "busy_percent": (
                100 * sum(compute_ms) / window_ms if window_ms > 0 else None
            ),
        }
    return summaries


def gil_shares(sqlite_path: Path, event_dir: Path) -> dict[str, JsonValue]:
    """Per server process, named by its stage when it wrote request events, from an Nsight Systems python-gil export:
    the share of the collection window some thread held the GIL, the holds of at least 20 ms, and the busiest threads'
    holding and waiting shares."""
    # note (0xtoward): py-spy's blocking sampler stops threads one at a time, so the threads still running pile up
    # waiting for the GIL (36% against nsys's 12% for one scheduler); nsys records every acquire for about 2% overhead,
    # but its injection breaks the MPS daemon, so collect with MPS off.
    names = stage_pids(event_dir) if event_dir.is_dir() else {}
    with contextlib.closing(sqlite3.connect(sqlite_path)) as connection:
        text_ids = dict(
            connection.execute(
                "select value, id from StringIds where value in ('Holding GIL', 'Waiting for GIL')"
            ).fetchall()
        )
        if len(text_ids) < 2:
            raise RuntimeError(
                f"{sqlite_path} has no Python GIL events; launch the server under nsys with --trace=python-gil"
            )
        else:
            pass
        holding_id = text_ids["Holding GIL"]
        waiting_id = text_ids["Waiting for GIL"]
        start_ns, end_ns = connection.execute(
            "select min(start), max(end) from NVTX_EVENTS where textId in (?, ?)",
            (holding_id, waiting_id),
        ).fetchone()
        rows = connection.execute(
            "select globalTid, textId, sum(end - start), count(*), sum(end - start >= ?) "
            "from NVTX_EVENTS where textId in (?, ?) group by globalTid, textId",
            (LONG_GIL_HOLD_NS, holding_id, waiting_id),
        ).fetchall()
        thread_names = dict(
            connection.execute(
                "select globalTid, value from ThreadNames join StringIds on ThreadNames.nameId = StringIds.id"
            ).fetchall()
        )
    window_ns = end_ns - start_ns
    threads: dict[int, dict[str, int]] = defaultdict(
        lambda: {"hold_ns": 0, "wait_ns": 0, "waits": 0, "long_holds": 0}
    )
    for thread, text_id, total_ns, count, long_count in rows:
        if text_id == holding_id:
            threads[thread]["hold_ns"] += total_ns
            threads[thread]["long_holds"] += long_count
        else:
            threads[thread]["wait_ns"] += total_ns
            threads[thread]["waits"] += count
    processes: dict[str, list[tuple[int, dict[str, int]]]] = defaultdict(list)
    for thread, totals in threads.items():
        # note (0xtoward): nsys packs the process ID into bits 24-47 of a global thread ID and the thread ID into the low 24.
        processes[str((thread >> 24) & 0xFFFFFF)].append((thread, totals))
    held_ns = {
        pid: sum(totals["hold_ns"] for _, totals in process_threads)
        for pid, process_threads in processes.items()
    }
    return {
        "window_s": window_ns / 1e9,
        "processes": {
            names.get(pid, f"pid {pid}"): {
                "held_percent": held_ns[pid] / window_ns * 100,
                "long_holds": sum(
                    totals["long_holds"] for _, totals in process_threads
                ),
                "busiest_threads": [
                    {
                        # note (0xtoward): Python threads are all named "python", so the thread ID tells them apart.
                        "thread": f"{thread_names.get(thread, 'thread')} {thread & 0xFFFFFF}",
                        "hold_percent": totals["hold_ns"] / window_ns * 100,
                        "wait_percent": totals["wait_ns"] / window_ns * 100,
                        "mean_wait_us": totals["wait_ns"]
                        / max(totals["waits"], 1)
                        / 1e3,
                    }
                    for thread, totals in sorted(
                        process_threads,
                        key=lambda item: -(item[1]["hold_ns"] + item[1]["wait_ns"]),
                    )[:BUSIEST_THREADS]
                ],
            }
            for pid, process_threads in sorted(
                processes.items(), key=lambda item: -held_ns[item[0]]
            )
        },
    }


def server_views(run_dir: Path) -> dict[str, JsonValue]:
    """Every view whose input was recorded beside run_dir, None for the others."""
    event_dir = level_event_dir(run_dir)
    memory_path = level_memory_path(run_dir)
    sqlite_path = level_trace_path(run_dir).with_suffix(".sqlite")
    return {
        "stages": stage_split(event_dir) if event_dir.is_dir() else None,
        "scheduler_batches": (
            scheduler_batches(event_dir) if event_dir.is_dir() else None
        ),
        "simple_stage_batches": (
            simple_stage_batches(event_dir) if event_dir.is_dir() else None
        ),
        "gpu_memory": (
            stage_memory(memory_path, event_dir) if memory_path.is_file() else None
        ),
        "gil": (gil_shares(sqlite_path, event_dir) if sqlite_path.is_file() else None),
    }


def format_server_tables(levels: dict[str, dict[str, JsonValue]]) -> str:
    """Markdown tables of the server-side views, one column per level that recorded them, keyed by the column label."""
    tables = ""
    staged = {label: level for label, level in levels.items() if level["stages"]}
    if staged:
        full_path = max(
            (stage_path for level in staged.values() for stage_path in level["stages"]),
            key=lambda stage_path: stage_path.count(">"),
        )
        groups = [level["stages"].get(full_path) for level in staged.values()]
        segment_names = list(
            dict.fromkeys(
                name for group in groups if group for name in group["segments_ms"]
            )
        )
        tables += (
            f"\nRequests through {full_path}, median (p90) ms:\n\n| segment | "
            + " | ".join(staged)
            + " |\n|---|"
            + "---:|" * len(staged)
            + "\n"
        )
        for segment_name in ["admission to response", *segment_names]:
            cells = []
            for group in groups:
                value = (
                    None
                    if group is None
                    else (
                        group["admission_to_response_ms"]
                        if segment_name == "admission to response"
                        else group["segments_ms"].get(segment_name)
                    )
                )
                cells.append(
                    "-"
                    if value is None or value["p50"] is None
                    else f"{value['p50']:.0f} ({value['p90']:.0f})"
                )
            tables += f"| {segment_name} | " + " | ".join(cells) + " |\n"
    else:
        pass
    sampled = {
        label: level
        for label, level in levels.items()
        if level["gpu_memory"] is not None
    }
    if sampled:
        process_names = sorted(
            {
                name
                for level in sampled.values()
                for name in level["gpu_memory"]["process_peak_mib"]
            }
        )
        tables += (
            "\nPeak GPU memory (MiB) and mean GPU busy share:\n\n| process | "
            + " | ".join(sampled)
            + " |\n|---|"
            + "---:|" * len(sampled)
            + "\n"
        )
        for name in ["card", *process_names]:
            cells = [
                str(
                    level["gpu_memory"]["card_peak_mib"]
                    if name == "card"
                    else level["gpu_memory"]["process_peak_mib"].get(name, "-")
                )
                for level in sampled.values()
            ]
            tables += f"| {name} | " + " | ".join(cells) + " |\n"
        tables += (
            "| busy | "
            + " | ".join(
                f"{level['gpu_memory']['card_busy_percent_mean']:.0f}%"
                for level in sampled.values()
            )
            + " |\n"
        )
    else:
        pass
    batched = {
        label: level for label, level in levels.items() if level["scheduler_batches"]
    }
    if batched:
        tables += (
            "\nRunning requests per scheduler, time-weighted mean (p90), and prefills per s "
            "x requests per prefill:\n\n| scheduler | "
            + " | ".join(batched)
            + " |\n|---|"
            + "---:|" * len(batched)
            + "\n"
        )
        for name in sorted(
            {name for level in batched.values() for name in level["scheduler_batches"]}
        ):
            cells = []
            for level in batched.values():
                batches = level["scheduler_batches"].get(name)
                cells.append(
                    "-"
                    if batches is None or batches["running_mean"] is None
                    else f"{batches['running_mean']:.1f} ({batches['running_p90']}), "
                    f"{batches['prefills_per_s']:.1f} x {batches['requests_per_prefill_mean']:.1f}"
                )
            tables += f"| {name} | " + " | ".join(cells) + " |\n"
    else:
        pass
    simple = {
        label: level for label, level in levels.items() if level["simple_stage_batches"]
    }
    if simple:
        tables += (
            "\nBatches of the simple and session stages: requests per batch mean (max) / "
            "requests that ran alone / busy share / p90 wait from dispatch to the batch:\n\n"
            "| stage | "
            + " | ".join(simple)
            + " |\n|---|"
            + "---:|" * len(simple)
            + "\n"
        )
        for name in sorted(
            {
                name
                for level in simple.values()
                for name in level["simple_stage_batches"]
            }
        ):
            cells = []
            for level in simple.values():
                batches = level["simple_stage_batches"].get(name)
                cells.append(
                    "-"
                    if batches is None
                    or batches["busy_percent"] is None
                    or batches["wait_ms"]["p90"] is None
                    else f"{batches['requests_per_batch_mean']:.1f} "
                    f"({batches['requests_per_batch_max']}) / "
                    f"{batches['requests_alone_percent']:.0f}% / "
                    f"{batches['busy_percent']:.0f}% / {batches['wait_ms']['p90']:.0f} ms"
                )
            tables += f"| {name} | " + " | ".join(cells) + " |\n"
    else:
        pass
    traced = {
        label: level for label, level in levels.items() if level["gil"] is not None
    }
    if traced:
        tables += (
            "\nGIL from Nsight Systems: share of the window some thread held it / busiest "
            "thread's wait share / holds of 20 ms or more:\n\n| process | "
            + " | ".join(traced)
            + " |\n|---|"
            + "---:|" * len(traced)
            + "\n"
        )
        # note (0xtoward): helper processes that barely run Python would only add rows of zeros.
        for name in sorted(
            {
                name
                for level in traced.values()
                for name, process in level["gil"]["processes"].items()
                if process["held_percent"] >= 1
            }
        ):
            cells = []
            for level in traced.values():
                process = level["gil"]["processes"].get(name)
                cells.append(
                    "-"
                    if process is None
                    else f"{process['held_percent']:.0f}% / "
                    f"{max(thread['wait_percent'] for thread in process['busiest_threads']):.0f}% / "
                    f"{process['long_holds']}"
                )
            tables += f"| {name} | " + " | ".join(cells) + " |\n"
    else:
        pass
    return tables


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run any benchmark command against a server on this host with its "
        "request events recorded, then print the server-side views.",
        usage="%(prog)s --server-url URL --output-dir DIR [--gpu-memory] -- COMMAND ...",
    )
    parser.add_argument(
        "--server-url",
        required=True,
        help="HTTP base URL of the server, such as http://127.0.0.1:8000",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="New directory for the request events (server.events), the GPU memory "
        "samples (server.gpu-memory.jsonl) and the views (server.json)",
    )
    parser.add_argument(
        "--gpu-memory",
        action="store_true",
        help="Also sample GPU memory and utilization once a second",
    )
    parser.add_argument(
        "command", nargs=argparse.REMAINDER, help="Benchmark command, after --"
    )
    args = parser.parse_args(argv)
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("give the benchmark command after --")
    else:
        pass
    args.output_dir.mkdir(parents=True)
    run_dir = args.output_dir / "server"
    with contextlib.ExitStack() as probes:
        probes.enter_context(request_events(args.server_url, level_event_dir(run_dir)))
        if args.gpu_memory:
            probes.enter_context(gpu_memory_samples(level_memory_path(run_dir)))
        else:
            pass
        returncode = subprocess.run(command, check=False).returncode
    views = server_views(run_dir)
    (args.output_dir / "server.json").write_text(json.dumps(views, indent=2) + "\n")
    print(format_server_tables({args.output_dir.name: views}))
    return returncode


if __name__ == "__main__":
    raise SystemExit(main())
else:
    pass
