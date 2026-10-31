# SPDX-License-Identifier: Apache-2.0
"""Server-side views of one serving level, for any pipeline: each request's time split by stage and each scheduler's real batches from the profiler's request events, GPU memory per stage, and GIL holding per process from Nsight Systems."""

from __future__ import annotations

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


@contextlib.contextmanager
def request_events(server_url: str, event_dir: Path) -> Iterator[None]:
    """Record every stage's request events into event_dir while the block runs; the server must share this filesystem."""
    run_id = event_dir.name
    post_profile_command(
        f"{server_url}/start_request_profile",
        {"run_id": run_id, "event_dir": str(event_dir.resolve())},
    )
    try:
        yield
    finally:
        post_profile_command(f"{server_url}/stop_request_profile", {"run_id": run_id})


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
