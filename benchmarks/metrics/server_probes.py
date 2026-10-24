# SPDX-License-Identifier: Apache-2.0
"""Server-side views of one serving level, for any pipeline: each request's time split by stage from the profiler's request events, and GPU memory per stage."""

from __future__ import annotations

import contextlib
import json
import re
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


def stage_split(event_dir: Path) -> dict[str, JsonValue]:
    """Requests grouped by the stages they went through, each group with its admission-to-response time and per-segment times."""
    requests: dict[str, dict[tuple[str, str], float]] = defaultdict(dict)
    for path in sorted(event_dir.glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            event = json.loads(line)
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
