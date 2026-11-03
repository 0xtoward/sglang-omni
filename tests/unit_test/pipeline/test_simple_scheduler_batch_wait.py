# SPDX-License-Identifier: Apache-2.0
"""Tests for SimpleScheduler batch coalescing window admission and the batch events it records.

Kept out of test_scheduler.py for the same reason as
test_simple_scheduler_concurrent.py: that module imports torch at top level.
"""

from __future__ import annotations

import json
import threading
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from sglang_omni.profiler.event_recorder import get_recorder
from sglang_omni.scheduling.message import IncomingMessage
from sglang_omni.scheduling.simple_scheduler import SimpleScheduler

WINDOW_MS = 200


def msg(request_id: str) -> IncomingMessage:
    return IncomingMessage(type="new_request", request_id=request_id, data=request_id)


def batching_scheduler(**kwargs: Any) -> SimpleScheduler:
    return SimpleScheduler(
        lambda payload: payload,
        batch_compute_fn=lambda payloads: list(payloads),
        max_batch_size=32,
        max_batch_wait_ms=WINDOW_MS,
        **kwargs,
    )


def run(
    scheduler: SimpleScheduler, messages: list[IncomingMessage], output_count: int
) -> tuple[list[Any], float]:
    """Return the results plus the milliseconds spent dispatching them.

    Note (wenyao): enqueue before start and stop the clock at the last result,
    or the scheduler's 100ms idle and shutdown polls swamp the window.
    """
    for message in messages:
        scheduler.inbox.put(message)
    thread = threading.Thread(target=scheduler.start, daemon=True)
    start = time.monotonic()
    thread.start()
    try:
        results = [scheduler.outbox.get(timeout=5.0) for _ in range(output_count)]
        return results, (time.monotonic() - start) * 1000
    finally:
        scheduler.stop()
        thread.join(timeout=2.0)


def test_idle_stage_does_not_wait_out_the_coalescing_window() -> None:
    scheduler = batching_scheduler(batch_wait_when_idle=False)
    _, elapsed_ms = run(scheduler, [msg("r1")], output_count=1)
    assert (
        elapsed_ms < WINDOW_MS / 2
    ), f"lone request waited {elapsed_ms:.1f}ms of the {WINDOW_MS}ms window"


def test_idle_batch_wait_remains_the_default_contract() -> None:
    scheduler = batching_scheduler()
    _, elapsed_ms = run(scheduler, [msg("r1")], output_count=1)
    assert (
        elapsed_ms >= WINDOW_MS / 2
    ), f"default batch wait dispatched after only {elapsed_ms:.1f}ms"


def test_backlog_still_coalesces_into_one_batch() -> None:
    seen_batches: list[int] = []
    scheduler = SimpleScheduler(
        lambda payload: payload,
        batch_compute_fn=lambda payloads: (
            seen_batches.append(len(payloads)) or list(payloads)
        ),
        max_batch_size=32,
        max_batch_wait_ms=WINDOW_MS,
        batch_wait_when_idle=False,
    )
    run(scheduler, [msg(f"r{i}") for i in range(8)], output_count=8)
    assert seen_batches, "batch compute never ran"
    assert max(seen_batches) > 1, f"backlog was not coalesced: {seen_batches}"


def test_late_arrival_joins_batch_once_a_backlog_exists() -> None:
    seen_batches: list[int] = []

    def batch_fn(payloads: list[Any]) -> list[Any]:
        seen_batches.append(len(payloads))
        return list(payloads)

    scheduler = SimpleScheduler(
        lambda payload: payload,
        batch_compute_fn=batch_fn,
        max_batch_size=32,
        max_batch_wait_ms=WINDOW_MS,
        batch_wait_when_idle=False,
    )
    scheduler.inbox.put(msg("r1"))
    scheduler.inbox.put(msg("r2"))
    thread = threading.Thread(target=scheduler.start, daemon=True)
    thread.start()
    try:
        time.sleep(WINDOW_MS / 4 / 1000)
        scheduler.inbox.put(msg("r3"))
        results = [scheduler.outbox.get(timeout=5.0) for _ in range(3)]
    finally:
        scheduler.stop()
        thread.join(timeout=2.0)
    assert len(results) == 3
    assert max(seen_batches) >= 3, f"straggler did not join: {seen_batches}"


def test_each_batch_records_a_shared_start_and_end_with_its_size(
    tmp_path: Path,
) -> None:
    scheduler = SimpleScheduler(
        lambda payload: payload,
        batch_compute_fn=lambda payloads: list(payloads),
        max_batch_size=32,
    )
    recorder = get_recorder()
    recorder.start("batch-events-test", str(tmp_path), "code2wav")
    for request_id in ("r0", "r1", "r2"):
        scheduler.inbox.put(msg(request_id))
    thread = threading.Thread(target=scheduler.start, daemon=True)

    thread.start()
    try:
        for _ in range(3):
            scheduler.outbox.get(timeout=5.0)
        scheduler.inbox.put(msg("r3"))
        scheduler.outbox.get(timeout=5.0)
    finally:
        scheduler.stop()
        thread.join(timeout=2.0)
        recorder.stop(wait=True)

    events = [
        json.loads(line)
        for path in tmp_path.glob("events_*.jsonl")
        for line in path.read_text().splitlines()
    ]
    members: dict[tuple[str, int], list[tuple[str, int]]] = defaultdict(list)
    for event in events:
        members[event["event_name"], event["timestamp_ns"]].append(
            (event["request_id"], event["metadata"]["batch_size"])
        )
    batches: dict[str, list[list[tuple[str, int]]]] = defaultdict(list)
    for (event_name, _), batch in members.items():
        batches[event_name].append(sorted(batch))
    expected = [[("r0", 3), ("r1", 3), ("r2", 3)], [("r3", 1)]]
    assert (
        sorted(batches["scheduler_batch_start"]) == expected
    ), "one start time per batch"
    assert sorted(batches["scheduler_batch_end"]) == expected, "one end time per batch"
    assert all(
        event["metadata"]["gather_ms"] >= 0
        for event in events
        if event["event_name"] == "scheduler_batch_start"
    ), "starts carry the gather time"
