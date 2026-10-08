# SPDX-License-Identifier: Apache-2.0
"""The ambient KV byte budget scope and its stage-worker wiring.

The budget must reach the engine bootstrap without touching model factory
signatures, and a declared budget that no engine consumes must fail startup
instead of being silently dropped.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager

import pytest

import sglang_omni.pipeline.stage_workers as stage_workers
from sglang_omni.pipeline.stage_workers import StageLaunchConfig, construct_scheduler
from sglang_omni.scheduling.stage_kv_budget import (
    consume_stage_kv_cache_bytes,
    peek_stage_kv_cache_bytes,
    stage_kv_cache_budget,
)
from tests.unit_test.fixtures.pipeline_fakes import fake_factory_path

LOG = logging.getLogger(__name__)


def test_consume_outside_scope_returns_none() -> None:
    assert consume_stage_kv_cache_bytes() is None
    assert peek_stage_kv_cache_bytes() is None


def test_scope_delivers_budget_once_consumed() -> None:
    with stage_kv_cache_budget("thinker", 2 * 1024**3):
        assert peek_stage_kv_cache_bytes() == 2 * 1024**3
        assert consume_stage_kv_cache_bytes() == 2 * 1024**3
    assert consume_stage_kv_cache_bytes() is None


def test_second_consume_in_one_scope_raises() -> None:
    """Two engines each taking the full stage budget would silently commit
    twice the declared bytes."""
    with stage_kv_cache_budget("thinker", 2 * 1024**3):
        assert consume_stage_kv_cache_bytes() == 2 * 1024**3
        with pytest.raises(RuntimeError, match="second SGLang engine"):
            consume_stage_kv_cache_bytes()
    assert peek_stage_kv_cache_bytes() is None


def test_unconsumed_scope_raises_on_exit() -> None:
    with pytest.raises(RuntimeError, match="'vocoder'.*did not build"):
        with stage_kv_cache_budget("vocoder", 1024**3):
            pass
    assert peek_stage_kv_cache_bytes() is None


def test_peek_does_not_count_as_consumption() -> None:
    with pytest.raises(RuntimeError, match="did not build"):
        with stage_kv_cache_budget("thinker", 1024**3):
            peek_stage_kv_cache_bytes()


def test_factory_exception_is_not_masked_by_consumption_check() -> None:
    with pytest.raises(ValueError, match="factory boom"):
        with stage_kv_cache_budget("thinker", 1024**3):
            raise ValueError("factory boom")
    assert peek_stage_kv_cache_bytes() is None


def test_nested_scopes_are_rejected() -> None:
    with pytest.raises(RuntimeError, match="cannot nest"):
        with stage_kv_cache_budget("thinker", 1024**3):
            with stage_kv_cache_budget("talker_ar", 1024**3):
                pass


def make_spec(
    factory_name: str, kv_cache_bytes: int | None = None
) -> StageLaunchConfig:
    return StageLaunchConfig(
        stage_name="thinker",
        factory=fake_factory_path(factory_name),
        kv_cache_bytes=kv_cache_bytes,
    )


def test_construct_scheduler_scopes_budget_around_factory() -> None:
    spec = make_spec(
        "make_scheduler_consuming_kv_budget",
        kv_cache_bytes=3 * 1024**3,
    )

    scheduler = construct_scheduler(spec, None, LOG)

    assert scheduler.consumed_kv_cache_bytes == 3 * 1024**3
    assert "kv_cache_bytes" not in scheduler.factory_kwargs


def test_construct_scheduler_fails_when_budget_is_not_consumed() -> None:
    spec = make_spec("make_scheduler", kv_cache_bytes=3 * 1024**3)

    with pytest.raises(RuntimeError, match="'thinker'.*did not build"):
        construct_scheduler(spec, None, LOG)


def test_construct_scheduler_without_budget_opens_no_scope() -> None:
    spec = make_spec("make_scheduler_consuming_kv_budget")

    scheduler = construct_scheduler(spec, None, LOG)

    assert scheduler.consumed_kv_cache_bytes is None


def test_total_reserve_cap_accumulates_across_colocated_stages(monkeypatch):
    import sys
    from types import SimpleNamespace

    import sglang_omni.pipeline.stage_workers as stage_workers

    calls: list[tuple[float, int]] = []
    fake_cuda = SimpleNamespace(
        is_available=lambda: True,
        get_device_properties=lambda device: SimpleNamespace(total_memory=100),
        set_per_process_memory_fraction=lambda fraction, device: calls.append(
            (round(fraction, 4), device)
        ),
    )
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=fake_cuda))
    monkeypatch.setattr(stage_workers, "_process_reserve_bytes", {})

    first = StageLaunchConfig(stage_name="a", total_reserve_bytes=30)
    second = StageLaunchConfig(stage_name="b", total_reserve_bytes=50)
    stage_workers.apply_total_reserve_cap(first, 0, LOG)
    stage_workers.apply_total_reserve_cap(second, 0, LOG)

    assert calls == [(0.3, 0), (0.8, 0)]


def test_total_reserve_cap_respects_opt_out_and_absence(monkeypatch):
    import sys
    from types import SimpleNamespace

    import sglang_omni.pipeline.stage_workers as stage_workers

    calls: list[tuple[float, int]] = []
    fake_cuda = SimpleNamespace(
        is_available=lambda: True,
        get_device_properties=lambda device: SimpleNamespace(total_memory=100),
        set_per_process_memory_fraction=lambda fraction, device: calls.append(
            (fraction, device)
        ),
    )
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=fake_cuda))
    monkeypatch.setattr(stage_workers, "_process_reserve_bytes", {})

    opted_out = StageLaunchConfig(
        stage_name="a", total_reserve_bytes=30, enforce_total_reserve=False
    )
    undeclared = StageLaunchConfig(stage_name="b")
    stage_workers.apply_total_reserve_cap(opted_out, 0, LOG)
    stage_workers.apply_total_reserve_cap(undeclared, 0, LOG)

    assert calls == []


@pytest.mark.parametrize("factory_fails", [False, True])
def test_startup_lock_wait_excludes_factory_and_releases_on_failure(
    monkeypatch: pytest.MonkeyPatch,
    factory_fails: bool,
) -> None:
    events: list[str] = []
    locked = False

    @contextmanager
    def phase(name: str, **labels) -> Iterator[None]:
        events.append(f"begin:{name}")
        try:
            yield
        finally:
            events.append(f"end:{name}")

    @contextmanager
    def startup_lock(device: int) -> Iterator[str]:
        nonlocal locked
        locked = True
        events.append("lock:acquired")
        try:
            yield "test-lock"
        finally:
            locked = False
            events.append("lock:released")

    def factory():
        assert locked
        events.append("factory")
        if factory_fails:
            raise ValueError("factory failed")
        else:
            return scheduler

    scheduler = construct_scheduler(make_spec("make_scheduler"), None, LOG)
    monkeypatch.setattr(stage_workers, "startup_phase", phase)
    monkeypatch.setattr(stage_workers, "gpu_startup_lock", startup_lock)
    monkeypatch.setattr(stage_workers, "import_string", lambda name: factory)
    spec = StageLaunchConfig(stage_name="codec", factory="test.factory")
    if factory_fails:
        with pytest.raises(ValueError, match="factory failed"):
            construct_scheduler(spec, 0, LOG)
    else:
        assert construct_scheduler(spec, 0, LOG) is scheduler
    assert not locked
    assert events == [
        "begin:scheduler.initialize",
        "begin:factory.import",
        "end:factory.import",
        "begin:gpu_lock.wait",
        "lock:acquired",
        "end:gpu_lock.wait",
        "begin:factory.build",
        "factory",
        "end:factory.build",
        "lock:released",
        "end:scheduler.initialize",
    ]
