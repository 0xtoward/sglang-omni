# SPDX-License-Identifier: Apache-2.0
"""Startup logs preserve context, compiler accounting, and failures."""

import asyncio
import json
import logging
import subprocess
import sys
from collections import deque
from collections.abc import Iterator
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from sglang_omni.utils.startup import startup_capture, startup_phase


@pytest.fixture
def timer_calls(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, bool]]:
    calls: list[tuple[str, bool]] = []

    @contextmanager
    def timer(name: str, *, log_only: bool) -> Iterator[None]:
        calls.append((name, log_only))
        yield

    monkeypatch.setitem(
        sys.modules,
        "sglang.srt.observability.startup_func_log_and_timer",
        SimpleNamespace(startup_timer=timer),
    )
    return calls


def test_stage_context_inherits_without_leaking(
    timer_calls: list[tuple[str, bool]]
) -> None:
    with (
        startup_phase("factory.build", stage="asr", tp_rank=1),
        startup_phase("model.resources"),
        startup_capture(backend="cuda"),
    ):
        pass
    with startup_phase("factory.build", stage="vocoder"):
        pass
    assert json.loads(timer_calls[2][0].split(" ", 1)[1]) == {
        "stage": "asr",
        "tp_rank": 1,
        "backend": "cuda",
    }
    assert json.loads(timer_calls[3][0].split(" ", 1)[1]) == {"stage": "vocoder"}
    assert all(log_only for _, log_only in timer_calls)


def test_failure_is_reported_and_context_is_reset(
    timer_calls: list[tuple[str, bool]], caplog: pytest.LogCaptureFixture
) -> None:
    failure = RuntimeError("failed warmup")
    with (
        caplog.at_level(logging.INFO),
        pytest.raises(RuntimeError) as caught,
        startup_phase("scheduler.warmup", stage="vocoder"),
    ):
        raise failure
    assert caught.value is failure
    assert (
        'Startup failed: omni.scheduler.warmup {"stage": "vocoder"} error=RuntimeError'
        in caplog.text
    )
    with startup_phase("next"):
        pass
    assert json.loads(timer_calls[-1][0].split(" ", 1)[1]) == {}


def test_concurrent_contexts_are_isolated(timer_calls: list[tuple[str, bool]]) -> None:
    async def stage(name: str) -> None:
        with startup_phase("factory.build", stage=name):
            await asyncio.sleep(0)
            with startup_capture(backend="cuda"):
                pass

    async def run() -> None:
        await asyncio.gather(stage("asr"), stage("vocoder"))

    asyncio.run(run())
    captures = [
        json.loads(name.split(" ", 1)[1])
        for name, _ in timer_calls
        if name.startswith("omni.graph.capture ")
    ]
    assert [context["stage"] for context in captures] == ["asr", "vocoder"]


def test_request_time_capture_is_not_startup(
    timer_calls: list[tuple[str, bool]]
) -> None:
    with startup_capture(backend="cuda"):
        pass
    assert timer_calls == []
    with startup_phase("factory.build", stage="codec"):
        with startup_capture(backend="cuda"):
            pass
    with startup_capture(backend="cuda"):
        pass
    assert len(timer_calls) == 2


def compiler_record(compile_id: str) -> SimpleNamespace:
    return SimpleNamespace(
        compile_id=compile_id,
        start_time_us=1234,
        is_forward=True,
        is_runtime=False,
        co_name="forward",
        co_filename="model.py",
        co_firstlineno=20,
        fail_type=None,
        dynamo_cumulative_compile_time_us=2_000_000,
        inductor_cumulative_compile_time_us=1_500_000,
        triton_compile_time_us=None,
        compile_time_autotune_time_us=0,
        runtime_triton_autotune_time_us=None,
    )


def test_compiler_records_are_native_deltas_with_missing_values(
    timer_calls: list[tuple[str, bool]],
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    records = [compiler_record("0/0")]
    monkeypatch.setitem(
        sys.modules,
        "torch._dynamo.utils",
        SimpleNamespace(get_compilation_metrics=lambda: records),
    )
    with caplog.at_level(logging.INFO):
        with startup_phase("factory.build", report_compilation=True, stage="codec"):
            records.append(compiler_record("1/0"))
        with startup_phase("factory.build", report_compilation=True):
            pass
    emitted = [
        json.loads(record.message.split(": ", 1)[1])
        for record in caplog.records
        if record.message.startswith("Startup compilation:")
    ]
    assert len(emitted) == 1
    assert emitted[0]["compile_id"] == "1/0"
    assert emitted[0]["dynamo_seconds"] == 2.0
    assert emitted[0]["inductor_seconds"] == 1.5
    assert emitted[0]["compile_autotune_seconds"] == 0.0
    assert emitted[0]["triton_compile_seconds"] is None
    assert emitted[0]["scope"] == "process_window"
    assert emitted[0]["retained_events_only"] is True
    assert emitted[0]["context"]["stage"] == "codec"


def test_bounded_buffer_does_not_claim_a_complete_total(
    timer_calls: list[tuple[str, bool]],
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    records = deque([compiler_record("0/0")], maxlen=1)
    monkeypatch.setitem(
        sys.modules,
        "torch._dynamo.utils",
        SimpleNamespace(get_compilation_metrics=lambda: records),
    )
    with caplog.at_level(logging.INFO):
        with startup_phase("factory.build", report_compilation=True):
            records.append(compiler_record("1/0"))
            records.append(compiler_record("2/0"))
    emitted = [
        json.loads(record.message.split(": ", 1)[1])
        for record in caplog.records
        if record.message.startswith("Startup compilation:")
    ]
    assert [record["compile_id"] for record in emitted] == ["2/0"]
    assert emitted[0]["retained_events_only"] is True


@pytest.mark.parametrize("body_fails", [False, True])
def test_compiler_diagnostic_failure_does_not_change_startup(
    body_fails: bool,
    timer_calls: list[tuple[str, bool]],
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    def unavailable() -> list[SimpleNamespace]:
        raise RuntimeError("compiler diagnostics unavailable")

    monkeypatch.setitem(
        sys.modules,
        "torch._dynamo.utils",
        SimpleNamespace(get_compilation_metrics=unavailable),
    )
    failure = ValueError("factory failed")
    if body_fails:
        with pytest.raises(ValueError) as caught:
            with startup_phase("factory.build", report_compilation=True):
                raise failure
        assert caught.value is failure
    else:
        with startup_phase("factory.build", report_compilation=True):
            pass
    assert "Cannot read startup compilation records" in caplog.text
    with startup_capture(backend="cuda"):
        pass
    assert len(timer_calls) == 1


def test_compiler_is_not_imported_for_diagnostics(
    timer_calls: list[tuple[str, bool]], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delitem(sys.modules, "torch._dynamo.utils", raising=False)
    with startup_phase("factory.build", report_compilation=True):
        pass
    assert "torch._dynamo.utils" not in sys.modules


def test_import_keeps_worker_bootstrap_lightweight() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import sglang_omni.utils.startup; assert 'torch' not in sys.modules; assert 'sglang' not in sys.modules",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def test_native_mlx_without_sglang_keeps_running(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unavailable(name: str) -> None:
        raise ModuleNotFoundError("No module named 'sglang'", name="sglang")

    monkeypatch.setattr("sglang_omni.utils.startup.import_module", unavailable)
    with startup_phase("factory.build", report_compilation=True):
        with startup_capture(backend="cuda"):
            pass


def test_broken_sglang_install_is_not_silently_ignored(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unavailable(name: str) -> None:
        raise ModuleNotFoundError("No module named 'dependency'", name="dependency")

    monkeypatch.setattr("sglang_omni.utils.startup.import_module", unavailable)
    with pytest.raises(ModuleNotFoundError, match="dependency"):
        with startup_phase("factory.build"):
            pytest.fail("A broken dependency must be reported")


def test_real_sglang_timer_keeps_omni_contexts_out_of_exporter(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    timers = pytest.importorskip("sglang.srt.observability.startup_func_log_and_timer")
    monkeypatch.setattr(timers, "enable_startup_metrics", True)
    monkeypatch.setattr(timers, "STARTUP_LATENCY_SECONDS", None)
    monkeypatch.setattr(timers, "_max_durations", {})
    with caplog.at_level(logging.INFO):
        with startup_phase("factory.build", stage="codec"):
            pass
    name = 'omni.factory.build {"stage": "codec"}'
    assert timers.get_max_duration(name) >= 0
    assert f"Startup timing: {name} took" in caplog.text


def test_failed_initial_snapshot_does_not_attribute_old_records(
    timer_calls: list[tuple[str, bool]],
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    attempts = 0

    def read_records() -> list[SimpleNamespace]:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("initial snapshot failed")
        else:
            return [compiler_record("old")]

    monkeypatch.setitem(
        sys.modules,
        "torch._dynamo.utils",
        SimpleNamespace(get_compilation_metrics=read_records),
    )
    with caplog.at_level(logging.INFO):
        with startup_phase("factory.build", report_compilation=True):
            pass
    assert "Startup compilation:" not in caplog.text
    assert "Cannot read startup compilation records" in caplog.text
