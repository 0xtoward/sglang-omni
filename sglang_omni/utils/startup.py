# SPDX-License-Identifier: Apache-2.0
"""Omni startup context for the SGLang timer and PyTorch compiler records."""

from __future__ import annotations

import json
import logging
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from importlib import import_module
from typing import Literal

logger = logging.getLogger(__name__)
StartupLabel = str | int | None
CompilationValue = str | int | float | bool | None
CompilationIdentity = tuple[str, int | None, bool | None, bool | None]
STARTUP_CONTEXT: ContextVar[dict[str, StartupLabel] | None] = ContextVar(
    "omni_startup_context", default=None
)


@contextmanager
def startup_phase(
    phase: str, *, report_compilation: bool = False, **labels: StartupLabel
) -> Iterator[None]:
    """Log inclusive wall time with inherited stage labels and no device sync."""
    # note (0xtoward): importing SGLang must follow worker device/cache setup.
    try:
        timers = import_module("sglang.srt.observability.startup_func_log_and_timer")
    except ModuleNotFoundError as error:
        if error.name != "sglang":
            raise
        else:
            # note (0xtoward): native MLX installs do not require SGLang.
            yield
            return

    context = {**(STARTUP_CONTEXT.get() or {}), **labels}
    name = f"omni.{phase} {json.dumps(context, sort_keys=True)}"
    token = STARTUP_CONTEXT.set(context)
    try:
        logger.info(f"Startup begin: {name}")
        before = compilation_snapshot() if report_compilation else {}
        try:
            with timers.startup_timer(name, log_only=True):
                yield
        except BaseException as error:
            logger.error(f"Startup failed: {name} error={type(error).__name__}")
            raise
        finally:
            if report_compilation and before is not None:
                for identity, record in (compilation_snapshot() or {}).items():
                    if identity in before:
                        continue
                    else:
                        payload = {
                            "phase": phase,
                            "context": context,
                            "scope": "process_window",
                            "retained_events_only": True,
                            **record,
                        }
                        logger.info(
                            f"Startup compilation: {json.dumps(payload, sort_keys=True)}"
                        )
            else:
                pass
    finally:
        STARTUP_CONTEXT.reset(token)


def compilation_snapshot() -> (
    dict[CompilationIdentity, dict[str, CompilationValue]] | None
):
    """Read retained native records without importing or configuring the compiler."""
    compiler = sys.modules.get("torch._dynamo.utils")
    if compiler is None:
        return {}
    else:
        pass
    try:
        records: dict[CompilationIdentity, dict[str, CompilationValue]] = {}
        for metric in list(compiler.get_compilation_metrics()):
            identity = (
                str(metric.compile_id),
                metric.start_time_us,
                metric.is_forward,
                metric.is_runtime,
            )
            record: dict[str, CompilationValue] = {
                "compile_id": str(metric.compile_id),
                "start_time_us": metric.start_time_us,
                "is_forward": metric.is_forward,
                "is_runtime": metric.is_runtime,
                "function": metric.co_name,
                "file": metric.co_filename,
                "line": metric.co_firstlineno,
                "fail_type": metric.fail_type,
            }
            for field, microseconds in (
                ("dynamo_seconds", metric.dynamo_cumulative_compile_time_us),
                ("inductor_seconds", metric.inductor_cumulative_compile_time_us),
                ("triton_compile_seconds", metric.triton_compile_time_us),
                ("compile_autotune_seconds", metric.compile_time_autotune_time_us),
                ("runtime_autotune_seconds", metric.runtime_triton_autotune_time_us),
            ):
                record[field] = (
                    None if microseconds is None else microseconds / 1_000_000
                )
            records[identity] = record
        return records
    except Exception:
        # note (0xtoward): optional compiler diagnostics must not abort startup.
        logger.warning("Cannot read startup compilation records", exc_info=True)
        return None


@contextmanager
def startup_capture(*, backend: Literal["cuda", "npu", "xpu"]) -> Iterator[None]:
    """Log shared graph capture only while a startup phase is active."""
    if STARTUP_CONTEXT.get() is None:
        yield
    else:
        with startup_phase("graph.capture", backend=backend):
            yield
