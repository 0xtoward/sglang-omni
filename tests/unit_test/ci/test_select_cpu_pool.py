# SPDX-License-Identifier: Apache-2.0
"""Route CPU unit tests without reserving an H100 needed by another workflow."""

import importlib
import subprocess
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock

import pytest


@pytest.fixture
def router(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    monkeypatch.syspath_prepend(
        str(Path(__file__).resolve().parents[3] / ".github/scripts")
    )
    return importlib.import_module("select_cpu_pool")


def setup_job(status: str, runner: int | None) -> dict:
    return {
        "name": "setup - omni venv",
        "labels": ["self-hosted", "h100"],
        "status": status,
        "conclusion": "success" if status == "completed" else None,
        "runner_id": runner,
    }


@pytest.mark.parametrize("status", ["in_progress", "completed"])
def test_uncontended_h100_setup_selects_the_self_hosted_cpu_job(
    router: ModuleType, status: str
) -> None:
    def read(path: str) -> dict:
        if path.startswith("runs/"):
            return {"jobs": [setup_job(status, 17)]}
        else:
            return {"workflow_runs": [{"id": 100}]}

    pool, _ = router.select_cpu_pool(100, read)
    assert pool == "self-hosted"


@pytest.mark.parametrize(
    "jobs", [[], [setup_job("queued", None)], [setup_job("in_progress", 0)]]
)
def test_unassigned_setup_sends_cpu_work_to_hosted_runners(
    router: ModuleType, jobs: list
) -> None:
    read = Mock(return_value={"jobs": jobs})
    pool, _ = router.select_cpu_pool(100, read)
    assert pool == "hosted"
    read.assert_called_once()


@pytest.mark.parametrize(
    "workflow", ["omni-ci.yaml", "cleanup-pr-ci-home-on-close.yaml"]
)
@pytest.mark.parametrize("status", ["queued", "in_progress", "pending"])
def test_competing_h100_work_sends_cpu_work_to_hosted_runners(
    router: ModuleType, workflow: str, status: str
) -> None:
    responses = {
        "runs/100/jobs?per_page=100": {"jobs": [setup_job("in_progress", 17)]},
        f"workflows/{workflow}/runs?status={status}&per_page=2": {
            "workflow_runs": [{"id": 99}]
        },
    }
    pool, reason = router.select_cpu_pool(
        100, lambda path: responses.get(path, {"workflow_runs": []})
    )
    assert pool == "hosted"
    assert "99" in reason


def test_unavailable_queue_defaults_to_hosted_runners(router: ModuleType) -> None:
    read = Mock(side_effect=subprocess.TimeoutExpired("gh", 8))
    pool, _ = router.select_cpu_pool(100, read)
    assert pool == "hosted"
