# SPDX-License-Identifier: Apache-2.0
"""Select the CPU unit-test pool using this repository's Actions queue."""

import json
import os
import subprocess
from collections.abc import Callable
from pathlib import Path


def github_actions(path: str) -> dict:
    endpoint = f"repos/{os.environ['GITHUB_REPOSITORY']}/actions/{path}"
    return json.loads(subprocess.check_output(["gh", "api", endpoint], timeout=8))


def action_rows(read: Callable[[str], dict], path: str, key: str) -> list[dict]:
    page = read(f"{path}{'&' if '?' in path else '?'}per_page=100")
    rows = page[key]
    if page["total_count"] > len(rows):
        raise ValueError("incomplete queue snapshot")
    return rows


def h100_jobs(read: Callable[[str], dict], run_id: int) -> list[dict]:
    return [
        job
        for job in action_rows(read, f"runs/{run_id}/jobs", "jobs")
        if {"self-hosted", "h100"}.issubset(job["labels"])
    ]


def select_cpu_pool(run_id: int, read: Callable[[str], dict]) -> tuple[str, str]:
    try:
        setup_started = any(
            job["name"] == "setup - omni venv"
            and job.get("runner_id")
            and (job["status"] == "in_progress" or job["conclusion"] == "success")
            for job in h100_jobs(read, run_id)
        )
        if not setup_started:
            return "hosted", "H100 setup has not acquired a runner"

        seen = {run_id}
        for workflow in ("omni-ci.yaml", "cleanup-pr-ci-home-on-close.yaml"):
            for status in ("in_progress", "queued", "pending"):
                runs = action_rows(
                    read, f"workflows/{workflow}/runs?status={status}", "workflow_runs"
                )
                for run in runs:
                    if run["id"] in seen:
                        continue
                    seen.add(run["id"])
                    for job in h100_jobs(read, run["id"]):
                        if job["status"] != "completed":
                            return (
                                "hosted",
                                f"H100 job {job['name']} in run {run['id']} is {job['status']}",
                            )
    except (
        OSError,
        subprocess.SubprocessError,
        ValueError,
        KeyError,
        TypeError,
    ) as exc:
        return "hosted", f"could not read the H100 queue ({type(exc).__name__})"
    return (
        "self-hosted",
        "H100 setup has a runner and no competing H100 job is waiting or running",
    )


if __name__ == "__main__":
    pool, reason = select_cpu_pool(int(os.environ["GITHUB_RUN_ID"]), github_actions)
    print(f"CPU unit-test pool: {pool}. {reason}.")
    with Path(os.environ["GITHUB_OUTPUT"]).open("a") as output:
        print(f"pool={pool}", file=output)
