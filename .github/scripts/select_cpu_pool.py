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


def select_cpu_pool(run_id: int, read: Callable[[str], dict]) -> tuple[str, str]:
    try:
        jobs = read(f"runs/{run_id}/jobs?per_page=100")["jobs"]
        setup_started = any(
            job["name"] == "setup - omni venv"
            and {"self-hosted", "h100"}.issubset(job["labels"])
            and job.get("runner_id")
            and (job["status"] == "in_progress" or job["conclusion"] == "success")
            for job in jobs
        )
        if not setup_started:
            return "hosted", "H100 setup has not acquired a runner"

        # note (0xtoward): A peer can still be in preflight before its GPU jobs appear.
        for workflow in ("omni-ci.yaml", "cleanup-pr-ci-home-on-close.yaml"):
            for status in ("queued", "in_progress", "pending"):
                runs = read(f"workflows/{workflow}/runs?status={status}&per_page=2")
                for run in runs["workflow_runs"]:
                    if run["id"] != run_id:
                        return "hosted", f"{workflow} run {run['id']} is {status}"
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
        "H100 setup has a runner and no competing H100 workflow is active",
    )


if __name__ == "__main__":
    pool, reason = select_cpu_pool(int(os.environ["GITHUB_RUN_ID"]), github_actions)
    print(f"CPU unit-test pool: {pool}. {reason}.")
    with Path(os.environ["GITHUB_OUTPUT"]).open("a") as output:
        print(f"pool={pool}", file=output)
