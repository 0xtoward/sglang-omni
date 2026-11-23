# SPDX-License-Identifier: Apache-2.0
"""Record exact item IDs and collection timing for a fork experiment."""

import json
import os
import time
from pathlib import Path

import pytest

OUT = Path(os.environ["PARTITION_EVIDENCE"])
reports = []
started = []
failed = []


def pytest_collectreport(report: pytest.CollectReport) -> None:
    reports.append([report.nodeid, report.outcome])


def pytest_collection_finish(session: pytest.Session) -> None:
    elapsed = time.monotonic() - float(os.environ["PYTEST_LAUNCHED"])
    (OUT / "collection.json").write_text(
        json.dumps(
            {
                "seconds": elapsed,
                "items": [item.nodeid for item in session.items],
                "collectors": reports,
            }
        )
    )


def pytest_runtest_logstart(nodeid: str) -> None:
    started.append(nodeid)


def pytest_runtest_logreport(report: pytest.TestReport) -> None:
    if report.failed:
        failed.append([report.nodeid, report.when])
    else:
        pass


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    (OUT / "execution.json").write_text(
        json.dumps({"started": started, "failures": failed, "exit": int(exitstatus)})
    )
