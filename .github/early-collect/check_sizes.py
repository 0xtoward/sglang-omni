"""Measure three/six-way collection on hosted VMs using a two-vCPU affinity mask."""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

cpu_ids = sorted(os.sched_getaffinity(0))
assert len(cpu_ids) >= 2
os.sched_setaffinity(0, cpu_ids[:2])
root = Path(os.environ["RUNNER_TEMP"]) / "evidence"
root.mkdir()
lane = int(os.environ["LANE"])


def collect(
    arm,
    part,
    total,
    path="tests/",
    expected=0,
    marker="not benchmark and not accelerator",
):
    folder = root / f"{total}-{part}-{arm}"
    folder.mkdir()
    command = [
        sys.executable,
        "-m",
        "pytest",
        path,
        "-q",
        "-m",
        marker,
        "--collect-only",
        "-p",
        "scripts.ci.utils.pytest_early_partition",
        "-p",
        "scripts.ci.utils.pytest_partition_measure",
        "--auto-partition-id",
        str(part),
        "--auto-partition-size",
        str(total),
        "--partition-model-file",
        ".github/unit_test_durations.json",
    ]
    if arm in ("fixed", "warmup"):
        command.append("--collect-other-partitions")
    else:
        pass
    if arm == "explicit":
        command = [
            sys.executable,
            ".github/early-collect/explicit_files.py",
            str(part),
            str(total),
            "-q",
            "-m",
            marker,
            "--collect-only",
            "-p",
            "scripts.ci.utils.pytest_partition_measure",
        ]
    else:
        pass
    started = time.monotonic()
    with (folder / "pytest.log").open("w") as log:
        result = subprocess.run(
            command,
            env=dict(
                os.environ, PYTEST_LAUNCHED=str(started), PARTITION_EVIDENCE=str(folder)
            ),
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    row = dict(
        arm=arm,
        part=part,
        total=total,
        wall=time.monotonic() - started,
        exit=result.returncode,
        cpus=sorted(os.sched_getaffinity(0)),
    )
    (folder / "time.json").write_text(json.dumps(row))
    print(row, flush=True)
    assert result.returncode == expected
    return json.loads((folder / "collection.json").read_text())


collect("warmup", lane, 3)
for total, part in ((3, lane), (6, lane), (6, lane + 3)):
    arms = ("fixed", "early") if part % 2 == 0 else ("early", "fixed")
    a, b = [collect(arm, part, total) for arm in arms]
    assert len(a["items"]) == len(set(a["items"]))
    assert set(a["items"]) == set(b["items"])
    explicit = collect("explicit", part, total)
    assert sorted(a["items"]) == sorted(explicit["items"])

for part in range(lane, 8, 3):
    collect("explicit", part, 8)
