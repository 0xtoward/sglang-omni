import json
import os
import subprocess
import sys
import time
from pathlib import Path

part = os.environ["PART"]
order = os.environ["ORDER"]
root = Path(os.environ["RUNNER_TEMP"]) / "evidence"
root.mkdir()
probe = Path("tests/unit_test/ci/test_unweighted_partition_probe.py")
assert not probe.exists()
assert str(probe) not in json.loads(
    Path(".github/unit_test_durations.json").read_text()
)
probe.write_text(
    """import pytest

@pytest.mark.parametrize("value", [1, 2])
def test_unweighted_cpu(value):
    assert value in (1, 2)

@pytest.mark.accelerator
def test_unweighted_accelerator():
    pytest.fail("accelerator marker was ignored")
"""
)
# Each VM runs two complete suites in reverse order on its paired VM.
full_arms = order.split("-")
explicit_trial = "explicit" in full_arms
collect_arms = ["fixed", "early"] if order == "late-early" else ["early", "fixed"]
verdicts = []
for mode, arm in [("full", a) for a in full_arms] + [
    ("collect", a) for a in collect_arms if not explicit_trial
]:
    out = root / f"{mode}-{arm}"
    out.mkdir()
    plugin = "pytest_partition" if arm == "late" else "pytest_early_partition"
    args = [
        sys.executable,
        "-m",
        "pytest",
        "tests/",
        "-q",
        "-m",
        "not benchmark and not accelerator",
        "-p",
        "scripts.ci.utils." + plugin,
        "-p",
        "scripts.ci.utils.pytest_partition_measure",
        "--auto-partition-id",
        part,
        "--auto-partition-size",
        "8",
        "--partition-model-file",
        ".github/unit_test_durations.json",
        "-o",
        f"cache_dir={out}/cache",
    ]
    if mode == "collect":
        args.append("--collect-only")
    else:
        args.append(f"--junitxml={out}/junit.xml")
    if arm == "fixed":
        args.append("--collect-other-partitions")
    else:
        pass
    if arm == "explicit":
        args = [
            sys.executable,
            ".github/early-collect/explicit_files.py",
            part,
            "8",
            "-q",
            "-m",
            "not benchmark and not accelerator",
            "-p",
            "scripts.ci.utils.pytest_partition_measure",
            "-o",
            f"cache_dir={out}/cache",
            f"--junitxml={out}/junit.xml",
        ]
    else:
        pass
    start = time.monotonic()
    with (out / "pytest.log").open("w") as log:
        result = subprocess.run(
            args,
            env=dict(
                os.environ, PARTITION_EVIDENCE=str(out), PYTEST_LAUNCHED=str(start)
            ),
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    elapsed = time.monotonic() - start
    row = dict(
        arm=arm,
        mode=mode,
        part=part,
        order=order,
        seconds=elapsed,
        exit=result.returncode,
    )
    print(row, flush=True)
    (out / "time.json").write_text(json.dumps(row))
    verdicts.append(result.returncode)
sys.exit(int(any(verdicts)))
