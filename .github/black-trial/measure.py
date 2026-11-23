import json
import os
import subprocess
import tempfile
import time
from pathlib import Path

root = Path(os.environ["RUNNER_TEMP"])
base = subprocess.check_output(["git", "rev-parse", "HEAD^1"], text=True).strip()
assert base == os.environ["EXPECTED_BASE"]
assert os.environ["CERTIFIED"] == "true"
assert subprocess.check_output(["git", "status", "--porcelain", "-uno"]) == b""
rows = []
for arm in os.environ["ORDER"].split("-"):
    if arm == "full":
        command = ["pre-commit", "run", "--all-files", "--show-diff-on-failure"]
    else:
        command = ["bash", str(root / "black-trial/lint_with_scoped_black.sh"), base]
    start = previous = time.monotonic()
    environment = dict(
        os.environ, BLACK_CACHE_DIR=tempfile.mkdtemp(), PYTHONUNBUFFERED="1"
    )
    proc = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=environment,
    )
    hook_seconds = None
    for line in proc.stdout:
        print(line, end="", flush=True)
        now = time.monotonic()
        if (
            line.startswith("black-jupyter")
            and "(SKIP=" not in line
            and "Skipped" not in line
        ):
            hook_seconds = now - previous
        previous = now
    assert proc.wait() == 0
    assert subprocess.check_output(["git", "diff", "--name-only"]) == b""
    rows.append(dict(arm=arm, wall=time.monotonic() - start, black=hook_seconds))
result = dict(pr=os.environ["PR"], order=os.environ["ORDER"], base=base, rows=rows)
print(result)
(root / "timings.json").write_text(json.dumps(result))
