"""Pass only owned test filenames as positional pytest arguments."""

import json
import sys
from pathlib import Path
from zlib import crc32

import pytest
from sglang.test.ci.ci_register import CIRegistry, HWBackend, auto_partition

part, total = map(int, sys.argv[1:3])
weights = json.loads(Path(".github/unit_test_durations.json").read_text())
known = [
    CIRegistry(backend=HWBackend.CPU, filename=path, est_time=weight)
    for path, weight in weights.items()
    if Path(path).is_file()
]
owned = {r.filename for r in auto_partition(known, part, total)}
# This checkout uses pytest's default python_files patterns.
candidates = sorted(
    {
        p.as_posix()
        for pat in ("test_*.py", "*_test.py")
        for p in Path("tests").rglob(pat)
    }
)
paths = [
    p
    for p in candidates
    if p in owned or (p not in weights and crc32(p.encode()) % total == part)
]
assert paths, "empty shards must never fall back to pytest's default discovery"
sys.exit(pytest.main([*sys.argv[3:], *paths]))
