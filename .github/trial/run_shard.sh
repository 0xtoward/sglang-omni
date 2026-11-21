#!/usr/bin/env bash
# Trial only: run one CPU unit-test partition and write a one-line summary.
# usage: run_shard.sh <activate-script> <partition-id> <partition-size>
set -u
activate=$1; partition=$2; size=$3
# shellcheck disable=SC1090
source "$activate"
export PYTHONPATH=$PWD CUDA_VISIBLE_DEVICES="" TMPDIR=/tmp
git config --global --add safe.directory '*' || true
start=$(date +%s)
timeout 50m python -m pytest tests/ -m "not benchmark and not accelerator" \
  --continue-on-collection-errors \
  -p scripts.ci.utils.pytest_partition \
  --auto-partition-id "$partition" --auto-partition-size "$size" \
  --partition-model-file .github/unit_test_durations.json \
  --durations=30 --junitxml=junit.xml > pytest.log 2>&1
status=$?
end=$(date +%s)
tail -15 pytest.log
python - <<'PY' > test_summary.txt
import xml.etree.ElementTree as ET
try:
    s = ET.parse("junit.xml").getroot().find("testsuite")
    print(f"tests={s.get('tests')} failures={s.get('failures')} errors={s.get('errors')} "
          f"skipped={s.get('skipped')} pytest_seconds={float(s.get('time')):.0f}")
except Exception as exc:
    print(f"junit_unreadable={exc!r}")
PY
echo "wall_seconds=$((end - start)) status=$status" >> test_summary.txt
cat test_summary.txt
