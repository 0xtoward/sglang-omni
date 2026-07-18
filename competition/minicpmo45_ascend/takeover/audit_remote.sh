#!/usr/bin/env bash
set -euo pipefail

REMOTE="${1:?usage: audit_remote.sh <ssh-alias>}"
[[ "${REMOTE}" =~ ^[A-Za-z0-9._-]+$ ]] || { echo "unsafe SSH alias" >&2; exit 2; }

ssh "${REMOTE}" 'bash -s' <<'REMOTE_SCRIPT'
set -euo pipefail
source /usr/local/Ascend/cann-9.0.0/set_env.sh
PY=/usr/local/python3.11.15/bin/python3

echo "== identity =="
hostname
uname -a

echo "== effective quota =="
if [[ -r /sys/fs/cgroup/cpu/cpu.cfs_quota_us ]]; then
  printf 'cpu_quota_us='; cat /sys/fs/cgroup/cpu/cpu.cfs_quota_us
  printf 'cpu_period_us='; cat /sys/fs/cgroup/cpu/cpu.cfs_period_us
  printf 'memory_limit_bytes='; cat /sys/fs/cgroup/memory/memory.limit_in_bytes
else
  printf 'cpu.max='; cat /sys/fs/cgroup/cpu.max
  printf 'memory.max='; cat /sys/fs/cgroup/memory.max
fi
ulimit -n

echo "== mounts =="
findmnt -T /workspace -o TARGET,SOURCE,FSTYPE,OPTIONS
findmnt -T /workspace/user_data -o TARGET,SOURCE,FSTYPE,OPTIONS
findmnt -T /workspace/shared_assets -o TARGET,SOURCE,FSTYPE,OPTIONS
df -hT /workspace /workspace/user_data /workspace/shared_assets

echo "== NPU =="
npu-smi info -l
npu-smi info -m
npu-smi info | sed -n '/Process id/,$p'

echo "== packages =="
"${PY}" - <<'PY'
import importlib.metadata as m
import platform
print("python", platform.python_version())
for name in ("torch", "torch-npu", "sglang", "transformers", "modelscope", "sgl-kernel-npu"):
    try:
        print(name, m.version(name))
    except m.PackageNotFoundError:
        print(name, "MISSING")
PY

echo "== image SGLang =="
git -C /sgl-workspace/sglang branch --show-current
git -C /sgl-workspace/sglang rev-parse HEAD
git -C /sgl-workspace/sglang status --short
git -C /sgl-workspace/sglang remote -v

echo "== persistent root =="
ls -la /workspace/user_data
if [[ -d /workspace/user_data/hidevlab ]]; then
  find /workspace/user_data/hidevlab -maxdepth 2 -mindepth 1 -type d -print | sort
fi
REMOTE_SCRIPT
