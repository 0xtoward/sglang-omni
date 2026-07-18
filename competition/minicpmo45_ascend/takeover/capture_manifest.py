#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import subprocess
import time
from pathlib import Path


def run(*args: str) -> dict[str, object]:
    proc = subprocess.run(args, text=True, capture_output=True, check=False)
    return {
        "argv": list(args),
        "returncode": proc.returncode,
        "stdout": proc.stdout.rstrip(),
        "stderr": proc.stderr.rstrip(),
    }


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def read_first(*paths: str) -> str | None:
    for value in paths:
        path = Path(value)
        if path.is_file():
            return path.read_text().strip()
    return None


def git_snapshot(path: Path) -> dict[str, object]:
    return {
        "path": str(path),
        "head": run("git", "-C", str(path), "rev-parse", "HEAD"),
        "branch": run("git", "-C", str(path), "branch", "--show-current"),
        "status": run("git", "-C", str(path), "status", "--short"),
        "remotes": run("git", "-C", str(path), "remote", "-v"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--root", default="/workspace/user_data/hidevlab")
    args = parser.parse_args()
    root = Path(args.root)

    torch_info: dict[str, object]
    try:
        import torch
        import torch_npu

        torch_info = {
            "torch": torch.__version__,
            "torch_npu": torch_npu.__version__,
            "device_count": torch.npu.device_count(),
            "devices": [repr(torch.npu.get_device_properties(i)) for i in range(torch.npu.device_count())],
        }
    except Exception as exc:
        torch_info = {"error": f"{type(exc).__name__}: {exc}"}

    manifest = {
        "schema_version": 1,
        "captured_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "uid": os.getuid(),
        "packages": {
            name: package_version(name)
            for name in ("torch", "torch-npu", "sglang", "transformers", "modelscope", "sgl-kernel-npu")
        },
        "torch_npu": torch_info,
        "cgroup": {
            "cpu_quota_us": read_first("/sys/fs/cgroup/cpu/cpu.cfs_quota_us"),
            "cpu_period_us": read_first("/sys/fs/cgroup/cpu/cpu.cfs_period_us"),
            "memory_limit_bytes": read_first("/sys/fs/cgroup/memory/memory.limit_in_bytes"),
            "cpu_max_v2": read_first("/sys/fs/cgroup/cpu.max"),
            "memory_max_v2": read_first("/sys/fs/cgroup/memory.max"),
        },
        "mounts": {
            path: run("findmnt", "-T", path, "-o", "TARGET,SOURCE,FSTYPE,OPTIONS")
            for path in ("/workspace", "/workspace/user_data", "/workspace/shared_assets")
        },
        "npu_smi": {
            "list": run("npu-smi", "info", "-l"),
            "map": run("npu-smi", "info", "-m"),
            "info": run("npu-smi", "info"),
        },
        "repositories": {
            "image_sglang": git_snapshot(Path("/sgl-workspace/sglang")),
            "competition_core": git_snapshot(root / "repos/sglang"),
            "competition_omni": git_snapshot(root / "repos/sglang-omni"),
        },
        "recovery_artifacts": {
            "bundle_sha256": run("sha256sum", str(root / "artifacts" / "sglang-omni-minicpmo45-ascend.bundle"), str(root / "artifacts" / "sglang-minicpmo45-ascend-core.bundle")),
        },
        "environment_whitelist": {
            key: os.environ.get(key)
            for key in ("ASCEND_HOME_PATH", "ASCEND_OPP_PATH", "LD_LIBRARY_PATH", "PATH")
        },
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(output.suffix + ".tmp")
    tmp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(output)
    print(output)


if __name__ == "__main__":
    main()
