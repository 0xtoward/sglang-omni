# SPDX-License-Identifier: Apache-2.0
"""Avoid importing test modules assigned to another partition."""

import json
from pathlib import Path
from zlib import crc32

import pytest
from sglang.test.ci.ci_register import CIRegistry, HWBackend, auto_partition

from scripts.ci.utils import pytest_partition

KNOWN_FILES = pytest.StashKey[set[str]]()
ASSIGNED_FILES = pytest.StashKey[set[str]]()


def pytest_addoption(parser: pytest.Parser) -> None:
    pytest_partition.pytest_addoption(parser)
    parser.addoption("--collect-other-partitions", action="store_true")


def pytest_configure(config: pytest.Config) -> None:
    pytest_partition.pytest_configure(config)
    weights = json.loads(config.getoption("--partition-model-file").read_text())
    files = [
        CIRegistry(backend=HWBackend.CPU, filename=name, est_time=seconds)
        for name, seconds in weights.items()
        if (config.rootpath / name).is_file()
    ]
    config.stash[KNOWN_FILES] = {f.filename for f in files}
    config.stash[ASSIGNED_FILES] = {
        f.filename
        for f in auto_partition(
            files,
            config.getoption("--auto-partition-id"),
            config.getoption("--auto-partition-size"),
        )
    }


def owns_file(config: pytest.Config, path: Path) -> bool:
    name = path.relative_to(config.rootpath).as_posix()
    if name in config.stash[KNOWN_FILES]:
        return name in config.stash[ASSIGNED_FILES]
    else:
        return crc32(name.encode()) % config.getoption(
            "--auto-partition-size"
        ) == config.getoption("--auto-partition-id")


def pytest_ignore_collect(collection_path: Path, config: pytest.Config) -> bool | None:
    if (
        not config.getoption("--collect-other-partitions")
        and collection_path.is_file()
        and not owns_file(config, collection_path)
    ):
        return True
    else:
        return None


@pytest.hookimpl(trylast=True)
def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    keep = []
    drop = []
    for item in items:
        if owns_file(config, item.path):
            keep.append(item)
        else:
            drop.append(item)
    config.hook.pytest_deselected(items=drop)
    items[:] = keep
