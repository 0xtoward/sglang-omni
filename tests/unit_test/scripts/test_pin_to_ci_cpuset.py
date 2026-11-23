# SPDX-License-Identifier: Apache-2.0
import subprocess
from pathlib import Path

import pytest

PIN_SCRIPT = Path(__file__).resolve().parents[3] / ".github/scripts/pin_to_ci_cpuset.sh"


@pytest.mark.parametrize(
    ("requested", "effective", "status"),
    [
        ("24-25,88-89", "24,25,88,89", 0),
        ("88-89,24,24-25", "24,25,88,89", 0),
        ("16-31,80-95", "16-31,80-95", 0),
        ("24-28:2,88", "24,26,28,88", 0),
        ("24-25,88-89", "24,25,88", 2),
    ],
)
def test_pin_readback(requested, effective, status):
    result = subprocess.run(
        [
            "bash",
            "-euc",
            r"""
            OMNI_CI_CPUSET=$2
            readback=$3
            taskset() {
                if [[ $# == 3 ]]; then
                    [[ $1 == -pc && $2 == "$OMNI_CI_CPUSET" && $3 == "$$" ]]
                else
                    [[ $# == 2 && $1 == -pc && $2 == "$$" ]] || return 1
                    printf 'pid %s current affinity list: %s\n' "$$" "$readback"
                fi
            }
            source "$1"
            """,
            "bash",
            str(PIN_SCRIPT),
            requested,
            effective,
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == status, result.stdout + result.stderr
    if status:
        assert f"pinned mask '{effective}'" in result.stdout
        assert f"requested OMNI_CI_CPUSET='{requested}'" in result.stdout
