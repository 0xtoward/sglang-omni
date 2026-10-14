# SPDX-License-Identifier: Apache-2.0
"""test only (xp): quiet / knorm / keydiff thinker windows make multi-range cuts that keep ids, spans and KV consistent."""

import pytest
import torch

from sglang_omni.models.minicpm_o import session_adapters
from sglang_omni.models.minicpm_o.native_config import MiniCPMODuplexWindow
from tests.unit_test.minicpm_o.test_native_vision_relay import (  # noqa: F401
    CUT_UNITS,
    IDENTITY,
    append_window_unit,
    expected_session_layout,
    finish_window_units,
    observed_session_layout,
    open_window_session,
    perception,
    read_session_kv,
    relative_error,
    rotate_keys_exactly,
    unit_token_ids,
)

UNITS = [CUT_UNITS[index % len(CUT_UNITS)] for index in range(8)]


@pytest.mark.parametrize(
    "policy", ["basic", "quiet", "user", "knorm", "keydiff", "quietkd", "quietkn"]
)
def test_xp_policy_cut_keeps_ids_spans_and_kv(perception, monkeypatch, policy) -> None:
    monkeypatch.setattr(session_adapters, "XP_WINDOW_POLICY", policy)
    monkeypatch.setattr(session_adapters, "XP_PROTECT_TOKENS", 0)
    scheduler, bridge, adapter = open_window_session(
        perception, MiniCPMODuplexWindow(high_tokens=900, low_tokens=800)
    )
    finish_window_units(perception, scheduler, bridge, adapter, UNITS[:7])
    state = adapter.states[IDENTITY]
    state.unit_quiet = [False, True, False, True, False, True, False]
    state.unit_kinds = [2, 0, 1, 0, 2, 1, 2]
    starts = list(state.unit_starts)
    lengths = [len(unit_token_ids(unit)) for unit in UNITS[:7]]
    next_unit_start = starts[6] + lengths[6]
    adapter.window = MiniCPMODuplexWindow(
        high_tokens=next_unit_start - 1,
        low_tokens=next_unit_start - lengths[1] - lengths[3],
    )
    kv_end = scheduler.tree_cache.slots[IDENTITY.id].kv.kv_committed_len
    keys_before, values_before = read_session_kv(scheduler, kv_end)
    recorded = []
    original = type(state).drop_units

    def recording(self, *args, **kwargs):
        ranges = original(self, *args, **kwargs)
        recorded.append(ranges)
        return ranges

    monkeypatch.setattr(type(state), "drop_units", recording)
    request_data = append_window_unit(perception, bridge, adapter, 7, UNITS[7])
    [ranges] = recorded
    dropped = {
        index
        for index in range(7)
        if any(start <= starts[index] < end for start, end in ranges)
    }
    if policy in ("quiet", "user", "quietkd", "quietkn"):
        assert dropped == {1, 3} and len(ranges) == 2, (dropped, ranges)
    elif policy == "basic":
        assert dropped == {0, 1, 2} or dropped == {0, 1}, dropped
    else:
        assert dropped and 6 not in dropped, dropped
    kept = [(index, UNITS[index]) for index in range(7) if index not in dropped]
    assert observed_session_layout(request_data) == expected_session_layout(
        kept, 7, UNITS[7]
    ), "ids and spans are the kept units"
    shift = 0
    total = sum(end - start for start, end in ranges)
    keys_after, values_after = read_session_kv(scheduler, kv_end - total)
    for position in range(ranges[0][0], kv_end):
        if any(start <= position < end for start, end in ranges):
            shift += 1
            continue
        moved = position - shift
        assert torch.equal(values_after[:, moved], values_before[:, position])
        exact = rotate_keys_exactly(keys_before[:, position], shift)
        assert (
            relative_error(keys_after[:, moved], exact)
            <= 2 * relative_error(exact.to(torch.bfloat16), exact) + 1e-12
        ), (policy, position, shift)
