"""CPU contracts for the isolated buffered-C1 penalty owner (no CUDA model)."""

import hashlib
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

import pytest


@pytest.fixture
def owner_module():
    path = Path(__file__).resolve().parents[3] / "experiments/cosy_async_c1/owner.py"
    spec = importlib.util.spec_from_file_location("cosy_owner_under_test", path)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {spec.name: module}):
        spec.loader.exec_module(module)
        yield module


@pytest.fixture
def state(owner_module):
    owner = owner_module.Owner(None, NS(device="cuda:0"), True)
    req = NS(
        sampling_params=NS(
            repetition_penalty=1.21,
            frequency_penalty=0,
            presence_penalty=0,
            min_new_tokens=2,
            sampling_seed=42,
        ),
        custom_logit_processor=None,
        return_logprob=False,
        is_retracted=False,
        _cosy_async_owner=owner,
        _cosy_async_epoch=1,
    )
    updates = []
    penalizers = {
        name: type(name, (), {"is_prepared": lambda self: True})()
        for name in ("BatchedMinNewTokensPenalizer", "BatchedRepetitionPenalizer")
    }
    orchestrator = NS(
        is_required=True, penalizers=penalizers, cumulate_output_tokens=updates.append
    )
    batch = NS(
        reqs=[req],
        forward_mode=NS(is_decode=lambda: True),
        sampling_info=NS(penalizer_orchestrator=orchestrator),
        input_ids=NS(ndim=1, shape=(1,), device="cuda:0"),
    )
    return NS(owner=owner, req=req, batch=batch, updates=updates)


def test_reservation_does_not_update_until_resolved_commit(state):
    state.owner.reserve(state.batch)
    assert state.updates == []
    state.owner.commit_resolved(state.batch)
    assert state.updates == [state.batch.input_ids]
    assert state.owner.counts["reserved"] == state.owner.counts["committed"] == 1


@pytest.mark.parametrize("duplicate", ["reserve", "commit"])
def test_duplicate_update_fails_closed(state, duplicate):
    state.owner.reserve(state.batch)
    if duplicate == "commit":
        state.owner.commit_resolved(state.batch)
    with pytest.raises(RuntimeError, match="COSY_ASYNC_OWNER_INVALID"):
        getattr(
            state.owner, "reserve" if duplicate == "reserve" else "commit_resolved"
        )(state.batch)
    assert len(state.updates) == (duplicate == "commit")


@pytest.mark.parametrize("changed", ["epoch", "row", "orchestrator", "device", "shape"])
def test_changed_ownership_rejected_before_update(state, changed):
    state.owner.reserve(state.batch)
    if changed == "epoch":
        state.req._cosy_async_epoch += 1
    elif changed == "row":
        state.batch.reqs = [NS()]
    elif changed == "orchestrator":
        state.batch.sampling_info.penalizer_orchestrator = NS()
    elif changed == "device":
        state.batch.input_ids.device = "cuda:1"
    else:
        state.batch.input_ids.shape = (2,)
    with pytest.raises(RuntimeError, match="COSY_ASYNC_OWNER_INVALID"):
        state.owner.commit_resolved(state.batch)
    assert state.updates == []


def test_partial_update_failure_cannot_be_retried(state):
    def fail_after_update(ids):
        state.updates.append(ids)
        raise ValueError("partial submission")

    state.batch.sampling_info.penalizer_orchestrator.cumulate_output_tokens = (
        fail_after_update
    )
    state.owner.reserve(state.batch)
    with pytest.raises(ValueError, match="partial submission"):
        state.owner.commit_resolved(state.batch)
    with pytest.raises(RuntimeError, match="COSY_ASYNC_OWNER_INVALID"):
        state.owner.commit_resolved(state.batch)
    assert len(state.updates) == 1


def test_invalidated_epoch_cannot_commit(state):
    state.owner.reserve(state.batch)
    state.owner.invalidate("abort observed")
    with pytest.raises(RuntimeError, match="abort observed"):
        state.owner.commit_resolved(state.batch)
    assert state.updates == []


def test_unknown_runtime_source_is_not_silently_accepted(owner_module):
    path = Path(owner_module.__file__)
    owner_module._assert_source(
        owner_module.Owner, hashlib.sha256(path.read_bytes()).hexdigest()
    )
    with pytest.raises(RuntimeError, match="unreviewed source"):
        owner_module._assert_source(owner_module.Owner, "0" * 64)
