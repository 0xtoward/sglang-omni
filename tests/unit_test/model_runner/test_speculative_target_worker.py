# SPDX-License-Identifier: Apache-2.0
"""Target adapter contracts shared by speculative prefill and verification."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from sglang.srt.managers.utils import GenerationBatchResult
from sglang.srt.model_executor.forward_batch_info import ForwardBatch

from sglang_omni.model_runner.speculative_target_worker import SpeculativeTargetWorker


@pytest.mark.parametrize("return_kv_loc_plan", [False, True])
@pytest.mark.parametrize("is_prefill_only", [False, True])
def test_prefill_returns_the_target_kv_plan_without_changing_sampled_tokens(
    monkeypatch: pytest.MonkeyPatch,
    return_kv_loc_plan: bool,
    is_prefill_only: bool,
) -> None:
    """Native drafts failed on their first prefill when requesting the KV plan."""
    kv_location_plan = SimpleNamespace()
    forward_batch = SimpleNamespace(
        kv_loc_plan=kv_location_plan,
        seq_lens=torch.tensor([5, 7]),
        input_ids=torch.tensor([1, 2]),
    )
    monkeypatch.setattr(ForwardBatch, "init_new", Mock(return_value=forward_batch))
    model_worker = SimpleNamespace(
        model_runner=None, model_config=None, device="cpu", random_seed=0
    )
    target_worker = SpeculativeTargetWorker(model_worker)
    target_worker.omni_runner = SimpleNamespace(
        retain_prefill_input_embeds=False,
        prepare_and_forward=Mock(
            return_value=GenerationBatchResult(next_token_ids=torch.tensor([7, 8]))
        ),
    )
    batch = SimpleNamespace(
        reqs=[
            SimpleNamespace(rid="request-a", omni_data=None),
            SimpleNamespace(rid="request-b", omni_data=None),
        ],
        is_prefill_only=is_prefill_only,
    )

    result = target_worker.forward_batch_generation(
        batch, return_kv_loc_plan=return_kv_loc_plan
    )

    assert result.kv_loc_plan is (kv_location_plan if return_kv_loc_plan else None)
    assert result.next_token_ids.tolist() == ([0, 0] if is_prefill_only else [7, 8])


@pytest.mark.parametrize("return_kv_loc_plan", [False, True])
def test_verify_returns_its_own_target_kv_plan(return_kv_loc_plan: bool) -> None:
    """Verification must relay the current allocation plan without rebuilding it."""
    kv_location_plan = SimpleNamespace()
    forward_batch = SimpleNamespace(
        kv_loc_plan=kv_location_plan,
        apply_deprecated_skip_attn_backend_init=Mock(),
    )
    output = SimpleNamespace(
        logits_output=SimpleNamespace(),
        can_run_graph=True,
        expert_distribution_metrics=None,
        routed_experts_output=None,
        indexer_topk_output=None,
    )
    model_worker = SimpleNamespace(
        model_runner=SimpleNamespace(forward=Mock(return_value=output)),
        model_config=None,
        device="cpu",
        random_seed=0,
    )
    target_worker = SpeculativeTargetWorker(model_worker)

    result = target_worker.forward_batch_generation(
        None,
        forward_batch=forward_batch,
        is_verify=True,
        return_kv_loc_plan=return_kv_loc_plan,
    )

    assert result.kv_loc_plan is (kv_location_plan if return_kv_loc_plan else None)
    assert result.logits_output is output.logits_output
    assert result.can_run_cuda_graph
