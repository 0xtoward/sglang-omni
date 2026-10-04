# SPDX-License-Identifier: Apache-2.0
"""MiniCPM-o talker sampling on per-slot device history against the host window."""

from __future__ import annotations

from types import SimpleNamespace

import torch

from sglang_omni.model_runner.base import rank_shared_unseeded_sampling_seed
from sglang_omni.models.minicpm_o.talker_model_runner import MiniCPMOTalkerModelRunner

VOCAB = 64
EOS_ID = VOCAB - 1
MIN_NEW_TOKENS = 5
SEED = 7


def make_request(index: int, penalty: float, seed: int | None) -> SimpleNamespace:
    req = SimpleNamespace(
        output_ids=[],
        sampling_params=SimpleNamespace(sampling_seed=seed),
        eos_token_ids={EOS_ID},
    )
    inputs = {"rep_penalty": penalty, "min_new_tokens": MIN_NEW_TOKENS}
    data = SimpleNamespace(req=req, talker_model_inputs=inputs, return_logprob=False)
    return SimpleNamespace(data=data, request_id=f"request-{index}")


def test_device_history_matches_host_penalty_and_min_new_tokens() -> None:
    torch.manual_seed(0)
    requests = [
        make_request(row, 1.05 if row % 2 else 1.0, SEED if row % 3 else None)
        for row in range(4)
    ]
    rows = torch.tensor([5, 0, 11, 2])
    runner = MiniCPMOTalkerModelRunner.__new__(MiniCPMOTalkerModelRunner)
    runner.tp_worker = SimpleNamespace(
        model_runner=SimpleNamespace(
            req_to_token_pool=SimpleNamespace(req_to_token=torch.zeros(12, 1)),
            sample=lambda logits_output, forward_batch: (
                logits_output.next_token_logits.argmax(dim=-1)
            ),
        )
    )
    runner.apply_codec_suppress_tokens = lambda logits_output, requests: None
    sampling_info = SimpleNamespace(
        sampling_seed=None,
        need_min_p_sampling=False,
        need_top_p_sampling=False,
        need_top_k_sampling=False,
    )
    for step in range(12):
        logits = torch.randn(len(requests), VOCAB) * 3
        logits[:, EOS_ID] += 4.0
        expected = logits.clone()
        runner.process_sampling_logits(
            SimpleNamespace(next_token_logits=expected), requests
        )
        for row, sched_req in enumerate(requests):
            if len(sched_req.data.req.output_ids) < MIN_NEW_TOKENS:
                expected[row, EOS_ID] = float("-inf")
            else:
                pass
        forward_batch = SimpleNamespace(
            req_pool_indices=rows,
            forward_mode=SimpleNamespace(is_extend=lambda first=step == 0: first),
            sampling_info=sampling_info,
        )
        next_token_ids = runner.sample_next_token_ids(
            SimpleNamespace(next_token_logits=logits), forward_batch, None, requests
        )
        torch.testing.assert_close(logits, expected, rtol=0, atol=0)
        for row, sched_req in enumerate(requests):
            sched_req.data.req.output_ids.append(int(next_token_ids[row]))
    expected_seeds = [
        SEED if row % 3 else rank_shared_unseeded_sampling_seed(requests[row], row)
        for row in range(4)
    ]
    assert sampling_info.sampling_seed.tolist() == expected_seeds
