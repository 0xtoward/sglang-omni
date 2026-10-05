# SPDX-License-Identifier: Apache-2.0
"""MiniCPM-o talker sampling on per-slot device history against the host window."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from sglang_omni.model_runner.base import rank_shared_unseeded_sampling_seed
from sglang_omni.models.minicpm_o.talker_model_runner import (
    MiniCPMOTalkerModelRunner,
    TalkerSampleGraphs,
    TalkerSlotState,
)

VOCAB = 64
EOS_ID = VOCAB - 1
MIN_NEW_TOKENS = 5
SEED = 7
SUPPRESSED = (1, 3)


def make_request(index: int, penalty: float, seed: int | None) -> SimpleNamespace:
    req = SimpleNamespace(
        output_ids=[],
        sampling_params=SimpleNamespace(sampling_seed=seed),
        eos_token_ids={EOS_ID},
    )
    inputs = {"rep_penalty": penalty, "min_new_tokens": MIN_NEW_TOKENS}
    data = SimpleNamespace(req=req, talker_model_inputs=inputs, return_logprob=False)
    return SimpleNamespace(data=data, request_id=f"request-{index}")


def suppress_one_token(logits_output: SimpleNamespace, requests: list) -> None:
    logits_output.next_token_logits[SUPPRESSED] = float("-inf")


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
    runner.apply_codec_suppress_tokens = suppress_one_token
    runner.samples_in_graph = lambda sampling_info: False
    sampling_info = SimpleNamespace(
        sampling_seed=None,
        need_min_p_sampling=False,
        need_top_p_sampling=False,
        need_top_k_sampling=False,
        temperatures=torch.ones(len(requests), 1),
        top_ps=torch.ones(len(requests)),
        top_ks=torch.full((len(requests),), VOCAB),
        min_ps=torch.zeros(len(requests)),
    )
    for step in range(12):
        logits = torch.randn(len(requests), VOCAB) * 3
        logits[:, EOS_ID] += 4.0
        expected = logits.clone()
        runner.process_sampling_logits(
            SimpleNamespace(next_token_logits=expected), requests
        )
        expected[SUPPRESSED] = float("-inf")
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


def test_reset_keeps_the_seeds_of_deterministic_inference() -> None:
    state = TalkerSlotState.allocate(12, VOCAB, EOS_ID, torch.device("cpu"))
    requests = [make_request(row, 1.0, None) for row in range(2)]
    rows = torch.tensor([3, 8])
    # SGLang seeds unseeded rows with 42 under deterministic inference.
    sampling_info = SimpleNamespace(
        sampling_seed=torch.tensor([42, 42]),
        temperatures=torch.ones(len(requests), 1),
        top_ps=torch.ones(len(requests)),
        top_ks=torch.full((len(requests),), VOCAB),
        min_ps=torch.zeros(len(requests)),
    )
    state.reset(
        rows,
        requests,
        sampling_info,
        torch.zeros(len(requests), VOCAB, dtype=torch.bool),
    )
    assert state.seeds[rows].tolist() == [42, 42]


def test_slot_state_follows_wrap_reuse_and_replay() -> None:
    """A slot's window wraps past 16 tokens, is cleared for its next request, and is
    rebuilt from the replayed history after a retract."""
    state = TalkerSlotState.allocate(12, VOCAB, EOS_ID, torch.device("cpu"))
    rows = torch.tensor([7])
    sampling_info = SimpleNamespace(
        sampling_seed=None,
        temperatures=torch.ones(1, 1),
        top_ps=torch.ones(1),
        top_ks=torch.full((1,), VOCAB),
        min_ps=torch.zeros(1),
    )
    suppress = torch.zeros(1, VOCAB, dtype=torch.bool)
    first = make_request(0, 1.05, None)
    state.reset(rows, [first], sampling_info, suppress)
    tokens = [(step * 7) % (VOCAB - 1) for step in range(20)]
    for token in tokens:
        state.append(rows, torch.tensor([token]))
    assert int(state.generated[7]) == 20
    assert sorted(state.windows[7].tolist()) == sorted(tokens[-16:])

    second = make_request(1, 1.0, None)
    state.reset(rows, [second], sampling_info, suppress)
    assert int(state.generated[7]) == 0
    assert state.windows[7].tolist() == [VOCAB] * 16

    first.data.req.output_ids = list(tokens)
    state.reset(rows, [first], sampling_info, suppress)
    assert int(state.generated[7]) == 20
    # Token i sits at ring position i % 16: the four oldest slots hold tokens 16..19.
    assert state.windows[7].tolist() == [
        tokens[slot + 16 if slot < 4 else slot] for slot in range(16)
    ]
    logits = torch.ones(1, VOCAB)
    state.apply(logits, rows)
    penalized = torch.zeros(VOCAB, dtype=torch.bool)
    penalized[tokens[4:]] = True
    assert torch.equal(logits[0] < 1.0, penalized)
    assert logits[0, EOS_ID] == 1.0


@pytest.mark.accelerator
def test_sample_graph_replays_the_eager_step() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA graphs need a CUDA device")
    else:
        pass
    torch.manual_seed(0)
    device = torch.device("cuda")

    def seeded_sampler(logits_output, sampling_info, *args) -> torch.Tensor:
        scores = logits_output.next_token_logits / sampling_info.temperatures
        noise = (sampling_info.sampling_seed % 5).unsqueeze(1).to(scores.dtype)
        return (scores + noise).argmax(dim=-1).to(torch.int32)

    states = [TalkerSlotState.allocate(12, VOCAB, EOS_ID, device) for _ in range(2)]
    windows = torch.randint(0, VOCAB, states[0].windows.shape)
    for state in states:
        state.windows.copy_(windows)
        state.generated.copy_(torch.arange(state.generated.shape[0]))
        state.min_new_tokens.fill_(MIN_NEW_TOKENS)
        state.penalties.fill_(1.05)
        state.seeds.copy_(torch.arange(state.seeds.shape[0]) * 13)
        state.temperatures.fill_(0.8)
        state.suppress[:, 3] = True
    graphs = TalkerSampleGraphs(states[0], seeded_sampler)
    eager = TalkerSampleGraphs(states[1], seeded_sampler)
    for rows in ([5, 0, 11], [2, 7, 9, 4, 1], [6], [8, 10, 3]):
        rows = torch.tensor(rows, device=device)
        logits = torch.randn(len(rows), VOCAB, device=device) * 3
        positions = torch.randint(0, 100, (len(rows),), device=device)
        replayed = graphs.sample(logits.clone(), rows, positions)
        expected = eager.step(logits.clone(), rows, positions)
        assert torch.equal(replayed, expected.to(replayed.dtype))
    spare = states[0].spare
    assert torch.equal(states[0].windows[:spare], states[1].windows[:spare])
    assert torch.equal(states[0].generated[:spare], states[1].generated[:spare])
