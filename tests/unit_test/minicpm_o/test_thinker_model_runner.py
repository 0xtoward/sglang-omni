# SPDX-License-Identifier: Apache-2.0
"""Thinker length penalty on the sync and lookahead paths, and prefill CUDA graph wiring."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch
from sglang.srt.model_loader.utils import resolve_language_model
from torch import nn

from sglang_omni.model_runner.prefill_inputs import get_omni_prefill_inputs
from sglang_omni.model_runner.thinker_model_runner import ThinkerModelRunner
from sglang_omni.models.minicpm_o import stages
from sglang_omni.models.minicpm_o.components.sglang_thinker import (
    MiniCPMOThinkerForCausalLM,
)
from sglang_omni.models.minicpm_o.thinker_model_runner import MiniCPMOThinkerModelRunner
from sglang_omni.platforms import current_platform
from sglang_omni.proto import OmniRequest, StagePayload

EOS_TOKEN_IDS = [1, 3]


def bare_runner() -> MiniCPMOThinkerModelRunner:
    runner = object.__new__(MiniCPMOThinkerModelRunner)
    runner.eos_token_ids = EOS_TOKEN_IDS
    runner.eos_token_id_cache = None
    return runner


def scheduler_request(params: dict[str, Any]) -> SimpleNamespace:
    payload = StagePayload(
        request_id="request-0",
        request=OmniRequest(
            inputs=None,
            params=params,
            metadata={"output_modalities": ["text"]},
        ),
        data=None,
    )
    return SimpleNamespace(data=SimpleNamespace(stage_payload=payload))


def penalized_request(length_penalty: float) -> SimpleNamespace:
    return scheduler_request(
        {"stage_params": {"thinker": {"length_penalty": length_penalty}}}
    )


def test_length_penalty_scales_eos_logits_per_request() -> None:
    original = torch.tensor(
        [
            [0.5, 2.0, -1.0, -4.0],
            [0.5, 2.0, -1.0, -4.0],
            [0.5, 2.0, -1.0, -4.0],
        ]
    )
    logits_output = SimpleNamespace(next_token_logits=original.clone())

    bare_runner().process_sampling_logits(
        logits_output,
        [
            penalized_request(1.0),
            penalized_request(2.0),
            penalized_request(1.0),
        ],
    )

    torch.testing.assert_close(logits_output.next_token_logits[0], original[0])
    torch.testing.assert_close(logits_output.next_token_logits[2], original[2])
    expected = original[1].clone()
    expected[EOS_TOKEN_IDS] = torch.tensor([2.0 / 2.0, -4.0 * 2.0])
    torch.testing.assert_close(logits_output.next_token_logits[1], expected)


def test_lookahead_samples_from_penalized_logits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = torch.tensor([[0.5, 2.0, -1.0, -4.0]])
    logits_output = SimpleNamespace(next_token_logits=original.clone())
    logits_seen_by_parent = []

    def parent_sample_lookahead(self, logits_output, forward_batch, requests):
        logits_seen_by_parent.append(logits_output.next_token_logits.clone())
        return torch.tensor([0])

    monkeypatch.setattr(ThinkerModelRunner, "sample_lookahead", parent_sample_lookahead)

    bare_runner().sample_lookahead(
        logits_output, forward_batch=None, requests=[penalized_request(2.0)]
    )

    expected = original.clone()
    expected[0, EOS_TOKEN_IDS] = torch.tensor([2.0 / 2.0, -4.0 * 2.0])
    torch.testing.assert_close(logits_seen_by_parent[0], expected)


def test_missing_length_penalty_leaves_logits_unchanged() -> None:
    original = torch.tensor([[0.5, 2.0, -1.0, -4.0]])
    logits_output = SimpleNamespace(next_token_logits=original.clone())

    bare_runner().process_sampling_logits(logits_output, [scheduler_request({})])

    torch.testing.assert_close(logits_output.next_token_logits, original)


def test_prefill_graphs_resolve_the_thinker_decoder() -> None:
    model = object.__new__(MiniCPMOThinkerForCausalLM)
    nn.Module.__init__(model)
    model.language_model = SimpleNamespace(model=nn.Identity())
    assert resolve_language_model(model) is model.language_model.model


@pytest.mark.parametrize(
    ("server_args_overrides", "is_cuda", "backend", "operator_selected"),
    [
        ({}, True, "breakable", False),
        ({"cuda_graph_backend_prefill": "disabled"}, True, "disabled", True),
        ({}, False, "disabled", False),
    ],
)
def test_thinker_stage_defaults_breakable_prefill_graphs_on_cuda(
    monkeypatch: pytest.MonkeyPatch,
    server_args_overrides: dict[str, object],
    is_cuda: bool,
    backend: str,
    operator_selected: bool,
) -> None:
    built: dict[str, object] = {}
    monkeypatch.setattr(current_platform, "is_cuda", lambda: is_cuda)
    monkeypatch.setattr(
        stages, "resolve_concrete_device", lambda device, gpu_id: torch.device("cpu")
    )
    monkeypatch.setattr(stages, "register_minicpm_o_hf_config", lambda: None)
    monkeypatch.setattr(
        stages,
        "build_sglang_server_args",
        lambda model_path, context_length, **overrides: built.update(overrides),
    )
    monkeypatch.setattr(
        stages,
        "resolved_view",
        lambda server_args: SimpleNamespace(mem_fraction_static=0.5),
    )
    monkeypatch.setattr(stages, "validate_generation_batch_policy", lambda **_: None)
    monkeypatch.setattr(stages, "avail_gpu_mem", lambda gpu_id: 0)
    monkeypatch.setattr(
        stages,
        "create_thinker_scheduler",
        lambda server_args, gpu_id, **kwargs: built.update(scheduler=kwargs),
    )

    stages.create_sglang_thinker_executor_from_config(
        "model", server_args_overrides=server_args_overrides
    )

    assert built["cuda_graph_backend_prefill"] == backend
    assert built["scheduler"]["operator_selected_prefill_backend"] is operator_selected


@pytest.mark.parametrize("has_media", [False, True])
def test_prefill_hands_its_embeddings_to_the_graph_sidecar(
    monkeypatch: pytest.MonkeyPatch, has_media: bool
) -> None:
    runner = bare_runner()
    runner.embed_tokens = nn.Embedding(8, 4)
    input_ids = torch.tensor([1, 2, 3])
    media_embeds = torch.full((3, 4), 7.0)
    monkeypatch.setattr(
        runner,
        "inject_multimodal_embeds",
        lambda forward_batch, schedule_batch: (
            (media_embeds, None, None) if has_media else None
        ),
    )
    forward_batch = SimpleNamespace(
        input_ids=input_ids, input_embeds=None, replace_embeds=None
    )
    schedule_batch = SimpleNamespace(
        forward_mode=SimpleNamespace(is_extend=lambda: True)
    )

    assert runner.custom_prefill_forward(forward_batch, schedule_batch, []) is None

    expected = media_embeds if has_media else runner.embed_tokens(input_ids)
    torch.testing.assert_close(
        get_omni_prefill_inputs(forward_batch).input_embeds, expected
    )
    assert forward_batch.input_embeds is None
