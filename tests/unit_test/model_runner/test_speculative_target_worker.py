# SPDX-License-Identifier: Apache-2.0
"""The speculative target worker's verify entry for each kind of SGLang draft worker."""

from __future__ import annotations

from types import SimpleNamespace

from sglang_omni.model_runner import speculative_target_worker as target_module
from sglang_omni.model_runner.speculative_target_worker import SpeculativeTargetWorker


class FakeForwardBatch:
    def __init__(self, source: object) -> None:
        self.source = source
        self.skip_attn_backend_init: bool | None = None

    def apply_deprecated_skip_attn_backend_init(self, value: bool | None) -> None:
        self.skip_attn_backend_init = value


def make_worker(forwarded: list[FakeForwardBatch]) -> SpeculativeTargetWorker:
    worker = object.__new__(SpeculativeTargetWorker)

    def forward(*, forward_batch: FakeForwardBatch) -> SimpleNamespace:
        forwarded.append(forward_batch)
        return SimpleNamespace(
            logits_output=SimpleNamespace(next_token_logits=None),
            can_run_graph=False,
            expert_distribution_metrics=None,
            routed_experts_output=None,
            indexer_topk_output=None,
        )

    worker.model_runner = SimpleNamespace(forward=forward)
    worker.omni_runner = None
    worker.verify_requests = []
    return worker


def test_verify_builds_the_forward_batch_of_a_schedule_batch(monkeypatch) -> None:
    # NGRAM verifies with the schedule batch; EAGLE and DFlash hand over a planned one.
    built: list[tuple[object, object]] = []

    def init_new(batch, model_runner, **kwargs):
        built.append((batch, kwargs["capture_hidden_mode"]))
        return FakeForwardBatch(batch)

    monkeypatch.setattr(target_module.ForwardBatch, "init_new", init_new)
    forwarded: list[FakeForwardBatch] = []
    worker = make_worker(forwarded)
    schedule_batch = object()

    worker.forward_batch_generation(
        schedule_batch, is_verify=True, capture_hidden_mode="full"
    )
    assert built == [(schedule_batch, "full")]
    assert forwarded[0].source is schedule_batch

    planned = FakeForwardBatch(None)
    worker.forward_batch_generation(
        None, planned, is_verify=True, skip_attn_backend_init=True
    )
    assert forwarded[1] is planned
    assert planned.skip_attn_backend_init is True
    assert len(built) == 1
