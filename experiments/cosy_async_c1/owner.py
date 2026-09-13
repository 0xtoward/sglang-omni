"""Explicit buffered-C1 async decode experiment.

Install after binding the Cosy runner and before starting its scheduler.
The default remains synchronous. See experiments/cosy_async_c1/README.md
for the pinned runtime, request contract, and measured comparison.
"""

from __future__ import annotations

import contextlib
import hashlib
import inspect
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from types import MethodType

LOG = logging.getLogger(__name__)


@dataclass
class Ticket:
    owner: Owner
    request: object
    epoch: int
    orchestrator: object
    consumed: bool = False


class Owner:
    def __init__(self, scheduler, runner, allow_lookahead):
        self.scheduler = scheduler
        self.runner = runner
        self.allow_lookahead = allow_lookahead
        self.epoch = 0
        self.invalid_reason = None
        self.counts = {
            "prefills": 0,
            "reserved": 0,
            "committed": 0,
            "launches": 0,
            "resolves": 0,
            "dropped_overrun": 0,
            "finishes": 0,
        }

    def invalidate(self, reason):
        if self.invalid_reason is None:
            self.invalid_reason = reason
            LOG.error("COSY_ASYNC_OWNER_INVALID %s", json.dumps(self.report()))

    def require(self, condition, reason):
        if not condition:
            self.invalidate(reason)
        self.check()

    def check(self):
        if self.invalid_reason is not None:
            raise RuntimeError("COSY_ASYNC_OWNER_INVALID: " + self.invalid_reason)

    def report(self):
        return dict(
            scope="isolated-C1-no-abort-no-retract",
            allow_lookahead=self.allow_lookahead,
            invalid_reason=self.invalid_reason,
            **self.counts,
        )

    def validate_request(self, req):
        sp = req.sampling_params
        self.require(
            abs(float(sp.repetition_penalty) - 1.21) < 1e-9,
            "this prototype only audited repetition_penalty=1.21",
        )
        self.require(
            sp.frequency_penalty == 0 and sp.presence_penalty == 0,
            "frequency/presence outside oracle",
        )
        self.require(
            sp.min_new_tokens > 0 and sp.sampling_seed is not None,
            "requires positive minimum length and explicit per-request seed",
        )
        self.require(
            req.custom_logit_processor is None and not req.return_logprob,
            "custom processors/logprobs outside oracle",
        )
        self.require(not req.is_retracted, "retracted request cannot be owned")

    def reserve(self, batch):
        self.check()
        self.require(
            len(batch.reqs) == 1 and batch.forward_mode.is_decode(), "not a C1 decode"
        )
        req = batch.reqs[0]
        self.validate_request(req)
        self.require(getattr(req, "_cosy_async_owner", None) is self, "owner mismatch")
        previous = getattr(batch, "_cosy_async_penalty_ticket", None)
        self.require(
            previous is None or previous.consumed,
            "second reservation before consumption",
        )
        orchestrator = batch.sampling_info.penalizer_orchestrator
        self.require(
            orchestrator is not None and orchestrator.is_required,
            "missing live penalizer",
        )
        active = {
            type(p).__name__
            for p in orchestrator.penalizers.values()
            if p.is_prepared()
        }
        self.require(
            active == {"BatchedMinNewTokensPenalizer", "BatchedRepetitionPenalizer"},
            "unexpected active penalizer set: " + str(active),
        )
        batch._cosy_async_penalty_ticket = Ticket(
            self, req, req._cosy_async_epoch, orchestrator
        )
        self.counts["reserved"] += 1

    def commit_resolved(self, batch):
        self.check()
        ticket = getattr(batch, "_cosy_async_penalty_ticket", None)
        self.require(
            ticket is not None and ticket.owner is self and not ticket.consumed,
            "missing or consumed penalty ticket",
        )
        self.require(
            len(batch.reqs) == 1 and batch.reqs[0] is ticket.request,
            "row changed after C1 reservation",
        )
        self.require(
            ticket.request._cosy_async_epoch == ticket.epoch, "request epoch changed"
        )
        self.require(
            batch.sampling_info.penalizer_orchestrator is ticket.orchestrator,
            "penalizer replaced after reservation",
        )
        ids = batch.input_ids
        self.require(
            ids is not None and ids.ndim == 1 and ids.shape[0] == 1,
            "expected exactly one resolved device token",
        )
        self.require(ids.device == self.runner.device, "resolved token device mismatch")
        # Consume before submission: an exception must fail the request/experiment,
        # never retry the same update or fall back to the old host update.
        ticket.consumed = True
        try:
            ticket.orchestrator.cumulate_output_tokens(ids)
        except Exception:
            self.invalidate("penalty update raised after ticket consumption")
            raise
        self.counts["committed"] += 1


def _assert_source(obj, expected):
    filename = inspect.getsourcefile(obj)
    actual = hashlib.sha256(Path(filename).read_bytes()).hexdigest()
    if actual != expected:
        raise RuntimeError(f"unreviewed source: {filename}, sha256={actual}")


def install(scheduler, *, allow_lookahead=False):
    """Prepare an explicit, fail-closed instance patch; do not start the scheduler."""
    import torch
    from sglang.srt.managers.overlap_utils import resolve_forward_inputs
    from sglang.srt.managers.schedule_batch import ScheduleBatch
    from sglang_omni.model_runner.base import ModelRunner
    from sglang_omni.model_runner.sglang_execution import SGLangExecutionBridge
    from sglang_omni.models.fun_cosyvoice3.model_runner import FunCosyVoice3ModelRunner
    from sglang_omni.models.fun_cosyvoice3.sglang_model import VOCAB_SIZE
    from sglang_omni.scheduling.omni_scheduler import OmniScheduler

    runner = scheduler._model_runner
    if type(runner) is not FunCosyVoice3ModelRunner:
        raise TypeError("CosyVoice3 only")
    if (
        scheduler._running
        or scheduler._async_pending is not None
        or scheduler.enable_overlap
    ):
        raise RuntimeError(
            "install before startup, with no pending work and overlap schedule disabled"
        )
    if scheduler.max_running_requests != 1:
        raise RuntimeError(
            "configure max_running_requests=1 before constructing this experiment"
        )
    if getattr(runner, "_cosy_async_owner_protocol", None) is not None:
        raise RuntimeError("already installed")
    _assert_source(
        ModelRunner, "1aa4d0978e96825ce042ef9102a354da3872b604b1afd55f9a8a82c8107496b7"
    )
    _assert_source(
        SGLangExecutionBridge,
        "92c0c802e1d813824bcbf5fd464979439bea54f54241b4939e70ec2d5a076a94",
    )
    _assert_source(
        FunCosyVoice3ModelRunner,
        "a236fc34165f0d988c943632c15839525b6a0685abdbf630ff6e26660682f6a7",
    )
    _assert_source(
        OmniScheduler,
        "59a864bdf773fc8e5f415be9accb65d359f950d2d7b87ef2c53984a4a3f1295f",
    )
    _assert_source(
        ScheduleBatch,
        "170f847aba6701d8d876a98c5edb8a66f188f5bc12fc30973596d132fcbb8c72",
    )

    owner = Owner(scheduler, runner, bool(allow_lookahead))
    bridge = runner._execution_bridge
    original_prefill = runner.post_prefill
    original_finished = runner.on_request_finished
    original_abort = scheduler.abort
    original_add_queue = scheduler._add_request_to_queue
    original_context = bridge.forward_context

    # The class-level seam delegates unchanged for every unowned request. No
    # global lookahead gate or general penalty implementation is replaced.
    original_host = ScheduleBatch.cumulate_penalty_output_tokens
    if not getattr(original_host, "_cosy_owner_dispatch", False):

        def host_dispatch(batch):
            owners = [getattr(req, "_cosy_async_owner", None) for req in batch.reqs]
            if not any(item is not None for item in owners):
                return original_host(batch)
            live_owner = next(item for item in owners if item is not None)
            live_owner.require(
                len(owners) == 1 and owners[0] is live_owner, "mixed/unowned batch"
            )
            return live_owner.reserve(batch)  # deliberately no old host update

        host_dispatch._cosy_owner_dispatch = True
        ScheduleBatch.cumulate_penalty_output_tokens = host_dispatch

    def post_prefill(self, result, forward, batch, requests):
        owner.check()
        owner.require(
            len(requests) == 1 and not batch.is_prefill_only,
            "unsupported prefill shape/mode",
        )
        req = requests[0].data.req
        owner.validate_request(req)
        owner.require(
            not req.output_ids
            and not req.is_retracted
            and req.inflight_middle_chunks == 0,
            "only fresh, final, unchunked prefill is supported",
        )
        owner.require(
            requests[0].data.stream_metadata is None, "buffered-only pricing experiment"
        )
        original_prefill(result, forward, batch, requests)
        # Prefill produces/collects its first token normally. Do not update the
        # penalty here; first decode consumes that token exactly once.
        owner.epoch += 1
        req._cosy_async_owner = owner
        req._cosy_async_epoch = owner.epoch
        owner.counts["prefills"] += 1

    @contextlib.contextmanager
    def forward_context(self, batch, *, isolate_sampling=False):
        owner.check()
        if not batch.forward_mode.is_decode():
            with original_context(batch, isolate_sampling=isolate_sampling):
                yield
            return
        # Mirror the current bridge, inserting the one writer before snapshot.
        sampling_info = batch.sampling_info
        owner.require(
            isolate_sampling and sampling_info is not None,
            "forward sampling must be isolated",
        )
        resolve_forward_inputs(batch, self.future_map)
        owner.commit_resolved(batch)
        try:
            batch.sampling_info = sampling_info.copy_for_forward()
            yield
        except Exception:
            owner.invalidate("forward failed after penalty submission")
            raise
        finally:
            batch.sampling_info = sampling_info

    def eligible(self, batch):
        owner.check()
        if (
            not owner.allow_lookahead
            or len(batch.reqs) != 1
            or not batch.forward_mode.is_decode()
        ):
            return False
        ticket = getattr(batch, "_cosy_async_penalty_ticket", None)
        if ticket is None or ticket.owner is not owner or ticket.consumed:
            return False
        req = batch.reqs[0]
        owner.validate_request(req)
        return (
            ticket.request is req
            and ticket.epoch == req._cosy_async_epoch
            and getattr(req, "_cosy_async_owner", None) is owner
        )

    def launch(self, result, forward, requests):
        owner.check()
        owner.require(len(requests) == 1, "launch is C1 only")
        req = requests[0].data.req
        buf = ModelRunner.post_decode_launch(self, result, forward, requests)
        owner.counts["launches"] += 1
        return buf, req, req._cosy_async_epoch

    def resolve(self, payload, result, forward, batch, requests):
        owner.check()
        buf, req, epoch = payload
        owner.require(
            len(requests) == 1 and requests[0].data.req is req,
            "resolve request identity changed",
        )
        owner.require(req._cosy_async_epoch == epoch, "stale pending generation")
        result.next_token_ids = buf[
            :1
        ]  # independent CPU snapshot, not reused GPU output
        result._host_token_ids = result.next_token_ids
        result._host_token_ids_event = (
            None  # base execute_resolve waited on its launch event
        )
        owner.counts["resolves"] += 1
        if req.finished() or req.is_retracted:
            owner.counts["dropped_overrun"] += 1
            return  # skip BEFORE Cosy's direct collect/outbox, not only in finalize
        token_id = int(result.next_token_ids[0])
        if token_id < VOCAB_SIZE:
            token = torch.tensor([token_id], dtype=torch.long)
            requests[0].data.output_codes.append(token)
            self._queue_or_emit_code_chunk(requests[0], token)

    def finished(self, request_id, data):
        owner.check()
        original_finished(request_id, data)
        owner.counts["finishes"] += 1
        LOG.info("COSY_ASYNC_OWNER_REQUEST %s", json.dumps(owner.report()))

    def abort(self, request_id, *, defer_running_cleanup=True):
        owner.invalidate("abort observed; discard this isolated experiment")
        return original_abort(request_id, defer_running_cleanup=defer_running_cleanup)

    def add_queue(self, req, is_retracted=False):
        if is_retracted or req.is_retracted:
            owner.invalidate("retract observed; no old pending step may commit")
        return original_add_queue(req, is_retracted=is_retracted)

    runner.post_prefill = MethodType(post_prefill, runner)
    bridge.forward_context = MethodType(forward_context, bridge)
    runner.lookahead_eligible = MethodType(eligible, runner)
    runner.post_decode_launch = MethodType(launch, runner)
    runner.post_decode_resolve = MethodType(resolve, runner)
    runner.on_request_finished = MethodType(finished, runner)
    scheduler.abort = MethodType(abort, scheduler)
    scheduler._add_request_to_queue = MethodType(add_queue, scheduler)
    runner._cosy_async_owner_protocol = owner
    scheduler.enable_async_decode = bool(allow_lookahead)
    scheduler.async_decode_min_batch_size = 1
    runner._async_enabled = bool(allow_lookahead)
    LOG.warning("COSY_ASYNC_OWNER_INSTALLED %s", json.dumps(owner.report()))
    return owner
