# CosyVoice3 buffered-C1 async decode experiment

This fork-only draft preserves the implementation exercised by the September 13
H200 comparison. It is not a default-on engine change. `owner.py` has no import
side effects and must be installed explicitly on a bound, stopped Cosy scheduler.

## Mechanism

The scheduler reserves a penalty-update ticket without reading lagged host token
history. Immediately after resolving the previous device token, the runner
consumes that ticket once, updates penalties, and snapshots sampling state for
the next forward. Launch stages output into the base runner's pinned ping-pong
buffers; resolve collects that independent snapshot. Finished-request overrun
is discarded before Cosy's token collection and outgoing chunk publication.

This is **not** upstream Omni PR #1669's switch from
`disable_overlap_schedule=True` to `False`. This experiment keeps that switch
disabled and uses `enable_async_decode` instead. The scheduler treats the two
loops as mutually exclusive. Existing SGLang/Omni future-token, staging, and
one-step-lookahead machinery is reused; it is not a new scheduler implementation.

## Reproduction scope

- One active request, buffered output, fresh unchunked prefill, explicit seed.
- Repetition penalty 1.21, positive minimum length, no frequency/presence penalty,
  custom logit processor, or returned logprobs.
- Abort/retraction invalidate this isolated experiment. Stop that test service;
  cancellation recovery, streaming, TP and Cn have not been accepted.
- The source guards deliberately target the tested runtime. The Omni base
  requires seed-cache commit `f6bde5d1aa6e5fcf67291194b24ffdf1deee7a60`
  (upstream PR #2128), included in this fork PR's review base. No additional
  seed, metadata-graph, capture-scope or queue change is included in the diff.
  A current main checkout without that dependency is expected
  to fail its guard, not silently run a different implementation.
- Guards also check the tested SGLang `ScheduleBatch` and three other Omni
  modules. Keep them until a new dependency revision passes the state oracle;
  do not replace hashes merely to make installation proceed.

After the normal builder's `post_scheduler_setup` has bound the runner, before
starting the scheduler loop, an explicit experiment launcher can call:

```python
from experiments.cosy_async_c1.owner import install

# Configure max_running_requests=1 and disable_overlap_schedule=True first.
owner = install(scheduler, allow_lookahead=True)
```

Installation includes a process-level `ScheduleBatch` dispatch hook, delegating
unchanged for unowned requests. Use a dedicated test worker process and restart
it between independently installed experiments. This draft does not expose a
public CLI option or provide production lifecycle cleanup.

## Evidence

H200, 16 CPU, same-service ABBA, seed + metadata + peer-aware queue common to
both arms. Control is synchronous fallback inside the same async-capable loop,
**not** the stock synchronous scheduler:

| C1 cohort | Sync fallback | Async | E2E change |
| --- | ---: | ---: | ---: |
| Fixed | 379.795 ms | 356.676 ms | -6.087% |
| Mixed lengths | 445.105 ms | 413.303 ms | -7.145% |

128 timed requests: codec sequences, sample counts and WAV hashes matched.
Both cohorts improved for all 16 paired input means. Warmups excluded from the
timed result; CPU throttling was zero. These numbers belong to the original
guarded runtime, not a GPU rerun of this review branch. The module differs from
that runtime only in its introductory docstring, formatting and equivalent
lint cleanups (type annotation, import order and counter dictionary syntax).

Focused CPU state-machine tests:

```bash
python -m pytest tests/unit_test/fun_cosyvoice3/test_async_owner_experiment.py -q
```

Next integration gate: move the owner seam into native code, replace whole-file
version guards with supported runtime contracts, then validate abort/retract,
streaming and mixed batches before adding a user-facing enablement flag.
