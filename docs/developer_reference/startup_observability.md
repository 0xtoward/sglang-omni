# Startup timing

Omni adds startup context to the existing SGLang timer. Ordinary server startup
emits these INFO logs when SGLang is installed:

```text
Startup begin: omni.factory.build {"gpu_id": 0, "stage": "code2wav", "tp_rank": 0}
Startup timing: omni.factory.build {"gpu_id": 0, "stage": "code2wav", "tp_rank": 0} took 13.562s
```

The elapsed time comes from `sglang.srt.observability.startup_func_log_and_timer.startup_timer`.
Omni calls it with `log_only=True`. This change adds no Prometheus metrics or
exporter. Installations without SGLang skip these diagnostic scopes.

A `Startup failed:` record identifies a phase that raised. SGLang logs elapsed
time on failure too. A duration alone does not prove that startup succeeded.
An unmatched begin record can help locate a stalled or terminated operation.

## Shared boundaries

| Phase | Scope |
| --- | --- |
| `pipeline.start` | The launcher's call to `MultiProcessPipelineRunner.start()`. Earlier CLI work and later HTTP startup are excluded. |
| `scheduler.initialize` | Stage factory import, argument resolution, GPU lock wait, and factory execution. |
| `factory.import` | Stage factory import and argument resolution. |
| `gpu_lock.wait` | Acquisition of the existing per-GPU startup lock. |
| `factory.build` | Stage factory execution and validation of its declared KV budget, under the GPU lock when applicable. |
| `checkpoint.resolve` | The shared engine builder's checkpoint-resolution hook. |
| `model.setup` | The shared engine builder's model-setup hook. |
| `model.compile_setup` | The shared builder's compile hook. This can install lazy wrappers without compiling a graph. |
| `model.resources` | The shared builder's resource hook, including model-owned encoder initialization. |
| `runtime.resources` | The shared builder's runtime-resource hook. |
| `scheduler.warmup` | `StreamingVocoderBase.on_serving_start()` in the scheduler thread. |
| `graph.capture` | Each capture through the shared CUDA, NPU, or XPU backend, within an active startup scope. |

Stage factory scopes carry stage, TP rank, and process-visible GPU labels.
Nested scopes inherit those labels. Scheduler-thread warmup uses the existing
active-stage binding. Context variables do not propagate to arbitrary new
threads. The graph scope also records its backend.

SGLang retains ownership of its AR weight-loading and generation-graph timings.
Omni does not add a second timer around those operations. These shared scopes
need no changes to model files, factory signatures, or graph callers.

Capture logs run before entry into the device capture context and after its
exit. The helper adds no device synchronization, stream operations, or graph
configuration. Request-time lazy captures outside startup are excluded.

The shared graph API has no input tensors or shape argument. Its timing logs
therefore do not claim a model shape or component. Paths that call device APIs
directly receive their enclosing factory or warmup time, not separate capture
time. A model-specific first forward has no new dedicated scope in this change.

## Native compiler records

`factory.build` and `scheduler.warmup` read PyTorch's existing
`torch._dynamo.utils.get_compilation_metrics()` buffer at entry and exit.
New retained records appear as `Startup compilation:` JSON. The helper reads
the compiler only if it is already imported. It installs no compiler callbacks
and changes no compiler, cache, or buffer settings.

Records contain the native compile ID, source location, failure type, and these
fields in seconds:

- `dynamo_seconds`
- `inductor_seconds`
- `triton_compile_seconds`
- `compile_autotune_seconds`
- `runtime_autotune_seconds`

Missing values remain `null`. Dynamo, Inductor, and Triton accounting can nest.
Do not add these fields to calculate a total. Compiler diagnostic failures log
a warning and preserve the factory result or original exception.

The native buffer is bounded and process-local. Each record includes
`scope="process_window"` and `retained_events_only=true`. Other threads can
contribute records during that window, and older records can be evicted.
These records are neither a complete compilation history nor exclusive stage
totals. This internal API is checked against the pinned PyTorch version and
must be checked again when that pin changes. cuDNN plan creation has no separate
measurement here.

## Read the numbers correctly

All phase durations are inclusive wall time at existing completion boundaries.
Warmup can include compilation, autotuning, library initialization, and forward
execution. Unsynchronized GPU work can continue after a Python scope returns.
Nested scopes and concurrent stages overlap. Do not sum them to calculate
process startup time.

For example, a stage can wait for another stage's GPU lock. Its
`scheduler.initialize` duration includes that wait, while `factory.build`
measures its own construction. Read `gpu_lock.wait` to separate those costs.

The existing pipeline-ready signal follows scheduler-thread launch. It can
precede completion of asynchronous vocoder warmup. This change preserves that
behavior. A complete `pipeline.start` record is not a warmup-completion barrier.

Process launch to HTTP ready requires an external timer and health probe.
Checkpoint download outside the server requires separate timing. A timeline
assembled from log-receipt timestamps is an external analysis, not another
server metric. No complete end-to-end decomposition is claimed here.
