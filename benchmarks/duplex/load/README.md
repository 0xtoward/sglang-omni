# Full-duplex open-loop load

Measures how a MiniCPM-o full-duplex server keeps up when many sessions talk at once.

The recorder in `benchmarks/duplex` scores what the model says. This directory measures how late it says
it. Sessions arrive as a Poisson process rather than in lock step, because synchronized starts align every
session's units and overstate contention, while evenly spaced starts understate it. Each session is a real
multi-turn conversation, streamed at real time.

## Inputs

On macOS, speak the 24 questions in `questions.txt`, then join them into 32 sessions. Each session has
four questions with 10 to 18 s pauses for the answers, about 70 s in total.

```bash
python benchmarks/duplex/load/sessions.py say --output-dir duplex-input/questions
python benchmarks/duplex/load/sessions.py build --questions-dir duplex-input/questions --output-dir duplex-input/sessions32
```

Copy `duplex-input` to the GPU host.

## One cell

```bash
benchmarks/duplex/load/run_cell.sh --model-path "$MODEL" --sessions-dir duplex-input/sessions32 \
  --warmup-wav duplex-input/questions/q0.wav --output-dir runs/main-s16 --sessions 16 \
  --profile-at 35 --profile-window 40 -- --mps off --max_sessions 32 --thinker.engine.max_total_tokens 139264
python benchmarks/duplex/load/report.py lag runs/main-s16/cell
python benchmarks/duplex/load/report.py stages runs/main-s16
```

What the run does:

- The server binds 127.0.0.1.
- One warm-up session runs first.
- With the default `--mean-gap-s 3 --seed 11`, the 16 sessions arrive within about 45 s, and the cell lasts about 2 min.
- `--profile-at` opens a `/start_profile` window that records only stage events. Add `--torch-trace` for kernel traces.
- `smi.csv` logs GPU memory and utilization every 2 s, for memory curves.

To compare two trees, run the cells in A B B A order on one GPU with the same seed.

## Metrics

**Unit lag** (`report.py lag`). Every second of input becomes one unit. Its lag is the time of its `sglang.unit.done` minus the time the client finished sending that second.

- A unit is *speaking* if output audio arrived since the previous `unit.done`, and *listening* otherwise.
- A unit is *late* if its lag exceeds 1 s, the length of a unit.

**Stage split** (`report.py stages`). Each unit is its own request through perception, thinker, talker and speech. Its stage events split the time from admission to response into:

- queue, compute and hop segments;
- for the thinker, also prefill and decode.

The report gives each segment's median and p90, separately for speaking and listening units.

## Traces

Use `--torch-trace` with a short window, for example `--profile-window 5`.

Do not use that run's lag numbers: `stop_profile` stalls every stage for more than 10 s while it writes the traces. Time the same configuration in a separate events-only run.

```bash
python benchmarks/duplex/load/traces.py summary runs/x/trace-window
python benchmarks/duplex/load/traces.py slice runs/x/trace-window --output slice.json.gz
python benchmarks/duplex/load/traces.py attribute runs/x/trace-window/speech/*.trace.json.gz "vocoder.py(372): stream_batch"
```

- **`summary`** gives, for every stage:
  - GPU busy time, and the union over stages, since all stages share one GPU;
  - kernel classes;
  - CUDA syncs and launches;
  - Python self time.
- **`slice`** lays all four stages on one timeline. Open it in ui.perfetto.dev.
- **`attribute`** charges each kernel to the child frame that launched it.

A frame's CPU wall is not its GPU cost. Kernels launch asynchronously, and the wait lands on the next sync. For example, HiFT shows 80 ms of CPU wall inside `stream_batch`, but it waits there for the flow's GPU work; its own kernels take about 4 ms.

## Traps

- **H100 80 GB at 32 sessions.** The default thinker KV pool, (max_sessions + 1) x 8192 tokens, does not fit. Pass `--thinker.engine.max_total_tokens 139264`; a 70 s session uses about 1.5k tokens.
- **MPS off (the default).** Processes time-slice the GPU, so a kernel's duration includes time it spent switched out.
- **The Python tracer leaves some builtin frames open until the trace ends.** `traces.py` drops frames that outlive their caller and reports how many it dropped.

## Reference

H100, main e6dff575e, 16 sessions, events-only, 2026-10-12. Speaking units, in ms, median (p90):

| Segment | main | with #2665 |
| --- | --- | --- |
| thinker prefill | 29 (60) | 23 (31) |
| thinker decode | 207 (457) | 108 (214) |
| talker | 173 (270) | 178 (321) |
| speech first chunk | 144 (205) | 149 (247) |
| admission to response | 622 (914) | 513 (767) |
| client: late units | 2.4% | 0.4% |
