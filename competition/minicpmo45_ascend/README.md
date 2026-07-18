# MiniCPM-o 4.5 Ascend competition branch

This directory defines the project boundary for bringing MiniCPM-o 4.5 to
SGLang-Omni on Huawei Ascend 910C/A3. The branch is a clean development base;
it is **not** evidence that the model currently runs on NPU.

## Locked baseline

- Branch: `codex/minicpmo45-ascend-competition`
- Upstream base: `10e7e54197af9b384af0c9fa5563723f09eb25bf`
- Companion core branch: `0xtoward/sglang:codex/minicpmo45-ascend-competition-core`
- Image-compatible core base: `f308abc05212c2f5f455de22a525e14afa63ee4f`
- Target image: CANN 9.0.0, torch/torch_npu 2.10.0, aarch64

See `BASELINE.lock.json` for machine-readable values.
See `TAKEOVER.md` for a server-loss-safe restore and handoff procedure.

## Current red lines

- Mainline SGLang-Omni has no MiniCPM-o model package.
- Upstream dependencies pin torch 2.11, a different SGLang release, and CUDA
  packages. Do not run ordinary `pip install -e .` on the target NPU image.
- Runtime code still contains CUDA/NCCL/device assumptions. A generic vendor
  helper is not end-to-end Ascend support.
- Turn-taking text+speech and model-native full-duplex are separate milestones.

## Integration gates

1. Add an Ascend-safe dependency/install contract without mutating the image repo.
2. Establish device/stream/communication abstractions and CPU-only unit coverage.
3. Integrate MiniCPM-o 4.5 thinker: text/image/audio input→text.
4. Bridge Qwen3 hidden states to MiniCPMTTS and validate hidden→S3 fixtures.
5. Port Token2wav and validate waveform correctness, first-audio latency and RTF.
6. Expose a turn-taking request pipeline.
7. Add session-lifetime state, continuous append and barge-in for full-duplex.
8. Add profiler-off correctness/performance runs and separate profiler-on attribution.

Every gate must preserve an environment manifest, exact launch command, logs,
fixtures and failure evidence. Shared competition branches use merges and must
not be force-pushed.
