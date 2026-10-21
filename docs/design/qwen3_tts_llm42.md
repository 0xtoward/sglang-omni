# Deterministic Qwen3-TTS decoding with verify windows (LLM-42)

## Problem

`enable_deterministic_inference` makes Qwen3-TTS return byte-identical PCM for the same prompt, reference and seed at any runtime batch size (#1475). It does so by running every talker step with SGLang's batch-invariant kernels, decoding the vocoder at batch size one without CUDA graphs, and serializing reference preprocessing.

Measured on one H200, Qwen3-TTS-12Hz-0.6B-Base, Seed-TTS English (30 prompts at concurrency 1, 300 at concurrency 16), requests per second:

| Mode | Concurrency 1 | Concurrency 16 |
|---|---|---|
| default | 4.46 | 21.78 |
| SGLang deterministic inference in the talker only | 2.62 | 12.89 |
| full deterministic mode | 2.11 | 13.22 |

The talker's batch-invariant decode alone costs 41% of throughput at both concurrencies. The vocoder and preprocessing add the rest at concurrency 1 and nothing measurable at concurrency 16. The mode is also server-wide, so a request that does not need determinism pays the same price.

## Idea

LLM-42 (arXiv 2601.17768) keeps the fast, batch-dependent decode path and makes a fixed-shape, batch-invariant verifier the source of truth. Every W steps the verifier recomputes the last W positions of each deterministic request, teacher-forced on the tokens the fast path produced. It commits the longest prefix on which both agree, plus the verifier's token at the first disagreement, and rolls the request back to that point. Because every committed token and every committed KV entry comes from the verifier, the output depends only on the request, not on what it was batched with.

This is speculative decoding whose draft is the target itself in fast mode. sglang-omni already has the verify half: `SpeculativeTargetWorker` runs a target-verify forward over several rows per request, and the omni scheduler commits accepted prefixes and releases the KV of rejected rows (#2937).

## How it maps onto Qwen3-TTS

One talker step produces a 16-codebook frame:

1. The talker forward samples codebook 0 with SGLang's sampler, seeded per request (`semantic_sampling_seed_tensor`, installed by `install_sampling_seeds`).
2. `code_predictor_forward` samples codebooks 1 to 15 from the talker hidden state and codebook 0, seeded by the subtalker seed.
3. The sum of the frame's codebook embeddings is fed back as the next step's input (`pending_feedback_queue`).

The verifier for a window of W frames:

1. Runs the talker over the W positions in batch-invariant mode, with the fast path's feedback embeddings as inputs, and writes those positions' KV.
2. Re-samples codebook 0 at each position with the same per-position seeds, and re-runs the code predictor per frame in batch-invariant mode with its seeds.
3. Compares all 16 codebooks frame by frame. Frames before the first mismatch commit as they are; the mismatching frame commits the verifier's codes; later frames, their KV and their queued feedback are dropped, and fast decoding resumes from the committed frame.

Requirements this adds:

- **Position-keyed sampling.** A frame's sample must be a function of (seed, position, logits) only, so the verifier reproduces it regardless of batch. SGLang's seeded sampling already keys on position; the code predictor's sampler has to do the same for each of its 15 inner steps.
- **Per-forward batch-invariant mode.** SGLang enables batch-invariant kernels server-wide. Verify forwards need them on and fast forwards off in one process: batch-invariant matmul and reduction ops toggled per forward, attention with fixed split sizes for verify only, and verify CUDA graphs captured with the mode on.
- **Deterministic prefill.** The prompt's KV is context for every verifier, so prefill runs in batch-invariant mode too. It is a single forward per request.
- **Committed frames only downstream.** The vocoder decodes only committed frames. Streaming requests lag the fast path by up to W frames, which bounds the extra time to first audio by W talker steps.
- **Per-request opt-in.** Determinism becomes a request field. Requests without it never verify and keep the default kernels and vocoder batching.

## Expected cost

With a window of W frames the verifier adds one batch-invariant forward of W rows per request every W steps, plus W code-predictor passes. Rollback adds the steps replayed after a mismatch. A fast step and a verifier disagree only when seeded sampling lands within the numeric gap between the two paths, so most windows should commit whole; the rate per frame has to be measured, since 16 codebooks are sampled per frame. If rollbacks stay rare, deterministic talker throughput should approach the default path's 21.8 requests per second at concurrency 16, against 12.9 today. The vocoder's batch-size-one decode at concurrency 1 is untouched by this and needs its own batch-invariant path.

## Validation plan

- CPU tests: window acceptance (whole window, first-frame mismatch, mid-window mismatch, end of speech inside a window), rollback of KV and feedback queues, and position-keyed sampling giving the same codes when a row moves within a batch.
- GPU: the existing batch-invariance test (`tests/test_model/test_qwen3_tts_batch_invariance.py`) must still produce byte-identical PCM across batch sizes 1, 6 and 8 with the new mode.
- Throughput against #1475's deterministic mode at concurrency 1 and 16, alternated with repeats; a mixed batch with half the requests deterministic, where the other half must keep default throughput.
- Rollback statistics: committed frames per window and mismatches per thousand frames, for W of 8, 16 and 32.
