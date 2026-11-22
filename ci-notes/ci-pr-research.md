# CI capacity changes and their dispositions

The source baseline is `dfb6b43b3` (2026-11-22). Two changes shipped on the fork branch `qun/ci-lane-prereqs`. The branch ends at `204cc7568`. Provisioning additional CPU runners is a separate host change. [Fork branch diff](https://github.com/0xtoward/sglang-omni/compare/qun/ci-base...qun/ci-lane-prereqs) · [Multi-lane provisioning handoff](multilane-docker-handoff.md)

Retry measurements use **72 upstream CPU `unit-test` jobs across 24 commits**, three partitions per commit. The sample took 17 commits with all three partitions green and seven with at least one red. Retry-time percentages use **57,008 s** as their denominator, the sum of the public **Run test** step durations. Attempt timings come from the wrapper's timestamped attempt groups and completion messages. They include the interval between groups. Of 61 green jobs, 57 had retrievable attempt logs. All 11 red jobs had attempt logs. Whole-fleet percentages below use a separate two-week job-duration inventory. [CPU job and 30-minute step limit](../.github/workflows/test.yaml#L55-L71) · [Wrapper attempt markers](../.github/scripts/run_flaky_pytest.sh#L89-L118)

## Shipped on the fork

| Candidate | File | Measured effect | Risk and deciding evidence |
|---|---|---|---|
| Gate model suites on PR Test | [.github/workflows/omni-ci.yaml:251–329](../.github/workflows/omni-ci.yaml#L251-L329) | For seven sampled commits with red CPU tests, the gate would have skipped 13 executed model jobs totaling 5,173 s (86.2 H100 runner minutes) across one run per commit, using its last attempt. | Eleven jobs were green and two were red. The change delays those two additional failure signals. Shipped in [`0019c4ef9`](https://github.com/0xtoward/sglang-omni/commit/0019c4ef9). |
| Compare cpusets by CPU ID | [.github/scripts/pin_to_ci_cpuset.sh:5–15](../.github/scripts/pin_to_ci_cpuset.sh#L5-L15) | Zero execution time saved. A correctly pinned two-core lane can now start. | Exact string comparison rejected `24-25,88-89` when Linux returned `24,25,88,89`. Expanding both sides to sorted CPU IDs accepts equivalent spellings and still rejects a missing CPU. Shipped in [`204cc7568`](https://github.com/0xtoward/sglang-omni/commit/204cc7568). |

**What the model gate saves.** On the base revision, `asr-ci`, `tts-ci`, `mps-ci`, `qwen3-omni-ci` and `duplex-ci` all declare `needs: [preflight, setup, pr-test]`. Their conditions require setup success, the selected stage and a non-cancelled run. `always()` lets each selected suite run after PR Test finishes even if a CPU partition fails. The branch adds `needs.pr-test.result == 'success'` to all five. The accelerator unit test runs **inside** PR Test alongside the CPU matrix. An accelerator failure now also skips the selected model suites. [Original workflow design](../.github/workflows/omni-ci.yaml#L3-L16) · [PR Test call](../.github/workflows/omni-ci.yaml#L341-L351) · [Accelerator job](../.github/workflows/test.yaml#L73-L101)

| Setup result | PR Test result | Suites run on base | Suites run on branch |
|---|---|---:|---:|
| Success | Success | 5/5 | 5/5 |
| Success | Failure | 5/5 | 0/5 |
| Success | Skipped | 5/5 | 0/5 |
| Failure | Skipped by dependency | 0/5 | 0/5 |

The table evaluates the real conditions with all five stage selections enabled and no cancellation. In the fleet sample, GPU unit tests accounted for 20.7% of machine time and model suites 16.0%, or 36.7% combined. Only the selected portion of the **16.0% model-suite share** following a final PR Test failure can be saved by this gate. The accelerator test has already consumed its runner time. In one red-CPU run, five green TTS jobs used 30 min 23 s. In another, the CPU partition failed during collection and Fun-ASR subsequently reported a p95 of 0.643 s against a 0.612 s limit. The latter is the kind of diagnostic the gate postpones. [TTS run](https://github.com/sgl-project/sglang-omni/actions/runs/44228709761) · [CPU collection error](https://github.com/sgl-project/sglang-omni/actions/runs/43820517761/job/133156802147) · [Fun-ASR result](https://github.com/sgl-project/sglang-omni/actions/runs/43820517761/job/133165878866)

## Prerequisite for CPU lanes

| Candidate | File | Measured effect | Risk and disposition |
|---|---|---|---|
| Explicitly empty `CUDA_VISIBLE_DEVICES` means no GPU | [.github/scripts/delete_gpu_process.sh:17–41](../.github/scripts/delete_gpu_process.sh#L17-L41) and [.github/workflows/test.yaml:47–67](../.github/workflows/test.yaml#L47-L67) | A CPU container with a visible card and a neighboring process holding 2,913 MiB failed setup after 122 s. The retry cleanup probe took 124 s and abandoned the retry. | An empty mask falls through to all visible cards under `GITHUB_ACTIONS=true`. This deterministically blocks the lane while a neighbor holds memory above the threshold. Prerequisite, not filed. |

Move the explicit empty mask to CPU **job** scope so it reaches setup as well as pytest. Return success before cleanup for that case. Preserve unset-variable behavior for GPU jobs, explicit physical IDs, `all` and `OMNI_CI_ALLOW_UNSCOPED_GPU_CLEAN=1`. The test for acceptance is a real CPU job beside a busy GPU job, including setup and a retry. [Bare setup cleanup](../.github/actions/omni-setup/action.yaml#L88-L91) · [Wrapper cleanup](../.github/scripts/run_flaky_pytest.sh#L78-L86) · [Probe evidence and job isolation](multilane-docker-handoff.md#1-gpu-cleanup-can-fail-a-cpu-job-before-pytest-starts)

## Rejected or not pursued

| Candidate | File or invocation | Measured effect | Risk and disposition |
|---|---|---|---|
| Drop the duplicate reuse gate | [.github/actions/omni-setup/action.yaml:62–80](../.github/actions/omni-setup/action.yaml#L62-L80) | A warm job took 29.1 s for the first validation and 15.7 s for the second. At 15.7 s/job × 72 jobs, the projected setup saving is 1,130 s, equivalent to 2.0% of the 57,008 s reference. | No action step intervenes on successful reuse. Removing the second check would also remove a chance to detect external drift in the shared home. Keep the existing assertion. Not pursued. |
| Cut `OMNI_CI_MAX_ATTEMPTS` from 3 to 2 | [.github/scripts/run_flaky_pytest.sh:5–8](../.github/scripts/run_flaky_pytest.sh#L5-L8) | Seven third attempts with measurable durations used 1,704 s (3.0%). None rescued a job. | Seven observations leave a one-sided 95% upper bound near 35% on the rescue rate. A false red costs a 25–110 min CI round trip on the observed queue. Rejected on this sample. |
| Retry only failed tests (`--lf`) | [.github/scripts/run_flaky_pytest.sh:92–104](../.github/scripts/run_flaky_pytest.sh#L92-L104) and the [`-x` invocation](../.github/workflows/test.yaml#L63-L71) | Would avoid much of the repeated prefix. Achieved saving **not measured**. The entire retry budget was 14.4%. | The suffix after the first failure never ran. A passing isolated retry can turn partial coverage green. Rejected. |
| Resume at the failure (`--sw`) | [.github/workflows/test.yaml:63–71](../.github/workflows/test.yaml#L63-L71) and [scripts/ci/utils/pytest_partition.py:32–64](../scripts/ci/utils/pytest_partition.py#L32-L64) | Would avoid much of the repeated prefix if membership were stable. Achieved saving with correct coverage **not measured**. | A nine-file replay gave a green job after running the wrong partition. Rejected for this plugin. |
| Parallel pytest workers within a lane | [.github/workflows/test.yaml:65–71](../.github/workflows/test.yaml#L65-L71), pytest options | 1,498 s serial, 674 s at four workers (2.22×), 643 s at 16. | Imports repeat per worker and socket path length changes test outcomes. Closed as a lane-rollout strategy. |

The reuse proposal only removed the repeated assertion on a successful warm path. Rebuild and refresh would still require final validation. The gate checks Python 3.12, the image-backed torch/FlashInfer imports, missing dependencies, the recorded deps hash, imports and exact pins. Its 15.7 s cost is in **setup**, outside the retry denominator. [validate_omni_env_reusable.sh:32–85](../.github/scripts/validate_omni_env_reusable.sh#L32-L85)

### Retry time and the value of the third attempt

| Quantity from the 72 jobs | Seconds | Share of Run test | Interpretation |
|---|---:|---:|---|
| Measurable time in attempts 2 and 3 | 8,188 | 14.4% | Includes successful second attempts. |
| Failed first attempts of later-green jobs plus retries of final-red jobs | 7,346 | 12.9% | 1,182 s + 6,164 s. Counts work that did not produce the final green result or change the red result. |
| Retries on jobs that stayed red | 6,164 | 10.8% | A subset of the 8,188 s. |
| Failed first attempts of later-green jobs | 1,182 | 2.1% | Three of 57 green jobs with attempt logs needed a second attempt. None needed a third. |
| Measurable third attempts | 1,704 | 3.0% | Seven timed third attempts. All seven jobs stayed red. |

Each retry incurs about 20–30 s between pytest attempts for GPU cleanup and the default 10 s sleep. The same overhead applies to an early collection error and a late runtime failure. Five jobs reached the 30 min step limit at 1,780–1,800 s. The deadline curtailed third attempts. An attempt with no completion timestamp contributes no duration to the retry numerator, although its time is in the step denominator. **14.4% is a floor.** [run_flaky_pytest.sh:78–86](../.github/scripts/run_flaky_pytest.sh#L78-L86) · [Example timeout](https://github.com/sgl-project/sglang-omni/actions/runs/44228709761/job/134557413106)

Failure position determines the price of `-x`. Among the 11 final-red jobs, the three collection failures each spent 58 s on retries. At the expensive end, one partition failed after 630 s and spent another 580 s on its second attempt alone. A flat cost per failure would miss this spread. [Collection failure](https://github.com/sgl-project/sglang-omni/actions/runs/43820517761/job/133156802147) · [Late failure run](https://github.com/sgl-project/sglang-omni/actions/runs/44973505927)

For zero rescues in seven observations, the exact one-sided 95% binomial upper bound is `1 − 0.05^(1/7) = 34.8%`. Reopen the attempt-limit proposal after counting `passed on attempt 3/3` across several hundred jobs. Count jobs that entered attempt 3, jobs with a recorded outcome, rescues and timeouts separately. An upstream proposal to run unit tests once and continue past the first failure changes both retries and coverage. [Upstream PR #3748](https://github.com/sgl-project/sglang-omni/pull/3748)

### Two different ways to report a false green

**Last-failed (`--lf`).** One real job reported `1 failed, 2501 passed` on each attempt out of roughly 4,300 tests in its partition. About 40% of the partition had never executed. With `-x`, a successful rerun of the one recorded failure can leave the entire suffix untested. Moving that test into isolation can also hide a failure that requires an earlier test's process state. A valid replacement would have to demonstrate that every selected test ran, including the suffix, and preserve the failure signal under the original ordering. [Run test command](../.github/workflows/test.yaml#L63-L71)

**Stepwise (`--sw`).** The partition plugin uses a late collection hook and balances whichever files remain. Stepwise removes the prefix first. In the replay, partition 1 owned B, E and H. Attempt 1 passed B and failed E. Attempt 2 passed F and I, returned 0, never retried E and never reached H. A correct implementation must determine membership from the full collection, then apply its resume state inside that fixed partition. Check node IDs across attempts against the original partition before timing it. [Partition hook](../scripts/ci/utils/pytest_partition.py#L32-L64) · [Replay output](multilane-docker-handoff.md#9-stepwise-retries-can-silently-change-a-lanes-partition)

The replay fails within one job. Each CPU job uses `actions/checkout@v4` with the default `clean: true`, which removes `.pytest_cache` on checkout. These jobs override neither `clean` nor pytest's `cache_dir`. The wrapper starts subsequent attempts without a checkout between them. [test.yaml:47–71](../.github/workflows/test.yaml#L47-L71) · [pytest configuration](../pyproject.toml#L169) · [checkout v4 clean option](https://github.com/actions/checkout/blob/v4/action.yml)

### Exit codes do not identify a collection failure in production

| Invocation or fixture | Exit code on collection error | Evidence |
|---|---:|---|
| Production serial pytest with `-x` | 1 | Three job logs show `import file mismatch` and `failed with exit code 1` on all three attempts. Runtime test failure also exits 1. |
| Same collection error without `-x` | 2 | Replay of the collection error. |
| With xdist | 2 | Worker-based replay. |
| Toy tree containing only failing files | 2 | Local reproduction with different collection behavior. |

In the real tree, the collection error counts toward `maxfail`. Under `-x`, pytest takes its `Failed` path and omits the `Interrupted: N error during collection` line. A classifier based on exit code 2 would miss all three production cases. [Partition A](https://github.com/sgl-project/sglang-omni/actions/runs/43820517761/job/133156802147) · [Partition B](https://github.com/sgl-project/sglang-omni/actions/runs/43820517761/job/133156802174) · [Partition C](https://github.com/sgl-project/sglang-omni/actions/runs/43820517761/job/133156802134)

The terminal pytest summary separated all 11 sampled red jobs. Three had errors and no passes or failures, and eight had runtime failures with passes. Every available attempt of a given job kept the same classification. Anchor a classifier to the pytest totals line. A loose search for `\d+ (passed|failed|error)` also matches the wrapper's own `attempt 3/3 failed` message. [Wrapper log format](../.github/scripts/run_flaky_pytest.sh#L95-L109)

### Why the xdist result stops at four workers

| Configuration on the test node | Wall time | Speedup over serial |
|---|---:|---:|
| Serial | 1,498 s | 1.00× |
| `-n 4` | 674 s | 2.22× |
| `-n 16` | 643 s | 2.33× |

Each worker pays its own import cost. Collection alone took 105 s. The longest `loadscope` group also limits the finish time. Quadrupling workers from four to 16 saved only 31 s. Nine spurious failures were traced to Linux's 107-byte pathname budget for AF_UNIX sockets. A short `--basetemp` reduced the failure count from 11 to two at unchanged wall time. The independent eight-lane stress run combined four workers per lane on a 64-physical-core host and produced 43 verdict divergences. The lane proposal uses serial pytest. [Socket budget in the repository](../tests/utils/tts_mps_runtime.py#L21-L26) · [Stress-run configuration and verdicts](multilane-docker-handoff.md#8-the-eight-lane-stress-run-tested-a-different-execution-shape)

## Deferred

| Candidate | File | Measured effect | Risk and evidence needed |
|---|---|---|---|
| Change-based selection for Omni CI unit tests | [.github/workflows/test.yaml:63–71](../.github/workflows/test.yaml#L63-L71), invoked by [omni-ci.yaml:341–351](../.github/workflows/omni-ci.yaml#L341-L351) | **not measured**. Intel CPU and XPU workflows already use path filters. | Shared code can affect tests outside the changed directory. Replay candidate selections against public full-partition verdicts across commits. Record missed failures, excluded tests and runner seconds saved before changing this path. [Intel CPU filter](../.github/workflows/omni-cpu-ci.yaml#L83) · [Intel XPU filter](../.github/workflows/omni-xpu-ci.yaml#L77) |
