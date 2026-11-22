# Separate CPU runner lanes on the H100 hosts

Register one GitHub Actions runner per lane. Keep one GPU-capable runner per machine with access to both cards. Give the serial CPU unit-test partitions their own runners, a separate scheduling label and disjoint CPU masks. A listener can run in its own container. The workflow's existing `container:` starts a separate container for each job.

Each of the two H100 hosts currently registers one agent and executes one job at a time. Omni CI wall time grew from 770 s on 2026-10-10 to 3,191 s on 2026-11-22 (4.1×). Two weeks of self-hosted job durations give the following shares. During GPU tests, measured GPU utilization was 7–13%.

| Work | Share of self-hosted machine time |
|---|---:|
| CPU unit tests | 59.6% |
| GPU unit tests | 20.7% |
| Model suites | 16.0% |
| Environment preparation | 3.8% |

The workload inventory counted 29 job definitions targeting `[self-hosted, h100]`. Four GPU-free definitions expanded to six jobs. The other 25 used the two-card pool. At the source revision cited here, `dfb6b43b3`, there are 27 literal `runs-on: [self-hosted, h100]` definitions, 29 jobs after expanding the CPU matrix, and the same six GPU-free jobs. Those six are `unit-test-a/b/c`, `setup - omni venv`, dispatch cleanup and PR-close cleanup. Only the CPU partitions need to move to free most of the occupied runner time. The GPU lane must own the pair because `stage-10-videoamme-talker-tp2` runs tensor parallelism across two cards. [test.yaml:24–38](../.github/workflows/test.yaml#L24-L38) · [omni-ci.yaml:150–165](../.github/workflows/omni-ci.yaml#L150-L165) · [omni-ci.yaml:354–372](../.github/workflows/omni-ci.yaml#L354-L372) · [cleanup-pr-ci-home-on-close.yaml:11–17](../.github/workflows/cleanup-pr-ci-home-on-close.yaml#L11-L17) · [test-qwen3-omni-ci.yaml:444–475](../.github/workflows/test-qwen3-omni-ci.yaml#L444-L475)

## What already runs in production

| Component | Existing behavior |
|---|---|
| Affinity | Each listener reads its own `.env` at startup. Restart that listener after changing `OMNI_CI_CPUSET`. The job container forwards the variable. The sourced script pins the calling shell with `taskset -pc`. Children inherit its mask. Workflow container options cannot interpolate the runner-local value into `--cpuset-cpus`. [pin_to_ci_cpuset.sh:1–17](../.github/scripts/pin_to_ci_cpuset.sh#L1-L17) · [test.yaml:39–46](../.github/workflows/test.yaml#L39-L46) · [run_flaky_pytest.sh:5](../.github/scripts/run_flaky_pytest.sh#L5) |
| Image | Both unit-test jobs use `hongccc/sglang-omni@sha256:d2fbd80ec42c8a6d9dff4b8a4fdbee35d50c5a1b41aa8dc872925da6e0c82f6a`. A cold pull of this digest took 146 s for 76 layers, 18.93 GB compressed and 45.6 GB on disk. [test.yaml:37–46](../.github/workflows/test.yaml#L37-L46) · [test.yaml:73–85](../.github/workflows/test.yaml#L73-L85) |
| Environment | `omni-setup` defaults to `install-deps: false`. Each checkout symlinks its venv to the prepared copy under `OMNI_CI_HOME`. The reuse gate checks the interpreter, image packages, requirements, imports, exact pins and the stored fingerprint when present. The fingerprint covers `pyproject.toml` and `prepare_omni_venv.sh`. [omni-setup/action.yaml:8–17](../.github/actions/omni-setup/action.yaml#L8-L17) · [omni-setup/action.yaml:43–80](../.github/actions/omni-setup/action.yaml#L43-L80) · [validate_omni_env_reusable.sh:27–85](../.github/scripts/validate_omni_env_reusable.sh#L27-L85) · [omni_ci_deps_hash.sh:5–11](../.github/scripts/omni_ci_deps_hash.sh#L5-L11) |
| Persistent state | PRs use `/data/omni-ci/pr-<N>` and manual runs use a run ID. The path validator rejects traversal and accepts only the listed PR, run and calibration paths. [omni-ci.yaml:54–58](../.github/workflows/omni-ci.yaml#L54-L58) · [validate_omni_ci_home.sh:9–22](../.github/scripts/validate_omni_ci_home.sh#L9-L22) |
| Scheduling | The existing pools include `h100`, `intel-omni-spr` and `intel-omni-bmg`. Omni CI, Intel CPU CI, Intel XPU CI, lint and docs-check already set `cancel-in-progress: true`. [omni-ci.yaml:45–47](../.github/workflows/omni-ci.yaml#L45-L47) · [omni-cpu-ci.yaml:49](../.github/workflows/omni-cpu-ci.yaml#L49) · [CPU pool:175](../.github/workflows/omni-cpu-ci.yaml#L175) · [omni-xpu-ci.yaml:60](../.github/workflows/omni-xpu-ci.yaml#L60) · [XPU pool:182](../.github/workflows/omni-xpu-ci.yaml#L182) · [lint.yaml:15](../.github/workflows/lint.yaml#L15) · [docs-check.yaml:20](../.github/workflows/docs-check.yaml#L20) |

Change-based test selection exists in Intel CPU and XPU CI and is out of scope for the Omni CI unit-test path. [omni-cpu-ci.yaml:83](../.github/workflows/omni-cpu-ci.yaml#L83) · [omni-xpu-ci.yaml:77](../.github/workflows/omni-xpu-ci.yaml#L77) · [test.yaml:63–71](../.github/workflows/test.yaml#L63-L71)

## CPU budget and measured partition times

The production masks and load readings come from public `[cpuset-contention]` lines. On both hosts, logical CPU *n* and *n* + 64 share a physical core.

| Host | Production `OMNI_CI_CPUSET` | Logical CPUs | Physical cores | Foreign load, mean / peak |
|---|---|---:|---:|---|
| [A](https://github.com/sgl-project/sglang-omni/actions/runs/44973505927/job/137059189098) | `16-31,80-95` | 32 | 16 | 3.21 / 6.24 core equivalents |
| [B](https://github.com/sgl-project/sglang-omni/actions/runs/44228709761/job/134557413106) | `2-15,66-79` | 28 | 14 | 1.30 / 2.19 core equivalents |

The sampler sums busy ticks across the logical CPUs in the mask and subtracts ticks charged to the pytest process tree. Its mean cannot be subtracted from the physical core count. It cannot identify how many physical cores carried the foreign work or whether that work shared an SMT sibling with a test. Unallocated cores give the scheduler headroom. Affinity grants no reservation. [tests/utils/ci_cpu_contention.py:55–67](../tests/utils/ci_cpu_contention.py#L55-L67) · [tree accounting:104–129](../tests/utils/ci_cpu_contention.py#L104-L129) · [sampling and log format:181–209](../tests/utils/ci_cpu_contention.py#L181-L209)

A test host configured to host A's cpuset shape ran three CPU-only partitions. Each lane had two physical cores, four logical CPUs and one serial pytest process. The run used the pinned image, the production marker expression, `-x`, the duration-weighted partition file and `run_flaky_pytest.sh`. The GPU lane was idle in this timing arm. The CPU containers exposed the NVIDIA CLI and a device interface, with CUDA hidden from pytest. [test.yaml:55–71](../.github/workflows/test.yaml#L55-L71) · [pytest_partition.py:32–64](../scripts/ci/utils/pytest_partition.py#L32-L64)

| Partition | Upstream verdict | Lane verdict | Lane time | Retries |
|---|---|---|---:|---:|
| 0 | Pass | Pass | 938 s | 0 |
| 1 | Pass | Pass | 838 s | 0 |
| 2 | Pass | Pass | 915 s | 0 |

Partition 1 contains the child-ownership test that failed 19 times under heavier load. It passed on the first attempt here.

The matched comparison ran on that same test host, holding hardware constant.

| Configuration | Wall clock |
|---|---:|
| One agent holding all 16 physical cores, three partitions in sequence | 2,599 s |
| Three CPU lanes of two physical cores each, concurrent | **1,012 s** |

That is **2.57x**. Moving one partition from 16 physical cores to two cost **+3.6%** on the same host. Serial pytest schedules one test at a time, so a wide mask sits mostly idle: the penalty for narrowing a lane is an order of magnitude smaller than the gain from running three lanes at once. This is the economic case for the proposition.

The same comparison through registered runner agents, rather than containers driven by hand, gives **2.84x**: 1,066 s for three lanes against 3,026 s for one wide agent, on the same machine and the same merge commit, with every job green on its first attempt. It lands higher than the container-direct figure because a single agent also pays the gap between jobs — 42 s and 117 s here — which three lanes do not. Per partition the two widths are nearly identical: 982, 995 and 890 s on sixteen physical cores against 879 to 1,066 s on two, which is the single-threaded ceiling showing up from the other direction.

Do not derive the narrowing penalty by comparing a lane time against upstream's own duration for the same partition. That is a cross-machine comparison, and it put the penalty at +26% to +37% instead of +3.6%.

## Rollout pitfalls

### 1. GPU cleanup can fail a CPU job before pytest starts

**Hard prerequisite, not yet filed.** Fix the CPU job's cleanup path before enabling additional lanes. A CPU partition sets `CUDA_VISIBLE_DEVICES: ""` to hide CUDA from pytest. Today that value applies only to the **Run test** step. The preceding `omni-setup` step also runs GPU cleanup. The script copies `${CUDA_VISIBLE_DEVICES:-}` into `target_gpu_ids`, unsets the original variable and skips its nonempty-ID branch for an empty value. With `GITHUB_ACTIONS=true`, the next branch enumerates every card visible to `nvidia-smi`. The script documents unscoped cleanup as a single-tenant operation. [test.yaml:47–67](../.github/workflows/test.yaml#L47-L67) · [delete_gpu_process.sh:4–41](../.github/scripts/delete_gpu_process.sh#L4-L41) · [omni-setup/action.yaml:88–91](../.github/actions/omni-setup/action.yaml#L88-L91)

On the test host, a CPU-lane container with a private PID namespace, explicitly empty `CUDA_VISIBLE_DEVICES` and `GITHUB_ACTIONS=true` ran cleanup while a neighboring process held 2,913 MiB on the card.

| Probe | Observed result |
|---|---|
| Device enumeration | `nvidia-smi --query-gpu=index` returned `0`. |
| Selection | `=== GPU cleanup scope (physical): 0 (input CVD='') ===` |
| Orphan sweep | `orphan kill UNSCOPED (single-tenant / explicit allow)` |
| Processes killed | None. The script printed `No processes found on GPU 0.` |
| Memory wait | 17 polls reported 2,913 MiB. The wait for less than 1,024 MiB timed out. |
| Cleanup with orphan sweep | 124 s wall clock. |

The compute-apps query returned no PIDs inside this container. The orphan sweep could see only its own PID namespace through `/proc`. The neighboring process survived, but its device-wide memory use remained visible. Holding memory above the threshold throughout the deadline deterministically fails cleanup in this configuration. [delete_gpu_process.sh:64–83](../.github/scripts/delete_gpu_process.sh#L64-L83) · [process query:153–170](../.github/scripts/delete_gpu_process.sh#L153-L170) · [memory wait:182–209](../.github/scripts/delete_gpu_process.sh#L182-L209)

Between attempts the wrapper abandoned the retry. [run_flaky_pytest.sh:78–86](../.github/scripts/run_flaky_pytest.sh#L78-L86) · [failure propagation:111–119](../.github/scripts/run_flaky_pytest.sh#L111-L119)

```text
::error::GPU cleanup failed before retry; not retrying with dirty GPU state
```

The setup probe reproduced the earlier failure point. The composite action invokes the script directly, with neither `|| true` nor `continue-on-error`. No test started. [omni-setup/action.yaml:88–91](../.github/actions/omni-setup/action.yaml#L88-L91)

```text
wall=122s
::error::Timed out waiting for GPU memory.used < 1024 MiB; max memory.used=2913 MiB.
SETUP_STEP_RC=1
```

Set `CUDA_VISIBLE_DEVICES: ""` at **CPU job scope** so setup and retries both receive it. In the script, distinguish explicitly empty from unset before capturing or unsetting the variable. Explicitly empty with no override should exit 0 before enumeration, orphan cleanup or polling. Preserve explicit `all`, the `OMNI_CI_ALLOW_UNSCOPED_GPU_CLEAN=1` override, explicit device IDs and existing GPU jobs that leave the variable unset. Repeat both probes with a live neighbor. The CPU setup and retry must proceed immediately, and the neighbor must keep running.

### 2. Correct affinity can fail its readback check

The tested Linux hosts produced these `taskset -pc` readback forms.

| Consecutive logical CPUs | Readback form |
|---|---|
| One | `a` |
| Exactly two | `a,b` |
| Three or more | `a-b` |

Main compares strings. It successfully pins `24-25,88-89`, reads back `24,25,88,89`, then exits 2. The production masks have long runs and avoided this mismatch. Use `24,25,88,89` on the cited main revision. The fix shipped on fork branch `qun/ci-lane-prereqs` (commit `204cc7568`): it expands both specifications to sorted CPU IDs. Either form then works, and a genuinely narrowed mask still fails. The pinning operation itself needs no change. [pin_to_ci_cpuset.sh:5–15](../.github/scripts/pin_to_ci_cpuset.sh#L5-L15) · [CPU-ID comparison](https://github.com/0xtoward/sglang-omni/commit/204cc7568)

### 3. Rootless Podman can lack a delegated cpuset controller

The rootless Podman test rejected `--cpuset-cpus`: its cgroup had no delegated cpuset controller. Source `pin_to_ci_cpuset.sh` inside the job as production does. Check `taskset -pc $$` in the sourcing shell and `os.sched_getaffinity(0)` in its Python child. This establishes process affinity even on a host without container-level cpuset delegation. [pin_to_ci_cpuset.sh:1–15](../.github/scripts/pin_to_ci_cpuset.sh#L1-L15) · [run_flaky_pytest.sh:5](../.github/scripts/run_flaky_pytest.sh#L5)

### 4. The venv lock covers builders only

Three lanes started against the same empty CI home. The per-home `flock` lives under the shared uv cache. After acquiring it, each builder validates the venv again. The later lanes logged that another job had prepared the venv. There was one rebuild and roughly 43 s per place in the queue. [prepare_omni_venv.sh:19–43](../.github/scripts/prepare_omni_venv.sh#L19-L43)

| Cold-start lane | Completion time from common start | Exit code |
|---|---:|---:|
| Builder | 43 s | 0 |
| First waiter | 86 s | 0 |
| Second waiter | 129 s | 0 |

Code inspection identifies three ways a reader can see a changing home. A fingerprint mismatch triggers `rm -rf "${OMNI_CI_HOME}"` during rebuild. Successful reconciliation runs `uv pip install --upgrade --no-deps -e .` outside the builder lock. PR-close cleanup deletes the home without waiting for pytest. The SGLang, JIT, uv and Hugging Face caches sit outside this per-PR directory. A running pytest process holds no read lock. [prepare_omni_venv.sh:45–55](../.github/scripts/prepare_omni_venv.sh#L45-L55) · [reconcile_omni_ci_env.sh:28–57](../.github/scripts/reconcile_omni_ci_env.sh#L28-L57) · [cleanup_ci_pr_home.sh:18–37](../.github/scripts/cleanup_ci_pr_home.sh#L18-L37) · [omni-setup/action.yaml:19–41](../.github/actions/omni-setup/action.yaml#L19-L41) · [test.yaml:43–46](../.github/workflows/test.yaml#L43-L46)

A second lane forced a deps-hash mismatch while the first was mid-pytest. The victim finished with 4,363 passed and zero failed. A tight spawn probe completed 118 of 118 launches. At 50 ms sampling intervals the venv was absent in **1 of 1,043 samples**, or one observed 0.05 s sampling window. The interpreter behind `omni/bin/python` and the installed torch package live in the image. A fresh child still resolves the venv path at launch. A spawn deliberately timed inside that gap is **not measured**. Pause the rebuilder after removal and launch the venv interpreter from the reader to exercise that exact interleaving. Concurrent first-time publication of a shared JIT entry is also **not measured**. Start two cold users of the same kernel key and load the resulting entry from both. [prepare_omni_venv.sh:47–52](../.github/scripts/prepare_omni_venv.sh#L47-L52) · [validate_omni_env_reusable.sh:38–62](../.github/scripts/validate_omni_env_reusable.sh#L38-L62)

### 5. CPU partitions share host POSIX shared memory

The CPU job bind-mounts host `/dev/shm`. CPU relay tests create, access and unlink POSIX shared-memory objects. A reused name could collide across lanes. Cross-lane collisions are **not measured**. Run two lanes through the shared-memory tests with the host bind in place. Drop the host bind from the CPU partitions and give each job a private `--shm-size` sized from peak use. Keep the GPU jobs' IPC configuration separate. [test.yaml:37–46](../.github/workflows/test.yaml#L37-L46) · [test_shm_relay.py:73–92](../tests/unit_test/relay/test_shm_relay.py#L73-L92)

### 6. Host B has room for two tested CPU lanes

Start host A with **one GPU + three CPU agents**. Start host B with **one GPU + two CPU agents**, and pilot its third. Allocate eight physical cores on A and seven on B to GPU data loading and replica processes. With two cores per host set aside for headroom, A has six cores for a 2/2/2 CPU split. B has five for a 2/2/1 split. The one-core lane is no longer a guess: it cost 3.1% against two cores in a paired run, so enable it as a pilot rather than leaving it off, and watch host B's flake rate, which is the part still unmeasured.

| Allocation | Host A logical CPU IDs | Physical cores | Host B logical CPU IDs | Physical cores |
|---|---|---:|---|---:|
| GPU lane, both cards | `16,17,18,19,20,21,22,23,80,81,82,83,84,85,86,87` | 8 | `2,3,4,5,6,7,8,66,67,68,69,70,71,72` | 7 |
| CPU lane 1 | `24,25,88,89` | 2 | `9,10,73,74` | 2 |
| CPU lane 2 | `26,27,90,91` | 2 | `11,12,75,76` | 2 |
| CPU lane 3 | `28,29,92,93` | 2 | Disabled (candidate `13,77`) | 0 |
| Unallocated at rollout | `30,31,94,95` | 2 | `13,14,15,77,78,79` | 3 |

Each row keeps SMT siblings together. On unpatched main, enter the long GPU masks as `16-23,80-87` and `2-8,66-72`. The owner must confirm CPU/core/NUMA numbering and GPU PCI locality on each host, and this is not advisory: copying host A's GPU mask verbatim onto a machine whose card sits on the other NUMA node fails outright with

```
MPS DP2 requires 4 allowed CPU cores on the GPU NUMA node, found 0
```

MPS DP2 refuses rather than degrading, so a GPU lane's cpuset has to be chosen for NUMA adjacency to the card and not for width. An equal-width remap onto the correct node produced working replica blocks, with the second block drawing the other SMT threads of four cores the first block already used — so replica blocks can overlap SMT siblings, which the table above does not model. MPS DP2 derives replica blocks from the intersection of the GPU's NUMA CPUs and `sched_getaffinity(0)`. Its minimum is four allowed logical CPU IDs. MPS job options at this revision forward `NVIDIA_VISIBLE_DEVICES` but omit `OMNI_CI_CPUSET`. Verify the actual GPU job's affinity and enforce the mask in the job container before measuring DP2 with the CPU lanes busy. [tts_mps_runtime.py:21–27](../tests/utils/tts_mps_runtime.py#L21-L27) · [NUMA intersection:158–177](../tests/utils/tts_mps_runtime.py#L158-L177) · [test-mps-ci.yaml:41–52](../.github/workflows/test-mps-ci.yaml#L41-L52)

### 7. A listener container and its job container are siblings

With the host Docker socket, the host daemon creates the job container. Bind sources resolve on the host. The listener's device limits, cpuset, private temporary filesystem and cgroup limits do not propagate to that sibling. A separate `_work` and `RUNNER_TEMP` per listener prevent workspace conflicts. Host-level commands using Python's default temporary directory can still share the host directory. The existing workflow `container:` supplies the per-job filesystem. [test.yaml:37–51](../.github/workflows/test.yaml#L37-L51) · [omni-ci.yaml:18–26](../.github/workflows/omni-ci.yaml#L18-L26)

The simplest initial deployment is multiple independent listener services on each machine, retaining the existing job containers. A container per listener also works if workspace mounts resolve at the same paths on the daemon host. Accept either topology with one real `container:` job, paired with a busy GPU job. Inspect CPU affinity, NVIDIA device access, PID and network namespaces, temporary files, shared-memory mounts and effective cgroup limits **inside the jobs**. The CPU partition must pass setup and exercise the MPS launch-script test with the intended device restrictions. A full partition with all GPU device nodes hidden is **not measured**.

### 8. The eight-lane stress run tested a different execution shape

A 64-physical-core test node ran eight lanes with eight physical cores each and `pytest -n 4` per lane. That launched 32 workers plus their spawned children. Production uses one serial job per agent. Across 72 partition comparisons, upstream passed and the stress run failed 43 times. The reverse happened zero times.

| Upstream history for the 43 divergent cases | Cases |
|---|---:|
| Passed on first attempt | 39 |
| Passed after a retry | 1 |
| Attempt log unavailable | 3 |

| Failure concentration | Mechanism |
|---|---|
| Native child ownership | A spawned child must finish within `process.join(timeout=15)`. [test_native_child_processes.py:49](../tests/unit_test/pipeline/test_native_child_processes.py#L49) |
| Worker hard-exit handling | Two child ready events each have a 15 s wait. [test_process_failure.py:196–201](../tests/unit_test/pipeline/test_process_failure.py#L196-L201) · [waits:254–255](../tests/unit_test/pipeline/test_process_failure.py#L254-L255) |
| MPS launch validation | The subprocess has a 120 s deadline. A missing `nvidia-smi` binary on the stress host also made the launcher exit 127 before the assertion it was meant to exercise. [test_mps_dp_support.py:378–392](../tests/unit_test/serve/test_mps_dp_support.py#L378-L392) · [launch.sh:120–128](../examples/mps_dp/launch.sh#L120-L128) |

All three tests passed at the two-physical-core serial lane width. The 43 divergences measure excess concurrent worker load and a missing test-host dependency. Use the serial partition results above to size lanes.

### 9. Stepwise retries can silently change a lane's partition

The partition plugin runs `pytest_collection_modifyitems(trylast=True)`. It builds the file list from the items left at that point. On retry, `--sw` truncates the list first, so the same partition ID selects different files. A replay used nine equal-weight files. Partition 1 owned B, E and H. Both E and H contained failing tests. [pytest_partition.py:32–64](../scripts/ci/utils/pytest_partition.py#L32-L64) · [run_flaky_pytest.sh:92–105](../.github/scripts/run_flaky_pytest.sh#L92-L105)

```text
attempt 1:  stepwise: no previously failed tests, not skipping.
            test_b.py PASSED   test_e.py FAILED        exit 2
attempt 2:  stepwise: skipping 4 already passed items
            test_f.py PASSED   test_i.py PASSED
            ::notice::pytest stage passed on attempt 2/3
            wrapper exit=0
```

E never ran again. H never ran at all. Keep the production command for lane rollout. Breakpoint resumption requires fixing partition membership before applying the breakpoint within that partition.

## Who does what

| Owner | Deliverable | Acceptance |
|---|---|---|
| Repo reviewer, first | Land the explicitly empty GPU cleanup fix and set the CPU job's environment at job scope. | With a neighbor holding GPU memory, setup exits 0, the retry starts and cleanup never enumerates the neighbor's devices. |
| Repo reviewer, routing | Route only `unit-test` through the proposed `OMNI_CI_CPU_RUNS_ON` repository variable, defaulting to the current labels. Give CPU jobs private shared memory sized from a run. | The routing PR can land with the variable absent. Set it to `["self-hosted","omni-cpu-lane"]` when agents are ready. Remove it to send newly scheduled runs back to `h100`. |
| Repo reviewer, affinity | Use canonical mask spellings or land the CPU-ID readback fix from `qun/ci-lane-prereqs`. Check the GPU workflow forwards or enforces the GPU lane mask. | Sourced shell and child have the requested IDs, including both SMT siblings. No structural change to `pin_to_ci_cpuset.sh` is needed. |
| Machine owner | Register independent identities with separate listener directories, `_work`, `RUNNER_TEMP` and local `.env` masks. Keep `h100` only on the GPU agents. Give CPU agents `omni-cpu-lane`. Restart listeners after mask changes. | Job logs identify different runners serving concurrent CPU partitions. Accelerator and model jobs land only on the two-card agents. |
| Machine owner | Expose both cards only to the GPU job container. Configure job mounts, NUMA placement, cgroups and Docker daemon storage. Retain the shared data paths required by the workflow. | One real CPU job completes beside a GPU job. Check DP2 throughput and the contention log under that mixed load. |

The proposed CPU job’s `runs-on` expression is `${{ fromJSON(vars.OMNI_CI_CPU_RUNS_ON || '["self-hosted","h100"]') }}`.

## Open measurements

| Question | Status | Measurement that settles it |
|---|---|---|
| Speedup through real runner agents | **measured: 2.84x** | Three registered agents, each listener in its own container, against one wide agent on the same host and the same merge commit, both through Actions. Narrow span 1,066 s with a 879 s three-way overlap; wide span 3,026 s with the partitions strictly sequential at 982, 995 and 890 s. Removing the wide arm's two 42 s and 117 s replacement gaps still leaves 2.69x. Runs [45058163756](https://github.com/0xtoward/sglang-omni-ci-sandbox/actions/runs/45058163756) and [45060106617](https://github.com/0xtoward/sglang-omni-ci-sandbox/actions/runs/45060106617). |
| Host shared-memory collisions | **measured: none observed** | Two lanes ran the seven relay tests concurrently, 14 of 14 both with the host bind and with a private 64 MiB mount. Each pair created 18 distinctly named objects with zero open or unlink failures, and the private mount peaked at 73,728 bytes. That peak is this test file's, not a general figure. |
| One-core lane on B | **measured: +3.1%, worth piloting** | The full timeout-sensitive partition took 931 s on `13,77` against 903 s on two cores, both first attempt, RC 0, zero retries, each with the other two partitions running alongside throughout. `child_ownership[True]`, the test that timed out upstream, passed in both arms. This is one pairing on a lease machine; host B's long-run flake rate remains unmeasured. |
| GPU lane under concurrent CPU load | not measured | Run MPS DP2 and a representative GPU stage with CPU lanes idle and busy. Compare replica core blocks, dataloading time, performance gates and foreign-load samples. |
| Image cost with one daemon per lane | not measured | Compare a shared daemon and independent empty daemon stores on the same host. Record pull time, transferred bytes and on-disk layer bytes after each lane's first real job. A shared daemon can reuse the digest's layers. |
