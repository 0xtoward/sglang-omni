# Ascend competition takeover SOP

This is the portable, public-safe recovery contract for the MiniCPM-o 4.5
Ascend competition branch.  It intentionally contains no SSH credentials,
host names, ports, tokens, or private model URLs.

## What this SOP guarantees

- The competition code is anchored by exact Git branches and commits.
- Re-created HiDevLab containers use persistent storage for all irreplaceable
  artifacts.
- The vendor image checkout is treated as evidence, not as a development repo.
- Long-running remote work survives a dropped SSH/browser session.
- A readiness log is never reported as an inference result: every result must
  retain the request, response, environment manifest and timing evidence.

It does **not** claim that MiniCPM-o 4.5 currently runs in SGLang-Omni on
Ascend.  The branch is an integration baseline.

The public-safe `takeover/` directory keeps the read-only audit, locked remote
bootstrap, and environment-manifest generator off the disposable server.  The
local orchestration wrapper and bundle SHA-256 lock live outside the bundle to
avoid a self-referential hash; pass those locked hashes to
`takeover/bootstrap_remote.sh` through `OMNI_BUNDLE_SHA256` and
`CORE_BUNDLE_SHA256`.

## Locked repositories

| Role | Repository | Branch | Required ancestor |
| --- | --- | --- | --- |
| Omni orchestration | `0xtoward/sglang-omni` | `codex/minicpmo45-ascend-competition` | `10e7e54197af9b384af0c9fa5563723f09eb25bf` |
| Image-compatible core | `0xtoward/sglang` | `codex/minicpmo45-ascend-competition-core` | `f308abc05212c2f5f455de22a525e14afa63ee4f` |

Use fast-forward merges on shared branches and never force-push them.

## Persistent layout

The HiDevLab UI may call the shared volume `/user_data`; inside the current
container contract it is mounted at `/workspace/user_data`.  Keep the
recoverable project here:

```text
/workspace/user_data/hidevlab/
├── repos/       # the two competition checkouts
├── models/      # ModelScope snapshots or symlinks plus revision manifests
├── runs/        # one immutable directory per benchmark run
├── logs/        # launch, server and profiler logs
├── artifacts/   # git bundles, traces and compact result packages
├── manifests/   # machine-readable environment snapshots
├── cache/       # disposable but persistent download/build caches
├── tmp/         # quarantine for incomplete restores
├── bin/         # bootstrap and manifest scripts
└── vendor/      # a patch describing the image-provided SGLang overlay
```

The following locations are ephemeral and must not be the only copy of any
work: `/workspace` outside the mounted subdirectory, `/sgl-workspace`, `/tmp`,
and the browser IDE workspace.

Do not use the Gluster mount root itself as a process working directory.  On a
re-created environment its backend can appear as a deleted inode and Git may
fail with `Unable to read current working directory`.  Use `/tmp`, `/root`, or
a real subdirectory under `/workspace/user_data/hidevlab` as `cwd` and refer to
the mount by absolute path.

## First 15 minutes on a replacement server

1. Use the platform's VS Code connection action once so that it writes a fresh
   local SSH alias.  Do not reuse a stale host key or old alias blindly.
2. Run only short, read-only probes over foreground SSH: `hostname`, `pwd`,
   `findmnt`, `npu-smi info`, package versions and Git status.
3. Confirm the effective cgroup limits instead of trusting host-wide `lscpu`
   and `free` output.
4. Confirm `/workspace/user_data` is writable and `/workspace/shared_assets`
   is read-only.
5. Restore both repositories from verified local Git bundles when direct
   GitHub access is slow.  Set `origin` and `upstream` to their canonical URLs
   after cloning the bundles.
6. Capture the image checkout's `git diff --binary`, image commit, CANN/driver,
   Python packages, NPU topology, cgroup limits and mount table before any
   installation.
7. Run repository/unit smoke tests before consuming NPU time.

## Remote execution rule

Any command that may exceed ten seconds, writes files, launches a server, or
uses the NPU must run in `tmux`, or under `nohup` when `tmux` is unavailable.
Record all five fields below immediately:

```text
remote_host=<ssh alias>
remote_cwd=<absolute working directory>
session_or_pid=<tmux session/window or PID>
log=<absolute log path on persistent storage>
launch_command=<exact command>
```

Foreground SSH is only for short read-only checks and log inspection.

## Model download contract

Prefer ModelScope inside mainland-network containers.  Pin and record the
model ID, revision, resolved snapshot path, file hashes and downloaded byte
count.  Download into the persistent `models/` or `cache/` tree with a durable
background job.  A directory name alone is not proof of a complete model.

## Runtime contract

The target image has a vendor-modified editable SGLang checkout under
`/sgl-workspace/sglang`.  Do not `git pull`, reset, or develop directly there.
Capture its patch, then make source changes only in the companion core branch.

SGLang-Omni mainline currently pins a different PyTorch/SGLang combination and
contains CUDA-specific dependencies.  Do not run an ordinary `pip install -e
.` against the target image.  Introduce an explicit Ascend dependency profile
and validate it before mutating the environment.

## Evidence ladder

Advance one gate at a time and retain failure evidence:

1. Environment and import preflight, with zero NPU workload.
2. MiniCPM-o thinker: text input to text output.
3. Image and audio inputs to text output.
4. Hidden-state-to-S3 TTS fixture.
5. Token2wav waveform correctness and first-audio latency.
6. Turn-taking text-and-speech service.
7. Continuous session state, append and barge-in for true full duplex.
8. Throughput/latency sweep without profiler, then a separate profiler run.

Each run directory must contain `manifest.json`, `launch.txt`, server/client
logs, request metadata, raw result data and a short verdict.  Separate
functional correctness, performance and profiling runs so instrumentation does
not contaminate the headline numbers.

## Minimum handoff checklist

Before ending a server session, verify that:

- both repositories are clean or every change is committed and pushed;
- fresh complete Git bundles exist and pass `git bundle verify`;
- model revisions and hashes are recorded;
- all jobs have PID/session, cwd, log and exact command records;
- the current environment manifest and vendor-image patch are persistent;
- all benchmark claims link to raw artifacts;
- no required file exists only on the container overlay or in `/tmp`.
