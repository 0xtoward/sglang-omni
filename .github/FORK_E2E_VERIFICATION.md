# Fork advisory bot E2E verification log

Repository: [fengxiaohu/sglang-omni](https://github.com/fengxiaohu/sglang-omni)
Date: 2026-11-16 (UTC+8)

`ADVISORY_REVIEW_ENABLED` is **false** again after the kill-switch check.

## Phase A

| Item | Evidence |
|------|----------|
| Workflows on default `main` | `pr-advisory-review.yaml`, `pr-documentation-labeler.yaml` |
| Variables | `DOCS_REVIEW_MODEL`, `DOCS_REVIEW_BUDGET`, `UNIT_TEST_REVIEW_MODEL` |
| Ledger | [`docs-review-ledger`](https://github.com/fengxiaohu/sglang-omni/blob/docs-review-ledger/docs-review-ledger.json) |
| Secret `DOCS_REVIEW_API_KEY` | Set from the local docs-review keychain item. A direct call to the configured endpoint returned **HTTP 401** (`invalid API key, or the requested model is not available to this key`). |

## Phase B — offline dispatch

| Scope | PR | Run | Result |
|-------|----|-----|--------|
| docs | [#41](https://github.com/fengxiaohu/sglang-omni/pull/41) | [43745789910](https://github.com/fengxiaohu/sglang-omni/actions/runs/43745789910) | success; artifact `docs-review-43745789910-1` is a one-element `docs` array; collection-only `report.md` |
| unit_tests | [#42](https://github.com/fengxiaohu/sglang-omni/pull/42) | [43745805350](https://github.com/fengxiaohu/sglang-omni/actions/runs/43745805350) | success; artifact `docs-review-43745805350-1` is `unit_tests` |

Neither pull request gained a comment containing `<!-- sglang-omni-advisory-review -->`.

## Phase C — live

### Documentation probe [#41](https://github.com/fengxiaohu/sglang-omni/pull/41)

- Labeler [43746101603](https://github.com/fengxiaohu/sglang-omni/actions/runs/43746101603) succeeded. The pull request has the `documentation` label.
- Omni CI for that push is a single `pull_request` run ([43746103767](https://github.com/fengxiaohu/sglang-omni/actions/runs/43746103767)). No second Omni CI appeared after `addLabels`. `omni-ci.yaml` listens for `labeled`, and the labeler uses `GITHUB_TOKEN`, which does not start that workflow.
- Advisory [43746101504](https://github.com/fengxiaohu/sglang-omni/actions/runs/43746101504) uploaded an artifact and then failed the incomplete-review step. `report.json` says `execution=incomplete`, `incomplete_reason=model request failed`. No sticky comment was posted, so the follow-up push could not show an in-place comment update.
- Ledger reserved that run as `execution: incomplete` (50 cents estimated, not a complete review).

### Unit-test probe [#42](https://github.com/fengxiaohu/sglang-omni/pull/42)

- First live run [43746125347](https://github.com/fengxiaohu/sglang-omni/actions/runs/43746125347): incomplete, `No contract source was collected`.
- After the test referenced `sglang_omni.pipeline.coordinator`, run [43746424991](https://github.com/fengxiaohu/sglang-omni/actions/runs/43746424991) recorded the oversized-module limitation in the artifact:

```text
sglang_omni/pipeline/coordinator.py: contract source exceeds the collection budget (38099 bytes, 36000 bytes left)
```

The report stayed incomplete, so the sticky unit-test section was not posted. Hash skip (`review input unchanged since the last complete run`) needs a prior **complete** reservation; this ledger entry is incomplete, so that skip was not exercised.

## Negative cases

| Case | PR | Run | Observed |
|------|----|-----|----------|
| Draft | [#43](https://github.com/fengxiaohu/sglang-omni/pull/43) | advisory [43746440505](https://github.com/fengxiaohu/sglang-omni/actions/runs/43746440505) **skipped** | Advisory does not run on drafts. The labeler still added `documentation` on `synchronize` (it has no draft filter). |
| Code only | [#44](https://github.com/fengxiaohu/sglang-omni/pull/44) | [43746447446](https://github.com/fengxiaohu/sglang-omni/actions/runs/43746447446) success | Log: `No in-scope changes; advisory review skipped.` No `documentation` label. |
| Kill switch | [#41](https://github.com/fengxiaohu/sglang-omni/pull/41) with `ADVISORY_REVIEW_ENABLED=false` | [43746526181](https://github.com/fengxiaohu/sglang-omni/actions/runs/43746526181) **skipped** | Same skip as the earlier disabled run [43745513734](https://github.com/fengxiaohu/sglang-omni/actions/runs/43745513734). |

## Checklist

- [x] Fork Actions allowed; default branch has both workflows and reviewer scripts
- [x] Offline docs + unit_tests dispatch, artifacts present, no sticky comment
- [x] Docs probe received `documentation`; labeling did not start a second Omni CI
- [ ] Sticky comment and in-place update (blocked: model endpoint HTTP 401)
- [x] Unit-test oversized contract limitation is in the artifact
- [ ] Unit-test sticky section and hash skip (blocked: no complete model run)
- [x] Draft advisory skipped; code-only skipped with no documentation label; kill switch skips the job

## Phase D — upstream

Merge #3971, then #3973, then #3976. #3993 can land in parallel. On `sgl-project/sglang-omni`, repeat configuration and the two probe pull requests with a maintainer-owned `DOCS_REVIEW_API_KEY`. Leave `ADVISORY_REVIEW_ENABLED=false` until that key is accepted by the model endpoint and a maintainer signs off.

Replace the fork secret before any further live run. The key currently stored returns HTTP 401.
