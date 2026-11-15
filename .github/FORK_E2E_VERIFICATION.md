# Fork advisory bot E2E verification log

Repository: [fengxiaohu/sglang-omni](https://github.com/fengxiaohu/sglang-omni)  
Date: 2026-11-16 (UTC+8)

## Phase A — completed

| Item | Status | Evidence |
|------|--------|----------|
| Fork `main` contains bot + labeler workflows | Done | [`pr-advisory-review.yaml`](workflows/pr-advisory-review.yaml), [`pr-documentation-labeler.yaml`](workflows/pr-documentation-labeler.yaml) on `main` @ `0a1ee1025` |
| `ADVISORY_REVIEW_ENABLED` | Done | `false` (kill-switch baseline; flip to `true` only for Phase C live) |
| `DOCS_REVIEW_MODEL` / `DOCS_REVIEW_BUDGET` | Done | Repository variables present |
| `UNIT_TEST_REVIEW_MODEL` | Done | Copied from pilot variable |
| `docs-review-ledger` branch + JSON | Done | [`docs-review-ledger.json`](https://github.com/fengxiaohu/sglang-omni/blob/docs-review-ledger/docs-review-ledger.json) |
| `DOCS_REVIEW_API_KEY` | **Missing** | Only `DOCS_REVIEW_REPORT_KEY` exists; live model runs need `DOCS_REVIEW_API_KEY` per workflow |

## Phase B — offline (GitHub blocked; local surrogate passed)

### GitHub `workflow_dispatch`

`gh workflow run pr-advisory-review.yaml …` returns:

```text
HTTP 422: Actions has been disabled for this repository.
```

`GET /repos/.../actions/permissions` reports `enabled: true`, but dispatch remains blocked (likely UI-level disable or account policy). Re-enable in **Settings → Actions → General** if workflows should run.

### Local collection-only runs (same entrypoint as CI)

Mirrors `offline=true` dispatch against open probe PRs (`AUTOMATIC_REVIEW=false`, numeric `GITHUB_RUN_ID`):

| Scope | PR | Run ID | Artifact |
|-------|-----|--------|----------|
| `docs` | [#41](https://github.com/fengxiaohu/sglang-omni/pull/41) | `999001` | `/tmp/docs-review/artifact/report.json`, `report.md` |
| `unit_tests` | [#42](https://github.com/fengxiaohu/sglang-omni/pull/42) | `999003` | `/tmp/unit-review-e2e/docs-review/artifact/report.json`, `report.md` |

**Pass criteria met locally:** `report.json` is a one-element array per scope; `report.md` includes collection-only notice; **no** PR comment with `<!-- sglang-omni-advisory-review -->` on #41 (checked via `gh pr view --comments`).

Example local command:

```sh
export GH_TOKEN="$(gh auth token)"
export GITHUB_REPOSITORY=fengxiaohu/sglang-omni GITHUB_WORKSPACE="$PWD"
export GITHUB_RUN_ID=999001 GITHUB_RUN_ATTEMPT=1 PULL_REQUEST_NUMBER=41
export OFFLINE_REVIEW=true AUTOMATIC_REVIEW=false REVIEW_SCOPE=docs
export RUNNER_TEMP=/tmp GITHUB_OUTPUT=/tmp/gh_out GITHUB_SERVER_URL=https://github.com
python3 -m scripts.docs_review.github_pilot
```

After Actions unblock, re-run on GitHub:

```sh
gh workflow run pr-advisory-review.yaml --repo fengxiaohu/sglang-omni \
  -f pull_request_number=41 -f offline=true -f review_scope=docs
gh workflow run pr-advisory-review.yaml --repo fengxiaohu/sglang-omni \
  -f pull_request_number=42 -f offline=true -f review_scope=unit_tests
```

## Phase C — live (pending Actions + API key)

| PR | Branch | Purpose | Prep |
|----|--------|---------|------|
| [#41](https://github.com/fengxiaohu/sglang-omni/pull/41) | `codex/e2e-docs-probe` | Docs scope + `documentation` label | Second push (`7b244c720`) for sticky **update** test |
| [#42](https://github.com/fengxiaohu/sglang-omni/pull/42) | `codex/e2e-unittest-probe` | `unit_tests` scope | Offline artifact validated locally |

**Label logic (static):** #41 changes `docs/e2e/advisory-bot-probe.md` → matches labeler rule → expect `documentation` once labeler runs.

**Live steps when unblocked:**

1. Add secret `DOCS_REVIEW_API_KEY`.
2. `gh variable set ADVISORY_REVIEW_ENABLED --repo fengxiaohu/sglang-omni --body true`
3. Close/reopen #41 or push again → expect **Label documentation pull requests** + **Advisory pull request review** on `pull_request_target`.
4. Confirm sticky marker comment; second #41 push should **PATCH** one comment (concurrency), not add a second bot comment.
5. On #42, sticky should include **### Advisory unit-test contract review**; repeat event with unchanged hash should log `review input unchanged since the last complete run; skipped` (ledger on `docs-review-ledger`).

**Not yet observed on GitHub (blocked):** label on #41/#42, sticky comments, Actions run URLs, artifact downloads from CI.

## Phase C3 — negative probes (created)

| Case | PR | State | Expected when Actions run |
|------|-----|-------|-----------------------------|
| Draft | [#43](https://github.com/fengxiaohu/sglang-omni/pull/43) | Draft, touches `docs/` | Advisory job `if` skips drafts; labeler may still run on open/sync—confirm no unwanted GPU CI |
| Kill switch | (any open PR) | `ADVISORY_REVIEW_ENABLED=false` | Advisory `pull_request_target` job skipped |
| Code only | [#44](https://github.com/fengxiaohu/sglang-omni/pull/44) | Only `sglang_omni/__init__.py` | No `documentation` label; advisory skipped (no in-scope paths) |

## Phase D — upstream

Merge order on `sgl-project/sglang-omni`: #3971 → #3973 → #3976; #3993 in parallel on `main`.  
Maintainers repeat **configuration + C1/C2** with org-owned secrets; keep `ADVISORY_REVIEW_ENABLED=false` until sign-off.

## Checklist

- [x] Fork `main` has workflows + reviewer scripts
- [x] Ledger branch and repository variables (except `DOCS_REVIEW_API_KEY`)
- [x] Probe PRs #41 / #42; #41 second push for sticky update
- [x] Negative PRs #43 (draft) / #44 (code-only)
- [x] Offline collection: local surrogate for docs + unit_tests
- [ ] GitHub Actions dispatch unblocked (owner: enable in Settings)
- [ ] GitHub offline dispatch artifacts (post-unblock)
- [ ] Live `documentation` label + sticky on #41
- [ ] Live unit-test sticky + hash skip on #42
- [ ] Negative runs confirmed in Actions UI

### Evidence links (fill after GitHub runs)

- Labeler run:
- Advisory run (docs):
- Advisory run (unit_tests):
- Sticky comment URL:
- Artifact run IDs: