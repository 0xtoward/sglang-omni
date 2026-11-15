# Fork advisory bot E2E verification log

Repository: [fengxiaohu/sglang-omni](https://github.com/fengxiaohu/sglang-omni)  
Date: 2026-11-16 (UTC)

## Phase A — completed

| Item | Status | Evidence |
|------|--------|----------|
| Fork `main` contains bot + labeler workflows | Done | [`pr-advisory-review.yaml`](workflows/pr-advisory-review.yaml), [`pr-documentation-labeler.yaml`](workflows/pr-documentation-labeler.yaml) on `main` @ `d364e988f` + labeler commit |
| `ADVISORY_REVIEW_ENABLED` | Done | `false` for Phase B (offline / kill-switch baseline) |
| `DOCS_REVIEW_MODEL` / `DOCS_REVIEW_BUDGET` | Done | Already set on fork |
| `UNIT_TEST_REVIEW_MODEL` | Done | Copied from `UNIT_TEST_REVIEW_PILOT_MODEL` |
| `docs-review-ledger` branch + JSON | Done | [`docs-review-ledger.json`](https://github.com/fengxiaohu/sglang-omni/blob/docs-review-ledger/docs-review-ledger.json) |

## Phase B–C — blocked (Actions disabled on fork)

`gh workflow run` returns:

```text
HTTP 422: Actions has been disabled for this repository.
```

Probe pull requests are open and ready once Actions are enabled:

| PR | Purpose |
|----|---------|
| [#41](https://github.com/fengxiaohu/sglang-omni/pull/41) | Docs path → `documentation` label + docs advisory scope |
| [#42](https://github.com/fengxiaohu/sglang-omni/pull/42) | `tests/unit_test/` change → unit_tests scope |

**Unblock (maintainer, one time):** GitHub → **Settings → Actions → General** → allow Actions for this repository (e.g. “Allow all actions”). Then re-run the commands below.

### After unblock — Phase B offline

```sh
gh workflow run pr-advisory-review.yaml --repo fengxiaohu/sglang-omni \
  -f pull_request_number=41 -f offline=true -f review_scope=docs
gh workflow run pr-advisory-review.yaml --repo fengxiaohu/sglang-omni \
  -f pull_request_number=42 -f offline=true -f review_scope=unit_tests
```

Expect: run success, artifact `docs-review-*`, **no** PR comment containing `<!-- sglang-omni-advisory-review -->`.

### After unblock — Phase C live

1. Ensure secret `DOCS_REVIEW_API_KEY` exists (model vars reference it).
2. `gh variable set ADVISORY_REVIEW_ENABLED --repo fengxiaohu/sglang-omni --body true`
3. Push to PR #41 or close/reopen to trigger `pull_request_target`.
4. Confirm label `documentation` on #41 and sticky advisory comment; push again to confirm comment **updates** not duplicates.

### Negative cases (after unblock)

| Case | How |
|------|-----|
| Kill switch | `ADVISORY_REVIEW_ENABLED=false`, push to open PR → advisory job skipped |
| Draft | Open a **draft** PR touching `docs/` → no automatic advisory run |
| Code only | PR changing only `sglang_omni/` → no `documentation` label, advisory skipped |

## Phase D — upstream

Merge order: #3971 → #3973 → #3976; #3993 can land in parallel.  
Repeat configuration and C1/C2 on `sgl-project/sglang-omni` with maintainer-owned secrets. Keep `ADVISORY_REVIEW_ENABLED=false` until sign-off.

## Checklist (copy for sign-off)

- [x] Fork `main` has workflows + reviewer scripts
- [x] Ledger branch and repository variables
- [x] Probe PRs #41 / #42 created
- [ ] Actions enabled on fork (manual)
- [ ] Offline dispatch docs + unit_tests
- [ ] Live sticky comment + label on #41
- [ ] Unit-test scope on #42
- [ ] Negative: draft / kill switch / code-only
- [ ] Evidence links filled in below

### Evidence links (fill after runs)

- Labeler run:
- Advisory run (docs):
- Advisory run (unit_tests):
- Sticky comment URL:
- Artifact run IDs: