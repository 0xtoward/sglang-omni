import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

WORKFLOW_PATH = (
    Path(__file__).resolve().parents[3]
    / ".github/workflows/pr-documentation-labeler.yml"
)
NODE_RUNNER = r"""
const {readFileSync} = require('node:fs');
const input = JSON.parse(readFileSync(0, 'utf8'));
let added = [];
let removed = [];
const comments = [];
const pullRequest = input.pullRequest;
const context = {
  payload: {action: input.action ?? 'opened', pull_request: pullRequest},
  repo: {owner: 'example', repo: 'example'},
};
const pages = input.filePages ?? [input.files ?? []];
async function* fileIterator() {
  for (const data of pages) {
    yield {data};
  }
}
async function* commentIterator() {
  yield {data: input.existingComments ?? []};
}
const listFiles = 'listFiles';
const listComments = 'listComments';
const github = {
  paginate: {
    iterator: endpoint =>
      endpoint === listComments ? commentIterator() : fileIterator(),
  },
  rest: {
    pulls: {get: async () => ({data: pullRequest}), listFiles},
    issues: {
      addLabels: async request => {added = request.labels},
      removeLabel: async request => {removed.push(request.name)},
      listComments,
      createComment: async request => {
        comments.push({action: 'create', body: request.body});
      },
      updateComment: async request => {
        comments.push({action: 'update', body: request.body});
      },
    },
  },
};
const core = {info: () => {}};
const AsyncFunction = Object.getPrototypeOf(async function () {}).constructor;
(async () => {
  await new AsyncFunction('github', 'context', 'core', input.script)(
    github,
    context,
    core
  );
  process.stdout.write(JSON.stringify({
    added: added.sort(),
    removed: removed.sort(),
    comments,
  }));
})().catch(error => {console.error(error); process.exit(1)});
"""


def labeler_script() -> str:
    workflow = yaml.safe_load(WORKFLOW_PATH.read_text())
    return workflow["jobs"]["label"]["steps"][0]["with"]["script"]


def run_bot(
    *,
    title: str,
    branch: str,
    files: list[dict[str, str]],
    existing_labels: list[str] | None = None,
    existing_comments: list[dict[str, object]] | None = None,
    action: str = "opened",
    draft: bool = False,
) -> dict[str, object]:
    pull_request = {
        "number": 42,
        "title": title,
        "draft": draft,
        "head": {"ref": branch},
        "labels": [{"name": name} for name in (existing_labels or [])],
    }
    payload = {
        "script": labeler_script(),
        "pullRequest": pull_request,
        "files": files,
        "action": action,
        "existingComments": existing_comments or [],
    }
    result = subprocess.run(
        ["node", "-e", NODE_RUNNER],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(result.stdout)


def run_labeler(
    *,
    title: str,
    branch: str,
    files: list[dict[str, str]],
    existing_labels: list[str] | None = None,
    action: str = "opened",
) -> tuple[list[str], list[str]]:
    changes = run_bot(
        title=title,
        branch=branch,
        files=files,
        existing_labels=existing_labels,
        action=action,
    )
    return changes["added"], changes["removed"]


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is required")
@pytest.mark.parametrize(
    ("title", "branch", "files", "existing_labels", "expected_labels"),
    [
        (
            "[Fix][Ming-ASR] Restore MoE expert probability normalization",
            "neutral",
            [{"filename": "sglang_omni/x.py", "status": "modified"}],
            None,
            ["bug"],
        ),
        (
            "Neutral title",
            "codex/fix/timeout",
            [{"filename": "sglang_omni/x.py", "status": "modified"}],
            None,
            [],
        ),
        (
            "[Docs] Add a Voxt Quickstart",
            "neutral",
            [{"filename": "sglang_omni/x.py", "status": "modified"}],
            None,
            ["documentation"],
        ),
        (
            "Neutral",
            "codex/docs/readme",
            [{"filename": "sglang_omni/x.py", "status": "modified"}],
            None,
            [],
        ),
        (
            "docs-only",
            "neutral",
            [{"filename": "docs/foo.md", "status": "modified"}],
            None,
            ["documentation", "run-lint-only"],
        ),
        (
            "docs plus code",
            "neutral",
            [
                {"filename": "docs/foo.md", "status": "modified"},
                {"filename": "sglang_omni/x.py", "status": "modified"},
            ],
            None,
            ["documentation"],
        ),
        (
            "docs plus code with run label",
            "neutral",
            [{"filename": "docs/foo.md", "status": "modified"}],
            ["run-unit-test-ci-only"],
            ["documentation"],
        ),
        (
            "[Feat] [NPU] MiniMax-Music3: NPU support",
            "neutral",
            [{"filename": "sglang_omni/x.py", "status": "modified"}],
            None,
            ["enhancement"],
        ),
        (
            "[Refactor] tighten imports",
            "codex/refactor/imports",
            [{"filename": "sglang_omni/x.py", "status": "modified"}],
            None,
            ["refactor"],
        ),
        (
            "[Good first issue] improve messages",
            "neutral",
            [{"filename": "sglang_omni/x.py", "status": "modified"}],
            None,
            ["good first issue"],
        ),
        (
            "Good first issue: improve messages",
            "codex/good-first-issue/messages",
            [{"filename": "sglang_omni/x.py", "status": "modified"}],
            None,
            [],
        ),
        (
            "[Help wanted] reproduce a failure",
            "neutral",
            [{"filename": "sglang_omni/x.py", "status": "modified"}],
            None,
            ["help wanted"],
        ),
        (
            "Help wanted: reproduce a failure",
            "codex/help-wanted/repro",
            [{"filename": "sglang_omni/x.py", "status": "modified"}],
            None,
            [],
        ),
        (
            "New model directory",
            "neutral",
            [
                {
                    "filename": "sglang_omni/models/labeler_probe/__init__.py",
                    "status": "added",
                }
            ],
            None,
            ["model support"],
        ),
        (
            "Unrelated change",
            "neutral",
            [{"filename": "sglang_omni/x.py", "status": "modified"}],
            None,
            [],
        ),
        (
            "$(touch /tmp/labeler-injection)",
            "neutral",
            [{"filename": "sglang_omni/x.py", "status": "modified"}],
            None,
            [],
        ),
        (
            "Already labeled",
            "neutral",
            [{"filename": "docs/foo.md", "status": "modified"}],
            ["documentation"],
            ["run-lint-only"],
        ),
    ],
)
def test_suggested_labels(
    title: str,
    branch: str,
    files: list[dict[str, str]],
    existing_labels: list[str] | None,
    expected_labels: list[str],
) -> None:
    added, removed = run_labeler(
        title=title,
        branch=branch,
        files=files,
        existing_labels=existing_labels,
    )
    assert added == sorted(expected_labels)
    assert removed == []


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is required")
def test_code_push_removes_stale_run_lint_only() -> None:
    added, removed = run_labeler(
        title="docs plus code",
        branch="neutral",
        files=[
            {"filename": "docs/foo.md", "status": "modified"},
            {"filename": "sglang_omni/models/minicpm_o/model.py", "status": "modified"},
        ],
        existing_labels=["documentation", "run-lint-only"],
        action="synchronize",
    )
    assert added == []
    assert removed == ["run-lint-only"]


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is required")
def test_synchronize_does_not_restore_removed_title_label() -> None:
    added, removed = run_labeler(
        title="[Fix] Restore MoE expert probability normalization",
        branch="codex/fix/timeout",
        files=[{"filename": "sglang_omni/x.py", "status": "modified"}],
        action="synchronize",
    )
    assert added == []
    assert removed == []


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is required")
def test_edited_title_still_adds_category_label() -> None:
    added, removed = run_labeler(
        title="[Fix] Restore MoE expert probability normalization",
        branch="neutral",
        files=[{"filename": "sglang_omni/x.py", "status": "modified"}],
        action="edited",
    )
    assert added == ["bug"]
    assert removed == []


RESOLVER_PATH = (
    Path(__file__).resolve().parents[3] / ".github/scripts/resolve_ci_labels.py"
)
SUGGESTION_MARKER = "<!-- ci-stage-suggestion -->"
MINICPM_FILE = {
    "filename": "sglang_omni/models/minicpm_o/session_adapters.py",
    "status": "modified",
}
QWEN3_ASR_FILE = {
    "filename": "sglang_omni/models/qwen3_asr/pipeline.py",
    "status": "modified",
}
# note (0xtoward): run-qwen3-tts-12hz-1.7b-customvoice gates no job until it is
# calibrated on the CI host, and the five stages below serve no model directory.
LABELS_OUTSIDE_THE_TABLE = frozenset(
    {
        "run-lint-only",
        "run-unit-test-ci-only",
        "run-intel-cpu-ci",
        "run-intel-xpu-ci",
        "run-mac-ci",
        "run-qwen3-tts-12hz-1.7b-customvoice",
    }
)


def make_bot_comment(body: str, comment_id: int = 1) -> dict[str, object]:
    return {
        "id": comment_id,
        "user": {"login": "github-actions[bot]", "type": "Bot"},
        "body": body,
    }


def build_suggestion_body(
    *,
    files: list[dict[str, str]],
    existing_labels: list[str] | None = None,
) -> str:
    comments = run_bot(
        title="[Fix] change a model",
        branch="topic",
        files=files,
        existing_labels=existing_labels,
    )["comments"]

    return comments[0]["body"] if comments else ""


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is required")
@pytest.mark.parametrize(
    ("files", "existing_labels", "draft", "expected_targets"),
    [
        ([MINICPM_FILE], None, False, ["minicpmo", "full-duplex"]),
        ([MINICPM_FILE], ["run-omni-ci"], False, ["full-duplex"]),
        ([MINICPM_FILE], ["run-omni-ci", "run-native-duplex"], False, []),
        ([MINICPM_FILE], None, True, []),
        (
            [
                QWEN3_ASR_FILE,
                {
                    "filename": "sglang_omni/models/qwen3_tts/stages.py",
                    "status": "modified",
                },
            ],
            None,
            False,
            ["qwen3-asr", "cascaded", "qwen3-tts"],
        ),
        ([QWEN3_ASR_FILE], ["run-qwen3-asr-1.7b"], False, ["cascaded"]),
        (
            [QWEN3_ASR_FILE],
            ["run-qwen3-asr-1.7b", "run-cascaded-duplex"],
            False,
            [],
        ),
        (
            [{"filename": "sglang_omni/models/moss_tts/stages.py", "status": "modified"}],
            None,
            False,
            [],
        ),
        (
            [
                {
                    "filename": "sglang_omni/models/streaming_cascade/base.py",
                    "status": "modified",
                }
            ],
            None,
            False,
            ["cascaded"],
        ),
        (
            [
                {
                    "filename": "sglang_omni/codec/vocoder.py",
                    "previous_filename": "sglang_omni/models/qwen3_tts/vocoder.py",
                    "status": "renamed",
                }
            ],
            None,
            False,
            ["qwen3-tts", "cascaded"],
        ),
    ],
)
def test_suggests_the_slash_targets_of_every_stage_that_serves_a_changed_model(
    files: list[dict[str, str]],
    existing_labels: list[str] | None,
    draft: bool,
    expected_targets: list[str],
) -> None:
    comments = run_bot(
        title="[Fix] change a model",
        branch="topic",
        files=files,
        existing_labels=existing_labels,
        draft=draft,
    )["comments"]

    if expected_targets:
        assert len(comments) == 1, comments
        assert SUGGESTION_MARKER in comments[0]["body"]
        assert (
            f"/tag-and-rerun-ci {' '.join(expected_targets)}" in comments[0]["body"]
        )
    else:
        assert comments == []


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is required")
@pytest.mark.parametrize(
    ("existing_body", "existing_labels", "expected_action", "expected_text"),
    [
        (None, None, "create", "minicpmo"),
        ("current", None, None, None),
        (f"{SUGGESTION_MARKER}\nstale text", None, "update", "minicpmo"),
        (
            "current",
            ["run-omni-ci", "run-native-duplex"],
            "update",
            "No model CI suggestions",
        ),
    ],
)
def test_one_sticky_comment_is_reused_and_only_written_when_it_changes(
    existing_body: str | None,
    existing_labels: list[str] | None,
    expected_action: str | None,
    expected_text: str | None,
) -> None:
    current = build_suggestion_body(files=[MINICPM_FILE])
    body = current if existing_body == "current" else existing_body

    comments = run_bot(
        title="[Fix] change a model",
        branch="topic",
        files=[MINICPM_FILE],
        existing_labels=existing_labels,
        existing_comments=[make_bot_comment(body)] if body else [],
    )["comments"]

    if expected_action is None:
        assert comments == []
    else:
        assert [comment["action"] for comment in comments] == [expected_action]
        assert expected_text in comments[0]["body"]


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is required")
def test_a_comment_from_a_person_is_never_edited_even_with_the_marker() -> None:
    human_comment = {
        "id": 7,
        "user": {"login": "contributor", "type": "User"},
        "body": f"{SUGGESTION_MARKER}\nI pasted the marker by hand.",
    }

    comments = run_bot(
        title="[Fix] change a model",
        branch="topic",
        files=[MINICPM_FILE],
        existing_comments=[human_comment],
    )["comments"]

    assert [comment["action"] for comment in comments] == ["create"]


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is required")
def test_a_push_still_suggests_the_stage_of_a_newly_pushed_model_file() -> None:
    comments = run_bot(
        title="[Fix] change a model",
        branch="topic",
        files=[MINICPM_FILE],
        action="synchronize",
    )["comments"]

    assert [comment["action"] for comment in comments] == ["create"]
    assert "/tag-and-rerun-ci minicpmo full-duplex" in comments[0]["body"]


def test_the_suggestion_table_names_every_label_the_resolver_gates_on() -> None:
    resolver = RESOLVER_PATH.read_text(encoding="utf-8")
    workflow = WORKFLOW_PATH.read_text(encoding="utf-8")

    resolver_labels = {
        label
        for label in re.findall(r'"(run-[a-z0-9.\-]+)"', resolver)
        if label not in LABELS_OUTSIDE_THE_TABLE
    }

    assert {"run-tts-ci", "run-minicpm-o-4.5"} <= resolver_labels
    assert sorted(label for label in resolver_labels if label not in workflow) == []
