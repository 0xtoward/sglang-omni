"""Compare Black's touched paths and the overall lint result on adversarial PR diffs."""

import json
import os
import subprocess
import tempfile
from pathlib import Path

import yaml

TOOLS = Path(__file__).resolve().parent
ROOT = TOOLS.parents[1]
CLEAN = "values = [1, 2]\n"
BAD = "values=[1,2]\n"
HEADER = "#!/usr/bin/env python3\n"
NOTEBOOK = (
    json.dumps(
        {
            "cells": [
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": ["values=[1,2]"],
                }
            ],
            "metadata": {"language_info": {"name": "python"}},
            "nbformat": 4,
            "nbformat_minor": 4,
        }
    )
    + "\n"
)


def git(root, *args):
    return subprocess.check_output(["git", *args], cwd=root)


def put(root, name, contents):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(contents)
    if contents.startswith("#!"):
        path.chmod(0o755)


def commit(root, name):
    git(root, "add", "-A")
    git(root, "commit", "--quiet", "--allow-empty", "-m", name)
    return git(root, "rev-parse", "HEAD").decode().strip()


def check(root, command):
    result = subprocess.run(
        command,
        cwd=root,
        env=dict(os.environ, BLACK_CACHE_DIR=tempfile.mkdtemp()),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    changed = sorted(
        os.fsdecode(v) for v in git(root, "diff", "--name-only", "-z").split(b"\0") if v
    )
    git(root, "restore", "--worktree", ".")
    return result.returncode, changed, result.stdout


def run(
    name,
    before,
    after,
    expected,
    changed,
    scope,
    *,
    rename=None,
    unverified=False,
    merge_only=False,
    missing_parent=False,
):
    config = yaml.safe_load((ROOT / ".pre-commit-config.yaml").read_text())
    black = next(repo for repo in config["repos"] if repo["repo"].endswith("/black"))
    hooks = next(
        repo for repo in config["repos"] if repo["repo"].endswith("/pre-commit-hooks")
    )
    config["repos"] = [dict(hooks, hooks=[{"id": "check-yaml"}]), black]
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        git(root, "init", "-q", "-b", "fixture")
        git(root, "config", "user.email", "trial@example.invalid")
        git(root, "config", "user.name", "Lint trial")
        put(root, ".pre-commit-config.yaml", yaml.safe_dump(config))
        for file, value in before.items():
            put(root, file, value)
        base = commit(root, "baseline")
        base_rc, _, _ = check(
            root, ["pre-commit", "run", "black-jupyter", "--all-files"]
        )
        if base_rc != 0 and not unverified:
            raise AssertionError(f"unexpected dirty fixture base: {name}")
        git(root, "checkout", "-qb", "feature")
        if rename:
            source, target = rename
            (root / target).parent.mkdir(parents=True, exist_ok=True)
            (root / source).rename(root / target)
        for file, value in after.items():
            if value is None:
                (root / file).unlink()
            elif file == ".pre-commit-config.yaml":
                put(
                    root,
                    file,
                    (root / file)
                    .read_text()
                    .replace("exclude: ^\\.claude/skills/", "exclude: ^never-match$"),
                )
            else:
                put(root, file, value)
        commit(root, "feature")
        git(root, "checkout", "-q", "fixture")
        git(root, "merge", "--no-ff", "--no-commit", "-q", "feature")
        if merge_only:
            put(root, "merge-only.py", BAD)
        commit(root, "test merge")
        if missing_parent:
            shallow = root / "shallow"
            git(root, "clone", "-q", "--depth=1", f"file://{root}", str(shallow))
            root = shallow
        full = check(root, ["pre-commit", "run", "--all-files"])
        candidate = check(
            root,
            [
                "bash",
                str(TOOLS / "lint_with_scoped_black.sh"),
                base if base_rc == 0 and not unverified else "",
            ],
        )
        expected_result = (expected, sorted(changed))
        wanted = "Running full lint" if scope == "full" else "Running Black on the diff"
        if (
            full[:2] != expected_result
            or candidate[:2] != expected_result
            or wanted not in candidate[2]
        ):
            raise AssertionError((name, expected_result, full, candidate))
        print(
            f"PASS {name}: {scope}, exit {candidate[0]}, changed={candidate[1]!r}",
            flush=True,
        )


run("python edit", {"a.py": CLEAN}, {"a.py": BAD}, 1, ["a.py"], "diff")
run("stub added", {}, {"a.pyi": "def f(a:int)->int: ...\n"}, 1, ["a.pyi"], "diff")
run("notebook added", {}, {"a.ipynb": NOTEBOOK}, 1, ["a.ipynb"], "diff")
run(
    "executable Python without suffix",
    {},
    {"bin/tool": HEADER + BAD},
    1,
    ["bin/tool"],
    "diff",
)
run(
    "non-Python renamed to Python",
    {"data.raw": BAD},
    {},
    1,
    ["renamed.py"],
    "diff",
    rename=("data.raw", "renamed.py"),
)
run(
    "file moved out of exclusion",
    {".claude/skills/a.py": BAD},
    {},
    1,
    ["a.py"],
    "diff",
    rename=(".claude/skills/a.py", "a.py"),
)
run("odd filename", {}, {"space\nand 中文.py": BAD}, 1, ["space\nand 中文.py"], "diff")
run("delete last Python", {"a.py": CLEAN}, {"a.py": None}, 0, [], "diff")
run("docs only", {"a.py": CLEAN}, {"README.md": "hello\n"}, 0, [], "diff")
run("still excluded", {}, {".claude/skills/a.py": BAD}, 0, [], "diff")
run(
    "hook config unmasks untouched Python",
    {".claude/skills/a.py": BAD},
    {".pre-commit-config.yaml": "change"},
    1,
    [".claude/skills/a.py"],
    "full",
)
long_line = 'items = ["aaaaaaaaaa", "bbbbbbbbbb", "cccccccccc"]\n'
run(
    "format config affects untouched file",
    {"a.py": long_line},
    {"pyproject.toml": "[tool.black]\nline-length = 20\n"},
    1,
    ["a.py"],
    "full",
)
run(
    "missing clean proof catches pre-existing violation",
    {"a.py": BAD},
    {"README.md": "hello\n"},
    1,
    ["a.py"],
    "full",
    unverified=True,
)
run(
    "parent unavailable",
    {"a.py": CLEAN},
    {"a.py": BAD},
    1,
    ["a.py"],
    "full",
    missing_parent=True,
)
run(
    "non-Black hook still scans unmodified YAML",
    {"bad.yaml": "a: [\n", "a.py": CLEAN},
    {"README.md": "hello\n"},
    1,
    [],
    "diff",
)
run(
    "nested config retained",
    {"pkg/pyproject.toml": "[tool.black]\nline-length=88\n"},
    {"pkg/a.py": BAD},
    1,
    ["pkg/a.py"],
    "full",
)

run(
    "merge-result-only change",
    {},
    {"README.md": "hello\n"},
    1,
    ["merge-only.py"],
    "diff",
    merge_only=True,
)
