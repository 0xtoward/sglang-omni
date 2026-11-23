#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

base=${1:-}
# The caller supplies the tested merge commit's base only after its full Black check succeeded.
if [[ -z "$base" || "$(git rev-parse --verify HEAD^1 2>/dev/null)" != "$base" ]] ||
   ! git diff --quiet --no-renames "$base" HEAD -- \
        '.pre-commit-config.yaml' ':(glob)**/pyproject.toml' ':(glob)**/.gitignore' \
        ':(glob)**/.gitattributes' '.gitmodules' '.github/workflows/lint.yaml' \
        '.github/scripts/lint_with_scoped_black.sh' ||
   [[ -n "$(git ls-files -- ':(glob)*/**/pyproject.toml')" ]]; then
    echo "Running full lint"
    exec pre-commit run --all-files --show-diff-on-failure
fi

echo "Running Black on the diff from $base"
status=0
SKIP=black-jupyter pre-commit run --all-files --show-diff-on-failure || status=$?
pre-commit run black-jupyter --from-ref "$base" --to-ref HEAD --show-diff-on-failure || status=$?
exit "$status"
