#!/usr/bin/env bash
set -euo pipefail

ROOT="${HIDEVLAB_ROOT:-/workspace/user_data/hidevlab}"
OMNI_URL="${OMNI_URL:-https://github.com/0xtoward/sglang-omni.git}"
OMNI_UPSTREAM="${OMNI_UPSTREAM:-https://github.com/sgl-project/sglang-omni.git}"
OMNI_BRANCH="${OMNI_BRANCH:-codex/minicpmo45-ascend-competition}"
OMNI_BASE="${OMNI_BASE:-10e7e54197af9b384af0c9fa5563723f09eb25bf}"
CORE_URL="${CORE_URL:-https://github.com/0xtoward/sglang.git}"
CORE_UPSTREAM="${CORE_UPSTREAM:-https://github.com/sgl-project/sglang.git}"
CORE_BRANCH="${CORE_BRANCH:-codex/minicpmo45-ascend-competition-core}"
CORE_BASE="${CORE_BASE:-f308abc05212c2f5f455de22a525e14afa63ee4f}"
OMNI_BUNDLE="${OMNI_BUNDLE:-/tmp/sglang-omni-minicpmo45-ascend.bundle}"
CORE_BUNDLE="${CORE_BUNDLE:-/tmp/sglang-minicpmo45-ascend-core.bundle}"
OMNI_BUNDLE_SHA256="${OMNI_BUNDLE_SHA256:?OMNI_BUNDLE_SHA256 is required}"
CORE_BUNDLE_SHA256="${CORE_BUNDLE_SHA256:?CORE_BUNDLE_SHA256 is required}"
EXPECTED_IMAGE_CORE="${EXPECTED_IMAGE_CORE:-f308abc05212c2f5f455de22a525e14afa63ee4f}"
EXPECTED_TORCH_PREFIX="${EXPECTED_TORCH_PREFIX:-2.10.0}"
EXPECTED_TORCH_NPU="${EXPECTED_TORCH_NPU:-2.10.0}"
PY="${PYTHON:-/usr/local/python3.11.15/bin/python3}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MANIFEST_SCRIPT="${MANIFEST_SCRIPT:-${SCRIPT_DIR}/capture_manifest.py}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
TAKEOVER_LOG="${TAKEOVER_LOG:-${ROOT}/logs/takeover_${STAMP}.log}"

source /usr/local/Ascend/cann-9.0.0/set_env.sh

mount_opts="$(findmnt -T /workspace/user_data -n -o OPTIONS)"
[[ ",${mount_opts}," == *,rw,* ]] || { echo "/workspace/user_data is not a rw mount" >&2; exit 3; }

mkdir -p "${ROOT}"/{repos,models,runs,logs,artifacts,manifests,cache,tmp,bin,vendor}
exec 9>"${ROOT}/.bootstrap.lock"
flock -n 9 || { echo "another takeover bootstrap holds ${ROOT}/.bootstrap.lock" >&2; exit 6; }
exec > >(tee -a "${TAKEOVER_LOG}") 2>&1

echo "takeover_started=${STAMP}"
echo "takeover_log=${TAKEOVER_LOG}"
cat > "${ROOT}/CURRENT" <<EOF
status=in_progress
captured_at=${STAMP}
takeover_log=${TAKEOVER_LOG}
EOF

image_head="$(git -C /sgl-workspace/sglang rev-parse HEAD)"
[[ "${image_head}" == "${EXPECTED_IMAGE_CORE}" ]] || {
  echo "image SGLang changed; audit and create a new compatibility baseline" >&2
  echo "expected=${EXPECTED_IMAGE_CORE} actual=${image_head}" >&2
  exit 7
}

read -r torch_version torch_npu_version < <(
  "${PY}" -c 'import torch, torch_npu; print(torch.__version__, torch_npu.__version__)'
)
[[ "${torch_version}" == "${EXPECTED_TORCH_PREFIX}"* ]] || {
  echo "unexpected torch version: ${torch_version}" >&2
  exit 7
}
[[ "${torch_npu_version}" == "${EXPECTED_TORCH_NPU}" ]] || {
  echo "unexpected torch_npu version: ${torch_npu_version}" >&2
  exit 7
}

verify_bundle() {
  local path="$1" expected="$2" actual
  [[ -r "${path}" ]] || { echo "bundle is unreadable: ${path}" >&2; exit 4; }
  actual="$(sha256sum "${path}" | awk '{print $1}')"
  [[ "${actual}" == "${expected}" ]] || {
    echo "bundle SHA-256 mismatch: ${path}" >&2
    echo "expected=${expected} actual=${actual}" >&2
    exit 4
  }
  git -C "${VERIFY_REPO}" bundle verify "${path}"
}

VERIFY_REPO="${ROOT}/tmp/bundle-verify_${STAMP}.git"
git init --bare "${VERIFY_REPO}" >/dev/null
verify_bundle "${OMNI_BUNDLE}" "${OMNI_BUNDLE_SHA256}"
verify_bundle "${CORE_BUNDLE}" "${CORE_BUNDLE_SHA256}"

sync_repo() {
  local name="$1" url="$2" upstream_url="$3" branch="$4" base="$5" bundle="$6"
  local path="${ROOT}/repos/${name}"
  local bundle_head
  bundle_head="$(git bundle list-heads "${bundle}" "refs/heads/${branch}" | awk '{print $1}')"
  [[ -n "${bundle_head}" ]] || { echo "branch missing from bundle: ${branch}" >&2; exit 4; }

  if [[ -d "${path}/.git" ]] && ! git -C "${path}" rev-parse --verify HEAD >/dev/null 2>&1; then
    mv "${path}" "${ROOT}/tmp/${name}.incomplete_${STAMP}"
  fi
  if [[ ! -d "${path}/.git" ]]; then
    git clone --single-branch --branch "${branch}" "${bundle}" "${path}"
    git -C "${path}" remote set-url origin "${url}"
  else
    [[ -z "$(git -C "${path}" status --porcelain)" ]] || { echo "refusing dirty repo: ${path}" >&2; exit 4; }
    [[ "$(git -C "${path}" remote get-url origin)" == "${url}" ]] || { echo "unexpected origin: ${path}" >&2; exit 4; }
    git -C "${path}" checkout "${branch}"
    if ! git -C "${path}" cat-file -e "${bundle_head}^{commit}" 2>/dev/null; then
      git -C "${path}" fetch "${bundle}" "refs/heads/${branch}:refs/remotes/bundle/${branch}"
    fi
    git -C "${path}" merge --ff-only "${bundle_head}"
  fi

  if git -C "${path}" remote get-url upstream >/dev/null 2>&1; then
    [[ "$(git -C "${path}" remote get-url upstream)" == "${upstream_url}" ]] || { echo "unexpected upstream: ${path}" >&2; exit 4; }
  else
    git -C "${path}" remote add upstream "${upstream_url}"
  fi
  git -C "${path}" remote set-url --push upstream DISABLED

  [[ "$(git -C "${path}" branch --show-current)" == "${branch}" ]] || { echo "wrong branch: ${path}" >&2; exit 5; }
  [[ "$(git -C "${path}" rev-parse HEAD)" == "${bundle_head}" ]] || { echo "repo is not at bundle head: ${path}" >&2; exit 5; }
  git -C "${path}" merge-base --is-ancestor "${base}" HEAD || { echo "base ${base} is not an ancestor of ${name}" >&2; exit 5; }
  [[ -z "$(git -C "${path}" status --porcelain)" ]] || { echo "repo became dirty: ${path}" >&2; exit 5; }
}

sync_repo sglang-omni "${OMNI_URL}" "${OMNI_UPSTREAM}" "${OMNI_BRANCH}" "${OMNI_BASE}" "${OMNI_BUNDLE}"
sync_repo sglang "${CORE_URL}" "${CORE_UPSTREAM}" "${CORE_BRANCH}" "${CORE_BASE}" "${CORE_BUNDLE}"

install -m 0644 "${OMNI_BUNDLE}" "${ROOT}/artifacts/sglang-omni-minicpmo45-ascend.bundle"
install -m 0644 "${CORE_BUNDLE}" "${ROOT}/artifacts/sglang-minicpmo45-ascend-core.bundle"
sha256sum "${ROOT}"/artifacts/*.bundle > "${ROOT}/artifacts/git-bundles.sha256"
grep -Fq "${OMNI_BUNDLE_SHA256}" "${ROOT}/artifacts/git-bundles.sha256"
grep -Fq "${CORE_BUNDLE_SHA256}" "${ROOT}/artifacts/git-bundles.sha256"

vendor_root="${ROOT}/vendor/ascend-image-${image_head:0:12}"
mkdir -p "${vendor_root}"
git -C /sgl-workspace/sglang diff --binary > "${vendor_root}/working-tree.patch"
git -C /sgl-workspace/sglang diff --name-status > "${vendor_root}/name-status.txt"
printf '%s\n' "${image_head}" > "${vendor_root}/image-head.txt"

install -m 0755 "${SCRIPT_DIR}/bootstrap_remote.sh" "${ROOT}/bin/bootstrap_remote.sh"
install -m 0755 "${MANIFEST_SCRIPT}" "${ROOT}/bin/capture_manifest.py"

manifest="${ROOT}/manifests/env_${STAMP}.json"
"${PY}" "${MANIFEST_SCRIPT}" --root "${ROOT}" --output "${manifest}"
cp "${manifest}" "${ROOT}/manifests/current_env.json"

cat > "${ROOT}/CURRENT" <<EOF
status=complete
captured_at=${STAMP}
omni_branch=${OMNI_BRANCH}
omni_head=$(git -C "${ROOT}/repos/sglang-omni" rev-parse HEAD)
omni_bundle_sha256=${OMNI_BUNDLE_SHA256}
core_branch=${CORE_BRANCH}
core_head=$(git -C "${ROOT}/repos/sglang" rev-parse HEAD)
core_bundle_sha256=${CORE_BUNDLE_SHA256}
image_core_head=${image_head}
torch=${torch_version}
torch_npu=${torch_npu_version}
manifest=${manifest}
takeover_log=${TAKEOVER_LOG}
EOF

echo "takeover_root=${ROOT}"
cat "${ROOT}/CURRENT"
echo "No packages installed and no NPU workload started."
