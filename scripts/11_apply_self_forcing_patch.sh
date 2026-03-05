#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SF_DIR="${ROOT_DIR}/third_party/Self-Forcing"
PATCH_FILE="${ROOT_DIR}/docs/patches/self_forcing_kv_quant.patch"

if [[ ! -d "${SF_DIR}/.git" ]]; then
  echo "Self-Forcing repo not found at ${SF_DIR}. Run scripts/10_clone_deps.sh first."
  exit 1
fi

if [[ ! -f "${PATCH_FILE}" ]]; then
  echo "Patch file not found: ${PATCH_FILE}"
  exit 1
fi

cd "${SF_DIR}"
if git apply --check "${PATCH_FILE}" >/dev/null 2>&1; then
  git apply "${PATCH_FILE}"
  echo "Applied KV quantization hook patch to Self-Forcing causal_model.py"
else
  if git apply --reverse --check "${PATCH_FILE}" >/dev/null 2>&1; then
    echo "Patch already applied in Self-Forcing causal_model.py"
  else
    echo "Patch cannot be applied cleanly; please verify third_party/Self-Forcing state."
    exit 1
  fi
fi
