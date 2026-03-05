#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${ROOT_DIR}/results/logs"
OUT_FILE="${OUT_DIR}/env_info.txt"
mkdir -p "${OUT_DIR}"

{
  echo "timestamp_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "hostname=$(hostname)"
  echo "pwd=$(pwd)"
  echo "git_branch=$(git -C "${ROOT_DIR}" branch --show-current || true)"
  echo "git_commit=$(git -C "${ROOT_DIR}" rev-parse HEAD || true)"
  echo
  echo "[python]"
  python --version 2>&1 || true
  echo
  echo "[pip]"
  pip --version 2>&1 || true
  echo
  echo "[nvidia-smi]"
  nvidia-smi || true
} | tee "${OUT_FILE}"
