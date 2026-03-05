#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HF_CLI="${HF_CLI:-huggingface-cli}"

mkdir -p "${ROOT_DIR}/wan_models" "${ROOT_DIR}/checkpoints"

"${HF_CLI}" download Wan-AI/Wan2.1-T2V-1.3B --local-dir-use-symlinks False --local-dir "${ROOT_DIR}/wan_models/Wan2.1-T2V-1.3B"
"${HF_CLI}" download gdhe17/Self-Forcing checkpoints/self_forcing_dmd.pt --local-dir "${ROOT_DIR}/checkpoints"

if [[ -f "${ROOT_DIR}/checkpoints/checkpoints/self_forcing_dmd.pt" ]]; then
  mv "${ROOT_DIR}/checkpoints/checkpoints/self_forcing_dmd.pt" "${ROOT_DIR}/checkpoints/self_forcing_dmd.pt"
  rmdir "${ROOT_DIR}/checkpoints/checkpoints" || true
fi

echo "Downloaded checkpoints to ${ROOT_DIR}/wan_models and ${ROOT_DIR}/checkpoints"
