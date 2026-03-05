#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HF_CLI="${HF_CLI:-}"

if [[ -z "${HF_CLI}" ]]; then
  if command -v huggingface-cli >/dev/null 2>&1; then
    HF_CLI="huggingface-cli"
  elif command -v hf >/dev/null 2>&1; then
    HF_CLI="hf"
  else
    echo "No huggingface CLI found. Set HF_CLI=/path/to/huggingface-cli or install huggingface_hub[cli]."
    exit 1
  fi
fi

mkdir -p "${ROOT_DIR}/wan_models" "${ROOT_DIR}/checkpoints"

if [[ "${HF_CLI}" == "hf" ]]; then
  "${HF_CLI}" download Wan-AI/Wan2.1-T2V-1.3B --local-dir "${ROOT_DIR}/wan_models/Wan2.1-T2V-1.3B"
  "${HF_CLI}" download gdhe17/Self-Forcing checkpoints/self_forcing_dmd.pt --local-dir "${ROOT_DIR}/checkpoints"
else
  "${HF_CLI}" download Wan-AI/Wan2.1-T2V-1.3B --local-dir-use-symlinks False --local-dir "${ROOT_DIR}/wan_models/Wan2.1-T2V-1.3B"
  "${HF_CLI}" download gdhe17/Self-Forcing checkpoints/self_forcing_dmd.pt --local-dir "${ROOT_DIR}/checkpoints"
fi

if [[ -f "${ROOT_DIR}/checkpoints/checkpoints/self_forcing_dmd.pt" ]]; then
  mv "${ROOT_DIR}/checkpoints/checkpoints/self_forcing_dmd.pt" "${ROOT_DIR}/checkpoints/self_forcing_dmd.pt"
  rmdir "${ROOT_DIR}/checkpoints/checkpoints" || true
fi

echo "Downloaded checkpoints to ${ROOT_DIR}/wan_models and ${ROOT_DIR}/checkpoints"
