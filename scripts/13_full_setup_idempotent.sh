#!/usr/bin/env bash
set -euo pipefail

# Idempotent full setup for kv-quant-longhorizon.
# Safe to rerun; installs only what is missing.
# Works on fresh systems too by bootstrapping Miniforge if conda is absent.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INFER_ENV="${INFER_ENV:-qvg_sf_infer}"
EVAL_ENV="${EVAL_ENV:-qvg_sf_eval}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu121}"
INSTALL_FLASH_ATTN="${INSTALL_FLASH_ATTN:-1}"
INSTALL_EVAL_ENV="${INSTALL_EVAL_ENV:-1}"
INSTALL_ASSETS="${INSTALL_ASSETS:-1}"
AUTO_INSTALL_CONDA="${AUTO_INSTALL_CONDA:-1}"

log() {
  echo ""
  echo "[setup] $1"
}

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Missing required command: $1"
    exit 1
  fi
}

ensure_conda() {
  if command -v conda >/dev/null 2>&1; then
    return
  fi
  if [[ "${AUTO_INSTALL_CONDA}" != "1" ]]; then
    echo "conda is missing and AUTO_INSTALL_CONDA=${AUTO_INSTALL_CONDA}."
    echo "Install conda/miniforge manually, then rerun."
    exit 1
  fi
  require_cmd wget
  require_cmd bash
  local installer="/tmp/Miniforge3.sh"
  local prefix="${HOME}/miniforge3"
  echo "[setup] conda not found; installing Miniforge to ${prefix}"
  wget -qO "${installer}" "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh"
  bash "${installer}" -b -p "${prefix}"
  export PATH="${prefix}/bin:${PATH}"
}

conda_env_exists() {
  conda env list | awk '{print $1}' | grep -qx "$1"
}

kernel_exists() {
  jupyter kernelspec list 2>/dev/null | awk '{print $1}' | grep -qx "$1"
}

pip_install_if_missing_import() {
  local env_name="$1"
  local import_name="$2"
  local pip_spec="$3"
  if conda run -n "${env_name}" python - <<PY >/dev/null 2>&1
import importlib
importlib.import_module("${import_name}")
PY
  then
    echo "[ok] ${env_name}: ${import_name}"
  else
    echo "[install] ${env_name}: ${pip_spec}"
    conda run -n "${env_name}" pip install ${pip_spec}
  fi
}

log "Sanity checks"
ensure_conda
require_cmd git
require_cmd python
require_cmd jupyter
require_cmd conda
cd "${ROOT_DIR}"
[ -f "README.md" ] || { echo "Run this from within repo context."; exit 1; }

log "Clone third_party deps if missing"
if [[ ! -d "${ROOT_DIR}/third_party/Self-Forcing/.git" ]] || [[ ! -d "${ROOT_DIR}/third_party/VBench/.git" ]]; then
  bash "${ROOT_DIR}/scripts/10_clone_deps.sh"
else
  echo "[ok] third_party repos already cloned"
fi

log "Create inference env if missing"
if conda_env_exists "${INFER_ENV}"; then
  echo "[ok] env ${INFER_ENV} exists"
else
  conda create -n "${INFER_ENV}" python="${PYTHON_VERSION}" -y
fi

log "Upgrade base packaging tools in inference env"
conda run -n "${INFER_ENV}" pip install --upgrade pip setuptools wheel

log "Install PyTorch only if missing"
if conda run -n "${INFER_ENV}" python - <<'PY' >/dev/null 2>&1
import torch
PY
then
  echo "[ok] torch already installed in ${INFER_ENV}"
else
  conda run -n "${INFER_ENV}" pip install torch torchvision torchaudio --index-url "${TORCH_INDEX_URL}"
fi

log "Install inference requirements only if key imports are missing"
if conda run -n "${INFER_ENV}" python - <<'PY' >/dev/null 2>&1
import omegaconf, einops, lpips, skimage, imageio, ffmpeg
PY
then
  echo "[ok] primary inference requirements already satisfied"
else
  conda run -n "${INFER_ENV}" pip install -r "${ROOT_DIR}/requirements-inference.txt"
fi

log "Install Self-Forcing requirements (idempotent)"
conda run -n "${INFER_ENV}" pip install -r "${ROOT_DIR}/third_party/Self-Forcing/requirements.txt"

log "Install additional known deps discovered during dry-run"
pip_install_if_missing_import "${INFER_ENV}" easydict "easydict"
pip_install_if_missing_import "${INFER_ENV}" diffusers "diffusers>=0.30"
pip_install_if_missing_import "${INFER_ENV}" transformers "transformers>=4.44"
pip_install_if_missing_import "${INFER_ENV}" accelerate "accelerate"
pip_install_if_missing_import "${INFER_ENV}" safetensors "safetensors"
pip_install_if_missing_import "${INFER_ENV}" sentencepiece "sentencepiece"
pip_install_if_missing_import "${INFER_ENV}" ftfy "ftfy"
pip_install_if_missing_import "${INFER_ENV}" huggingface_hub "huggingface_hub[cli]"

if [[ "${INSTALL_FLASH_ATTN}" == "1" ]]; then
  log "Install flash-attn if missing (optional)"
  if conda run -n "${INFER_ENV}" python - <<'PY' >/dev/null 2>&1
import flash_attn
PY
  then
    echo "[ok] flash_attn present"
  else
    echo "[info] installing flash-attn (will continue if build fails)"
    conda run -n "${INFER_ENV}" pip install flash-attn --no-build-isolation || true
  fi
fi

log "Install Self-Forcing package in editable mode"
conda run -n "${INFER_ENV}" bash -lc "cd '${ROOT_DIR}/third_party/Self-Forcing' && python setup.py develop"

log "Apply Self-Forcing KV patch only if not yet applied"
if ( cd "${ROOT_DIR}/third_party/Self-Forcing" && git apply --check ../../docs/patches/self_forcing_kv_quant.patch >/dev/null 2>&1 ); then
  bash "${ROOT_DIR}/scripts/11_apply_self_forcing_patch.sh"
else
  echo "[ok] patch already applied (or overlapping content already present)"
fi

log "Download model/checkpoint assets only if missing"
if [[ "${INSTALL_ASSETS}" == "1" ]]; then
  if [[ -f "${ROOT_DIR}/checkpoints/self_forcing_dmd.pt" ]] && [[ -d "${ROOT_DIR}/wan_models/Wan2.1-T2V-1.3B" ]]; then
    echo "[ok] assets already present"
  else
    echo "[info] if auth is needed, run: conda run -n ${INFER_ENV} huggingface-cli login"
    bash "${ROOT_DIR}/scripts/12_download_checkpoints.sh"
  fi
else
  echo "[skip] INSTALL_ASSETS=${INSTALL_ASSETS}"
fi

log "Register inference kernel if missing"
conda run -n "${INFER_ENV}" pip install ipykernel
if kernel_exists "${INFER_ENV}"; then
  echo "[ok] kernel ${INFER_ENV} already exists"
else
  conda run -n "${INFER_ENV}" python -m ipykernel install --user --name "${INFER_ENV}" --display-name "Python (${INFER_ENV})"
fi

if [[ "${INSTALL_EVAL_ENV}" == "1" ]]; then
  log "Create eval env if missing"
  if conda_env_exists "${EVAL_ENV}"; then
    echo "[ok] env ${EVAL_ENV} exists"
  else
    conda create -n "${EVAL_ENV}" python="${PYTHON_VERSION}" -y
  fi

  log "Install eval requirements only if missing"
  if conda run -n "${EVAL_ENV}" python - <<'PY' >/dev/null 2>&1
import skimage, lpips
PY
  then
    echo "[ok] base eval requirements already satisfied"
  else
    conda run -n "${EVAL_ENV}" pip install -r "${ROOT_DIR}/requirements-eval.txt"
  fi

  log "Install detectron2 if missing (required for some VBench dimensions)"
  if conda run -n "${EVAL_ENV}" python - <<'PY' >/dev/null 2>&1
import detectron2
PY
  then
    echo "[ok] detectron2 already installed"
  else
    conda run -n "${EVAL_ENV}" pip install "detectron2@git+https://github.com/facebookresearch/detectron2.git" || true
  fi

  log "Install VBench package if missing"
  if conda run -n "${EVAL_ENV}" python - <<'PY' >/dev/null 2>&1
import vbench
PY
  then
    echo "[ok] vbench import works"
  else
    conda run -n "${EVAL_ENV}" bash -lc "cd '${ROOT_DIR}/third_party/VBench' && pip install -e ."
  fi

  log "Register eval kernel if missing"
  conda run -n "${EVAL_ENV}" pip install ipykernel
  if kernel_exists "${EVAL_ENV}"; then
    echo "[ok] kernel ${EVAL_ENV} already exists"
  else
    conda run -n "${EVAL_ENV}" python -m ipykernel install --user --name "${EVAL_ENV}" --display-name "Python (${EVAL_ENV})"
  fi
else
  echo "[skip] INSTALL_EVAL_ENV=${INSTALL_EVAL_ENV}"
fi

log "Final verification"
echo "[envs]"
conda env list | grep -E "${INFER_ENV}|${EVAL_ENV}" || true
echo "[kernels]"
jupyter kernelspec list 2>/dev/null | grep -E "${INFER_ENV}|${EVAL_ENV}" || true
echo "[assets]"
[[ -f "${ROOT_DIR}/checkpoints/self_forcing_dmd.pt" ]] && ls -lh "${ROOT_DIR}/checkpoints/self_forcing_dmd.pt" || echo "checkpoint missing"
[[ -d "${ROOT_DIR}/wan_models/Wan2.1-T2V-1.3B" ]] && du -sh "${ROOT_DIR}/wan_models/Wan2.1-T2V-1.3B" || echo "wan model missing"

log "Smoke test dry-run"
conda run -n "${INFER_ENV}" python "${ROOT_DIR}/scripts/01_generate.py" --method BF16 --max-prompts 1 --dry-run

log "Setup complete"
echo "You can now run generation/evaluation with ${INFER_ENV} and ${EVAL_ENV}."
