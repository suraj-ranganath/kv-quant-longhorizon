#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/suraj/miniforge3/envs/dit_mem/bin/python}"
GPU_ID="${GPU_ID:-6}"  # physical GPU id, constrained to 6 or 7 by project policy
MAX_PROMPTS="${MAX_PROMPTS:-5}"
SEED="${SEED:-0}"

METHODS=(
  BF16
  RTN_INT4
  RTN_INT2
  KIVI_INT4
  KIVI_INT2
  QUAROT_KV_INT4
  QUAROT_KV_INT2
)

for method in "${METHODS[@]}"; do
  echo "[run] ${method}"
  CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" "${ROOT_DIR}/scripts/01_generate.py" \
    --method "${method}" \
    --max-prompts "${MAX_PROMPTS}" \
    --seed "${SEED}" \
    --device cuda:0 \
    --use-ema

done

for method in "${METHODS[@]}"; do
  if [[ "${method}" == "BF16" ]]; then
    continue
  fi

  echo "[fidelity] ${method}"
  "${PYTHON_BIN}" "${ROOT_DIR}/scripts/02_eval_fidelity.py" \
    --bf16-dir "${ROOT_DIR}/results/videos/BF16" \
    --candidate-dir "${ROOT_DIR}/results/videos/${method}" \
    --output "${ROOT_DIR}/results/metrics/fidelity_${method}.json" \
    --device cpu

  echo "[vbench] ${method}"
  CUDA_VISIBLE_DEVICES="${GPU_ID}" "${ROOT_DIR}/scripts/03_eval_vbench.sh" "${method}" "${ROOT_DIR}/results/videos/${method}" "${ROOT_DIR}/prompts/MovieGenVideoBench_extended.txt"
done

"${PYTHON_BIN}" "${ROOT_DIR}/scripts/05_summarize_results.py"
