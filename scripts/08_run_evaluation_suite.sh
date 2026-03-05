#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INFER_PYTHON="${INFER_PYTHON:-/home/suraj/miniforge3/envs/qvg_sf_infer/bin/python}"
EVAL_PYTHON="${EVAL_PYTHON:-/home/suraj/miniforge3/envs/qvg_sf_eval/bin/python}"
RUN_ROOT="${RUN_ROOT:-${ROOT_DIR}/results}"
GPU_ID="${GPU_ID:-0}"

METHODS=(
  BF16
  RTN_INT4
  RTN_INT2
  KIVI_INT4
  KIVI_INT2
  QUAROT_KV_INT4
  QUAROT_KV_INT2
)

mkdir -p "${RUN_ROOT}/logs" "${RUN_ROOT}/metrics" "${RUN_ROOT}/tables"
echo "[run_root] ${RUN_ROOT}"

for method in "${METHODS[@]}"; do
  echo "[vbench] ${method}"
  CUDA_VISIBLE_DEVICES="${GPU_ID}" PYTHON_BIN="${EVAL_PYTHON}" RUN_ROOT="${RUN_ROOT}" \
    "${ROOT_DIR}/scripts/03_eval_vbench.sh" "${method}" "${RUN_ROOT}/videos/${method}" "${ROOT_DIR}/prompts/MovieGenVideoBench_extended.txt" "${RUN_ROOT}/metrics/vbench_${method}" "${RUN_ROOT}/metrics/vbench_${method}.json" \
    >"${RUN_ROOT}/logs/vbench_${method}.log" 2>&1

done

for method in "${METHODS[@]}"; do
  if [[ "${method}" == "BF16" ]]; then
    continue
  fi
  echo "[fidelity] ${method}"
  "${INFER_PYTHON}" "${ROOT_DIR}/scripts/02_eval_fidelity.py" \
    --bf16-dir "${RUN_ROOT}/videos/BF16" \
    --candidate-dir "${RUN_ROOT}/videos/${method}" \
    --output "${RUN_ROOT}/metrics/fidelity_${method}.json" \
    --device cpu \
    >"${RUN_ROOT}/logs/fidelity_${method}.log" 2>&1

done

"${INFER_PYTHON}" "${ROOT_DIR}/scripts/05_summarize_results.py" --results-root "${RUN_ROOT}"

echo "Evaluation suite completed."
