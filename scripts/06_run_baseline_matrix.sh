#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INFER_PYTHON="${INFER_PYTHON:-/home/suraj/miniforge3/envs/qvg_sf_infer/bin/python}"
EVAL_PYTHON="${EVAL_PYTHON:-/home/suraj/miniforge3/envs/qvg_sf_eval/bin/python}"
RUN_ROOT="${RUN_ROOT:-${ROOT_DIR}/results}"
GPU_ID="${GPU_ID:-2}"
MAX_PROMPTS="${MAX_PROMPTS:-}"
NUM_OUTPUT_FRAMES="${NUM_OUTPUT_FRAMES:-42}"
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

EXTRA_ARGS=()
if [[ -n "${MAX_PROMPTS}" ]]; then
  EXTRA_ARGS+=(--max-prompts "${MAX_PROMPTS}")
fi
if [[ -n "${NUM_OUTPUT_FRAMES}" ]]; then
  EXTRA_ARGS+=(--num-output-frames "${NUM_OUTPUT_FRAMES}")
fi

for method in "${METHODS[@]}"; do
  echo "[run] ${method}"
  CUDA_VISIBLE_DEVICES="${GPU_ID}" "${INFER_PYTHON}" "${ROOT_DIR}/scripts/01_generate.py" \
    --method "${method}" \
    --seed "${SEED}" \
    --device cuda:0 \
    --use-ema \
    --results-root "${RUN_ROOT}" \
    "${EXTRA_ARGS[@]}"

done

for method in "${METHODS[@]}"; do
  echo "[vbench] ${method}"
  CUDA_VISIBLE_DEVICES="${GPU_ID}" PYTHON_BIN="${EVAL_PYTHON}" RUN_ROOT="${RUN_ROOT}" "${ROOT_DIR}/scripts/03_eval_vbench.sh" "${method}" "${RUN_ROOT}/videos/${method}" "${ROOT_DIR}/prompts/MovieGenVideoBench_extended.txt" "${RUN_ROOT}/metrics/vbench_${method}" "${RUN_ROOT}/metrics/vbench_${method}.json"
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
    --device cpu
done

"${INFER_PYTHON}" "${ROOT_DIR}/scripts/05_summarize_results.py" --results-root "${RUN_ROOT}"
