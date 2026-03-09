#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INFER_PYTHON="${INFER_PYTHON:-/home/suraj/miniforge3/envs/qvg_sf_infer/bin/python}"
RUN_ROOT="${RUN_ROOT:-${ROOT_DIR}/results}"
SEED="${SEED:-0}"
MAX_PROMPTS="${MAX_PROMPTS:-}"
NUM_OUTPUT_FRAMES="${NUM_OUTPUT_FRAMES:-42}"
PYTORCH_ALLOC="${PYTORCH_ALLOC:-expandable_segments:True}"

# Comma-separated, e.g. "0,1,2,3,4,5"
GPU_LIST_RAW="${GPU_LIST:-0,1,2,3,4,5}"
IFS=',' read -r -a GPUS <<< "${GPU_LIST_RAW}"

METHODS=(
  BF16
  RTN_INT4
  RTN_INT2
  KIVI_INT4
  KIVI_INT2
  QUAROT_KV_INT4
  QUAROT_KV_INT2
)

mkdir -p "${RUN_ROOT}/logs" "${RUN_ROOT}/videos" "${RUN_ROOT}/metrics" "${RUN_ROOT}/tables" "${RUN_ROOT}/plots"
echo "[run_root] ${RUN_ROOT}"

EXTRA_ARGS=()
if [[ -n "${MAX_PROMPTS}" ]]; then
  EXTRA_ARGS+=(--max-prompts "${MAX_PROMPTS}")
fi
if [[ -n "${NUM_OUTPUT_FRAMES}" ]]; then
  EXTRA_ARGS+=(--num-output-frames "${NUM_OUTPUT_FRAMES}")
fi

failed=0
num_methods="${#METHODS[@]}"
num_gpus="${#GPUS[@]}"
start=0

while [[ "${start}" -lt "${num_methods}" ]]; do
  pids=()
  for ((slot=0; slot<num_gpus && start+slot<num_methods; slot++)); do
    idx=$((start + slot))
    method="${METHODS[$idx]}"
    gpu="${GPUS[$slot]}"
    log_file="${RUN_ROOT}/logs/generate_${method}.log"

    echo "[launch] method=${method} gpu=${gpu} log=${log_file}"
    (
      cd "${ROOT_DIR}"
      PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_ALLOC}" CUDA_VISIBLE_DEVICES="${gpu}" \
        "${INFER_PYTHON}" scripts/01_generate.py \
        --method "${method}" \
        --seed "${SEED}" \
        --device cuda:0 \
        --use-ema \
        --results-root "${RUN_ROOT}" \
        "${EXTRA_ARGS[@]}"
    ) >"${log_file}" 2>&1 &
    pids+=("$!")
  done

  for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
      failed=1
    fi
  done
  start=$((start + num_gpus))
done

if [[ "${failed}" -ne 0 ]]; then
  echo "One or more generation jobs failed. Check ${RUN_ROOT}/logs/generate_*.log"
  exit 1
fi

echo "All generation jobs completed successfully."
