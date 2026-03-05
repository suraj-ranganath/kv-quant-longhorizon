#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INFER_PYTHON="${INFER_PYTHON:-/home/suraj/miniforge3/envs/qvg_sf_infer/bin/python}"
SEED="${SEED:-0}"
MAX_PROMPTS="${MAX_PROMPTS:-}"
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

mkdir -p "${ROOT_DIR}/results/logs"

EXTRA_ARGS=()
if [[ -n "${MAX_PROMPTS}" ]]; then
  EXTRA_ARGS+=(--max-prompts "${MAX_PROMPTS}")
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
    log_file="${ROOT_DIR}/results/logs/generate_${method}.log"

    echo "[launch] method=${method} gpu=${gpu} log=${log_file}"
    (
      cd "${ROOT_DIR}"
      PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_ALLOC}" CUDA_VISIBLE_DEVICES="${gpu}" \
        "${INFER_PYTHON}" scripts/01_generate.py \
        --method "${method}" \
        --seed "${SEED}" \
        --device cuda:0 \
        --use-ema \
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
  echo "One or more generation jobs failed. Check results/logs/generate_*.log"
  exit 1
fi

echo "All generation jobs completed successfully."
