#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INFER_PYTHON="${INFER_PYTHON:-/home/suraj/miniforge3/envs/qvg_sf_infer/bin/python}"
EVAL_PYTHON="${EVAL_PYTHON:-/home/suraj/miniforge3/envs/qvg_sf_eval/bin/python}"
RUN_NAME="${RUN_NAME:-spatial_mixed_balanced}"
RUN_TS="${RUN_TS:-$(date +%s)}"
GPU_ID="${GPU_ID:-}"
PYTORCH_ALLOC="${PYTORCH_ALLOC:-expandable_segments:True}"
MAX_PROMPTS="${MAX_PROMPTS:-}"
NUM_OUTPUT_FRAMES="${NUM_OUTPUT_FRAMES:-}"
SEED="${SEED:-0}"
RUN_ROOT="${RUN_ROOT:-}"
SPATIAL_MASK_POLICY="${SPATIAL_MASK_POLICY:-hybrid}"
SPATIAL_VARIANCE_THRESHOLD="${SPATIAL_VARIANCE_THRESHOLD:-0.02}"
SPATIAL_MIN_FOREGROUND_RATIO="${SPATIAL_MIN_FOREGROUND_RATIO:-0.45}"
SPATIAL_MAX_FOREGROUND_RATIO="${SPATIAL_MAX_FOREGROUND_RATIO:-0.85}"
SPATIAL_TARGET_FOREGROUND_RATIO="${SPATIAL_TARGET_FOREGROUND_RATIO:-0.65}"
INCLUDE_QUALITY_VARIANT="${INCLUDE_QUALITY_VARIANT:-0}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-name)
      RUN_NAME="$2"
      shift 2
      ;;
    --run-ts)
      RUN_TS="$2"
      shift 2
      ;;
    --run-root)
      RUN_ROOT="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1"
      echo "Usage: $0 [--run-name NAME] [--run-ts UNIX_TS] [--run-root PATH]"
      exit 1
      ;;
  esac
done

if [[ -z "${GPU_ID}" ]]; then
  GPU_ID="$(
    nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
      | sort -t',' -k2 -nr \
      | head -n1 \
      | cut -d',' -f1 \
      | tr -d ' '
  )"
fi

SAFE_RUN_NAME="$(echo "${RUN_NAME}" | tr ' ' '_' | tr -cd '[:alnum:]_.-')"
RUN_ID="${SAFE_RUN_NAME}_${RUN_TS}"
RUN_ROOT="${RUN_ROOT:-${ROOT_DIR}/results/runs/${RUN_ID}}"
mkdir -p "${RUN_ROOT}/videos" "${RUN_ROOT}/metrics" "${RUN_ROOT}/logs" "${RUN_ROOT}/tables" "${RUN_ROOT}/plots"

cat > "${RUN_ROOT}/run_meta.json" <<EOF
{
  "run_name": "${RUN_NAME}",
  "safe_run_name": "${SAFE_RUN_NAME}",
  "run_timestamp_unix": ${RUN_TS},
  "run_id": "${RUN_ID}",
  "run_root": "${RUN_ROOT}",
  "experiment_type": "spatial_mixed_iter2"
}
EOF

METHODS=(
  BF16
  RTN_INT4
  SPATIAL_MIXED_FG_RTN_INT4_BG_RTN_INT4
  SPATIAL_MIXED_FG_RTN_INT4_BG_RTN_INT2
)
if [[ "${INCLUDE_QUALITY_VARIANT}" == "1" ]]; then
  METHODS+=(SPATIAL_MIXED_FG_QUAROT_KV_INT4_BG_RTN_INT2)
fi

EXTRA_ARGS=()
if [[ -n "${MAX_PROMPTS}" ]]; then
  EXTRA_ARGS+=(--max-prompts "${MAX_PROMPTS}")
fi
if [[ -n "${NUM_OUTPUT_FRAMES}" ]]; then
  EXTRA_ARGS+=(--num-output-frames "${NUM_OUTPUT_FRAMES}")
fi

echo "[run_id] ${RUN_ID}"
echo "[run_root] ${RUN_ROOT}"

for method in "${METHODS[@]}"; do
  echo "[run] ${method}"
  log_file="${RUN_ROOT}/logs/generate_${method}.log"
  if [[ "${method}" == SPATIAL_MIXED* ]]; then
    fg_method="RTN"
    fg_bits="4"
    bg_method="RTN"
    bg_bits="2"
    if [[ "${method}" == "SPATIAL_MIXED_FG_RTN_INT4_BG_RTN_INT4" ]]; then
      bg_bits="4"
    elif [[ "${method}" == "SPATIAL_MIXED_FG_QUAROT_KV_INT4_BG_RTN_INT2" ]]; then
      fg_method="QUAROT_KV"
    fi
    PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_ALLOC}" CUDA_VISIBLE_DEVICES="${GPU_ID}" "${INFER_PYTHON}" "${ROOT_DIR}/scripts/01_generate.py" \
      --method SPATIAL_MIXED \
      --spatial-fg-method "${fg_method}" \
      --spatial-fg-bits "${fg_bits}" \
      --spatial-bg-method "${bg_method}" \
      --spatial-bg-bits "${bg_bits}" \
      --spatial-mask-policy "${SPATIAL_MASK_POLICY}" \
      --spatial-variance-threshold "${SPATIAL_VARIANCE_THRESHOLD}" \
      --spatial-min-foreground-ratio "${SPATIAL_MIN_FOREGROUND_RATIO}" \
      --spatial-max-foreground-ratio "${SPATIAL_MAX_FOREGROUND_RATIO}" \
      --spatial-target-foreground-ratio "${SPATIAL_TARGET_FOREGROUND_RATIO}" \
      --seed "${SEED}" \
      --device cuda:0 \
      --use-ema \
      --results-root "${RUN_ROOT}" \
      "${EXTRA_ARGS[@]}" >"${log_file}" 2>&1
  else
    PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_ALLOC}" CUDA_VISIBLE_DEVICES="${GPU_ID}" "${INFER_PYTHON}" "${ROOT_DIR}/scripts/01_generate.py" \
      --method "${method}" \
      --seed "${SEED}" \
      --device cuda:0 \
      --use-ema \
      --results-root "${RUN_ROOT}" \
      "${EXTRA_ARGS[@]}" >"${log_file}" 2>&1
  fi
done

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
    --device cpu >"${RUN_ROOT}/logs/fidelity_${method}.log" 2>&1
done

"${INFER_PYTHON}" "${ROOT_DIR}/scripts/05_summarize_results.py" --results-root "${RUN_ROOT}" --methods "${METHODS[@]}"

echo "Spatial mixed experiment completed for ${RUN_ID}."
