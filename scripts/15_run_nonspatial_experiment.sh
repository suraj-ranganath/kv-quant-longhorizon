#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INFER_PYTHON="${INFER_PYTHON:-/home/suraj/miniforge3/envs/qvg_sf_infer/bin/python}"
EVAL_PYTHON="${EVAL_PYTHON:-/home/suraj/miniforge3/envs/qvg_sf_eval/bin/python}"
RUN_NAME="${RUN_NAME:-nonspatial_bundle}"
RUN_TS="${RUN_TS:-$(date +%s)}"
GPU_ID="${GPU_ID:-}"
PYTORCH_ALLOC="${PYTORCH_ALLOC:-expandable_segments:True}"
MAX_PROMPTS="${MAX_PROMPTS:-1}"
NUM_OUTPUT_FRAMES="${NUM_OUTPUT_FRAMES:-21}"
SEED="${SEED:-0}"
RUN_ROOT="${RUN_ROOT:-}"

PRQ_RESIDUAL_BITS="${PRQ_RESIDUAL_BITS:-4}"
QAQ_OUTLIER_THRESHOLD="${QAQ_OUTLIER_THRESHOLD:-6.0}"
AGE_TIER_RECENT_RATIO="${AGE_TIER_RECENT_RATIO:-0.3}"
AGE_TIER_RECENT_BITS="${AGE_TIER_RECENT_BITS:-4}"
AGE_TIER_RECENT_METHOD="${AGE_TIER_RECENT_METHOD:-RTN}"
AGE_TIER_OLD_METHOD="${AGE_TIER_OLD_METHOD:-RTN}"

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
  "experiment_type": "nonspatial_bundle"
}
EOF

METHODS=(
  BF16
  RTN_INT4
  PRQ_INT4
  PRQ_INT2
  QAQ_INT4
  QAQ_INT2
  AGE_TIER_INT4
  AGE_TIER_INT2
)

echo "[run_id] ${RUN_ID}"
echo "[run_root] ${RUN_ROOT}"

for method in "${METHODS[@]}"; do
  echo "[run] ${method}"
  log_file="${RUN_ROOT}/logs/generate_${method}.log"
  base_cmd=(
    "${INFER_PYTHON}" "${ROOT_DIR}/scripts/01_generate.py"
    --method "${method}"
    --seed "${SEED}"
    --device cuda:0
    --use-ema
    --results-root "${RUN_ROOT}"
    --max-prompts "${MAX_PROMPTS}"
    --num-output-frames "${NUM_OUTPUT_FRAMES}"
  )

  if [[ "${method}" == PRQ_* ]]; then
    base_cmd+=(--prq-residual-bits "${PRQ_RESIDUAL_BITS}")
  elif [[ "${method}" == QAQ_* ]]; then
    base_cmd+=(--qaq-outlier-threshold "${QAQ_OUTLIER_THRESHOLD}")
  elif [[ "${method}" == AGE_TIER_* ]]; then
    base_cmd+=(
      --age-tier-recent-ratio "${AGE_TIER_RECENT_RATIO}"
      --age-tier-recent-bits "${AGE_TIER_RECENT_BITS}"
      --age-tier-recent-method "${AGE_TIER_RECENT_METHOD}"
      --age-tier-old-method "${AGE_TIER_OLD_METHOD}"
    )
  fi

  PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_ALLOC}" CUDA_VISIBLE_DEVICES="${GPU_ID}" "${base_cmd[@]}" >"${log_file}" 2>&1
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

echo "Non-spatial experiment completed for ${RUN_ID}."
