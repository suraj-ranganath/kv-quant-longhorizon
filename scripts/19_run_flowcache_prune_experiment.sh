#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INFER_PYTHON="${INFER_PYTHON:-/home/suraj/miniforge3/envs/qvg_sf_infer/bin/python}"
EVAL_PYTHON="${EVAL_PYTHON:-/home/suraj/miniforge3/envs/qvg_sf_eval/bin/python}"
RUN_NAME="${RUN_NAME:-flowcache_prune_smoke}"
RUN_TS="${RUN_TS:-$(date +%s)}"
GPU_ID="${GPU_ID:-}"
PYTORCH_ALLOC="${PYTORCH_ALLOC:-expandable_segments:True}"
MAX_PROMPTS="${MAX_PROMPTS:-1}"
NUM_OUTPUT_FRAMES="${NUM_OUTPUT_FRAMES:-21}"
SEED="${SEED:-0}"
RUN_ROOT="${RUN_ROOT:-}"

FLOWCACHE_RECENT_RATIO="${FLOWCACHE_RECENT_RATIO:-0.25}"
FLOWCACHE_RECENT_BITS="${FLOWCACHE_RECENT_BITS:-4}"
FLOWCACHE_RECENT_METHOD="${FLOWCACHE_RECENT_METHOD:-RTN}"
FLOWCACHE_OLD_METHOD="${FLOWCACHE_OLD_METHOD:-RTN}"
FLOWCACHE_IMPORTANT_OLD_RATIO="${FLOWCACHE_IMPORTANT_OLD_RATIO:-0.10}"
FLOWCACHE_IMPORTANCE_ALPHA="${FLOWCACHE_IMPORTANCE_ALPHA:-0.7}"
FLOWCACHE_IMPORTANCE_BETA="${FLOWCACHE_IMPORTANCE_BETA:-0.3}"
FLOWCACHE_MIN_LAYER_BUDGET_SCALE="${FLOWCACHE_MIN_LAYER_BUDGET_SCALE:-0.70}"
FLOWCACHE_MAX_LAYER_BUDGET_SCALE="${FLOWCACHE_MAX_LAYER_BUDGET_SCALE:-1.30}"
FLOWCACHE_PROFILE_MIN_SCALE="${FLOWCACHE_PROFILE_MIN_SCALE:-0.70}"
FLOWCACHE_PROFILE_MAX_SCALE="${FLOWCACHE_PROFILE_MAX_SCALE:-1.30}"
FLOWCACHE_PRUNE_RETAINED_OLD_RATIO="${FLOWCACHE_PRUNE_RETAINED_OLD_RATIO:-0.30}"
FLOWCACHE_PRUNE_REFRESH_GAP_CHUNKS="${FLOWCACHE_PRUNE_REFRESH_GAP_CHUNKS:-1}"

AGE_TIER_RECENT_RATIO="${AGE_TIER_RECENT_RATIO:-${FLOWCACHE_RECENT_RATIO}}"
AGE_TIER_RECENT_BITS="${AGE_TIER_RECENT_BITS:-${FLOWCACHE_RECENT_BITS}}"
AGE_TIER_RECENT_METHOD="${AGE_TIER_RECENT_METHOD:-${FLOWCACHE_RECENT_METHOD}}"
AGE_TIER_OLD_METHOD="${AGE_TIER_OLD_METHOD:-${FLOWCACHE_OLD_METHOD}}"

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
PROFILE_ROOT="${RUN_ROOT}/profile_pass"
mkdir -p "${RUN_ROOT}/videos" "${RUN_ROOT}/metrics" "${RUN_ROOT}/logs" "${RUN_ROOT}/tables" "${RUN_ROOT}/plots" "${PROFILE_ROOT}"

cat > "${RUN_ROOT}/run_meta.json" <<EOF
{
  "run_name": "${RUN_NAME}",
  "safe_run_name": "${SAFE_RUN_NAME}",
  "run_timestamp_unix": ${RUN_TS},
  "run_id": "${RUN_ID}",
  "run_root": "${RUN_ROOT}",
  "experiment_type": "flowcache_prune_experiment",
  "flowcache_recent_ratio": ${FLOWCACHE_RECENT_RATIO},
  "flowcache_recent_bits": ${FLOWCACHE_RECENT_BITS},
  "flowcache_recent_method": "${FLOWCACHE_RECENT_METHOD}",
  "flowcache_old_method": "${FLOWCACHE_OLD_METHOD}",
  "flowcache_important_old_ratio": ${FLOWCACHE_IMPORTANT_OLD_RATIO},
  "flowcache_prune_retained_old_ratio": ${FLOWCACHE_PRUNE_RETAINED_OLD_RATIO},
  "flowcache_importance_alpha": ${FLOWCACHE_IMPORTANCE_ALPHA},
  "flowcache_importance_beta": ${FLOWCACHE_IMPORTANCE_BETA}
}
EOF

echo "[run_id] ${RUN_ID}"
echo "[run_root] ${RUN_ROOT}"
echo "[profile] FLOWCACHE_PROFILE"
PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_ALLOC}" CUDA_VISIBLE_DEVICES="${GPU_ID}" \
  "${INFER_PYTHON}" "${ROOT_DIR}/scripts/01_generate.py" \
  --method FLOWCACHE_PROFILE \
  --seed "${SEED}" \
  --device cuda:0 \
  --use-ema \
  --results-root "${PROFILE_ROOT}" \
  --max-prompts "${MAX_PROMPTS}" \
  --num-output-frames "${NUM_OUTPUT_FRAMES}" \
  --flowcache-recent-ratio "${FLOWCACHE_RECENT_RATIO}" \
  >"${RUN_ROOT}/logs/profile_FLOWCACHE_PROFILE.log" 2>&1

LAYER_BUDGET_PATH="${RUN_ROOT}/metrics/flowcache_profile_layer_budget.json"
PROFILE_EFF_PATH="${PROFILE_ROOT}/metrics/efficiency_FLOWCACHE_PROFILE.json"
python3 - <<PY
import json
from pathlib import Path

profile_eff_path = Path(${PROFILE_EFF_PATH@Q})
output_path = Path(${LAYER_BUDGET_PATH@Q})
payload = json.loads(profile_eff_path.read_text(encoding="utf-8"))
scores = payload.get("flowcache_profile_layer_scores", {})
min_scale = float(${FLOWCACHE_PROFILE_MIN_SCALE@Q})
max_scale = float(${FLOWCACHE_PROFILE_MAX_SCALE@Q})
if not isinstance(scores, dict) or not scores:
    table = {}
else:
    items = {int(k): float(v) for k, v in scores.items()}
    min_score = min(items.values())
    max_score = max(items.values())
    table = {}
    for layer_id, score in sorted(items.items()):
        if max_score > min_score:
            norm = (score - min_score) / (max_score - min_score)
        else:
            norm = 0.5
        table[layer_id] = min_scale + norm * (max_scale - min_scale)
output_path.write_text(json.dumps(table, indent=2), encoding="utf-8")
print(json.dumps(table, indent=2))
PY

METHODS=(
  BF16
  RTN_INT4
  AGE_TIER_INT2
  FLOWCACHE_HYBRID_INT2
  FLOWCACHE_ADAPTIVE_INT2
  FLOWCACHE_PRUNE_INT2
)

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

  if [[ "${method}" == "AGE_TIER_INT2" ]]; then
    base_cmd+=(
      --age-tier-recent-ratio "${AGE_TIER_RECENT_RATIO}"
      --age-tier-recent-bits "${AGE_TIER_RECENT_BITS}"
      --age-tier-recent-method "${AGE_TIER_RECENT_METHOD}"
      --age-tier-old-method "${AGE_TIER_OLD_METHOD}"
    )
  elif [[ "${method}" == "FLOWCACHE_HYBRID_INT2" || "${method}" == "FLOWCACHE_ADAPTIVE_INT2" || "${method}" == "FLOWCACHE_PRUNE_INT2" ]]; then
    base_cmd+=(
      --flowcache-recent-ratio "${FLOWCACHE_RECENT_RATIO}"
      --flowcache-recent-bits "${FLOWCACHE_RECENT_BITS}"
      --flowcache-recent-method "${FLOWCACHE_RECENT_METHOD}"
      --flowcache-old-method "${FLOWCACHE_OLD_METHOD}"
      --flowcache-important-old-ratio "${FLOWCACHE_IMPORTANT_OLD_RATIO}"
      --flowcache-importance-alpha "${FLOWCACHE_IMPORTANCE_ALPHA}"
      --flowcache-importance-beta "${FLOWCACHE_IMPORTANCE_BETA}"
      --flowcache-layer-budget-path "${LAYER_BUDGET_PATH}"
      --flowcache-min-layer-budget-scale "${FLOWCACHE_MIN_LAYER_BUDGET_SCALE}"
      --flowcache-max-layer-budget-scale "${FLOWCACHE_MAX_LAYER_BUDGET_SCALE}"
    )
    if [[ "${method}" == "FLOWCACHE_PRUNE_INT2" ]]; then
      base_cmd+=(
        --flowcache-prune-retained-old-ratio "${FLOWCACHE_PRUNE_RETAINED_OLD_RATIO}"
        --flowcache-prune-refresh-gap-chunks "${FLOWCACHE_PRUNE_REFRESH_GAP_CHUNKS}"
      )
    fi
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

echo "FlowCache prune experiment completed for ${RUN_ID}."
