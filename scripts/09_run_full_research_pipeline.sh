#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_NAME="${RUN_NAME:-baseline}"
RUN_TS="${RUN_TS:-$(date +%s)}"
NUM_OUTPUT_FRAMES="${NUM_OUTPUT_FRAMES:-}"

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
  "num_output_frames": "${NUM_OUTPUT_FRAMES}"
}
EOF

echo "[run_id] ${RUN_ID}"
echo "[run_root] ${RUN_ROOT}"

RUN_ROOT="${RUN_ROOT}" "${ROOT_DIR}/scripts/07_run_generation_multi_gpu.sh"
RUN_ROOT="${RUN_ROOT}" "${ROOT_DIR}/scripts/08_run_evaluation_suite.sh"

echo "Full baseline replication pipeline completed for ${RUN_ID}."
