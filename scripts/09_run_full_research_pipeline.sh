#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

"${ROOT_DIR}/scripts/07_run_generation_multi_gpu.sh"
"${ROOT_DIR}/scripts/08_run_evaluation_suite.sh"

echo "Full baseline replication pipeline completed."
