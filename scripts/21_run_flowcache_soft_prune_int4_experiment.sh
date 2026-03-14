#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export RUN_NAME="${RUN_NAME:-flowcache_soft_prune_int4_multiprompt}"
export MAX_PROMPTS="${MAX_PROMPTS:-5}"
export METHODS_CSV="${METHODS_CSV:-BF16,RTN_INT4,FLOWCACHE_PRUNE_INT4,FLOWCACHE_SOFT_PRUNE_INT2,FLOWCACHE_SOFT_PRUNE_INT4}"

exec bash "${ROOT_DIR}/scripts/20_run_flowcache_soft_prune_experiment.sh" "$@"
