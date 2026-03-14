#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_ROOT="${RUN_ROOT:-${ROOT_DIR}/results}"
METHOD="${1:?usage: 03_eval_vbench.sh <METHOD> [VIDEOS_DIR]}"
VIDEOS_DIR="${2:-${RUN_ROOT}/videos/${METHOD}}"
PROMPT_FILE="${3:-${ROOT_DIR}/prompts/MovieGenVideoBench_extended.txt}"
OUT_DIR="${4:-${RUN_ROOT}/metrics/vbench_${METHOD}}"
FINAL_JSON="${5:-${RUN_ROOT}/metrics/vbench_${METHOD}.json}"
MASTER_PORT="${MASTER_PORT:-$((15000 + ($$ % 20000)))}"

mkdir -p "${OUT_DIR}"
export MASTER_PORT
echo "Using MASTER_PORT=${MASTER_PORT}"

PROMPT_MAP_JSON="${OUT_DIR}/prompt_map.json"
"${PYTHON_BIN}" - <<'PY' "${VIDEOS_DIR}" "${PROMPT_FILE}" "${PROMPT_MAP_JSON}"
import json
import re
import sys
from pathlib import Path

videos_dir = Path(sys.argv[1])
prompt_file = Path(sys.argv[2])
out_json = Path(sys.argv[3])

prompts = [x.rstrip("\n") for x in prompt_file.read_text(encoding="utf-8").splitlines()]
mapping = {}
for p in sorted(videos_dir.glob("prompt_*_seed_*.mp4")):
    m = re.search(r"prompt_(\d+)_seed_", p.name)
    if not m:
        continue
    idx = int(m.group(1))
    if idx >= len(prompts):
        continue
    mapping[str(p)] = prompts[idx]
out_json.write_text(json.dumps(mapping, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"prompt_map_size={len(mapping)}")
PY

"${PYTHON_BIN}" "${ROOT_DIR}/third_party/VBench/evaluate.py" \
  --videos_path "${VIDEOS_DIR}" \
  --mode custom_input \
  --prompt_file "${PROMPT_MAP_JSON}" \
  --dimension background_consistency imaging_quality subject_consistency aesthetic_quality \
  --output_path "${OUT_DIR}"

LATEST_JSON="$(ls -1t "${OUT_DIR}"/*_eval_results.json | head -n 1)"
cp "${LATEST_JSON}" "${FINAL_JSON}"
echo "Saved ${FINAL_JSON}"
