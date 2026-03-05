#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DASHBOARD_PYTHON="${DASHBOARD_PYTHON:-python}"
PORT="${PORT:-8501}"

cd "${ROOT_DIR}"
"${DASHBOARD_PYTHON}" -m streamlit run dashboard/app.py --server.port "${PORT}" --server.headless true
