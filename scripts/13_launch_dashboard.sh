#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DASHBOARD_PYTHON="${DASHBOARD_PYTHON:-python}"
PORT="${PORT:-8501}"
ADDRESS="${ADDRESS:-0.0.0.0}"

if command -v hostname >/dev/null 2>&1; then
  HOST_IP="$(hostname -I 2>/dev/null | awk '{print $1}')"
else
  HOST_IP=""
fi

cd "${ROOT_DIR}"
echo "Launching dashboard from ${ROOT_DIR}"
echo "Bind address: ${ADDRESS}"
echo "Local URL:   http://localhost:${PORT}"
if [[ -n "${HOST_IP}" ]]; then
  echo "Network URL: http://${HOST_IP}:${PORT}"
fi

"${DASHBOARD_PYTHON}" -m streamlit run dashboard/app.py \
  --server.address "${ADDRESS}" \
  --server.port "${PORT}" \
  --server.headless true
