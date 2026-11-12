#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then
  echo "Usage: PYTHON_BIN=python3 bash scripts/setup_client.sh"
  echo "Installs the robot client into .venv using Python 3.11 or 3.12."
  exit 0
fi
PYTHON_BIN="${PYTHON_BIN:-python3}"
"${PYTHON_BIN}" -c 'import sys; assert (3, 11) <= sys.version_info[:2] < (3, 13), "Use Python 3.11 or 3.12 (set PYTHON_BIN if needed)"'
if [[ ! -x "${ROOT}/.venv/bin/python" ]]; then
  "${PYTHON_BIN}" -m venv "${ROOT}/.venv"
fi
"${ROOT}/.venv/bin/python" -m pip install -r "${ROOT}/requirements-client.txt"
"${ROOT}/.venv/bin/python" -m pip install -e "${ROOT}/legacy/openpi/openpi-client" --no-deps
"${ROOT}/.venv/bin/python" -m pip install -e "${ROOT}/pi05" --no-deps
"${ROOT}/.venv/bin/python" -c 'import cv2, numpy, PIL, pyrealsense2; from openpi_client.websocket_client_policy import WebsocketClientPolicy; import robot_client_runtime.production_pi05; print("Client imports OK")'
echo "Client environment ready. Launchers will use .venv automatically."
