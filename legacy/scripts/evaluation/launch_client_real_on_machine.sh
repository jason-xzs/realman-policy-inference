#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

export REAL_SERVER_HOST="${REAL_SERVER_HOST:-127.0.0.1}"
export REAL_SERVER_PORT="${REAL_SERVER_PORT:-18080}"
export REAL_DATASET_PATH="${REAL_DATASET_PATH:-/data1/public_data/xzs_data/real_data/lerobot}"
export REAL_WEBSOCKET_PING_INTERVAL="${REAL_WEBSOCKET_PING_INTERVAL:-0}"
export REAL_WEBSOCKET_PING_TIMEOUT="${REAL_WEBSOCKET_PING_TIMEOUT:-600}"
export REAL_WEBSOCKET_CLOSE_TIMEOUT="${REAL_WEBSOCKET_CLOSE_TIMEOUT:-120}"

CLIENT_PY="${REAL_CLIENT_PY:-${SCRIPT_DIR}/client_realman_on_machine.py}"
if [[ ! -f "${CLIENT_PY}" && -f "${SCRIPT_DIR}/client_realman.py" ]]; then
  CLIENT_PY="${SCRIPT_DIR}/client_realman.py"
fi

python "${CLIENT_PY}" \
  --server-host "${REAL_SERVER_HOST}" \
  --server-port "${REAL_SERVER_PORT}" \
  "$@"
