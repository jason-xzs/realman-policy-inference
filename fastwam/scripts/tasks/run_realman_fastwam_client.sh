#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."

FASTWAM_PYTHON="${FASTWAM_PYTHON:-python}"

export PYTHONPATH="$(pwd)/src${PYTHONPATH:+:${PYTHONPATH}}"
SERVER_URL="${SERVER_URL:-http://127.0.0.1:18000}"
NO_PROXY="${NO_PROXY:+${NO_PROXY},}127.0.0.1,localhost"
no_proxy="${no_proxy:+${no_proxy},}127.0.0.1,localhost"
export NO_PROXY
export no_proxy
LEFT_ARM_IP="${LEFT_ARM_IP:-192.168.1.18}"
RIGHT_ARM_IP="${RIGHT_ARM_IP:-192.168.1.19}"
ARM_PORT="${ARM_PORT:-8080}"
HIGH_CAMERA_ID="${HIGH_CAMERA_ID:-420122071413}"
LEFT_CAMERA_ID="${LEFT_CAMERA_ID:-323622272614}"
RIGHT_CAMERA_ID="${RIGHT_CAMERA_ID:-230422272673}"

MOCK_HARDWARE=0
CLI_EXECUTE=0
for arg in "$@"; do
  case "${arg}" in
    --mock-hardware) MOCK_HARDWARE=1 ;;
    --execute) CLI_EXECUTE=1 ;;
  esac
done

if [[ "${MOCK_HARDWARE}" == "0" ]]; then
  : "${HIGH_CAMERA_ID:?Set HIGH_CAMERA_ID to the overhead RealSense serial number}"
  : "${LEFT_CAMERA_ID:?Set LEFT_CAMERA_ID to the left-wrist RealSense serial number}"
  : "${RIGHT_CAMERA_ID:?Set RIGHT_CAMERA_ID to the right-wrist RealSense serial number}"
fi

HIGH_CAMERA_ID="${HIGH_CAMERA_ID:-}"
LEFT_CAMERA_ID="${LEFT_CAMERA_ID:-}"
RIGHT_CAMERA_ID="${RIGHT_CAMERA_ID:-}"

EXECUTE_ARGS=()
if [[ "${EXECUTE:-0}" == "1" ]]; then
  EXECUTE_ARGS+=(--execute)
elif [[ "${CLI_EXECUTE}" == "0" ]]; then
  echo "[safety] dry-run mode; set EXECUTE=1 only after validating camera mapping and actions"
fi

exec "${FASTWAM_PYTHON}" -m realman_deployment.client \
  --server-url "${SERVER_URL}" \
  --left-ip "${LEFT_ARM_IP}" \
  --right-ip "${RIGHT_ARM_IP}" \
  --arm-port "${ARM_PORT}" \
  --high-camera-id "${HIGH_CAMERA_ID}" \
  --left-camera-id "${LEFT_CAMERA_ID}" \
  --right-camera-id "${RIGHT_CAMERA_ID}" \
  "${EXECUTE_ARGS[@]}" \
  "$@"
