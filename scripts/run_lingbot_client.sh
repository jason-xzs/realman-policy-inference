#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${ROOT}/fastwam/src/realman_deployment${PYTHONPATH:+:${PYTHONPATH}}"
export REAL_CAMERA_SDK_ROOT="${REAL_CAMERA_SDK_ROOT:-${ROOT}/legacy/scripts/evaluation}"
export REAL_DATASET_PATH="${REAL_DATASET_PATH:-${ROOT}/datasets/lerobot}"
exec "${REALMAN_PYTHON:-python}" "${ROOT}/legacy/scripts/evaluation/client_realman_on_machine.py" "$@"
