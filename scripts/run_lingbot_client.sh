#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${ROOT}/fastwam/src/realman_deployment${PYTHONPATH:+:${PYTHONPATH}}"
export REAL_CAMERA_SDK_ROOT="${REAL_CAMERA_SDK_ROOT:-${ROOT}/legacy/scripts/evaluation}"
export REAL_DATASET_PATH="${REAL_DATASET_PATH:-${ROOT}/datasets/lerobot}"
export REAL_HIGH_CAMERA_SERIAL="${REAL_HIGH_CAMERA_SERIAL:-420122071413}"
export REAL_WRIST_CAMERA_SERIAL="${REAL_WRIST_CAMERA_SERIAL:-323622272614}"
PYTHON="${ROOT}/.venv/bin/python"
[[ -x "${PYTHON}" ]] || PYTHON=python3
exec "${REALMAN_PYTHON:-${PYTHON}}" "${ROOT}/legacy/scripts/evaluation/client_realman_on_machine.py" \
  --action-stats-path "${REAL_ACTION_STATS_PATH:-${ROOT}/configs/lingbot/action_norm_stats.json}" "$@"
