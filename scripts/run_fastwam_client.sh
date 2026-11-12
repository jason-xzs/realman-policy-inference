#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${ROOT}/.venv/bin/python"
[[ -x "${PYTHON}" ]] || PYTHON=python3
export FASTWAM_PYTHON="${FASTWAM_PYTHON:-${REALMAN_PYTHON:-${PYTHON}}}"
exec bash "${ROOT}/fastwam/scripts/tasks/run_realman_fastwam_client.sh" --record-dir "" "$@"
