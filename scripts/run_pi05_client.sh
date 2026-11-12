#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${ROOT}/.venv/bin/python"
[[ -x "${PYTHON}" ]] || PYTHON=python3
PYTHON="${REALMAN_PYTHON:-${PYTHON}}"
CONFIG="${PI05_CONFIG:-${ROOT}/pi05/configs/pi05_realman_presets.local.toml}"
for arg in "$@"; do
  if [[ "${arg}" == --help || "${arg}" == -h ]]; then
    exec "${PYTHON}" "${ROOT}/pi05/scripts/run_pi05_realman.py" --help
  fi
done
if [[ ! -f "${CONFIG}" ]]; then
  "${PYTHON}" "${ROOT}/scripts/prepare_pi05_config.py" --output "${CONFIG}"
fi
exec "${PYTHON}" "${ROOT}/pi05/scripts/run_pi05_realman.py" --config "${CONFIG}" "$@"
