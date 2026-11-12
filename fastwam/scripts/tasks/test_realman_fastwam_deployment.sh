#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."
FASTWAM_PYTHON="${FASTWAM_PYTHON:-python}"

export PYTHONPATH="$(pwd)/src${PYTHONPATH:+:${PYTHONPATH}}"

"${FASTWAM_PYTHON}" -m unittest discover -s src/realman_deployment/tests -v
"${FASTWAM_PYTHON}" -m realman_deployment.simulate --replans 3 --actions-per-replan 4
