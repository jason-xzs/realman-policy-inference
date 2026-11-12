#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec bash "${ROOT}/fastwam/scripts/tasks/run_realman_fastwam_client.sh" "$@"
