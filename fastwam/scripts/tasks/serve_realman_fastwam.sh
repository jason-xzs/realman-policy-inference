#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."

FASTWAM_PYTHON="${FASTWAM_PYTHON:-python}"

export PYTHONPATH="$(pwd)/src${PYTHONPATH:+:${PYTHONPATH}}"
export DIFFSYNTH_MODEL_BASE_PATH="${DIFFSYNTH_MODEL_BASE_PATH:-$(pwd)/checkpoints}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

CHECKPOINT="${CHECKPOINT:-runs/realman_demo_organize_finetune_3cam_384_1e-5/2026-08-17_22-06-55/checkpoints/weights/step_015000.pt}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
REPLAN_STEPS="${REPLAN_STEPS:-32}"
INFERENCE_STEPS="${INFERENCE_STEPS:-10}"

exec "${FASTWAM_PYTHON}" -m realman_deployment.server \
  --host "${HOST}" \
  --port "${PORT}" \
  --checkpoint "${CHECKPOINT}" \
  --replan-steps "${REPLAN_STEPS}" \
  --num-inference-steps "${INFERENCE_STEPS}" \
  "$@"
