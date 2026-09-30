#!/usr/bin/env bash
set -euo pipefail
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
data="${HUMANOID_DATA_ROOT:-$repo/data}"
python="${FLUX_MODEL_PYTHON:-$data/venvs/flux-model/bin/python}"
export PYTHONPATH="$repo/src:$repo/third_party/flux/flux-training/src${PYTHONPATH:+:$PYTHONPATH}"
host="${FLUX_EVAL_HOST:-$(tailscale ip -4 | head -1)}"
exec "$python" -m uvicorn humanoid_lab.evaluation.app:app --host "$host" --port "${FLUX_EVAL_PORT:-7860}"
