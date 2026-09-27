#!/usr/bin/env bash
# Host-side GPU model server for the independent Flux ROS process.
#
# Runs the vendored zmq_server.py under the prepared model environment (the
# LeRobot/PEFT interpreter that holds the Flux3 runtime), binds loopback only,
# and defaults to port 5561: the server's own 5557 default is the SONIC
# g1_debug state port on this machine, so the simulation names an explicit port
# and the robot defaults stay untouched.
#
# Interpreter resolution (first hit wins):
#   1. --python PATH
#   2. $FLUX_MODEL_PYTHON
#   3. the persistent environment $HUMANOID_DATA_ROOT/venvs/flux-model
#      (build it with scripts/flux-bootstrap-model-env.sh)
#   4. the validated ephemeral environment /tmp/lerobot-peft-env, when present
# FLUX_MERGE_ADAPTER=1 bakes the LoRA adapter into the base weights at load
# (in memory only; identical math up to bf16 rounding, ~22% faster sampling).
# `--plan` prints paths; `--check` verifies dependencies and checkpoint identity.
# Normal operation remains in this terminal and streams the server logs.
set -euo pipefail

repo_root="$(cd "$(dirname "$0")/.." && pwd)"
data_root="${HUMANOID_DATA_ROOT:-$repo_root/data}"
port="${FLUX_MODEL_PORT:-5561}"
checkpoint="${FLUX_CHECKPOINT:-$data_root/models/flux-dex3/checkpoint-2500}"
model_python="${FLUX_MODEL_PYTHON:-}"
persistent_python="$data_root/venvs/flux-model/bin/python"
ephemeral_python="${FLUX_MODEL_FALLBACK_PYTHON:-/tmp/lerobot-peft-env/bin/python}"
merge_adapter="${FLUX_MERGE_ADAPTER:-0}"
action=start

usage() {
  cat >&2 <<EOF
usage: $0 [--port PORT] [--checkpoint DIR] [--python INTERPRETER] [--check | --plan]

  --port PORT          model endpoint (default $port)
  --checkpoint DIR     prepared read-only adapter (default $checkpoint)
  --python PATH        model interpreter (default persistent environment, then fallback)
  --check              verify interpreter and checkpoint without loading the model
  --plan               show resolved paths without starting the model

Set FLUX_MERGE_ADAPTER=1 to bake the LoRA adapter into the base weights at load
(in memory only; ~22% faster sampling, see the header note).
Start without a mode flag to run the model server in this terminal. Ctrl-C stops it.
EOF
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --port) port="${2:-}"; shift 2 ;;
    --checkpoint) checkpoint="${2:-}"; shift 2 ;;
    --python) model_python="${2:-}"; shift 2 ;;
    --check) action=check; shift ;;
    --plan) action=plan; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "error: unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done

python_source=override
if [ -z "$model_python" ]; then
  python_source=missing
  if [ -x "$persistent_python" ]; then
    model_python="$persistent_python"
    python_source=persistent
  elif [ -x "$ephemeral_python" ]; then
    model_python="$ephemeral_python"
    python_source=ephemeral-fallback
  fi
elif [ ! -x "$model_python" ]; then
  echo "error: interpreter not executable: $model_python" >&2
  exit 2
fi
if ! [[ "$port" =~ ^[0-9]+$ ]] || [ "$port" -lt 1 ] || [ "$port" -gt 65535 ]; then
  echo "error: invalid port: $port" >&2
  exit 2
fi

export PYTHONPATH="$repo_root/third_party/flux/flux-inference:$repo_root/third_party/flux/flux-inference/ros2/flux_dex3:$repo_root/third_party/flux/flux-training/src${PYTHONPATH:+:$PYTHONPATH}"
server="$repo_root/third_party/flux/flux-inference/examples/dex3/zmq_server.py"
plan_json() {
  # Pure path/pin resolution: runs without an interpreter so a missing
  # environment is diagnosable, and never touches docker or the GPU.
  python3 - "$model_python" "$python_source" "$persistent_python" "$ephemeral_python" \
           "$checkpoint" "$port" "$repo_root" "$merge_adapter" <<'PY'
import hashlib
import json
import os
import sys

(model_python, source, persistent, ephemeral, checkpoint, port, repo_root, merge) = sys.argv[1:9]
lock = os.path.join(repo_root, "scripts/flux-model-env.lock.txt")
plan = {
    "model_python": model_python or None,
    "model_python_source": source,
    "persistent_python": persistent,
    "persistent_exists": os.access(persistent, os.X_OK),
    "fallback_python": ephemeral,
    "fallback_exists": os.access(ephemeral, os.X_OK),
    "checkpoint": checkpoint,
    "checkpoint_exists": os.path.isdir(checkpoint),
    "port": int(port),
    "merge_adapter": merge == "1",
    "bootstrap": "bash %s/scripts/flux-bootstrap-model-env.sh" % repo_root,
    "lock_file": lock,
    "lock_exists": os.path.isfile(lock),
    "lerobot_commit": "e624f3f7f8411ec3a02635d06e79373341e5ef35",
    "protocol": "flux_dex3.protocol v1 (DEALER/ROUTER, unchanged)",
}
if os.path.isfile(lock):
    with open(lock, "rb") as stream:
        plan["lock_sha256"] = hashlib.sha256(stream.read()).hexdigest()
print(json.dumps(plan, indent=2, sort_keys=True))
PY
}

preflight() {
  echo "flux-model-server: interpreter=$model_python (source=$python_source)" >&2
  "$model_python" - <<'PY'
import importlib

for name in ("torch", "numpy", "zmq", "flux_action", "lerobot", "peft", "av"):
    importlib.import_module(name)
PY
  if [ ! -d "$checkpoint" ]; then
    echo "error: no prepared adapter copy at $checkpoint" >&2
    echo "       prepare it once with: ./dev.sh flux-checkpoint --source <training checkpoint>" >&2
    return 2
  fi
  "$model_python" "$repo_root/scripts/flux-prepare-checkpoint.py" --check-only --dest "$checkpoint"
}

if [ "$python_source" = missing ] && [ "$action" != plan ]; then
  echo "error: no model interpreter found (checked \$FLUX_MODEL_PYTHON, $persistent_python, $ephemeral_python)" >&2
  echo "       build the persistent one: bash scripts/flux-bootstrap-model-env.sh" >&2
  exit 2
fi

case "$action" in
  plan) plan_json; exit 0 ;;
  check) preflight; echo "flux-model-server: preflight OK"; exit 0 ;;
esac

preflight
model_args=(--bind-ip 127.0.0.1 --port "$port" --checkpoint "$checkpoint")
if [ "$merge_adapter" = "1" ]; then
  model_args+=(--merge-adapter)
fi
exec "$model_python" "$server" "${model_args[@]}"
