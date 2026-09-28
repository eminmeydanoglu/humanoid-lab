#!/usr/bin/env bash
# Host-side GPU model server for the independent Flux ROS process.
#
# Runs the vendored zmq_server.py under the prepared model environment (the
# LeRobot/PEFT interpreter that holds the Flux3 runtime), binds loopback by
# default, and defaults to port 5561: the server's own 5557 default is the
# SONIC g1_debug state port on this machine, so the simulation names an
# explicit port and the robot defaults stay untouched.
#
# Remote use (wired GPU-host/robot network) selects the host's explicit
# interface IP with --bind-ip and makes CURVE mandatory: --server-secret-key
# (server .key_secret certificate) and --client-keys-dir (allowlisted robot
# client .key certificates) are both required and validated before interpreter
# or GPU work, while wildcard, multicast, hostname and unauthenticated remote
# binds are refused. Key files travel to zmq_server as paths only; their
# material is never echoed. The robot side configures the flux_dex3 node with
# endpoint=tcp://IP:PORT, client_certificate (its private .key_secret) and
# server_public_key (the server's public .key text).
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
bind_ip="${FLUX_MODEL_BIND_IP:-127.0.0.1}"
server_secret_key="${FLUX_MODEL_SERVER_SECRET_KEY:-}"
client_keys_dir="${FLUX_MODEL_CLIENT_KEYS_DIR:-}"
checkpoint="${FLUX_CHECKPOINT:-$data_root/models/flux-dex3/checkpoint-2500}"
model_python="${FLUX_MODEL_PYTHON:-}"
persistent_python="$data_root/venvs/flux-model/bin/python"
ephemeral_python="${FLUX_MODEL_FALLBACK_PYTHON:-/tmp/lerobot-peft-env/bin/python}"
merge_adapter="${FLUX_MERGE_ADAPTER:-0}"
action=start

usage() {
  cat >&2 <<EOF
usage: $0 [--port PORT] [--bind-ip IP] [--checkpoint DIR] [--python INTERPRETER]
          [--server-secret-key FILE] [--client-keys-dir DIR] [--check | --plan]

  --port PORT          model endpoint (default $port)
  --bind-ip IP         explicit IP literal to bind (default 127.0.0.1); wildcard
                       addresses are refused and any non-loopback IP requires both
                       CURVE flags below
  --server-secret-key FILE  CURVE server .key_secret certificate (remote binds)
  --client-keys-dir DIR     directory of allowlisted robot client .key certificates
  --checkpoint DIR     prepared read-only adapter (default $checkpoint)
  --python PATH        model interpreter (default persistent environment, then fallback)
  --check              verify interpreter and checkpoint without loading the model
  --plan               show resolved paths, endpoint and server arguments

Remote robots set the flux_dex3 node parameters endpoint=tcp://IP:PORT,
client_certificate=<robot private .key_secret> and server_public_key=<server
public .key text>; key material never appears in this script's output.

Set FLUX_MERGE_ADAPTER=1 to bake the LoRA adapter into the base weights at load
(in memory only; ~22% faster sampling, see the header note).
Start without a mode flag to run the model server in this terminal. Ctrl-C stops it.
EOF
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --port) port="${2:-}"; shift 2 ;;
    --bind-ip) bind_ip="${2:-}"; shift 2 ;;
    --server-secret-key) server_secret_key="${2:-}"; shift 2 ;;
    --client-keys-dir) client_keys_dir="${2:-}"; shift 2 ;;
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

# The vendored server binds one explicit IP, refuses wildcards and demands
# CURVE for non-loopback binds; mirror that here so a bad deployment fails
# before interpreter resolution, checkpoint checks or GPU work.
bind_kind="$(python3 - "$bind_ip" <<'PY'
import ipaddress
import sys

try:
    address = ipaddress.ip_address(sys.argv[1])
except ValueError:
    kind = "invalid"
else:
    if address.is_unspecified:
        kind = "wildcard"
    elif address.is_multicast:
        kind = "multicast"
    elif address.is_loopback:
        kind = "loopback"
    else:
        kind = "remote"
print(kind)
PY
)"
case "$bind_kind" in
  invalid)
    echo "error: --bind-ip must be an explicit IP literal, got: ${bind_ip:-<empty>}" >&2
    exit 2 ;;
  wildcard)
    echo "error: refusing wildcard bind ${bind_ip}; select the host's explicit interface IP" >&2
    exit 2 ;;
  multicast)
    echo "error: refusing multicast bind ${bind_ip}; select the host's explicit interface IP" >&2
    exit 2 ;;
esac
if [ "$bind_kind" = remote ]; then
  if [ -z "$server_secret_key" ]; then
    echo "error: non-loopback bind $bind_ip requires --server-secret-key (CURVE server .key_secret)" >&2
    exit 2
  fi
  if [ -z "$client_keys_dir" ]; then
    echo "error: non-loopback bind $bind_ip requires --client-keys-dir (allowlisted robot client .key certificates)" >&2
    exit 2
  fi
fi
if [ -n "$server_secret_key" ] && { [ ! -f "$server_secret_key" ] || [ ! -r "$server_secret_key" ]; }; then
  echo "error: --server-secret-key must be a readable file: $server_secret_key" >&2
  exit 2
fi
if [ -n "$client_keys_dir" ]; then
  if [ ! -d "$client_keys_dir" ]; then
    echo "error: --client-keys-dir must be a directory: $client_keys_dir" >&2
    exit 2
  fi
  if ! compgen -G "$client_keys_dir/*.key" >/dev/null; then
    echo "error: --client-keys-dir contains no allowlisted *.key certificates: $client_keys_dir" >&2
    exit 2
  fi
fi

export PYTHONPATH="$repo_root/third_party/flux/flux-inference:$repo_root/third_party/flux/flux-inference/ros2/flux_dex3:$repo_root/third_party/flux/flux-training/src${PYTHONPATH:+:$PYTHONPATH}"
server="$repo_root/third_party/flux/flux-inference/examples/dex3/zmq_server.py"

# CURVE arguments are passed only when configured; remote binds require both,
# and existing-but-unused files are still validated above.
model_args=(--bind-ip "$bind_ip" --port "$port" --checkpoint "$checkpoint")
if [ -n "$server_secret_key" ]; then
  model_args+=(--server-secret-key "$server_secret_key")
fi
if [ -n "$client_keys_dir" ]; then
  model_args+=(--client-keys-dir "$client_keys_dir")
fi
if [ "$merge_adapter" = "1" ]; then
  model_args+=(--merge-adapter)
fi

plan_json() {
  # Pure path/pin resolution: runs without an interpreter so a missing
  # environment is diagnosable, and never touches docker or the GPU. The
  # server arguments are resolved by the caller and echoed verbatim; key
  # files appear as paths only.
  python3 - "$model_python" "$python_source" "$persistent_python" "$ephemeral_python" \
           "$checkpoint" "$port" "$repo_root" "$merge_adapter" "$bind_ip" "$server" \
           -- "${model_args[@]}" <<'PY'
import hashlib
import ipaddress
import json
import os
import sys

marker = sys.argv.index("--")
(model_python, source, persistent, ephemeral, checkpoint, port,
 repo_root, merge, bind_ip, server) = sys.argv[1:marker]
model_args = sys.argv[marker + 1:]


def flag_value(flag):
    return model_args[model_args.index(flag) + 1] if flag in model_args else None


address = ipaddress.ip_address(bind_ip)
host = "[{}]".format(bind_ip) if ":" in bind_ip else bind_ip
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
    "bind_ip": bind_ip,
    "remote": not address.is_loopback,
    "server": server,
    "model_args": model_args,
    "server_secret_key": flag_value("--server-secret-key"),
    "client_keys_dir": flag_value("--client-keys-dir"),
    "robot_endpoint": "tcp://{}:{}".format(host, int(port)),
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

remote_note() {
  # Operational guidance only: the robot's private certificate lives on the
  # robot and no key material is printed here.
  [ "$bind_kind" = remote ] || return 0
  local host="$bind_ip"
  case "$bind_ip" in *:*) host="[$bind_ip]" ;; esac
  echo "flux-model-server: remote bind ${bind_ip}:${port} requires CURVE (client allowlist: ${client_keys_dir})" >&2
  echo "  robot side: endpoint=tcp://${host}:${port} client_certificate=<robot private .key_secret>" >&2
  echo "  server_public_key=<server public .key text>; see third_party/flux/flux-inference/docs/dex3_ros_bridge.md" >&2
}

case "$action" in
  plan) plan_json; exit 0 ;;
  check) remote_note; preflight; echo "flux-model-server: preflight OK"; exit 0 ;;
esac

remote_note
preflight
exec "$model_python" "$server" "${model_args[@]}"
