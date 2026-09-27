#!/usr/bin/env bash
# Host-side GPU model server for the flux-ros evaluation loop.
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
# `--plan` prints the resolution and the pins without touching anything;
# `--check` verifies the interpreter, the adapter copy's checkpoint_identity and
# the base policy the adapter references.
set -euo pipefail

repo_root="$(cd "$(dirname "$0")/.." && pwd)"
data_root="${HUMANOID_DATA_ROOT:-$repo_root/data}"
port="${FLUX_MODEL_PORT:-5561}"
checkpoint="${FLUX_CHECKPOINT:-$data_root/models/flux-dex3/checkpoint-2500}"
model_python="${FLUX_MODEL_PYTHON:-}"
persistent_python="$data_root/venvs/flux-model/bin/python"
ephemeral_python="${FLUX_MODEL_FALLBACK_PYTHON:-/tmp/lerobot-peft-env/bin/python}"
log_dir="${FLUX_LOG_DIR:-$data_root/outputs/flux-dex3}"
ready_timeout="${FLUX_MODEL_READY_TIMEOUT:-300}"
action=start
foreground=0

usage() {
  cat >&2 <<EOF
usage: $0 [--port PORT] [--checkpoint DIR] [--python INTERPRETER] [--log-dir DIR]
          [--foreground | --stop | --status | --check | --plan]

  --port PORT          loopback port for the ROUTER server (default $port)
  --checkpoint DIR     immutable adapter copy (default $checkpoint)
  --python PATH        model environment interpreter (default \$FLUX_MODEL_PYTHON,
                       else $persistent_python,
                       else the validated fallback $ephemeral_python)
  --foreground         stay in the foreground instead of detaching
  --stop               stop the server started through this script
  --status             probe STATUS and print the reply
  --check              preflight only: interpreter imports, adapter
                       checkpoint_identity, base policy reachability (no model
                       is loaded, no server is started)
  --plan               print the resolved interpreter/pins as JSON and exit
EOF
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --port) port="${2:-}"; shift 2 ;;
    --checkpoint) checkpoint="${2:-}"; shift 2 ;;
    --python) model_python="${2:-}"; shift 2 ;;
    --log-dir) log_dir="${2:-}"; shift 2 ;;
    --foreground) foreground=1; shift ;;
    --stop) action=stop; shift ;;
    --status) action=status; shift ;;
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
pidfile="$log_dir/model-server-$port.pid"
logfile="$log_dir/model-server-$port.log"

# `plan` and `stop` resolve paths only; everything else needs the interpreter.
if [ "$python_source" = missing ] && [ "$action" != plan ] && [ "$action" != stop ]; then
  echo "error: no model interpreter found (checked \$FLUX_MODEL_PYTHON, $persistent_python, $ephemeral_python)" >&2
  echo "       build the persistent one: bash scripts/flux-bootstrap-model-env.sh" >&2
  echo "       or point at an existing environment: --python PATH / FLUX_MODEL_PYTHON" >&2
  exit 2
fi

stop_server() {
  [ -r "$pidfile" ] || { echo "flux-model-server: no pidfile at $pidfile" >&2; return 0; }
  read -r pid <"$pidfile" || pid=""
  case "$pid" in (*[!0-9]*|"") rm -f -- "$pidfile"; return 0 ;; esac
  if [ -r "/proc/$pid/cmdline" ] && tr '\0' ' ' <"/proc/$pid/cmdline" | grep -Fq "zmq_server.py"; then
    echo "flux-model-server: stopping pid=$pid" >&2
    kill -TERM "$pid" 2>/dev/null || true
    for _ in $(seq 1 30); do [ -d "/proc/$pid" ] || break; sleep 0.1; done
    if [ -d "/proc/$pid" ]; then kill -KILL "$pid" 2>/dev/null || true; fi
  fi
  rm -f -- "$pidfile"
}

probe_status() {
  "$model_python" - "$port" <<'PY'
import json
import sys

import zmq
from flux_dex3 import protocol

# DEALER, exactly like the node's own client: a REQ socket would add the empty
# delimiter frame the ROUTER-side protocol does not expect.
context = zmq.Context()
socket = context.socket(zmq.DEALER)
socket.setsockopt(zmq.LINGER, 0)
socket.setsockopt(zmq.RCVTIMEO, 3000)
socket.setsockopt(zmq.SNDTIMEO, 3000)
socket.connect("tcp://127.0.0.1:%d" % int(sys.argv[1]))
try:
    socket.send_multipart(protocol.encode_status_request())
    reply = protocol.decode_reply(socket.recv_multipart())
finally:
    socket.close(linger=0)
    context.term()
print(json.dumps({"status": reply["status"], "checkpoint": reply["checkpoint"], "error": reply["error"]}))
PY
}

wait_ready() {
  "$model_python" - "$port" "$ready_timeout" <<'PY'
import json
import sys
import time

import zmq
from flux_dex3 import protocol

port, timeout = int(sys.argv[1]), float(sys.argv[2])
context = zmq.Context()
socket = context.socket(zmq.DEALER)
socket.setsockopt(zmq.LINGER, 0)
socket.setsockopt(zmq.RCVTIMEO, 2000)
socket.setsockopt(zmq.SNDTIMEO, 2000)
socket.connect("tcp://127.0.0.1:%d" % port)
deadline = time.monotonic() + timeout
last = None
try:
    while time.monotonic() < deadline:
        try:
            socket.send_multipart(protocol.encode_status_request())
            reply = protocol.decode_reply(socket.recv_multipart())
            status = {"status": reply["status"], "checkpoint": reply["checkpoint"], "error": reply["error"]}
        except Exception as exc:  # not up yet, or a malformed/unreachable server
            status = {"status": "UNREACHABLE", "checkpoint": "", "error": str(exc)[:200]}
            socket.close(linger=0)
            socket = context.socket(zmq.DEALER)
            socket.setsockopt(zmq.LINGER, 0)
            socket.setsockopt(zmq.RCVTIMEO, 2000)
            socket.setsockopt(zmq.SNDTIMEO, 2000)
            socket.connect("tcp://127.0.0.1:%d" % port)
        if status["status"] != last:
            print(json.dumps(status, sort_keys=True), flush=True)
            last = status["status"]
        if status["status"] == "READY":
            raise SystemExit(0)
        if status["status"] == "ERROR":
            raise SystemExit(1)
        time.sleep(1)
finally:
    socket.close(linger=0)
    context.term()
print(json.dumps({"status": "TIMEOUT", "timeout_s": timeout}), flush=True)
raise SystemExit(2)
PY
}

plan_json() {
  # Pure path/pin resolution: runs without an interpreter so a missing
  # environment is diagnosable, and never touches docker or the GPU.
  python3 - "$model_python" "$python_source" "$persistent_python" "$ephemeral_python" \
           "$checkpoint" "$port" "$ready_timeout" "$repo_root" <<'PY'
import hashlib
import json
import os
import sys

(model_python, source, persistent, ephemeral, checkpoint, port, ready_timeout, repo_root) = sys.argv[1:9]
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
    "ready_timeout_s": float(ready_timeout),
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

case "$action" in
  plan)
    plan_json
    exit 0
    ;;
  stop)
    stop_server
    exit 0
    ;;
  status)
    probe_status
    exit 0
    ;;
  check)
    preflight
    echo "flux-model-server: preflight OK"
    exit 0
    ;;
esac

if [ -r "$pidfile" ]; then
  read -r pid <"$pidfile" || pid=""
  if [ -n "$pid" ] && [ -r "/proc/$pid/cmdline" ] && tr '\0' ' ' <"/proc/$pid/cmdline" | grep -Fq "zmq_server.py"; then
    if probe_status 2>/dev/null | grep -q '"status": "READY"'; then
      echo "flux-model-server: reusing the READY server pid=$pid on port $port" >&2
      exit 0
    fi
    echo "error: a model server is already running but not READY (pid=$pid, see $pidfile)" >&2
    echo "       stop it with: $0 --port $port --stop" >&2
    exit 2
  fi
  rm -f -- "$pidfile"
fi
if ss -ltn 2>/dev/null | grep -Eq ":${port}[[:space:]]"; then
  # A server may already be serving this port (for example started by an
  # earlier session); reuse it only when it is the same protocol and READY.
  if probe_status 2>/dev/null | grep -q '"status": "READY"'; then
    echo "flux-model-server: reusing the READY server already bound to port $port" >&2
    exit 0
  fi
  echo "error: port $port is already in use by an unready/unrelated process; pick another --port" >&2
  exit 2
fi

if [ "$foreground" -eq 1 ]; then
  exec "$model_python" "$server" --bind-ip 127.0.0.1 --port "$port" --checkpoint "$checkpoint"
fi

mkdir -p "$log_dir"
preflight
setsid "$model_python" "$server" --bind-ip 127.0.0.1 --port "$port" --checkpoint "$checkpoint" \
  >>"$logfile" 2>&1 </dev/null &
pid=$!
printf '%s\n' "$pid" >"$pidfile"
echo "flux-model-server: pid=$pid port=$port checkpoint=$checkpoint log=$logfile" >&2
wait_ready
