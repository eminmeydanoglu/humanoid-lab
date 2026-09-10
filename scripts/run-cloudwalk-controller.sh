#!/usr/bin/env bash
# Start GR00T, VLA scheduling, and native SONIC without owning Isaac Timeline.
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
container=${CLOUDWALK_CONTAINER:-humanoid-lab-dev}
project=${CLOUDWALK_PROJECT_ROOT:-/workspace/humanoid-lab}
log=${CLOUDWALK_CONTROLLER_LOG:-$root/outputs/cloudwalk-controller.log}
session_args=${CLOUDWALK_CONTROLLER_SESSION_ARGS:-}
mkdir -p "$(dirname "$log")"
: >"$log"

pids=()
cleanup() {
  local pid
  for pid in "${pids[@]:-}"; do kill "$pid" 2>/dev/null || true; done
  docker exec "$container" pkill -f 'run_gr00t_server.py.*56111|cloudwalk-vla-worker.py.*56111|sonic-closed-loop-native|cloudwalk-controller-session.py' 2>/dev/null || true
  wait "${pids[@]:-}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM HUP

docker inspect "$container" >/dev/null
if ! docker exec "$container" python3 -c 'import socket; s=socket.socket(); s.settimeout(1.0); s.connect(("127.0.0.1", 56115)); s.close()'
then
  echo "error: CloudWalk simulator is not running or control endpoint 56115 is unavailable" >&2
  exit 2
fi
cleanup
start() {
  docker exec "$container" bash -lc "$1" >>"$log" 2>&1 &
  pids+=("$!")
}
start "source /opt/humanoid-lab/entrypoint.sh && use-groot && exec /opt/venvs/groot-n17/bin/python /opt/src/isaac-groot/gr00t/eval/run_gr00t_server.py --model-path /data/models/cloudwalk-gr00t-n17-g1-grab-bottle-rh-371ep-v10-finetune/checkpoint-30000 --embodiment-tag unitree_g1_sonic --port 56111"
start "source /opt/humanoid-lab/entrypoint.sh && use-groot && PYTHONPATH=/opt/src/isaac-groot:/opt/src/sonic exec /opt/venvs/groot-n17/bin/python $project/scripts/cloudwalk-vla-worker.py --policy-port 56111"
start "source /opt/humanoid-lab/entrypoint.sh && use-isaac-sonic && SONIC_NATIVE_HARNESS=closed-loop PROJECT_ROOT=$project SONIC_ROOT=/opt/src/sonic $project/scripts/build-sonic-isolated-native.sh && exec env CLOUDWALK_ACTION_PORT=56114 CLOUDWALK_STATE_PORT=56112 CLOUDWALK_BODY_PORT=56113 DECODER_MODEL=/data/runtime/sonic-deploy-models/sonic_v1_1/model_decoder.onnx LD_LIBRARY_PATH=/opt/onnxruntime-linux-x64-1.20.1/lib:/usr/local/cuda-12.8/lib64 /tmp/sonic-closed-loop-native"

# Arm Isaac only after the native decoder has bound its body endpoint and its
# state subscriber has had time to join the simulator publisher.
for _ in $(seq 1 100); do
  if docker exec "$container" python3 -c 'import socket; s=socket.socket(); s.settimeout(0.2); s.connect(("127.0.0.1", 56113)); s.close()' 2>/dev/null; then
    sleep 1
    break
  fi
  sleep 0.1
done
if ! docker exec "$container" python3 -c 'import socket; s=socket.socket(); s.settimeout(0.5); s.connect(("127.0.0.1", 56113)); s.close()' 2>/dev/null; then
  echo "error: native SONIC did not become ready on port 56113" >&2
  exit 1
fi

docker exec "$container" bash -lc "source /opt/humanoid-lab/entrypoint.sh && use-isaac-sonic && exec /opt/venvs/isaac-sonic/bin/python $project/scripts/cloudwalk-controller-session.py $session_args" 2>&1 | tee -a "$log"
