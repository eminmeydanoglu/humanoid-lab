#!/usr/bin/env bash
# Launch the incompatible GR00T, Isaac/SONIC, and native SONIC runtimes separately.
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
container=${CLOUDWALK_CONTAINER:-humanoid-lab-dev}
project=${CLOUDWALK_PROJECT_ROOT:-/workspace/humanoid-lab}
log=${CLOUDWALK_LOG:-$root/outputs/isaac-closed-loop.log}
steps=${CLOUDWALK_STEPS:-250}
metrics=${CLOUDWALK_ROLLOUT_METRICS:-$project/outputs/nominal-rollout.json}
video=${CLOUDWALK_ROLLOUT_VIDEO:-$project/outputs/nominal-rollout.mp4}
mkdir -p "$(dirname "$log")"
: >"$log"

pids=()
cleanup() {
  local pid
  for pid in "${pids[@]:-}"; do kill "$pid" 2>/dev/null || true; done
  docker exec "$container" pkill -f 'run_gr00t_server.py.*56111|cloudwalk-vla-worker.py.*56111|sonic-closed-loop-native' 2>/dev/null || true
  wait "${pids[@]:-}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM HUP

docker inspect "$container" >/dev/null
# A prior interrupted docker exec can leave its setsid child alive; clear only
# this launcher's uniquely named workers before binding the fixed loopback ports.
cleanup
# Do not use docker exec foreground processes as the process groups escape on interruption.
start() {
  docker exec "$container" bash -lc "$1" >>"$log" 2>&1 &
  pids+=("$!")
}
start "source /opt/humanoid-lab/entrypoint.sh && use-groot && exec /opt/venvs/groot-n17/bin/python /opt/src/isaac-groot/gr00t/eval/run_gr00t_server.py --model-path /data/models/cloudwalk-gr00t-n17-g1-grab-bottle-rh-371ep-v10-finetune/checkpoint-30000 --embodiment-tag unitree_g1_sonic --port 56111"
start "source /opt/humanoid-lab/entrypoint.sh && use-groot && PYTHONPATH=/opt/src/isaac-groot:/opt/src/sonic exec /opt/venvs/groot-n17/bin/python $project/scripts/cloudwalk-vla-worker.py --policy-port 56111"
start "source /opt/humanoid-lab/entrypoint.sh && use-isaac-sonic && SONIC_NATIVE_HARNESS=closed-loop PROJECT_ROOT=$project SONIC_ROOT=/opt/src/sonic $project/scripts/build-sonic-isolated-native.sh && exec env CLOUDWALK_ACTION_PORT=56114 CLOUDWALK_STATE_PORT=56112 CLOUDWALK_BODY_PORT=56113 DECODER_MODEL=/data/runtime/sonic-deploy-models/sonic_v1_1/model_decoder.onnx LD_LIBRARY_PATH=/opt/onnxruntime-linux-x64-1.20.1/lib:/usr/local/cuda-12.8/lib64 /tmp/sonic-closed-loop-native"

# Isaac is the foreground owner of the closed loop. Its stdout is the durable run record.
docker exec "$container" bash -lc "source /opt/humanoid-lab/entrypoint.sh && use-isaac-sonic && exec /opt/venvs/isaac-sonic/bin/python $project/scripts/run-cloudwalk-isaac.py --headless --closed-loop --steps $steps --capture-path $project/outputs/isaac-closed-loop.jpg --rollout-metrics-path $metrics --video-path $video" 2>&1 | tee -a "$log"
