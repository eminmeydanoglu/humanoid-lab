#!/usr/bin/env bash
# ROMP upper-body (+ SONIC IDLE standing) on the REAL G1 via g1_tail.
#
# MSI side only: Stage-1 ROMP webcam streamer (host) + upper-body bridge
# (in $ROMP_CONTAINER). It deliberately does NOT start MuJoCo or a local deploy,
# and it never touches the robot. Start the robot-side deploy yourself using the
# command printed by `robot-cmd` (robot supported, E-stop in hand).
#
# Usage:
#   romp_real.sh start        # Stage 1 + bridge only
#   romp_real.sh stop
#   romp_real.sh status
#   romp_real.sh logs
#   romp_real.sh robot-cmd    # print the exact g1_tail deploy command
#
# Env: ROMP_CONTAINER, ROMP_WORKTREE, ROMP_DISPLAY, ROMP_SOURCE,
#      ROMP_STAGE1_PYTHON, ROMP_ROBOT_HOST.
set -euo pipefail

CONTAINER="${ROMP_CONTAINER:-romp-faruk-dev}"
WORKTREE="${ROMP_WORKTREE:-$HOME/code/romp_faruk}"
HOST_DIR="$WORKTREE/romp_teleop"
CTR_DIR="/workspace/humanoid-lab/romp_teleop"
PYTHON="/opt/venvs/sonic-sim/bin/python"
DISPLAY_ID="${ROMP_DISPLAY:-:1}"
XAUTH_HOST="$HOST_DIR/.xdg_auth"
STAGE1_PY="${ROMP_STAGE1_PYTHON:-$HOME/romp-venv/bin/python}"
CAMERA="${ROMP_SOURCE:-0}"
ROBOT_HOST="${ROMP_ROBOT_HOST:-100.126.18.76}"
RAW_PORT=5558
SONIC_PORT=5556
LOG_BRIDGE=/tmp/romp_real_bridge.log        # container path
LOG_STAGE1=/tmp/romp_real_stage1.log        # host path
PID_STAGE1=/tmp/romp_real_stage1.pid

log() { printf '[romp_real] %s\n' "$*" >&2; }

kill_in() {
  local container="$1" patterns="$2"
  docker exec -e KILL_PATTERNS="$patterns" "$container" bash -c '
    for pat in $KILL_PATTERNS; do pkill -f "$pat" 2>/dev/null || true; done
  ' >/dev/null 2>&1 || true
}

stop_host_stage1() {
  if [[ -f "$PID_STAGE1" ]]; then
    kill "$(cat "$PID_STAGE1")" 2>/dev/null || true
    rm -f "$PID_STAGE1"
  fi
  pkill -f 'romp_pose_streamer.py.*--port 5558' 2>/dev/null || true
}

stop_all() {
  stop_host_stage1
  # Do not leave the simulator/deploy stack running: it would steal port 5556.
  kill_in "$CONTAINER" "romp_to_sonic_bridge.py run_deploy_zmq.py sim_mujoco_domain42.py"
}

start_bridge() {
  docker exec -d -e PYTHONUNBUFFERED=1 "$CONTAINER" sh -c '
    exec "$0" "$1" --sonic-root /opt/src/sonic --control-mode upper-body \
      --subscribe --raw-port '"$RAW_PORT"' --port '"$SONIC_PORT"' \
      --fps 50 --smooth 0.75 > '"$LOG_BRIDGE"' 2>&1
  ' "$PYTHON" "$CTR_DIR/romp_to_sonic_bridge.py"
  log "bridge started, binds tcp://*:$SONIC_PORT (container log: $LOG_BRIDGE)"
}

start_stage1() {
  nohup env DISPLAY="$DISPLAY_ID" XAUTHORITY="$XAUTH_HOST" PYTHONUNBUFFERED=1 \
    "$STAGE1_PY" "$HOST_DIR/romp_pose_streamer.py" \
    --source "$CAMERA" --publish --port "$RAW_PORT" --show \
    --width 1280 --height 720 > "$LOG_STAGE1" 2>&1 < /dev/null &
  echo $! > "$PID_STAGE1"
  log "Stage-1 ROMP started (pid $(cat "$PID_STAGE1"), log: $LOG_STAGE1)"
}

robot_cmd() {
  cat <<EOF
# --- On g1_tail (robot supported, E-stop in hand) ---
cd ~/sonic_jp5/GR00T-WholeBodyControl/gear_sonic_deploy
source /opt/ros/foxy/setup.bash
source ~/cyclonedds_ws/install/setup.bash
source ~/unitree_ros2/cyclonedds_ws/install/setup.bash
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
./target/release/g1_deploy_onnx_ref eth0 \\
  policy/sonic_v1_1/model_decoder.onnx reference/example/ \\
  --encoder-file policy/sonic_v1_1/model_encoder.onnx \\
  --obs-config policy/sonic_v1_1/observation_config.yaml \\
  --planner-file planner/target_vel/V2/planner_sonic_trt85.onnx \\
  --planner-precision 32 --policy-precision 32 \\
  --input-type zmq_manager --zmq-host $ROBOT_HOST --zmq-port $SONIC_PORT \\
  --output-type zmq --zmq-verbose \\
  --logs-dir /tmp/sonic_romp_upper
# -----------------------------------------------------
EOF
}

do_start() {
  if [[ "$(docker inspect -f '{{.State.Running}}' "$CONTAINER" 2>/dev/null)" != true ]]; then
    log "container $CONTAINER is not running"; return 1
  fi
  if [[ ! -f "$XAUTH_HOST" ]]; then log "missing X auth: $XAUTH_HOST"; return 1; fi
  log "stopping leftovers (simulator/deploy/bridge/stage1)"
  stop_all
  sleep 1
  start_bridge
  start_stage1
  sleep 2
  do_status
  echo
  robot_cmd
}

do_status() {
  echo "[stage1 host]"
  pgrep -af "romp_pose_streamer.py.*--port $RAW_PORT" || echo "  (not running)"
  echo "[container $CONTAINER]"
  docker exec "$CONTAINER" ps -eo pid,etime,args 2>/dev/null \
    | grep -E 'romp_to_sonic_bridge|run_deploy_zmq|sim_mujoco_domain42' \
    | grep -v grep || echo "  (none)"
  echo "[port $SONIC_PORT]"
  ss -tlnp 2>/dev/null | grep ":$SONIC_PORT" || echo "  (not listening)"
}

do_logs() {
  echo "===== $LOG_STAGE1 (host) ====="
  [[ -f "$LOG_STAGE1" ]] && tail -n 20 "$LOG_STAGE1" || echo "(missing)"
  echo
  echo "===== $LOG_BRIDGE (container) ====="
  docker exec "$CONTAINER" bash -c "tail -n 20 '$LOG_BRIDGE'" 2>/dev/null || echo "(missing)"
}

ACTION="${1:-start}"
case "$ACTION" in
  start)     do_start ;;
  stop)      stop_all; log "stopped (MSI side only; robot untouched)" ;;
  status)    do_status ;;
  logs)      do_logs ;;
  robot-cmd) robot_cmd ;;
  *) echo "usage: romp_real.sh [start|stop|status|logs|robot-cmd]" >&2; exit 2 ;;
esac
