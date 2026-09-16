#!/usr/bin/env bash
# ROMP upper-body -> SONIC (zmq_manager, teleop encoder mode 1) -> MuJoCo G1.
#
# Full-stack launcher for the romp_faruk worktree. It starts, in order:
#
#   1. MuJoCo sim on DDS domain 42 / interface 'lo' (sim_mujoco_domain42.py)
#   2. SONIC deploy with --input-type zmq_manager     (run_deploy_zmq.py)
#   3. ROMP->SONIC bridge in upper-body mode          (romp_to_sonic_bridge.py)
#   4. Stage-1 ROMP webcam streamer on the host       (romp_pose_streamer.py)
#
# Steps 1-3 run in $ROMP_CONTAINER (default: romp-faruk-dev); step 4 runs on the
# host in ~/romp-venv because only that venv has ROMP/torch/cv2.
#
# Usage:
#   romp_upper.sh start [extra bridge args...]   # default action is start
#   romp_upper.sh stop
#   romp_upper.sh status
#   romp_upper.sh logs
#
# Env overrides: ROMP_CONTAINER, ROMP_WORKTREE, ROMP_DISPLAY, ROMP_SOURCE,
#                ROMP_STAGE1_PYTHON, ROMP_CONFLICT_CONTAINER.
set -euo pipefail

CONTAINER="${ROMP_CONTAINER:-romp-faruk-dev}"
CONFLICT_CONTAINER="${ROMP_CONFLICT_CONTAINER:-humanoid-lab-dev}"
WORKTREE="${ROMP_WORKTREE:-$HOME/code/romp_faruk}"
HOST_DIR="$WORKTREE/romp_teleop"
CTR_DIR="/workspace/humanoid-lab/romp_teleop"
PYTHON="/opt/venvs/sonic-sim/bin/python"
DISPLAY_ID="${ROMP_DISPLAY:-:1}"
XAUTH_HOST="$HOST_DIR/.xdg_auth"
XAUTH_CTR="/tmp/romp_xauth"
MARKER="/tmp/romp_drop_robot"
RAW_PORT=5558
SONIC_PORT=5556
STAGE1_PY="${ROMP_STAGE1_PYTHON:-$HOME/romp-venv/bin/python}"
CAMERA="${ROMP_SOURCE:-0}"

LOG_SIM=/tmp/romp_upper_sim.log
LOG_DEPLOY=/tmp/romp_upper_deploy.log
LOG_DEPLOY_OUT=/tmp/romp_upper_deploy.stdout
LOG_BRIDGE=/tmp/romp_upper_bridge.log
LOG_STAGE1=/tmp/romp_upper_stage1.log
PID_STAGE1=/tmp/romp_upper_stage1.pid

log() { printf '[romp_upper] %s\n' "$*" >&2; }

# Kill processes matching the patterns in $1 (space separated) inside a
# container. Patterns travel through the environment so the pkill wrapper's own
# command line never contains them.
kill_in() {
  local container="$1" patterns="$2"
  docker exec -e KILL_PATTERNS="$patterns" "$container" bash -c '
    for pat in $KILL_PATTERNS; do
      pkill -f "$pat" 2>/dev/null || true
    done
  ' >/dev/null 2>&1 || true
}

stop_host_stage1() {
  if [[ -f "$PID_STAGE1" ]]; then
    kill "$(cat "$PID_STAGE1")" 2>/dev/null || true
    rm -f "$PID_STAGE1"
  fi
  pkill -f 'romp_pose_streamer.py.*--port 5558' 2>/dev/null || true
}

stop_stack() {
  stop_host_stage1
  kill_in "$CONTAINER" "romp_to_sonic_bridge.py run_deploy_zmq.py sim_mujoco_domain42.py"
  docker exec "$CONTAINER" rm -f "$MARKER" >/dev/null 2>&1 || true
}

stop_conflicts() {
  # A stray keyboard deploy / Isaac-targeted sim on the same DDS domain and GPU
  # will win the DDS race and ignore the ROMP ZMQ stream. Drop them first.
  kill_in "$CONFLICT_CONTAINER" "g1_deploy_onnx_ref run_sim_loop"
}

require_container() {
  if [[ "$(docker inspect -f '{{.State.Running}}' "$CONTAINER" 2>/dev/null)" != true ]]; then
    log "container '$CONTAINER' is not running; start it from $WORKTREE first"
    return 1
  fi
  if [[ ! -d "$HOST_DIR" ]]; then
    log "worktree not found: $HOST_DIR"
    return 1
  fi
  if [[ ! -f "$XAUTH_HOST" ]]; then
    log "X authority not found: $XAUTH_HOST"
    return 1
  fi
}

start_sim() {
  docker exec -d -e DISPLAY="$DISPLAY_ID" -e XAUTHORITY="$XAUTH_CTR" \
    -e PYTHONUNBUFFERED=1 "$CONTAINER" sh -c '
      exec "$0" "$1" \
        --interface lo --simulator mujoco --enable-onscreen --no-enable-offscreen \
        --no-enable-real-device --waist-pitch-limit 30 \
        --no-data-collection --no-verbose > '"$LOG_SIM"' 2>&1
    ' "$PYTHON" "$CTR_DIR/sim_mujoco_domain42.py"
  log "MuJoCo sim starting (domain 42 / lo), log: $LOG_SIM"
}

start_deploy() {
  docker exec "$CONTAINER" sh -c ": > $LOG_DEPLOY; : > $LOG_DEPLOY_OUT" 2>/dev/null || true
  docker exec -d -e PYTHONUNBUFFERED=1 "$CONTAINER" sh -c '
    exec "$0" "$1" --input-type zmq_manager --port '"$SONIC_PORT"' \
      --drop-marker '"$MARKER"' --log '"$LOG_DEPLOY"' --print \
      > '"$LOG_DEPLOY_OUT"' 2>&1
  ' "$PYTHON" "$CTR_DIR/run_deploy_zmq.py"
  log "SONIC deploy starting (zmq_manager), log: $LOG_DEPLOY"
}

wait_deploy() {
  local ready=0
  for _ in $(seq 1 240); do
    if docker exec "$CONTAINER" bash -c "grep -aq 'Init Done' '$LOG_DEPLOY'" 2>/dev/null; then
      ready=1; break
    fi
    if ! docker exec "$CONTAINER" pgrep -f run_deploy_zmq.py >/dev/null 2>&1; then
      log "deploy exited during init; tail of $LOG_DEPLOY:"
      docker exec "$CONTAINER" bash -c "tail -n 20 '$LOG_DEPLOY'" || true
      return 1
    fi
    sleep 0.5
  done
  [[ "$ready" -eq 1 ]] || { log "timed out waiting for deploy 'Init Done'"; return 1; }
  log "deploy initialised"
}

start_bridge() {
  docker exec -d -e PYTHONUNBUFFERED=1 "$CONTAINER" sh -c '
    py="$0"; script="$1"; shift
    exec "$py" "$script" --sonic-root /opt/src/sonic \
      --control-mode upper-body --subscribe \
      --raw-port '"$RAW_PORT"' --port '"$SONIC_PORT"' \
      --fps 50 --smooth 0.75 "$@" > '"$LOG_BRIDGE"' 2>&1
  ' "$PYTHON" "$CTR_DIR/romp_to_sonic_bridge.py" "$@"
  log "upper-body bridge started, log: $LOG_BRIDGE"
}

start_stage1() {
  nohup env DISPLAY="$DISPLAY_ID" XAUTHORITY="$XAUTH_HOST" PYTHONUNBUFFERED=1 \
    "$STAGE1_PY" "$HOST_DIR/romp_pose_streamer.py" \
    --source "$CAMERA" --publish --port "$RAW_PORT" --show \
    --width 1280 --height 720 > "$LOG_STAGE1" 2>&1 < /dev/null &
  echo $! > "$PID_STAGE1"
  log "Stage-1 ROMP streamer started (pid $(cat "$PID_STAGE1")), log: $LOG_STAGE1"
}

do_start() {
  require_container
  log "stopping previous/conflicting controllers"
  stop_stack
  stop_conflicts
  sleep 1

  docker cp "$XAUTH_HOST" "$CONTAINER:$XAUTH_CTR" >/dev/null
  docker exec "$CONTAINER" chmod 600 "$XAUTH_CTR"
  docker exec "$CONTAINER" rm -f "$MARKER" >/dev/null 2>&1 || true

  start_sim
  sleep 3
  start_deploy
  wait_deploy
  start_bridge "$@"
  start_stage1
  sleep 2
  do_status
  log "started: click the MuJoCo window and press 9 if the elastic band lingers"
}

do_stop() {
  log "stopping"
  stop_stack
  log "stopped"
}

do_status() {
  echo "[stage1 host]"
  pgrep -af "romp_pose_streamer.py.*--port $RAW_PORT" || echo "  (not running)"
  echo "[container $CONTAINER]"
  docker exec "$CONTAINER" ps -eo pid,etime,args 2>/dev/null \
    | grep -E 'sim_mujoco_domain42|run_deploy_zmq|romp_to_sonic_bridge|run_sim_loop' \
    | grep -v grep || echo "  (none)"
  echo "[ports]"
  ss -tlnp 2>/dev/null | grep -E ":$RAW_PORT|:$SONIC_PORT" || echo "  (none)"
  echo "[conflicts in $CONFLICT_CONTAINER]"
  docker exec "$CONFLICT_CONTAINER" ps -eo pid,etime,args 2>/dev/null \
    | grep -E 'g1_deploy_onnx_ref|run_sim_loop' | grep -v grep || echo "  (none)"
}

do_logs() {
  for f in "$LOG_SIM" "$LOG_DEPLOY" "$LOG_DEPLOY_OUT" "$LOG_BRIDGE" "$LOG_STAGE1"; do
    echo "===== $f ====="
    if [[ -f "$f" ]]; then tail -n 25 "$f"; else echo "(missing)"; fi
    echo
  done
}

ACTION="${1:-start}"
case "$ACTION" in
  start)   shift || true; do_start "$@" ;;
  stop)    do_stop ;;
  restart) shift || true; do_stop; do_start "$@" ;;
  status)  do_status ;;
  logs)    do_logs ;;
  *) echo "usage: romp_upper.sh [start [bridge args...]|stop|restart|status|logs]" >&2; exit 2 ;;
esac
