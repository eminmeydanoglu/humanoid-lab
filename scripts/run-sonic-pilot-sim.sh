#!/usr/bin/env bash
# Run the frame-exact reference replay and offline-latent SONIC simulation for
# one prepared pilot directory.
#
# The official deployment (./dev.sh sonic-controller zmq_manager) owns the DDS
# topics, Isaac G1 follows them through its sonic_dds controller, and
# scripts/replay-sonic-latent.py feeds action_tokens.npz through SONIC's official
# Protocol v4 input. This tests the persisted offline latent rather than asking
# the deployment's live C++ encoder to encode reference.npz again.
#
# Usage: scripts/run-sonic-pilot-sim.sh <pilot_dir> [--duration SECONDS] [--no-direct] [--no-kinematic]
set -euo pipefail
cd "$(dirname "$0")/.."

PILOT_DIR="${1:?usage: run-sonic-pilot-sim.sh <pilot_dir> [--duration SECONDS] [--no-direct] [--no-kinematic]}"
shift
DURATION=60
RUN_DIRECT=1
RUN_KINEMATIC=1
while [ "$#" -gt 0 ]; do
  case "$1" in
    --duration) DURATION="$2"; shift 2 ;;
    --no-direct) RUN_DIRECT=0; shift ;;
    --no-kinematic) RUN_KINEMATIC=0; shift ;;
    *) echo "error: unknown argument $1" >&2; exit 2 ;;
  esac
done
[ -d "$PILOT_DIR" ] || { echo "error: pilot directory not found: $PILOT_DIR" >&2; exit 2; }
[ -f "$PILOT_DIR/reference.npz" ] || { echo "error: $PILOT_DIR/reference.npz missing" >&2; exit 2; }
[ -f "$PILOT_DIR/action_tokens.npz" ] || { echo "error: $PILOT_DIR/action_tokens.npz missing" >&2; exit 2; }

# dev.sh expects container paths; the caller passes the host path it sees.
container_path() {
  case "$1" in
    /*) printf '%s' "$1" ;;
    *) printf '%s/%s' "$PWD" "$1" ;;
  esac
}
PILOT_HOST="$(cd "$PILOT_DIR" && pwd)"
case "$PILOT_HOST" in
  "$PWD"/*) PILOT_CONTAINER="/workspace/humanoid-lab/${PILOT_HOST#"$PWD"/}" ;;
  *) PILOT_CONTAINER="$PILOT_HOST" ;;
esac
DATA_ROOT="${HUMANOID_DATA_ROOT:-$PWD/data}"
case "$PILOT_CONTAINER" in
  "$DATA_ROOT"/*) PILOT_CONTAINER="/data/${PILOT_CONTAINER#"$DATA_ROOT"/}" ;;
esac
echo "[pilot-sim] pilot: $PILOT_HOST -> $PILOT_CONTAINER"

NAME="$(basename "$(dirname "$PILOT_HOST")")-$(basename "$PILOT_HOST")"
LOG_DIR="$PWD/data/outputs/sonic-isaac"
mkdir -p "$LOG_DIR"

# Primary reference evidence: write the completed 50 Hz body/hand trajectory
# directly into Isaac. No controller, PD drive or physics tracking sits between
# reference.npz and the rendered robot.
if [ "$RUN_KINEMATIC" = 1 ]; then
  echo "[pilot-sim] frame-exact completed-reference replay"
  ./dev.sh isaac-g1-kinematic dex3 --headless \
    --kinematic-reference "$PILOT_CONTAINER/reference.npz" \
    --kinematic-label "completed_reference_50hz" \
    --record-video "$PILOT_CONTAINER/completed_reference_kinematic.mp4" \
    --tracking-output "$PILOT_CONTAINER/completed_reference_kinematic_tracking.parquet" \
    --metrics-output "$PILOT_CONTAINER/completed_reference_kinematic_metrics.json" \
    >"$LOG_DIR/$NAME-kinematic.log" 2>&1
fi

if [ "$RUN_DIRECT" = 1 ]; then
  echo "[pilot-sim] fixed-base direct joint oracle"
  ./dev.sh isaac-g1-direct-reference dex3 --headless \
    --trajectory-reference "$PILOT_CONTAINER/reference.npz" \
    --record-video "$PILOT_CONTAINER/direct_fixed.mp4" \
    --tracking-output "$PILOT_CONTAINER/direct_tracking.parquet" \
    --metrics-output "$PILOT_CONTAINER/direct_metrics.json" \
    --duration "$((DURATION / 4 + 4))" >"$LOG_DIR/$NAME-direct.log" 2>&1 \
    || echo "[pilot-sim] warning: direct oracle run exited non-zero (kept as evidence)"
fi

# The deployment refuses to start control without a live LowState, so the
# simulator publishes robot state first and the deployment attaches to it; the
# debug stream then marks the moment the ZMQ replay may start.
echo "[pilot-sim] starting Isaac G1 with offline-latent SONIC control"
./dev.sh isaac-g1-sonic dex3 --headless \
  --record-video "$PILOT_CONTAINER/sonic_latent_free.mp4" \
  --tracking-output "$PILOT_CONTAINER/sonic_latent_tracking.parquet" \
  --metrics-output "$PILOT_CONTAINER/sonic_latent_metrics.json" \
  --duration "$DURATION" >"$LOG_DIR/$NAME-sonic-latent.log" 2>&1 &
sim_pid=$!

for _ in $(seq 1 60); do
  grep -q '\[isaac-g1\] physics=' "$LOG_DIR/$NAME-sonic-latent.log" 2>/dev/null && break
  sleep 1
done
sleep 2

echo "[pilot-sim] starting the official deployment (ZMQ manager input)"
./dev.sh sonic-controller zmq_manager >"$LOG_DIR/$NAME-controller.log" 2>&1 &
controller_pid=$!
cleanup() {
  kill "$controller_pid" 2>/dev/null || true
  wait "$controller_pid" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# The deployment only publishes its debug stream once control has started, and
# the replay's start handshake drives that, so wait for its init line instead of
# waiting for a token that by definition cannot arrive yet.
for _ in $(seq 1 90); do
  grep -q 'Init Done' "$LOG_DIR/$NAME-controller.log" 2>/dev/null && break
  sleep 1
done
if grep -q 'Init Done' "$LOG_DIR/$NAME-controller.log" 2>/dev/null; then
  echo "[pilot-sim] deployment is initialized"
else
  echo "[pilot-sim] warning: deployment did not report 'Init Done' (see $LOG_DIR/$NAME-controller.log)"
fi
sleep 2
echo "[pilot-sim] replaying persisted offline tokens into the deployment"
replay_status=0
docker exec humanoid-lab-dev bash -lc '
  source /opt/humanoid-lab/entrypoint.sh && use-sonic-sim &&
  cd /workspace/humanoid-lab && PYTHONPATH=src exec python3 scripts/replay-sonic-latent.py \
    "'"$PILOT_CONTAINER"'/action_tokens.npz" --endpoint tcp://*:5556' || replay_status=$?
if [ "$replay_status" -ne 0 ]; then
  echo "[pilot-sim] latent replay failed with exit $replay_status" >&2
  kill "$sim_pid" 2>/dev/null || true
  wait "$sim_pid" 2>/dev/null || true
  exit "$replay_status"
fi

if ! wait "$sim_pid"; then
  echo "[pilot-sim] warning: free SONIC run exited non-zero (kept as evidence)"
fi

# The simulator video includes model startup and the supported IDLE takeover.
# Keep that full artifact, and create a review clip beginning at support release
# so time zero aligns with the source/reference motion rather than startup.
if [ -f "$PILOT_HOST/sonic_latent_metrics.json" ] && [ -f "$PILOT_HOST/sonic_latent_free.mp4" ]; then
  release_s="$(python3 - "$PILOT_HOST/sonic_latent_metrics.json" <<'PY'
import json, sys
payload = json.load(open(sys.argv[1], encoding="utf-8"))
tick = payload.get("controller", {}).get("support", {}).get("release_tick")
print("" if tick is None else float(tick) * 0.005)
PY
)"
  source_duration="$(ffprobe -v error -show_entries format=duration -of default=nw=1:nk=1 "$PILOT_HOST/source.mp4")"
  if [ -n "$release_s" ] && [ -n "$source_duration" ]; then
    ffmpeg -hide_banner -loglevel error -y -ss "$release_s" -i "$PILOT_HOST/sonic_latent_free.mp4" \
      -t "$source_duration" -an -c:v libx264 -pix_fmt yuv420p -g 25 -keyint_min 25 \
      -sc_threshold 0 -movflags +faststart "$PILOT_HOST/sonic_latent_motion.mp4"
  fi
fi
echo "[pilot-sim] done: $PILOT_HOST"
