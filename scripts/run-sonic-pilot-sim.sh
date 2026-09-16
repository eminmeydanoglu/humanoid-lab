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
DURATION=0
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

# --duration belongs to the whole simulator process, including model startup.
# A short fixed 60 s budget cut the 822-frame Dex3 replay at frame 481.  Give
# startup its own budget and size the run to the episode; caller duration is a
# minimum, never permission to produce an incomplete review artifact.
episode_frames="$(python3 - "$PILOT_DIR/run_manifest.json" <<'PY'
import json, sys
print(int(json.load(open(sys.argv[1], encoding="utf-8"))["frames"]))
PY
)"
minimum_duration="$((90 + 2 * ((episode_frames + 49) / 50)))"
if [ "$DURATION" -lt "$minimum_duration" ]; then
  DURATION="$minimum_duration"
fi
echo "[pilot-sim] wall-time duration budget: ${DURATION}s (${episode_frames} token frames)"

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
  --replay-clock-output "$PILOT_CONTAINER/sonic_sim_clock.txt" \
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
    "'"$PILOT_CONTAINER"'/action_tokens.npz" --endpoint tcp://*:5556 \
    --timeline-output "'"$PILOT_CONTAINER"'/sonic_latent_stream.npz" \
    --sim-clock "'"$PILOT_CONTAINER"'/sonic_sim_clock.txt"' || replay_status=$?
if [ "$replay_status" -ne 0 ]; then
  echo "[pilot-sim] latent replay failed with exit $replay_status" >&2
  kill "$sim_pid" 2>/dev/null || true
  wait "$sim_pid" 2>/dev/null || true
  exit "$replay_status"
fi

delivery_status=0
python3 scripts/check-sonic-token-delivery.py \
  --log "$LOG_DIR/$NAME-controller.log" \
  --expected-frames "$episode_frames" \
  --output "$PILOT_HOST/sonic_transport_coverage.json" || delivery_status=$?

if ! wait "$sim_pid"; then
  echo "[pilot-sim] warning: free SONIC run exited non-zero (kept as evidence)"
fi
if [ "$delivery_status" -ne 0 ]; then
  echo "[pilot-sim] error: controller did not receive every latent frame" >&2
  exit "$delivery_status"
fi

# The simulator video includes model startup and the supported IDLE takeover.
# Keep that full artifact, and create a review clip beginning at support release
# so time zero aligns with the source/reference motion rather than startup.
docker exec humanoid-lab-dev bash -lc '
  source /opt/humanoid-lab/entrypoint.sh && use-isaac-sonic &&
  cd /workspace/humanoid-lab && PYTHONPATH=src exec python3 scripts/evaluate-sonic-latent-replay.py \
    "'"$PILOT_CONTAINER"'"'
if [ "$(jq -r '.coverage.full_motion' "$PILOT_HOST/sonic_fidelity.json")" != true ]; then
  echo "[pilot-sim] error: incomplete latent replay; refusing to make a review clip" >&2
  exit 1
fi
if [ -f "$PILOT_HOST/sonic_latent_free.mp4" ]; then
  clip_start="$(jq -r '.coverage.clip_start_sim_s' "$PILOT_HOST/sonic_fidelity.json")"
  clip_end="$(jq -r '.coverage.clip_end_sim_s' "$PILOT_HOST/sonic_fidelity.json")"
  clip_duration="$(awk -v a="$clip_start" -v b="$clip_end" 'BEGIN {print b-a}')"
  ffmpeg -hide_banner -loglevel error -y -ss "$clip_start" -i "$PILOT_HOST/sonic_latent_free.mp4" \
    -t "$clip_duration" -an -c:v libx264 -pix_fmt yuv420p -g 25 -keyint_min 25 \
    -sc_threshold 0 -movflags +faststart "$PILOT_HOST/sonic_latent_motion.mp4"
  recorded_duration="$(ffprobe -v error -show_entries format=duration -of default=nw=1:nk=1 "$PILOT_HOST/sonic_latent_motion.mp4")"
  if ! awk -v got="$recorded_duration" -v expected="$clip_duration" 'BEGIN {exit !(got + 0.12 >= expected)}'; then
    echo "[pilot-sim] error: review clip is short (${recorded_duration}s vs ${clip_duration}s expected)" >&2
    exit 1
  fi
fi
python3 scripts/write-sonic-pilot-page.py --pilot-dir "$PILOT_HOST" --output "$PILOT_HOST/review.html"
echo "[pilot-sim] done: $PILOT_HOST"
if [ "$(jq -r '.result' "$PILOT_HOST/sonic_fidelity.json")" != PASS ]; then
  echo "[pilot-sim] A/B/C fidelity gate FAILED; recordings and metrics are retained for review" >&2
  exit 1
fi
