#!/usr/bin/env bash
# One-command simulation loop for the Flux 3 / Dex3 evaluation.
#
# Canonical run:  ./dev.sh flux-sim e2e        (or `up` to stay running)
#   scene:        configs/profiles/isaac-g1-flux-dex3-pickapple.json
#                 (fixed-root calibrated PickApple table)
#   actuation:    configs/flux/flux-dex3-sim-motor-config.json
#   --no-motor-commands selects the command-disabled dry run; --profile and
#   --motor-output-config replace either file per run.  No SONIC controller is
#   ever started: the simulator's flux_dds adapter owns the command topics.
#
# Lifecycle this script owns (all of it host-side):
#   Isaac (fixed-root scene profile, SONIC-format camera feed on :5555),
#   the profile-gated flux-ros container with the robot's own ROS node,
#   the GPU model server on the explicit port 5561,
#   and the ROS launch (camera bridge + flux_dex3 node).
#
# Evidence lives in $HUMANOID_DATA_ROOT/outputs/flux-dex3/<tag>: the ROS and
# Isaac logs, the simulator's tracking Parquet (measured joint motion) and the
# JSON reports of the client exercise and the tracking verification.
set -euo pipefail

repo_root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$repo_root"
if [ -f .env ]; then
  set -a
  # shellcheck source=/dev/null
  source .env
  set +a
fi

data_root="${HUMANOID_DATA_ROOT:-$repo_root/data}"
state_root="$data_root/outputs/flux-dex3"
latest_file="$state_root/latest"
prompt="${FLUX_PROMPT:-Put the apple into the plate.}"
port="${FLUX_MODEL_PORT:-5561}"
camera_port="${FLUX_CAMERA_PORT:-5555}"
profile="${FLUX_ISAAC_PROFILE:-}"
#: The `e2e` run waits for Isaac to finish its own duration, so the default is
#: bounded; raise it for interactive work.
duration="${FLUX_DURATION:-180}"
run_s="${FLUX_RUN_S:-8}"
#: The canonical evaluation scene (fixed-root calibrated PickApple table) and
#: the simulator's Flux actuation contract.  Both are ordinary repository paths;
#: --profile / --motor-output-config replace them, --no-motor-commands selects
#: the command-disabled dry run.
default_profile="configs/profiles/isaac-g1-flux-dex3-pickapple.json"
default_motor_config="configs/flux/flux-dex3-sim-motor-config.json"
motor_config="${FLUX_MOTOR_CONFIG:-$default_motor_config}"
tag="$(date +%Y%m%d-%H%M%S)-$$"
declared_tag=0
declared_keys=" "
motor_config_declared=0
no_motor_commands=0
#: How the simulator's UI is delivered.  The WebRTC livestream is the default:
#: it needs no local X display and can be watched from any Isaac Sim WebRTC
#: client; --gui is the local window and --headless the lightest mode for
#: measurement runs.  `livestream` passes no mode flag on to `flux-isaac`,
#: which owns the endpoint resolution (this host's Tailscale address when it
#: has one) and prints the connection banner.
display_mode="${FLUX_ISAAC_MODE:-livestream}"
mode_flags=""
no_verify=0
no_task=0
keep=0
keep_model_server=0
follow_logs=1

usage() {
  cat >&2 <<EOF
usage: $0 <up|logs|status|task|verify|stop|config|e2e> [options]

Canonical evaluation run (no flags needed):
  scene:     $default_profile   (fixed-root calibrated PickApple table)
  actuation: $default_motor_config

  --profile PATH            Isaac scene profile (\$FLUX_ISAAC_PROFILE)
  --prompt TEXT             task caption (default: "$prompt")
  --port N                  model server port (default $port; 5561 because the
                            SONIC state port owns 5557)
  --duration S              Isaac run length (default $duration s); the e2e run
                            waits for it to finish so Isaac writes its tracking
                            Parquet and run summary
  --run-s S                 task window the client exercises (default $run_s s)
  --motor-output-config P   node motor config (\$FLUX_MOTOR_CONFIG)
  --no-motor-commands       command-disabled dry run: the node creates no motor
                            publishers at all.  Use it for negative controls;
                            it conflicts with --motor-output-config.
  --livestream              Isaac's WebRTC livestream, no local window: connect
                            the Isaac Sim WebRTC client to this host (default)
  --gui                     run Isaac with the local X11 window
  --headless                no UI at all: the lightest render load, for
                            measurement runs.  At most one of these three.
  --no-verify               e2e: skip the tracking verification
  --no-task                 e2e: negative control -- start nothing and require
                            zero controlled rows in the tracking file
  --keep                    e2e: leave the session running at the end
  --detach                  up: return after preflight instead of following logs
  --keep-model-server       stop: leave the READY GPU server running (the model
                            reload costs minutes)
  --tag NAME                run tag (default: timestamp-pid)

What the launcher starts and checks (never a SONIC controller process):
  * docker compose: the dev simulator container and the profile-gated flux-ros
    container are started if they are not running;
  * the ROS overlay is rebuilt from the mounted checkout
    (scripts/flux-ros-build.sh over third_party/flux/flux-inference/ros2);
  * the GPU model server on 127.0.0.1:<port> must report READY
    (scripts/flux-model-server.sh); it accepts only an immutable, read-only,
    symlink-free adapter copy verified by its own checkpoint_identity
    (prepare it once with: ./dev.sh flux-checkpoint --source <training checkpoint>);
  * the scene profile exists; unless --no-motor-commands, the motor config
    exists and passes the node's own load_motor_config() before the model is
    loaded.
  \`config\` prints the resolved plan and these file checks without touching
  docker, the simulator or the model server.
EOF
}

subcommand="${1:-}"
[ -n "$subcommand" ] || { usage; exit 2; }
shift
case "$subcommand" in
  -h|--help|help) usage; exit 0 ;;
esac

while [ "$#" -gt 0 ]; do
  case "$1" in
    --profile) profile="${2:-}"; declared_keys="$declared_keys profile "; shift 2 ;;
    --prompt) prompt="${2:-}"; declared_keys="$declared_keys prompt "; shift 2 ;;
    --port) port="${2:-}"; declared_keys="$declared_keys port "; shift 2 ;;
    --duration) duration="${2:-}"; declared_keys="$declared_keys duration "; shift 2 ;;
    --run-s) run_s="${2:-}"; declared_keys="$declared_keys run_s "; shift 2 ;;
    --motor-output-config) motor_config="${2:-}"; motor_config_declared=1; declared_keys="$declared_keys motor_config "; shift 2 ;;
    --no-motor-commands) no_motor_commands=1; declared_keys="$declared_keys motor_config "; shift ;;
    --tag) tag="${2:-}"; declared_tag=1; declared_keys="$declared_keys tag "; shift 2 ;;
    --livestream) display_mode=livestream; mode_flags="$mode_flags --livestream"; shift ;;
    --gui) display_mode=gui; mode_flags="$mode_flags --gui"; shift ;;
    --headless) display_mode=headless; mode_flags="$mode_flags --headless"; shift ;;
    --no-verify) no_verify=1; shift ;;
    --no-task) no_task=1; shift ;;
    --keep) keep=1; shift ;;
    --detach) follow_logs=0; shift ;;
    --keep-model-server) keep_model_server=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "error: unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done
if [ "$(printf '%s' "$mode_flags" | wc -w)" -gt 1 ]; then
  echo "error: only one of --livestream, --gui, --headless may be given:$mode_flags" >&2
  exit 2
fi
case "$display_mode" in
  livestream|gui|headless) ;;
  *) echo "error: display mode must be livestream, gui or headless, got '$display_mode'" >&2; exit 2 ;;
esac
if [ "$no_motor_commands" -eq 1 ]; then
  if [ "$motor_config_declared" -eq 1 ]; then
    echo "error: --no-motor-commands conflicts with --motor-output-config" >&2
    exit 2
  fi
  motor_config=""
fi
profile="${profile:-$default_profile}"
for pair in "duration:$duration" "run_s:$run_s" "port:$port" "camera_port:$camera_port"; do
  name="${pair%%:*}"
  value="${pair#*:}"
  if ! [[ "$value" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
    echo "error: $name must be a number, got '$value'" >&2
    exit 2
  fi
done

# Derived per-session paths, recomputed after a state load.
set_paths() {
  run_dir="$state_root/$tag"
  host_log_dir="$run_dir"
  state_file="$run_dir/state.env"
  # The same tree under the two containers' own spellings.
  flux_dir="/data/outputs/flux-dex3/$tag"
  dev_dir="/outputs/flux-dex3/$tag"
}
set_paths

require_command() {
  command -v "$1" >/dev/null 2>&1 || { echo "error: $1 is required" >&2; exit 2; }
}
# `config` only resolves paths and checks files: it must run without docker.
if [ "$subcommand" != config ]; then
  require_command docker
fi

flux_ros_exec() { # bash command string inside flux-ros
  docker compose --env-file .env --profile flux exec -T flux-ros \
    bash -lc "source /opt/humanoid-lab/flux-ros/entrypoint.sh && $1"
}

save_state() {
  {
    printf 'tag=%q\n' "$tag"
    printf 'profile=%q\n' "$profile"
    printf 'prompt=%q\n' "$prompt"
    printf 'port=%q\n' "$port"
    printf 'duration=%q\n' "$duration"
    printf 'run_s=%q\n' "$run_s"
    printf 'motor_config=%q\n' "$motor_config"
  } >"$state_file"
}

load_state() {
  if [ "$declared_tag" -eq 0 ]; then
    [ -f "$latest_file" ] || { echo "error: no flux-sim session recorded; run '$0 up' first" >&2; exit 2; }
    tag="$(cat "$latest_file")"
  fi
  set_paths
  [ -f "$state_file" ] || { echo "error: no session state at $state_file" >&2; exit 2; }
  # Values named on this command line win over the recorded session.
  local key value
  while IFS='=' read -r key value; do
    case "$key" in
      tag|profile|prompt|port|duration|run_s|motor_config) ;;
      *) continue ;;
    esac
    case "$declared_keys" in
      *" $key "*) continue ;;
    esac
    eval "$key=$value"
  done <"$state_file"
}

container_motor_config() {
  [ -n "$motor_config" ] || return 0
  case "$motor_config" in
    /data/*|/outputs/*|/workspace/*) printf '%s\n' "$motor_config" ;;
    "$data_root"/*) printf '/data/%s\n' "${motor_config#"$data_root"/}" ;;
    "$repo_root"/*) printf '/workspace/humanoid-lab/%s\n' "${motor_config#"$repo_root"/}" ;;
    /*) printf '%s\n' "$motor_config" ;;
    *) printf '/workspace/humanoid-lab/%s\n' "$motor_config" ;;
  esac
}

isaac_alive() {
  [ -r "$run_dir/isaac.pid" ] || return 1
  local pid
  read -r pid <"$run_dir/isaac.pid" || return 1
  [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null
}

start_isaac() {
  mkdir -p "$run_dir"
  local -a base=(--duration "$duration"
                 --tracking-output "$dev_dir/tracking.parquet"
                 --metrics-output "$dev_dir/summary.json")
  # `livestream` intentionally passes no mode flag: `flux-isaac` resolves the
  # WebRTC endpoint and prints the banner, so that resolution lives in one place.
  local -a mode_args=()
  case "$display_mode" in
    gui) mode_args=(--gui) ;;
    headless) mode_args=(--headless) ;;
  esac
  ./dev.sh flux-isaac "$profile" "${mode_args[@]}" "${base[@]}" >"$host_log_dir/isaac.log" 2>&1 &
  printf '%s\n' "$!" >"$run_dir/isaac.pid"
  echo "[flux-sim] Isaac started: profile=$profile mode=$display_mode pid=$(cat "$run_dir/isaac.pid") log=$host_log_dir/isaac.log" >&2
  if [ "$display_mode" = livestream ]; then
    echo "[flux-sim] WebRTC livestream: point the Isaac Sim WebRTC client at this host, port ${ISAAC_LIVESTREAM_PORT:-49100}" >&2
    echo "[flux-sim] the resolved address and banner are in $host_log_dir/isaac.log" >&2
  fi
}

start_model_server() {
  # The server's pidfile/log stay in the shared flux-dex3 directory so a later
  # session can stop the same process; only the launch log is per-tag.
  bash scripts/flux-model-server.sh --port "$port" \
    >"$host_log_dir/model-server-launch.log" 2>&1
  cat "$host_log_dir/model-server-launch.log" >&2
}

start_ros_launch() {
  local container_config enable launch_args matches
  container_config="$(container_motor_config)"
  enable=false
  launch_args=""
  if [ -n "$container_config" ]; then
    enable=true
    launch_args="motor_output_config:='$container_config' enable_motor_commands:=true"
  fi
  if flux_ros_exec "pgrep -f '[r]os2 launch flux_sim_camera flux_sim.launch.py' >/dev/null 2>&1" >/dev/null 2>&1; then
    # A launch from an earlier `up` may still be alive.  It is kept only when
    # its command mode matches this session; otherwise it is replaced, so a
    # stale command-disabled launch can never serve an enabled session.
    if [ "$enable" = true ]; then
      matches="pgrep -af '[r]os2 launch flux_sim_camera flux_sim.launch.py' | grep -Fq 'motor_output_config:=$container_config'"
    else
      matches="! pgrep -af '[r]os2 launch flux_sim_camera flux_sim.launch.py' | grep -Fq 'enable_motor_commands:=true'"
    fi
    if flux_ros_exec "$matches" >/dev/null 2>&1; then
      echo "[flux-sim] ROS launch already running (command publishing $enable)" >&2
      return 0
    fi
    echo "[flux-sim] replacing the running ROS launch (command publishing $enable)" >&2
    stop_ros_launch
  fi
  flux_ros_exec "mkdir -p '$flux_dir' && cd /workspace/humanoid-lab && \
    exec bash -c 'setsid ros2 launch flux_sim_camera flux_sim.launch.py \
      model_endpoint:=tcp://127.0.0.1:$port \
      camera_endpoint:=tcp://127.0.0.1:$camera_port \
      $launch_args \
      > \"$flux_dir/ros-launch.log\" 2>&1 < /dev/null & echo \$! > \"$flux_dir/ros-launch.pid\"'" >/dev/null
  echo "[flux-sim] ROS launch started in flux-ros (camera bridge + flux_dex3 node, command publishing $enable)" >&2
}

stop_ros_launch() {
  # The detached launch is matched by its command line as well as by the
  # pidfile: the pid recorded at start can exit while `ros2 launch` lives on,
  # which would otherwise leave a duplicate node behind.
  flux_ros_exec "
    killed=0
    for pid in \$(pgrep -f '[r]os2 launch flux_sim_camera flux_sim.launch.py' 2>/dev/null); do
      kill -- \"-\$pid\" 2>/dev/null || kill \"\$pid\" 2>/dev/null || true
      killed=1
    done
    if [ \"\$killed\" -eq 1 ]; then
      for _ in \$(seq 1 40); do
        pgrep -f '[r]os2 launch flux_sim_camera flux_sim.launch.py' >/dev/null 2>&1 || break
        sleep 0.1
      done
      for pid in \$(pgrep -f '[r]os2 launch flux_sim_camera flux_sim.launch.py' 2>/dev/null); do
        kill -KILL -- \"-\$pid\" 2>/dev/null || kill -KILL \"\$pid\" 2>/dev/null || true
      done
    fi
    pkill -f '[f]lux_sim_camera/lib/flux_sim_camera/camera_bridge' 2>/dev/null || true
    pkill -f '[f]lux_dex3/lib/flux_dex3/dex3_node' 2>/dev/null || true
    rm -f -- '$flux_dir/ros-launch.pid'" >/dev/null 2>&1 || true
}

stop_isaac() {
  [ -r "$run_dir/isaac.pid" ] || return 0
  local pid
  read -r pid <"$run_dir/isaac.pid" || pid=""
  if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
    echo "[flux-sim] stopping Isaac launcher pid=$pid" >&2
    kill -TERM "$pid" 2>/dev/null || true
    for _ in $(seq 1 100); do kill -0 "$pid" 2>/dev/null || break; sleep 0.2; done
    kill -KILL "$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
  fi
  rm -f -- "$run_dir/isaac.pid"
}

eval_client() { # mode, extra argument string
  local mode="$1" extra="${2:-}"
  flux_ros_exec "cd /workspace/humanoid-lab && exec python3 scripts/flux-sim-eval.py --mode $mode --model-endpoint tcp://127.0.0.1:$port $extra"
}

validate_motor_config() {
  # The node's own loader is the acceptance check: 28 joint limits, gains,
  # tracking bound and the two confirmation flags, exactly as the robot node
  # validates them.  Fail here, before the model server spends minutes loading.
  local container_config
  container_config="$(container_motor_config)"
  [ -n "$container_config" ] || return 0
  if [ ! -f "$motor_config" ]; then
    echo "error: motor config missing: $motor_config" >&2
    echo "       pass --motor-output-config PATH or --no-motor-commands for a dry run" >&2
    return 2
  fi
  if ! flux_ros_exec "python3 -c 'import sys; from flux_dex3.command_output import load_motor_config; load_motor_config(sys.argv[1]); print(\"motor config OK\")' '$container_config'" >/dev/null; then
    echo "error: the node rejected the motor config: $motor_config" >&2
    echo "       (see the node's load_motor_config contract; --no-motor-commands runs without it)" >&2
    return 2
  fi
}

cmd_config() {
  # The resolved plan for this invocation, without touching docker, the
  # simulator or the model server: exactly what `up`/`e2e` would run and which
  # files they would check first.
  local mode="$display_mode"
  python3 - "$profile" "$motor_config" "$port" "$camera_port" "$duration" "$run_s" \
           "$repo_root" "$run_dir" "$mode" "$prompt" <<'PY'
import json
import os
import sys

profile, motor_config, port, camera_port, duration, run_s, repo_root, run_dir, mode, prompt = sys.argv[1:11]
enabled = bool(motor_config)
plan = {
    "profile": profile,
    "profile_exists": os.path.isfile(profile),
    "motor_config": motor_config,
    "motor_config_exists": os.path.isfile(motor_config) if enabled else None,
    "motor_commands": enabled,
    "mode": mode,
    "prompt": prompt,
    "model_endpoint": "tcp://127.0.0.1:%s" % port,
    "camera_endpoint": "tcp://127.0.0.1:%s" % camera_port,
    "duration_s": float(duration),
    "run_s": float(run_s),
    "evidence_dir": run_dir,
    "sonic_process": False,
    "prerequisites": [
        "docker compose starts the dev simulator and the profile-gated flux-ros container",
        "scripts/flux-ros-build.sh rebuilds the ROS overlay from the mounted checkout",
        "scripts/flux-model-server.sh serves the immutable read-only adapter copy on the model endpoint and must report READY",
        "the motor config (when enabled) passes the node's own load_motor_config",
        "no SONIC controller process is started; the simulator's flux_dds adapter owns the command topics",
    ],
}
print(json.dumps(plan, indent=2, sort_keys=True))
PY
}

cmd_logs() {
  load_state
  echo "[flux-sim] following Isaac, ROS and model logs for $tag; Ctrl-C detaches without stopping the session" >&2
  exec tail -n 12 -F "$run_dir/isaac.log" "$run_dir/ros-launch.log" \
    "${FLUX_LOG_DIR:-$state_root}/model-server-$port.log"
}

cmd_up() {
  # Without an explicit tag, `up` resumes the recorded session (starting a
  # second Isaac against the same scene is never what the caller wants); a
  # fresh session is requested with --tag or after `stop`.
  if [ "$declared_tag" -eq 0 ] && [ -f "$latest_file" ]; then
    local candidate
    candidate="$(cat "$latest_file")"
    if [ -f "$state_root/$candidate/state.env" ]; then
      tag="$candidate"
    fi
  fi
  set_paths
  [ -f "$profile" ] || {
    echo "error: Isaac profile missing: $profile" >&2
    echo "       the canonical scene is $default_profile (--profile PATH overrides)" >&2
    exit 2
  }
  if [ -f "$state_file" ]; then
    # Re-entrant: `up` again brings the same session back up (for example
    # after a partial start) instead of creating a second one.  Explicit
    # options from this invocation are folded into the recorded session.
    load_state
    save_state
    printf '%s\n' "$tag" >"$latest_file"
    echo "[flux-sim] resuming session $tag" >&2
  else
    mkdir -p "$run_dir"
    printf '%s\n' "$tag" >"$latest_file"
    save_state
  fi
  docker compose --env-file .env up -d dev >/dev/null
  docker compose --env-file .env --profile flux up -d flux-ros >/dev/null
  echo "[flux-sim] building the ROS overlay from the mounted checkout ..." >&2
  flux_ros_exec "exec bash /workspace/humanoid-lab/scripts/flux-ros-build.sh" \
    >"$host_log_dir/flux-ros-build.log" 2>&1
  validate_motor_config || return $?
  if ! isaac_alive; then
    start_isaac
  fi
  start_model_server
  start_ros_launch
  echo "[flux-sim] waiting for model READY, camera and joint state ..." >&2
  if eval_client preflight "--status-timeout 300 --window-s 5 --report $flux_dir/preflight.json" \
       >"$host_log_dir/preflight.log" 2>&1; then
    cat "$host_log_dir/preflight.log"
    echo "[flux-sim] preflight PASS" >&2
  else
    local rc=$?
    cat "$host_log_dir/preflight.log" >&2 || true
    echo "[flux-sim] preflight FAIL (exit $rc); logs in $host_log_dir" >&2
    return 1
  fi
}

cmd_status() {
  load_state
  eval_client preflight "--status-timeout 5 --window-s 3 --report $flux_dir/preflight.json"
}

cmd_task() {
  load_state
  local expect=""
  [ -n "$motor_config" ] && expect="--expect-motor-commands"
  eval_client task "--prompt '$prompt' --run-s $run_s $expect --report $flux_dir/task-report.json"
}

cmd_verify() {
  load_state
  # The task report carries the StartTask/StopTask epochs; the tracking file
  # must not contain controlled rows outside that window (negative control).
  # A no-task session instead requires zero controlled rows at all.
  local window_args=""
  if [ "$no_task" -eq 1 ]; then
    window_args="--expect-no-commands"
  elif [ -f "$run_dir/task-report.json" ]; then
    window_args="$(python3 - "$run_dir/task-report.json" <<'PY'
import json
import sys

report = json.loads(open(sys.argv[1], encoding="utf-8").read())
args = []
if report.get("start_response_wall_unix"):
    args.extend(["--first-row-after", "%.3f" % report["start_response_wall_unix"]])
if report.get("stop_wall_unix"):
    args.extend(["--last-row-before", "%.3f" % report["stop_wall_unix"]])
print(" ".join(args))
PY
)"
  fi
  docker compose --env-file .env exec -T dev bash -lc "
    source /opt/humanoid-lab/entrypoint.sh
    use-isaac-sonic
    cd /workspace/humanoid-lab
    exec python3 scripts/flux-verify-tracking.py \
      --tracking '$dev_dir/tracking.parquet' \
      --metrics '$dev_dir/summary.json' \
      --json-out '$dev_dir/tracking-report.json' $window_args"
}

wait_isaac_exit() {
  # The tracking Parquet and the run summary are written when the simulator
  # exits its own duration; a killed simulator writes neither.
  local pid deadline
  isaac_alive || return 0
  read -r pid <"$run_dir/isaac.pid" || return 0
  deadline=$(( $(date +%s) + duration + 180 ))
  echo "[flux-sim] waiting for Isaac to finish its ${duration}s run (pid=$pid) ..." >&2
  while kill -0 "$pid" 2>/dev/null; do
    [ "$(date +%s)" -lt "$deadline" ] || {
      echo "[flux-sim] Isaac did not exit before the deadline" >&2
      return 1
    }
    sleep 2
  done
  rm -f -- "$run_dir/isaac.pid"
  return 0
}

cmd_stop() {
  load_state
  stop_ros_launch
  if [ "$keep_model_server" -eq 0 ]; then
    bash scripts/flux-model-server.sh --port "$port" --stop >&2 || true
  fi
  stop_isaac
  echo "[flux-sim] session $tag stopped" >&2
}

cmd_e2e() {
  local task_rc=0 verify_rc=0 up_rc=0
  # Every end-to-end run gets its own configuration and evidence.  Reusing the
  # latest `up` session could silently restore an earlier dry-run motor mode.
  if [ "$declared_tag" -eq 0 ]; then
    declared_tag=1
    set_paths
  fi
  cmd_up || up_rc=$?
  if [ "$up_rc" -ne 0 ]; then
    # The session is left running on purpose: a failed preflight is inspected
    # from its logs before the next attempt.
    echo "[flux-sim] e2e: session did not come up (rc=$up_rc); it is left running for inspection" >&2
    echo "[flux-sim] e2e: stop it with '$0 stop' (evidence in $host_log_dir)" >&2
    return "$up_rc"
  fi
  if [ "$no_task" -eq 0 ]; then
    cmd_task || task_rc=$?
  fi
  wait_isaac_exit || task_rc=$?
  if [ "$no_verify" -eq 0 ]; then
    cmd_verify || verify_rc=$?
  fi
  [ "$keep" -eq 1 ] || cmd_stop
  echo "[flux-sim] e2e: task_rc=$task_rc verify_rc=$verify_rc evidence=$host_log_dir" >&2
  [ "$task_rc" -eq 0 ] && [ "$verify_rc" -eq 0 ]
}

case "$subcommand" in
  up) cmd_up; [ "$follow_logs" -eq 0 ] || cmd_logs ;;
  logs) cmd_logs ;;
  status) cmd_status ;;
  task) cmd_task ;;
  verify) cmd_verify ;;
  stop) cmd_stop ;;
  config) cmd_config ;;
  e2e) cmd_e2e ;;
  *) usage; exit 2 ;;
esac
