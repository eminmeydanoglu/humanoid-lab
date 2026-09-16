#!/usr/bin/env bash
# Interactive keyboard driver for the InstinctLab G1 parkour playback.
#
# The Isaac Sim window already accepts W/S/A/D/F/G/X when it has focus.  This
# driver exists for when it does not: it forwards the same key events straight to
# the Isaac window by id, so you can drive the robot from any terminal -- an SSH
# session, a RustDesk session, or a second monitor -- without fighting for focus.
#
# Usage:
#   scripts/instinct-parkour-keyboard.sh                 # drive, show status
#   scripts/instinct-parkour-keyboard.sh --log FILE      # also tail the sim's [perf] line
#   scripts/instinct-parkour-keyboard.sh --status        # print one status line and exit
#   scripts/instinct-parkour-keyboard.sh --send w f      # send keys once, no UI
#
# Keys (identical to the ones the Isaac window itself listens for):
#   w / s   forward / back      (+/- 0.25 m/s per press, clamped to the trained range)
#   a / d   strafe left / right (clamped to 0 by default; see --lat below)
#   f / g   turn left / right   (+/- 0.5 rad/s, clamped to +/-1.0)
#   x       stop (zero the whole command)
#   q       quit this driver (the simulation keeps running)

set -uo pipefail

DISPLAY="${DISPLAY:-:1}"
WINDOW_NAME="${WINDOW_NAME:-^Isaac Sim 5\.1\.0$}"
LOG=""
MODE="drive"

# Mirror of the playback script's defaults, so the local readout matches the sim.
STEP=0.25
VX_MIN=0.0
VX_MAX=1.0
VY_MAX=0.0
WZ_MAX=1.0

while [ "$#" -gt 0 ]; do
  case "$1" in
    --log) LOG="${2:-}"; shift 2 ;;
    --status) MODE="status"; shift ;;
    --send) MODE="send"; shift; SEND_KEYS=("$@"); break ;;
    --step) STEP="${2:-0.25}"; shift 2 ;;
    --lat) VY_MAX="${2:-0.0}"; shift 2 ;;
    --display) DISPLAY="${2:-:1}"; shift 2 ;;
    -h|--help) sed -n '2,25p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

export DISPLAY

find_window() {
  xdotool search --name "$WINDOW_NAME" 2>/dev/null | tail -1
}

print_status() {
  # The simulation prints its own command/pose line; surface the newest one.
  if [ -n "$LOG" ] && [ -f "$LOG" ]; then
    local line
    line="$(grep -E '^\[perf\]' "$LOG" | tail -1)"
    if [ -n "$line" ]; then
      echo "$line"
      return 0
    fi
    echo "(no [perf] line in $LOG yet)"
    return 0
  fi
  echo "(pass --log FILE to show the simulation's own status line)"
}

WID="$(find_window)"
if [ -z "$WID" ]; then
  echo "error: no Isaac Sim window found on DISPLAY=$DISPLAY (looked for '$WINDOW_NAME')." >&2
  echo "       start the run first:  ./dev.sh instinct-parkour" >&2
  exit 1
fi

if [ "$MODE" = "status" ]; then
  echo "window: $WID"
  print_status
  exit 0
fi

if [ "$MODE" = "send" ]; then
  for k in "${SEND_KEYS[@]}"; do
    xdotool key --window "$WID" --clearmodifiers "$k"
    sleep 0.25
  done
  echo "sent: ${SEND_KEYS[*]}"
  exit 0
fi

# ---- interactive drive mode -------------------------------------------------
# The sim owns the real command state; this is a local mirror so the readout
# tells you what you have asked for, including the clamps the sim applies.
vx=0.00; vy=0.00; wz=0.00

mirror() {
  case "$1" in
    w) vx=$(awk -v v="$vx" -v s="$STEP" -v m="$VX_MAX" 'BEGIN{v+=s; if(v>m)v=m; printf "%.2f", v}') ;;
    s) vx=$(awk -v v="$vx" -v s="$STEP" -v m="$VX_MIN" 'BEGIN{v-=s; if(v<m)v=m; printf "%.2f", v}') ;;
    a) vy=$(awk -v v="$vy" -v s="$STEP" -v m="$VY_MAX" 'BEGIN{v+=s; if(v>m)v=m; printf "%.2f", v}') ;;
    d) vy=$(awk -v v="$vy" -v s="$STEP" -v m="$VY_MAX" 'BEGIN{v-=s; if(v<-m)v=-m; printf "%.2f", v}') ;;
    f) wz="0.50" ;;
    g) wz="-0.50" ;;
    x) vx=0.00; vy=0.00; wz=0.00 ;;
  esac
}

render() {
  printf '\r  command  vx %+5s   vy %+5s   wz %+5s      ' "$vx" "$vy" "$wz"
}

cat <<EOF
Driving Isaac Sim window $WID on DISPLAY=$DISPLAY
  w/s forward/back   a/d strafe (limit ${VY_MAX})   f/g turn   x stop   q quit
  envelope the checkpoint was trained on: vx [${VX_MIN}, ${VX_MAX}]  vy [$(awk -v m="$VY_MAX" 'BEGIN{printf "%.2f", -m}'), ${VY_MAX}]  wz [-${WZ_MAX}, ${WZ_MAX}]
EOF
[ -n "$LOG" ] && echo "  status line: $LOG"
echo

trap 'printf "\n"; exit 0' INT TERM
render
while true; do
  IFS= read -rsn1 key || break
  case "$key" in
    q|Q) printf '\n'; exit 0 ;;
    w|s|a|d|f|g|x) mirror "$key"; xdotool key --window "$WID" --clearmodifiers "$key"; render ;;
    *) : ;;
  esac
done
