#!/usr/bin/env bash
# Watch this host's Isaac Sim livestream from another machine.
#
# The client runs where the person looks, never on the streaming host: a
# livestreamed run has no local window.  The default target is emin-1; give
# another tailnet name (raider, or a user@host) to watch from somewhere else.
#
#   ./dev.sh webrtc-view                # emin-1
#   ./dev.sh webrtc-view raider         # the MSI laptop
#   ISAAC_VIEW_RAIDER=emin@host ./dev.sh webrtc-view raider
#
# The viewing machine needs the client (scripts/install-isaac-webrtc-client.sh)
# and python3-websocket; both are checked here and reported when missing.
set -euo pipefail
cd "$(dirname "$0")/.."

VIEW_PORT="${ISAAC_LIVESTREAM_PORT:-49100}"
DEBUG_PORT="${ISAAC_VIEW_DEBUG_PORT:-9223}"
CONNECT_HELPER="scripts/webrtc-client-connect.py"
CONNECT_HELPER_REMOTE="/tmp/humanoid-lab-webrtc-connect.py"

target="${1:-${ISAAC_VIEW_TARGET:-emin-1}}"
case "$target" in
  emin-1|emin1|emin)
    destination="${ISAAC_VIEW_EMIN1:-emin@emin-1}"
    address_name="emin-1"
    ;;
  raider|raider16)
    # The laptop's username is not known here; name it explicitly with
    # ISAAC_VIEW_RAIDER (or pass user@host) the first time.
    destination="${ISAAC_VIEW_RAIDER:-aksoy-msi-raider-16-max-hx-b2wj}"
    address_name="aksoy-msi-raider-16-max-hx-b2wj"
    ;;
  *@*) destination="$target"; address_name="${target#*@}" ;;
  *) destination="$target"; address_name="$target" ;;
esac

endpoint="${ISAAC_LIVESTREAM_ENDPOINT:-}"
if [ -z "$endpoint" ] && command -v tailscale >/dev/null 2>&1; then
  endpoint="$(tailscale ip -4 2>/dev/null | head -n1 || true)"
fi
[ -n "$endpoint" ] || { echo "error: no streaming address; set ISAAC_LIVESTREAM_ENDPOINT" >&2; exit 2; }

if ss -tln 2>/dev/null | grep -q ":${VIEW_PORT} "; then
  echo "[webrtc-view] this host is serving a livestream on ${endpoint}:${VIEW_PORT}"
else
  echo "[webrtc-view] warning: nothing is listening on port ${VIEW_PORT} yet." >&2
  echo "[webrtc-view] start a streamed run first, e.g. ./dev.sh isaac-g1-sonic-rough dex3" >&2
fi

remote_address="$(tailscale ip -4 "$address_name" 2>/dev/null | head -n1 || true)"
echo "[webrtc-view] target: $destination (watching ${endpoint}:${VIEW_PORT})"

# accept-new: a viewer that has never been reached from here has no host key
# yet, and a non-interactive run cannot answer the confirmation prompt.
SSH=(ssh -o ConnectTimeout=10 -o StrictHostKeyChecking=accept-new)

# The helper has to live on the viewing machine, next to the client it drives.
if ! "${SSH[@]}" "$destination" "cat > $CONNECT_HELPER_REMOTE" < "$CONNECT_HELPER"; then
  echo "[webrtc-view] could not reach $destination over ssh." >&2
  echo "[webrtc-view] for a machine that has never been used as a viewer, name the" >&2
  echo "[webrtc-view] account once: ISAAC_VIEW_RAIDER=user@host $0 raider" >&2
  exit 2
fi

remote_launch() {
  "${SSH[@]}" -T "$destination" bash -s -- "$DEBUG_PORT" "$endpoint" "$CONNECT_HELPER_REMOTE" <<'REMOTE'
set -euo pipefail
debug_port="$1"
server="$2"
helper="$3"

client="$HOME/Applications/IsaacSim/run-isaac-webrtc-client.sh"
if [ ! -x "$client" ]; then
  echo "error: the WebRTC client is not installed on this machine." >&2
  echo "       install it with: scripts/install-isaac-webrtc-client.sh" >&2
  exit 3
fi
if ! python3 -c "import websocket" 2>/dev/null; then
  echo "error: python3-websocket is missing on this machine." >&2
  echo "       install it with: sudo apt-get install -y python3-websocket" >&2
  exit 3
fi

# Wayland sessions run X clients through Xwayland, which is where the client's
# window (and its input) live; its own -auth file is the one that works over SSH.
display=""
auth=""
for pattern in '(^|/)Xwayland ' '(^|/)Xorg '; do
  line="$(ps -eo args --no-headers | grep -m1 -E "$pattern" || true)"
  [ -n "$line" ] || continue
  display="$(echo "$line" | awk '{print $2}')"
  auth="$(echo "$line" | sed -n 's/.*-auth \([^ ]*\).*/\1/p')"
  break
done
[ -n "$display" ] || display=":0"
if [ -n "$auth" ] && [ ! -r "$auth" ]; then auth="${HOME}/.Xauthority"; fi
export DISPLAY="${ISAAC_VIEW_DISPLAY:-$display}"
export XAUTHORITY="${ISAAC_VIEW_XAUTHORITY:-${auth:-$HOME/.Xauthority}}"
echo "[webrtc-view] remote display ${DISPLAY} (auth ${XAUTHORITY})"

# One client at a time: an older instance would keep holding the DevTools port.
if command -v xdotool >/dev/null 2>&1; then
  old="$(xdotool search --name 'Isaac Sim WebRTC Streaming Client' 2>/dev/null | tail -1 || true)"
  if [ -n "$old" ]; then
    old_pid="$(xdotool getwindowpid "$old" 2>/dev/null || true)"
    if [ -n "$old_pid" ]; then kill "$old_pid" 2>/dev/null || true; sleep 3; fi
  fi
fi

setsid nohup "$client" --remote-debugging-port="$debug_port" >/tmp/webrtc-client.log 2>&1 </dev/null &
sleep 6
python3 "$helper" --port "$debug_port" --server "$server"
REMOTE
}

remote_connect() {
  "${SSH[@]}" -T "$destination" python3 "$CONNECT_HELPER_REMOTE" \
    --port "$DEBUG_PORT" --server "$endpoint"
}

wait_for_connection() { # $1 = seconds
  # `ss` prints the accepted socket as "local:PORT peer:PORT", so the viewer
  # shows up as the peer of our own listening port.
  local second
  [ -n "$remote_address" ] || return 1
  for second in $(seq 1 "$1"); do
    if ss -Htn state established 2>/dev/null |
      grep -F ":${VIEW_PORT} " |
      grep -qF "${remote_address}:"; then
      return 0
    fi
    sleep 1
  done
  return 1
}

if wait_for_connection 1; then
  echo "[webrtc-view] $destination is already watching ${endpoint}:${VIEW_PORT}"
  [ "${ISAAC_VIEW_RESTART:-0}" = "1" ] || exit 0
  echo "[webrtc-view] ISAAC_VIEW_RESTART=1: restarting its client"
fi

remote_launch

if wait_for_connection 15; then
  echo "[webrtc-view] connected: ${remote_address} is streaming from ${endpoint}:${VIEW_PORT}"
  exit 0
fi

echo "[webrtc-view] no stream yet; asking the client once more" >&2
remote_connect || true

if wait_for_connection 15; then
  echo "[webrtc-view] connected: ${remote_address} is streaming from ${endpoint}:${VIEW_PORT}"
  exit 0
fi

if [ -n "$remote_address" ]; then
  echo "[webrtc-view] ${remote_address} never reached ${endpoint}:${VIEW_PORT}; check the client" >&2
  echo "[webrtc-view] window on ${destination}, and that the run is streaming." >&2
  exit 1
fi
echo "[webrtc-view] client started; connection unconfirmed (no address known for ${address_name})"
