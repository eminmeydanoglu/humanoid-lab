#!/usr/bin/env bash
# Keep the F310 bridge private: local TCP -> SSH -> Unitree loopback only.
set -euo pipefail

port="${F310_BRIDGE_PORT:-49051}"
case "$port" in
  ''|*[!0-9]*) echo "F310_BRIDGE_PORT must be numeric" >&2; exit 64 ;;
esac
if (( port < 1024 || port > 65535 )); then
  echo "F310_BRIDGE_PORT must be between 1024 and 65535" >&2
  exit 64
fi

exec ssh -o ExitOnForwardFailure=yes -o ServerAliveInterval=5 -o ServerAliveCountMax=2 \
  -N -L "127.0.0.1:${port}:127.0.0.1:${port}" unitree
