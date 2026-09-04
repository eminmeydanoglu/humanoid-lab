#!/usr/bin/env bash
# Install the official Isaac Sim 5.1 WebRTC client on the host, not in Docker.
set -euo pipefail

client_version="1.1.5"
client_name="isaacsim-webrtc-streaming-client-${client_version}-linux-x64.AppImage"
client_dir="${ISAAC_WEBRTC_CLIENT_DIR:-$HOME/Applications/IsaacSim}"
client_path="$client_dir/$client_name"
client_url="https://downloads.isaacsim.nvidia.com/$client_name"

if ! dpkg-query -W -f='${Status}' libfuse2t64 2>/dev/null | grep -q 'install ok installed'; then
  echo "Installing libfuse2t64, required by the AppImage..."
  sudo apt-get update -qq
  sudo apt-get install -y --no-install-recommends libfuse2t64
fi

mkdir -p "$client_dir"
if [ ! -s "$client_path" ]; then
  tmp_path="$client_path.partial"
  rm -f -- "$tmp_path"
  echo "Downloading official Isaac Sim WebRTC client ${client_version}..."
  curl --fail --location --retry 3 --output "$tmp_path" "$client_url"
  chmod 755 "$tmp_path"
  mv -- "$tmp_path" "$client_path"
fi

chmod 755 "$client_path"
echo "Installed: $client_path"
sha256sum "$client_path"
