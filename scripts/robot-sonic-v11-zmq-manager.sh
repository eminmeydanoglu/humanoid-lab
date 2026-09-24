#!/usr/bin/env bash
set -euo pipefail

# Run on Unitree only after the patched binary has passed supervised review.
# Default prints the exact command and never starts the controller.
usage() {
  echo "usage: $0 --print|--execute PATCHED_BINARY RAIDER_IP DDS_INTERFACE" >&2
  exit 2
}
[[ $# -eq 4 ]] || usage
mode="$1"; binary="$2"; raider_ip="$3"; dds_interface="$4"
[[ "$mode" == --print || "$mode" == --execute ]] || usage
[[ -x "$binary" ]] || { echo "patched SONIC binary is not executable: $binary" >&2; exit 2; }
[[ "$raider_ip" =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]] || { echo "RAIDER_IP must be IPv4" >&2; exit 2; }

source_dir="/home/unitree/sonic_jp5/GR00T-WholeBodyControl/gear_sonic_deploy"
cd "$source_dir"
command=(
  "$binary" "$dds_interface"
  policy/sonic_v1_1/model_decoder.onnx reference/example/
  --encoder-file policy/sonic_v1_1/model_encoder.onnx
  --obs-config policy/sonic_v1_1/observation_config.yaml
  --planner-file planner/target_vel/V2/planner_sonic_trt85.onnx
  --planner-precision 32 --policy-precision 32
  --input-type zmq_manager --zmq-host "$raider_ip" --zmq-port 5556
  --zmq-topic pose --zmq-out-port 5557 --output-type zmq
  --logs-dir /home/unitree/sonic_jp5/physical_logs/vla_robot
)
printf 'source directory: %s\npatched binary sha256: ' "$source_dir"
sha256sum "$binary" | cut -d' ' -f1
printf 'command:'
printf ' %q' "${command[@]}"
printf '\n'
if [[ "$mode" == --execute ]]; then
  exec "${command[@]}"
fi
