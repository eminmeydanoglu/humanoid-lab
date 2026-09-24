#!/usr/bin/env bash
set -euo pipefail

# Check or apply the robot-side SONIC watchdog to a *copy* or an explicitly
# selected source tree. Never modifies the installed robot source by default.
usage() {
  echo "usage: $0 --check|--apply SONIC_SOURCE_ROOT" >&2
  exit 2
}
[[ $# -eq 2 ]] || usage
mode="$1"
source_root="$2"
[[ "$mode" == --check || "$mode" == --apply ]] || usage
[[ -d "$source_root/gear_sonic_deploy" ]] || { echo "SONIC source root not found: $source_root" >&2; exit 2; }

repo_root="$(cd "$(dirname "$0")/.." && pwd)"
patch_file="$repo_root/patches/sonic/0001-return-to-planner-idle-on-stale-v4-token.patch"
status_patch="$repo_root/patches/sonic/0002-publish-zmq-manager-status.patch"
helper_file="$repo_root/patches/sonic/pose_watchdog.hpp"
header_dir="$source_root/gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include/input_interface"
relative_dir="gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include/input_interface"

[[ ! -e "$header_dir/pose_watchdog.hpp" ]] || { echo "pose_watchdog.hpp already exists" >&2; exit 1; }
scratch="$(mktemp -d)"
trap 'rm -rf "$scratch"' EXIT
mkdir -p "$scratch/$relative_dir"
cp "$header_dir/zmq_manager.hpp" "$header_dir/zmq_endpoint_interface.hpp" "$scratch/$relative_dir/"
git -C "$scratch" apply "$patch_file"
git -C "$scratch" apply "$status_patch"

if [[ "$mode" == --check ]]; then
  echo "PASS: both patches apply to a source copy and helper is absent"
else
  git -C "$source_root" apply "$patch_file"
  git -C "$source_root" apply "$status_patch"
  install -m 0644 "$helper_file" "$header_dir/pose_watchdog.hpp"
  echo "Applied SONIC idle watchdog and status publisher to $source_root; rebuild required"
fi
