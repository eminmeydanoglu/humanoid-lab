#!/usr/bin/env bash
# Build the repository's ROS packages into the flux-ros container's overlay.
#
# The sources are the bind-mounted checkout -- the very files the robot build
# uses -- not an image copy, so a change is picked up by rerunning this script.
# The pinned unitree_hg overlay baked into the image stays underneath.
set -euo pipefail

repo="${HUMANOID_SOURCE_ROOT:-/workspace/humanoid-lab}"
src="${FLUX_ROS_SOURCES:-$repo/third_party/flux/flux-inference/ros2}"
ws="${HUMANOID_FLUX_WS:-/data/flux-ros/ws}"

# shellcheck source=/dev/null
source /opt/humanoid-lab/flux-ros/entrypoint.sh

packages=(flux_dex3_interfaces flux_dex3 flux_sim_camera flux_sim_viz)
paths=()
for package in "${packages[@]}"; do
  [ -f "$src/$package/package.xml" ] || { echo "error: missing package: $src/$package" >&2; exit 2; }
  paths+=("$src/$package")
done

mkdir -p "$ws"
cd "$ws"
colcon build --paths "${paths[@]}" --symlink-install --event-handlers console_cohesion+

# Re-source through the entrypoint: colcon's own setup scripts reference
# optional variables, which this script's nounset would abort on.
# shellcheck source=/dev/null
source /opt/humanoid-lab/flux-ros/entrypoint.sh
ros2 pkg prefix flux_dex3 >/dev/null
for package in "${packages[@]}"; do
  ros2 pkg executables "$package" >/dev/null
done
echo "flux-ros build OK: ws=$ws"
