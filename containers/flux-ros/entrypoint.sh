#!/usr/bin/env bash
# Simulation-only Flux ROS 2 environment.
#
# Sources the pinned Jazzy install, the image's Unitree HG message overlay and,
# when it exists, the workspace built from the bind-mounted checkout.  The
# repository copy is authoritative: the overlay built from the mounted source
# is sourced last so a stale image layer can never win over it.
#
# This file is sourced by the container command and by every flux-ros helper,
# so it leaves the caller's shell options exactly as it found them.  The ament
# setup scripts reference optional variables, which nounset callers would abort.

_flux_nounset=0
if [ -o nounset ]; then
  _flux_nounset=1
  set +u
fi

source /opt/ros/jazzy/setup.bash
if [ -f /opt/humanoid-lab/flux-ros/unitree-ws/install/setup.bash ]; then
  # shellcheck source=/dev/null
  source /opt/humanoid-lab/flux-ros/unitree-ws/install/setup.bash
fi

export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_cyclonedds_cpp}"
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-42}"
export CYCLONEDDS_URI="${CYCLONEDDS_URI:-file:///opt/humanoid-lab/flux-ros/cyclonedds-sim.xml}"
export PYTHONDONTWRITEBYTECODE=1

flux_ws="${HUMANOID_FLUX_WS:-/data/flux-ros/ws}"
if [ -f "$flux_ws/install/setup.bash" ]; then
  # shellcheck source=/dev/null
  source "$flux_ws/install/setup.bash"
fi

if [ "$_flux_nounset" = 1 ]; then
  set -u
fi
unset _flux_nounset

flux_ros_show() {
  printf 'flux-ros : ROS_DOMAIN_ID=%s RMW=%s\n' "$ROS_DOMAIN_ID" "$RMW_IMPLEMENTATION"
  printf 'flux-ros : workspace=%s\n' "$flux_ws"
  printf 'flux-ros : %s\n' "$(head -n2 /opt/humanoid-lab/flux-ros/deps.txt 2>/dev/null | tr '\n' ' ')"
}
