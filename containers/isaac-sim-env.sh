#!/usr/bin/env bash
set -euo pipefail

# Isaac Sim's setup scripts interpolate these variables without defaults.  This
# wrapper itself is sourced from a strict shell, so give them defined values
# before delegating to the upstream script.
export PYTHONPATH="${PYTHONPATH:-}"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"

if [[ ! -f /isaac-sim/setup_conda_env.sh ]]; then
  echo "error: Isaac Sim environment setup file is missing" >&2
  return 1
fi

# setup_python_env.sh only supplies Python paths.  AppLauncher also requires
# ISAAC_PATH, CARB_APP_PATH, and EXP_PATH, all supplied by setup_conda_env.sh.
# NVIDIA's script reads ZSH_VERSION and OLDPWD without defaults; temporarily
# disable nounset while it runs and retain the caller's working directory.
_hl_isaac_start_dir="$PWD"
set +u
# This image-owned setup script is supplied by the pinned Isaac Sim base image.
# shellcheck source=/dev/null
source /isaac-sim/setup_conda_env.sh
cd "$_hl_isaac_start_dir"
set -u
unset _hl_isaac_start_dir
