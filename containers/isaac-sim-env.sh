#!/usr/bin/env bash
# Source this only for the Isaac Lab/SONIC interpreter.  Isaac Sim's Kit
# libraries must not leak into the separate GR00T or standalone MuJoCo envs.
set -euo pipefail

if [[ ! -f /isaac-sim/setup_python_env.sh ]]; then
  echo "error: Isaac Sim Python setup file is missing" >&2
  return 1
fi

# shellcheck source=/dev/null
source /isaac-sim/setup_python_env.sh
