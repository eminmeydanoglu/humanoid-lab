#!/usr/bin/env bash
set -euo pipefail

if [[ ! -f /isaac-sim/setup_python_env.sh ]]; then
  echo "error: Isaac Sim Python setup file is missing" >&2
  return 1
fi

source /isaac-sim/setup_python_env.sh
