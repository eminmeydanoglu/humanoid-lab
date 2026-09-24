#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "$0")/.." && pwd)"
main_checkout="${VLA_MAIN_CHECKOUT:-$(cd "$repo_root/../humanoid-lab-main" && pwd)}"
env_file="${VLA_ENV_FILE:-$main_checkout/.env}"
[[ -f "$env_file" ]] || { echo "missing compose environment: $env_file" >&2; exit 2; }
export HUMANOID_SOURCE_ROOT="$repo_root"
export HUMANOID_DATA_ROOT="${VLA_DATA_ROOT:-$main_checkout/data}"
export COMPOSE_PROJECT_NAME=vla-robot

# Reuse the cached image and persistent model/data mounts in a separate
# container. The existing main checkout/container stays untouched.
exec docker compose --env-file "$env_file" -f "$repo_root/compose.yaml" \
  run --rm --no-deps -T --entrypoint /bin/bash dev \
  -lc 'source /opt/humanoid-lab/entrypoint.sh && use-psi0 && cd /workspace/humanoid-lab && PYTHONPATH=src exec python scripts/run-robot-vla.py "$@"' \
  bash "$@"
