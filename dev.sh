#!/usr/bin/env bash
# dev.sh — manages the development container.
# No container is created per call: docker compose exec is used.
#
#   ./dev.sh               create/start main container + interactive shell
#   ./dev.sh isaac         isaac-sonic environment shell
#   ./dev.sh sonic-sim     MuJoCo environment shell
#   ./dev.sh groot         GR00T environment shell
#   ./dev.sh doctor        container + host validation (doctor.sh)
#   ./dev.sh smoke         hard smoke tests inside the running container
#   ./dev.sh fetch-models  download pinned models into the persistent dir
#   ./dev.sh hf-login      login to host-persistent HF cache (token never written to image/Git)
#   ./dev.sh stop          stop the container; keeps data
#   ./dev.sh rebuild       rebuild image with the same lock
#   ./dev.sh foxy          Foxy shell when the profile is enabled (not yet)
set -euo pipefail
cd "$(dirname "$0")"

[ -f .env ] || { echo "error: .env missing — run ./setup.sh first" >&2; exit 2; }
set -a; source .env; set +a

DC() { docker compose --env-file .env "$@"; }

up_once() {
  # Create/start if missing, else exec (compose exec does not auto-start).
  if ! docker compose --env-file .env ps -q dev >/dev/null 2>&1; then
    DC up -d dev
  else
    state=$(docker inspect -f '{{.State.Running}}' "${COMPOSE_PROJECT_NAME:-humanoid-lab}-dev" 2>/dev/null || echo false)
    [ "$state" = "true" ] || DC up -d dev
  fi
}

shell_env() { # $1 = use-... function name
  up_once
  DC exec dev bash -lc "source /opt/humanoid-lab/entrypoint.sh && $1 && exec bash"
}

case "${1:-}" in
  "")
    up_once
    DC exec dev bash -lc "source /opt/humanoid-lab/entrypoint.sh && exec bash"
    ;;
  isaac)      shell_env use-isaac-sonic ;;
  sonic-sim)  shell_env use-sonic-sim ;;
  groot)      shell_env use-groot ;;
  doctor)
    ./doctor.sh
    ;;
  smoke)
    up_once
    DC exec -T dev bash -lc 'source /opt/humanoid-lab/entrypoint.sh && /opt/humanoid-lab/smoke-test.sh'
    ;;
  fetch-models)
    up_once
    # The HF CLI lives in the venvs, not the base image — select one first.
    DC exec dev bash -lc "source /opt/humanoid-lab/entrypoint.sh && use-isaac-sonic && scripts/fetch-models.sh"
    ;;
  hf-login)
    up_once
    # Token is written to HF_HOME (/cache/huggingface — host bind mount),
    # shared by every env; 'hf' is the modern CLI (huggingface-cli is
    # deprecated upstream and removed in huggingface_hub >= 1.0).
    DC exec dev bash -lc "source /opt/humanoid-lab/entrypoint.sh && use-groot && hf login || use-groot && huggingface-cli login"
    ;;
  stop)
    DC stop
    ;;
  rebuild)
    DC build --progress=plain dev
    DC up -d dev
    ;;
  foxy)
    echo "error: foxy profile not enabled" >&2
    exit 2
    ;;
  *)
    echo "usage: $0 [isaac|sonic-sim|groot|doctor|smoke|fetch-models|hf-login|stop|rebuild|foxy]" >&2
    exit 2
    ;;
esac