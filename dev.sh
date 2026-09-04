#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

[ -f .env ] || { echo "error: .env missing — run ./setup.sh first" >&2; exit 2; }
set -a; source .env; set +a

DC() { docker compose --env-file .env "$@"; }

up_once() {
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
    DC exec dev bash -lc "source /opt/humanoid-lab/entrypoint.sh && use-isaac-sonic && scripts/fetch-models.sh"
    ;;
  hf-login)
    up_once
    DC exec dev bash -lc '
      source /opt/humanoid-lab/entrypoint.sh && use-groot
      if hf auth login --help >/dev/null 2>&1; then exec hf auth login; fi
      if hf login --help >/dev/null 2>&1; then exec hf login; fi
      exec huggingface-cli login'
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
