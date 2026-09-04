#!/usr/bin/env bash
set -u

export HL_ACTIVE_ENV="${HL_ACTIVE_ENV:-none}"
export HL_SOURCE_ROOT="/workspace/humanoid-lab"
export HL_VENVS_ROOT="/opt/venvs"

_hl_require_env() {
  local venv="$1"
  if [ ! -x "$venv/bin/python" ]; then
    echo "error: environment not installed: $venv (image build may be incomplete)" >&2
    return 1
  fi
}

use-isaac-sonic() {
  _hl_require_env "$HL_VENVS_ROOT/isaac-sonic" || return 1
  # Kit python shared-lib/extension path'leri sadece bu env'e
  source /opt/humanoid-lab/isaac-sim-env.sh
  export HL_ACTIVE_ENV="isaac-sonic"
  export PATH="$HL_VENVS_ROOT/isaac-sonic/bin:$PATH"
  export VIRTUAL_ENV="$HL_VENVS_ROOT/isaac-sonic"
  unset HL_NO_ENV
}

use-sonic-sim() {
  _hl_require_env "$HL_VENVS_ROOT/sonic-sim" || return 1
  export HL_ACTIVE_ENV="sonic-sim"
  export PATH="$HL_VENVS_ROOT/sonic-sim/bin:$PATH"
  export VIRTUAL_ENV="$HL_VENVS_ROOT/sonic-sim"
  unset ISAAC_PATH CARB_APP_PATH EXP_PATH PYTHONPATH
  unset HL_NO_ENV
}

use-groot() {
  _hl_require_env "$HL_VENVS_ROOT/groot-n17" || return 1
  export HL_ACTIVE_ENV="groot-n17"
  export PATH="$HL_VENVS_ROOT/groot-n17/bin:$PATH"
  export VIRTUAL_ENV="$HL_VENVS_ROOT/groot-n17"
  unset ISAAC_PATH CARB_APP_PATH EXP_PATH PYTHONPATH
  unset HL_NO_ENV
}

use-none() {
  export HL_ACTIVE_ENV="none"
  export HL_NO_ENV=1
}

show-env() {
  printf 'active env : %s\n' "${HL_ACTIVE_ENV:-none}"
  printf 'python     : %s\n' "$(command -v python || echo '(global alias off)')"
  if [ -n "${VIRTUAL_ENV:-}" ]; then
    "$VIRTUAL_ENV/bin/python" --version
  fi
}

# global python alias yok; aktif env prompt'ta gorunur
if [ -n "${PS1:-}" ] && [ -n "${BASH_VERSION:-}" ]; then
  PS1='\u@\h [\e[1;34m${HL_ACTIVE_ENV:-none}\e[0m] \w\$ '
fi

