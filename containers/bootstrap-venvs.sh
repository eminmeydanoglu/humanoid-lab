#!/usr/bin/env bash
# Provision lock-pinned environments in the persistent /opt/venvs bind mount.
set -euo pipefail

VENV_ROOT=/opt/venvs
STATE_ROOT="$VENV_ROOT/.humanoid-lab-state"
LOCKS_ROOT=/workspace/humanoid-lab/locks
[[ -f "$LOCKS_ROOT/isaac-sonic/uv.lock" ]] || LOCKS_ROOT=/opt/locks
# A runtime container is an unprivileged user; managed Python downloads must
# live in the host-mounted uv cache rather than the image's /opt prefix.
export UV_PYTHON_INSTALL_DIR=/cache/uv/python
mkdir -p "$STATE_ROOT"

fingerprint() {
  local lockfile="$1"
  shift
  {
    sha256sum "$lockfile"
    local repository
    for repository in "$@"; do
      git -C "$repository" rev-parse HEAD
    done
  } | sha256sum | awk '{print $1}'
}

is_current() {
  local name="$1"
  local value="$2"
  [[ -x "$VENV_ROOT/$name/bin/python" ]] \
    && [[ -f "$STATE_ROOT/$name.sha256" ]] \
    && [[ "$(<"$STATE_ROOT/$name.sha256")" == "$value" ]]
}

record_current() {
  local name="$1"
  local value="$2"
  local temporary="$STATE_ROOT/$name.sha256.tmp"
  printf '%s\n' "$value" > "$temporary"
  mv -f "$temporary" "$STATE_ROOT/$name.sha256"
}

sync_isaac_sonic() {
  local name=isaac-sonic
  local lockfile="$LOCKS_ROOT/isaac-sonic/uv.lock"
  local current
  current="$(fingerprint "$lockfile" /opt/src/isaaclab /opt/src/sonic)"
  if is_current "$name" "$current"; then
    echo "[venv] $name: lock and sources unchanged"
    return
  fi

  echo "[venv] $name: provisioning lock-pinned environment"
  uv venv --allow-existing --python /isaac-sim/kit/python/bin/python3 "$VENV_ROOT/$name"
  UV_PROJECT_ENVIRONMENT="$VENV_ROOT/$name" uv sync --frozen --no-dev --project "$LOCKS_ROOT/isaac-sonic"
  # Archive distributions omit data files; the checked-out pinned sources are authoritative.
  uv pip uninstall --python "$VENV_ROOT/$name/bin/python" \
    isaaclab isaaclab-assets isaaclab-tasks isaaclab-rl gear-sonic || true
  uv pip install --python "$VENV_ROOT/$name/bin/python" --no-deps \
    -e /opt/src/isaaclab/source/isaaclab \
    -e /opt/src/isaaclab/source/isaaclab_assets \
    -e /opt/src/isaaclab/source/isaaclab_tasks \
    -e /opt/src/isaaclab/source/isaaclab_rl \
    -e '/opt/src/sonic/gear_sonic[training]'
  "$VENV_ROOT/$name/bin/python" -c 'import isaaclab, gear_sonic, rsl_rl, torch; assert torch.__version__.startswith("2.7.0")'
  record_current "$name" "$current"
}

sync_sonic_sim() {
  local name=sonic-sim
  local lockfile="$LOCKS_ROOT/sonic-sim/uv.lock"
  local current
  current="$(fingerprint "$lockfile" /opt/src/sonic)"
  if is_current "$name" "$current"; then
    echo "[venv] $name: lock and sources unchanged"
    return
  fi

  echo "[venv] $name: provisioning lock-pinned environment"
  uv venv --allow-existing --python 3.11 "$VENV_ROOT/$name"
  UV_PROJECT_ENVIRONMENT="$VENV_ROOT/$name" uv sync --frozen --no-dev --project "$LOCKS_ROOT/sonic-sim"
  # cyclonedds has no CPython 3.11 wheel and must build against the image prefix.
  uv pip install --python "$VENV_ROOT/$name/bin/python" cyclonedds==0.10.2
  cp -a /opt/src/sonic/external_dependencies/unitree_sdk2_python/unitree_sdk2py \
    "$VENV_ROOT/$name/lib/python3.11/site-packages/"
  # Unitree SDK ships legacy CycloneDDS XML. CycloneDDS 0.10 on Ubuntu 24.04
  # requires the current namespace/schema spelling; simulation stays loopback-only.
  python3 /workspace/humanoid-lab/containers/patch-unitree-cyclonedds-config.py "$VENV_ROOT/$name/lib/python3.11/site-packages/unitree_sdk2py/core/channel_config.py"
  uv pip uninstall --python "$VENV_ROOT/$name/bin/python" gear-sonic || true
  uv pip install --python "$VENV_ROOT/$name/bin/python" --no-deps -e '/opt/src/sonic/gear_sonic[sim]'
  "$VENV_ROOT/$name/bin/python" -c 'import mujoco, gear_sonic, unitree_sdk2py'
  record_current "$name" "$current"
}

remove_unusable_groot_deepspeed() {
  local python="$VENV_ROOT/groot-n17/bin/python"
  if "$python" -c 'import importlib.metadata; importlib.metadata.version("deepspeed")' >/dev/null 2>&1; then
    # Accelerate imports an installed DeepSpeed package even when the GR00T
    # single-GPU launcher does not select DeepSpeed. This Isaac Sim runtime has
    # no CUDA toolkit/nvcc, so DeepSpeed 0.17.6 fails during that import.
    uv pip uninstall --python "$python" deepspeed
  fi
}

sync_groot() {
  local name=groot-n17
  local lockfile=/opt/src/isaac-groot/uv.lock
  local current
  current="$(fingerprint "$lockfile" /opt/src/isaac-groot)"
  if is_current "$name" "$current"; then
    echo "[venv] $name: lock and sources unchanged"
    remove_unusable_groot_deepspeed
    return
  fi

  echo "[venv] $name: provisioning lock-pinned environment"
  uv venv --allow-existing --python 3.12 "$VENV_ROOT/$name"
  UV_PROJECT_ENVIRONMENT="$VENV_ROOT/$name" uv sync --frozen --no-dev --project /opt/src/isaac-groot
  remove_unusable_groot_deepspeed
  "$VENV_ROOT/$name/bin/python" -c 'import flash_attn, gr00t, torch; assert torch.__version__.startswith("2.9.0")'
  record_current "$name" "$current"
}

sync_isaac_sonic
sync_sonic_sim
sync_groot
