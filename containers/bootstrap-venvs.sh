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

# Unitree DDS bindings (unitree_sdk2py + cyclonedds). The Isaac-side simulator
# bridge speaks the same Unitree topics as the official MuJoCo loop, so the
# Isaac environment needs these bindings too.
install_unitree_dds() {
  local venv="$VENV_ROOT/$1"
  # cyclonedds has no CPython wheel and must build against the image prefix.
  uv pip install --python "$venv/bin/python" cyclonedds==0.10.2
  cp -a /opt/src/sonic/external_dependencies/unitree_sdk2_python/unitree_sdk2py \
    "$venv/lib/python3.11/site-packages/"
  # Unitree SDK ships legacy CycloneDDS XML. CycloneDDS 0.10 on Ubuntu 24.04
  # requires the current namespace/schema spelling; simulation stays loopback-only.
  python3 /opt/humanoid-lab/patch-unitree-cyclonedds-config.py "$venv/lib/python3.11/site-packages/unitree_sdk2py/core/channel_config.py"
}

has_unitree_dds() {
  "$VENV_ROOT/$1/bin/python" -c 'import cyclonedds, unitree_sdk2py' >/dev/null 2>&1
}

# The Dex3 G1 body is a git-lfs file inside the pinned SONIC checkout. Image
# builds cannot fetch lfs content on this host, so the asset is materialized
# here, into the persistent data root, and verified by content hash; the run
# profile references the copy rather than /opt/src.
materialize_sonic_asset() {
  local dest="$1"
  local want="$2"
  if [ -f "$dest" ] && [ "$(sha256sum "$dest" | cut -d' ' -f1)" = "$want" ]; then
    echo "[asset] sonic Dex3 G1 usd: present ($(basename "$dest"))"
    return 0
  fi
  local source=/opt/src/sonic/gear_sonic/data/robots/g1/g1_29dof_with_hand_rev_1_0.usd
  [ -f "$source" ] || { echo "[asset] source missing: $source" >&2; return 1; }
  git -C /opt/src/sonic lfs pull --include "gear_sonic/data/robots/g1/g1_29dof_with_hand_rev_1_0.usd" >/dev/null 2>&1 || true
  mkdir -p "$(dirname "$dest")"
  cp "$source" "$dest"
  local got
  got="$(sha256sum "$dest" | cut -d' ' -f1)"
  if [ "$got" != "$want" ]; then
    echo "[asset] hash mismatch for $dest: $got != $want" >&2
    return 1
  fi
  echo "[asset] sonic Dex3 G1 usd: materialized ($(basename "$dest"))"
}

sync_isaac_sonic() {
  local name=isaac-sonic
  local lockfile="$LOCKS_ROOT/isaac-sonic/uv.lock"
  local current
  current="$(fingerprint "$lockfile" /opt/src/isaaclab /opt/src/sonic)"
  if is_current "$name" "$current" && has_unitree_dds "$name"; then
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
  install_unitree_dds "$name"
  "$VENV_ROOT/$name/bin/python" -c 'import isaaclab, gear_sonic, rsl_rl, torch; assert torch.__version__.startswith("2.7.0")'
  record_current "$name" "$current"
}

sync_sonic_sim() {
  local name=sonic-sim
  local lockfile="$LOCKS_ROOT/sonic-sim/uv.lock"
  local current
  current="$(fingerprint "$lockfile" /opt/src/sonic)"
  if is_current "$name" "$current" && has_unitree_dds "$name"; then
    echo "[venv] $name: lock and sources unchanged"
    return
  fi

  echo "[venv] $name: provisioning lock-pinned environment"
  uv venv --allow-existing --python 3.11 "$VENV_ROOT/$name"
  UV_PROJECT_ENVIRONMENT="$VENV_ROOT/$name" uv sync --frozen --no-dev --project "$LOCKS_ROOT/sonic-sim"
  install_unitree_dds "$name"
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

remove_unusable_psi0_deepspeed() {
  local python="$VENV_ROOT/psi0/bin/python"
  if "$python" -c 'import importlib.metadata; importlib.metadata.version("deepspeed")' >/dev/null 2>&1; then
    # Same limitation as GR00T above: deepspeed's package init calls
    # installed_cuda_version() and this runtime has no CUDA toolkit, so
    # `import deepspeed` raises before accelerate can fall back to DDP.
    uv pip uninstall --python "$python" deepspeed
  fi
}

sync_psi0() {
  local name=psi0
  # Upstream Psi0 is a git submodule of this checkout, not an image layer; the
  # workspace mount makes it visible here without rebuilding the image.
  local source=/workspace/humanoid-lab/third_party/Psi0
  local lockfile="$LOCKS_ROOT/psi0/uv.lock"
  local current
  [ -f "$source/uv.lock" ] || { echo "[venv] $name: checkout missing: $source" >&2; return 1; }
  current="$(fingerprint "$lockfile" "$source")"
  if is_current "$name" "$current"; then
    echo "[venv] $name: lock and sources unchanged"
    remove_unusable_psi0_deepspeed
    return
  fi

  echo "[venv] $name: provisioning lock-pinned environment"
  uv venv --allow-existing --python 3.11 "$VENV_ROOT/$name"
  # The dependency graph comes from locks/psi0, not from the checkout's own
  # uv.lock: that file has a duplicate TOML key (upstream bug) and this repo
  # already keeps frozen locks for every environment. See locks/psi0/pyproject.toml
  # for the two deliberate deltas (cu128 torch, no deepspeed).
  UV_PROJECT_ENVIRONMENT="$VENV_ROOT/$name" uv sync --frozen --no-dev --project "$LOCKS_ROOT/psi0"
  # psi itself is installed from the pinned checkout, editable and without
  # dependency resolution, so the sources under third_party/Psi0 stay authoritative.
  uv pip install --python "$VENV_ROOT/$name/bin/python" --no-deps -e "$source"
  remove_unusable_psi0_deepspeed
  "$VENV_ROOT/$name/bin/python" -c 'import psi, torch; assert torch.__version__.startswith("2.7.0"); assert torch.cuda.is_available()'
  record_current "$name" "$current"
}

sync_isaac_sonic
sync_sonic_sim
sync_groot
sync_psi0

# The container already exports HUMANOID_DATA_ROOT with the in-container path
# (/data); the repository's .env holds the host path and must not be used here.
DATA_ROOT="${HUMANOID_DATA_ROOT:-/data}"
if ! materialize_sonic_asset \
  "$DATA_ROOT/models/sonic-assets/g1_29dof_with_hand_rev_1_0.usd" \
  "a7a2bab76981d19a1d76adecdfffec9b52afa34df9ba8e288ccedf410d3ce6bd"; then
  echo "[asset] WARN: the Dex3 G1 asset is not available; profiles using it will fail." >&2
  echo "[asset] WARN: retry with ./dev.sh sync (needs network access to github.com)." >&2
fi
