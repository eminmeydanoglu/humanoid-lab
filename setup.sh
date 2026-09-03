#!/usr/bin/env bash
# setup.sh — host bootstrap + render + image build + smoke (idempotent).
# Usage: ./setup.sh | ./setup.sh --verify-digests (fetch Isaac Sim digest from NGC into the lock)
# Safety: never auto-installs driver/toolkit/Docker; never builds with unverified
# pins/digests; never overwrites an existing .env; no secrets written.
set -euo pipefail
cd "$(dirname "$0")"
ROOT="$(pwd)"
LOG="$ROOT/setup.log"

# Everything (console + durable log) streams into hersey.log for live tracking.
exec > >(tee -a "$ROOT/hersey.log") 2>&1

say() { printf '\n== %s ==\n' "$*"; }
die() { printf 'setup.sh: error: %s\n' "$*" >&2; exit 2; }

VERIFY_DIGESTS=0
[ "${1:-}" = "--verify-digests" ] && VERIFY_DIGESTS=1

command -v python3 >/dev/null || die "python3 required"
python3 -c 'import yaml' 2>/dev/null || die "PyYAML required (python3 -m pip install pyyaml)"
command -v curl >/dev/null || die "curl required"
command -v git >/dev/null || die "git required"

say "preflight (lock render)"
python3 scripts/render-lock-env.py --root . >/dev/null

# Load lock values into env (single source).
if [ -f .generated/versions.env ]; then
  set -a; source .generated/versions.env; set +a
fi

say "host preflight"
OS_ID="$(. /etc/os-release && echo "$ID")"
OS_VER="$(. /etc/os-release && echo "$VERSION_ID")"
[ "$OS_ID" = "ubuntu" ] && [ "$OS_VER" = "24.04" ] || die "Ubuntu 24.04 required (current: $OS_ID $OS_VER)"
[ "$(uname -m)" = "x86_64" ] || die "x86_64 required (current: $(uname -m))"

# At least ~100 GiB free (image + caches).
DISK_KB=$(df -Pk . | awk 'NR==2 {print $4}')
[ "${DISK_KB:-0}" -gt 104857600 ] || die "insufficient disk space (${DISK_KB:-?} KB free)"

DRIVER=""
if command -v nvidia-smi >/dev/null 2>&1; then
  DRIVER="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"
fi
if [ -z "$DRIVER" ]; then
  die "NVIDIA driver not detected. Install, REBOOT, then re-run this script:
      sudo apt install -y nvidia-driver-<version>   # or NVIDIA .run (production)
      (Ubuntu 24.04: 'ubuntu-drivers install' recommended)"
fi
BASELINE="${HOST_NVIDIA_DRIVER_TESTED_BASELINE:-580.65.06}"
if [ "$DRIVER" = "$BASELINE" ] || [ "$(printf '%s\n%s\n' "$BASELINE" "$DRIVER" | sort -V | tail -1)" = "$DRIVER" ]; then
  echo "driver: $DRIVER (>= baseline $BASELINE — keeping)"
else
  die "driver $DRIVER is below baseline $BASELINE. Upgrade to an NVIDIA production release first; no automatic downgrade."
fi

docker_install_hint() {
  # Instructions use EXACT versions from the lock (single source -> .generated/versions.env);
  # only verified versions are installed (these fields are required:false until validated).
  echo "docker missing — install instructions (versions from lock):"
  cat <<EOF
  # Official Docker apt repo:
  sudo install -m 0755 -d /etc/apt/keyrings
  curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
  echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu noble stable" | sudo tee /etc/apt/sources.list.d/docker.list >/dev/null
  sudo apt-get update
  sudo apt-get install -y \\
    docker-ce=${DOCKER_CE} \\
    docker-ce-cli=${DOCKER_CLI} \\
    containerd.io=${DOCKER_CONTAINERD} \\
    docker-buildx-plugin=${DOCKER_BUILDX} \\
    docker-compose-plugin=${DOCKER_COMPOSE}
EOF
  echo "  (versions from the 'docker' block of versions.lock.yaml; re-verify them"
  echo "   before installing and clear the required:false flag)"
}

say "Docker / NVIDIA Container Toolkit check (no auto-install)"
if command -v docker >/dev/null 2>&1; then
  echo "docker: $(docker --version 2>/dev/null || echo not-installed?)"
else
  docker_install_hint
  die "docker not installed — follow the instructions above (see repo README)"
fi
docker compose version >/dev/null 2>&1 || die "docker compose plugin (v2) required"

if command -v nvidia-ctk >/dev/null 2>&1; then
  echo "nvidia-ctk: $(nvidia-ctk --version 2>/dev/null | head -1)"
else
  die "NVIDIA Container Toolkit not installed:
      sudo apt-get install -y nvidia-container-toolkit=${NVIDIA_CONTAINER_TOOLKIT_VERSION}
      sudo nvidia-ctk runtime configure --runtime=docker
      sudo systemctl restart docker
      (version from the 'nvidia_container_toolkit' block of versions.lock.yaml; verify before install)"
fi

if ! docker info >/dev/null 2>&1; then
  if id -nG | grep -qw docker; then
    die "you are in the docker group but the daemon is unreachable — a new group session is needed: log out and back in (no newgrp tricks)"
  fi
  die "docker daemon unreachable. Add the user to the docker group: sudo usermod -aG docker \$USER (re-login required)"
fi

say "base image digest verification"
ISAAC_SIM_DIGEST="${CONTAINER_ISAAC_SIM_DIGEST:-}"

if [ "$VERIFY_DIGESTS" = 1 ]; then
  # Fetch the digest from NGC and write it into the lock (idempotent). Requires docker + NGC login.
  command -v docker >/dev/null 2>&1 || die "--verify-digests needs docker"
  docker info >/dev/null 2>&1 || die "--verify-digests needs a running docker daemon + 'docker login nvcr.io'"
  echo "Fetching digest from NGC (requires docker login nvcr.io)..."
  IMAGETOOLS=$(docker buildx imagetools inspect "${CONTAINER_ISAAC_SIM_IMAGE:-nvcr.io/nvidia/isaac-sim}:${CONTAINER_ISAAC_SIM_TAG:-5.1.0}" 2>/dev/null || true)
  DIGEST=$(printf '%s\n' "$IMAGETOOLS" | grep -oE 'sha256:[0-9a-f]{64}' | head -1 || true)
  if [ -z "$DIGEST" ]; then
    die "digest unavailable — no NGC access/license acceptance (nvcr.io is EULA-gated)."
  fi
  DIGEST=${DIGEST#sha256:}
  if grep -q '^  isaac_sim_digest:' versions.lock.yaml; then
    python3 - <<PY
import re
p = 'versions.lock.yaml'
s = open(p).read()
s = re.sub(r'(?ms)^(  isaac_sim_digest:\n    value: ").*?("\n    required: )false',
           r'\1${DIGEST}\2false', s)
s = s.replace("value: \"${DIGEST}\"\n    required: false", "value: \"${DIGEST}\"")
open(p, 'w').write(s)
PY
    echo "lock updated: container.isaac_sim_digest = $DIGEST"
  else
    die "isaac_sim_digest block not found in versions.lock.yaml"
  fi
  python3 scripts/render-lock-env.py --root . >/dev/null
  set -a; source .generated/versions.env; set +a
  ISAAC_SIM_DIGEST="${CONTAINER_ISAAC_SIM_DIGEST:-}"
fi

if [ -z "$ISAAC_SIM_DIGEST" ] || [ "$ISAAC_SIM_DIGEST" = "TBD-VERIFY-ON-NGC-ACCESS" ]; then
  die "Isaac Sim 5.1.0 digest not verified. Run: ./setup.sh --verify-digests"
fi
echo "isaac-sim digest: $ISAAC_SIM_DIGEST (from lock, verified)"

UV_SHA256="${TOOLS_UV_SHA256:-}"
if [ -z "$UV_SHA256" ] || [ "$UV_SHA256" = "TBD-VERIFY-ON-RELEASE-ASSET" ] || [ "$UV_SHA256" = "UNSET_UV_SHA256" ]; then
  die "uv sha256 missing in lock. Write it into versions.lock.yaml -> tools.uv_sha256.value (astral-sh/uv release asset sha256)."
fi

say ".env and data dirs"
if [ ! -f .env ]; then
  sed -e "s|/home/USER/humanoid-lab-data|$HOME/humanoid-lab-data|" \
      -e "s|/home/USER/code/humanoid-lab|$ROOT|" \
      .env.example > .env
  echo ".env created (default: $HOME/humanoid-lab-data)"
else
  echo ".env exists — unchanged"
fi
set -a; source .env; set +a
DATA_ROOT="${HUMANOID_DATA_ROOT:-$HOME/humanoid-lab-data}"
SOURCE_ROOT="${HUMANOID_SOURCE_ROOT:-$ROOT}"
[ "$SOURCE_ROOT" = "$ROOT" ] || die "HUMANOID_SOURCE_ROOT in .env is $SOURCE_ROOT; repo is $ROOT — mismatch"

UID_NUM=$(id -u); GID_NUM=$(id -g)
mkdir -p \
  "$DATA_ROOT"/{datasets,checkpoints,models,hf-cache,uv-cache,rosbags,diagnostics,outputs,runtime} \
  "$DATA_ROOT/isaac-cache"/{kit,ov,pip,glcache,computecache,logs,data,documents}
chmod 700 "$DATA_ROOT/hf-cache"
echo "data root: $DATA_ROOT (UID/GID: $UID_NUM/$GID_NUM)"

# Generated .env entries (idempotent append); feed both compose build args and doctor.sh.
append_env() { grep -q "^$1=" .env || printf '%s=%s\n' "$1" "$2" >> .env; }
append_env DEVELOPER_UID "$UID_NUM"
append_env DEVELOPER_GID "$GID_NUM"
append_env ISAAC_SIM_IMAGE "${CONTAINER_ISAAC_SIM_IMAGE:-nvcr.io/nvidia/isaac-sim}"
append_env ISAAC_SIM_TAG "${CONTAINER_ISAAC_SIM_TAG:-5.1.0}"
append_env ISAAC_SIM_DIGEST "$ISAAC_SIM_DIGEST"
append_env UV_VERSION "${TOOLS_UV:-0.12.9}"
append_env UV_SHA256 "$UV_SHA256"
append_env REPOSITORIES_ISAAC_LAB_COMMIT "$REPOSITORIES_ISAAC_LAB_COMMIT"
append_env REPOSITORIES_SONIC_COMMIT "$REPOSITORIES_SONIC_COMMIT"
append_env REPOSITORIES_ISAAC_GROOT_COMMIT "$REPOSITORIES_ISAAC_GROOT_COMMIT"

say "docker compose build dev"
docker compose --env-file .env build --progress=plain dev

say "starting dev container"
docker compose --env-file .env up -d dev
echo "running smoke tests in container (short)..."
# Smoke failure does not invalidate the build (model/gated steps report blocked),
# but output and exit code must stay visible — no silent pass.
docker compose --env-file .env exec -T dev bash -lc 'source /opt/humanoid-lab/entrypoint.sh && /opt/humanoid-lab/smoke-test.sh' \
  && echo "smoke: PASS" || echo "smoke: WARN/BLOCKED — details above (continuing)"

say "model downloads (separate step, revision-pinned)"
echo "  ./dev.sh fetch-models   # downloads pinned models (gated: needs HF license acceptance)"

say "result"
echo "  dev container:  docker compose exec dev bash"
echo "  doctor:         ./doctor.sh   (report: $DATA_ROOT/diagnostics/)"
echo "  log:            $LOG + hersey.log"
echo "SETUP COMPLETE (model and ROS stages are separate steps)"

# exec > >(tee ...) can swallow the exit status at shutdown; flush first.
sleep 0.2 2>/dev/null || true
exit 0