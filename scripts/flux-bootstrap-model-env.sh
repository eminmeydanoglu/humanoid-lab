#!/usr/bin/env bash
# Reproducible GPU-side model environment for the Flux 3 / Dex3 server.
#
# Creates the persistent environment the model server prefers
# ($HUMANOID_DATA_ROOT/venvs/flux-model) from the pinned lock file, plus the
# pinned LeRobot checkout it installs editable.  `--plan` prints what would
# happen and `--check` verifies an existing environment against the pins;
# neither downloads anything.
#
# What this does NOT provide (both are external artifacts, see the Flux Dex3
# section of README.md):
#   * the trained Dex3 adapter checkpoint (~450 MiB) -- prepare it with
#     ./dev.sh flux-checkpoint --source <training checkpoint>;
#   * the base policy export the adapter references (~13 GiB) -- read-only,
#     referenced by path, never copied automatically.
set -euo pipefail

repo_root="$(cd "$(dirname "$0")/.." && pwd)"
data_root="${HUMANOID_DATA_ROOT:-$repo_root/data}"
lock_file="${FLUX_MODEL_LOCK:-$repo_root/scripts/flux-model-env.lock.txt}"
venv="${FLUX_MODEL_VENV:-$data_root/venvs/flux-model}"
lerobot_src="${FLUX_LEROBOT_SRC:-$data_root/venvs/flux-model-src/lerobot}"
uv_bin="${FLUX_UV:-uv}"

#: The validated pins.  The lock file carries every package version; the
#: LeRobot source is pinned here and installed editable, exactly as the
#: validated environment had it.
lerobot_repo="${FLUX_LEROBOT_REPO:-https://github.com/huggingface/lerobot.git}"
lerobot_commit="${FLUX_LEROBOT_COMMIT:-e624f3f7f8411ec3a02635d06e79373341e5ef35}"
python_version="${FLUX_MODEL_PYTHON_VERSION:-3.12}"
uv_version_expected="${FLUX_UV_VERSION:-0.12.15}"
torch_index="${FLUX_TORCH_INDEX:-https://download.pytorch.org/whl/cu128}"
natten_index="${FLUX_NATTEN_INDEX:-https://whl.natten.org/}"

action=install
force=0
usage() {
  cat >&2 <<EOF
usage: $0 [--plan | --check | --force]

  --plan     print the resolved paths, pins and commands (no downloads, no writes)
  --check    verify the persistent environment against the pins (no writes)
  (default)  create the environment: uv venv + pinned lock install + pinned
             editable LeRobot checkout.  Requires network and disk (~7 GiB);
             refuses to touch an existing environment without --force.
  --force    with install: move an existing environment aside instead of deleting it

Pins:
  LeRobot      $lerobot_repo @ $lerobot_commit (editable)
  Python       $python_version
  uv           $uv_version_expected (built the validated environment)
  lock file    $lock_file
  torch index  $torch_index
  NATTEN index $natten_index
EOF
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --plan) action=plan; shift ;;
    --check) action=check; shift ;;
    --force) force=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "error: unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done

lock_sha256() {
  [ -f "$lock_file" ] || return 1
  sha256sum "$lock_file" | cut -d' ' -f1
}

plan_json() {
  python3 - "$venv" "$lerobot_src" "$lock_file" "$lerobot_repo" "$lerobot_commit" \
           "$python_version" "$uv_version_expected" "$torch_index" "$natten_index" <<'PY'
import json
import os
import sys

(venv, lerobot_src, lock_file, lerobot_repo, lerobot_commit,
 python_version, uv_version, torch_index, natten_index) = sys.argv[1:10]
install = [
    "uv venv --python %s %s" % (python_version, venv),
    "git init %s && git fetch --depth=1 %s %s" % (lerobot_src, lerobot_repo, lerobot_commit),
    "uv pip install --python %s/bin/python --extra-index-url %s -f %s -r %s"
    % (venv, torch_index, natten_index, lock_file),
    "uv pip install --python %s/bin/python -e %s" % (venv, lerobot_src),
]
plan = {
    "venv": venv,
    "venv_exists": os.path.isdir(venv),
    "venv_python_exists": os.access(os.path.join(venv, "bin", "python"), os.X_OK),
    "lerobot_src": lerobot_src,
    "lerobot_src_exists": os.path.isdir(os.path.join(lerobot_src, ".git")),
    "lerobot_repo": lerobot_repo,
    "lerobot_commit": lerobot_commit,
    "python_version": python_version,
    "uv_version": uv_version,
    "lock_file": lock_file,
    "lock_exists": os.path.isfile(lock_file),
    "torch_index": torch_index,
    "natten_index": natten_index,
    "install_commands": install,
    "model_server_prefers": venv + "/bin/python",
    "external_artifacts": [
        "trained adapter checkpoint (~450 MiB): ./dev.sh flux-checkpoint --source <training checkpoint>",
        "base policy export the adapter references (~13 GiB): pass --base-model-dir to flux-checkpoint; never copied automatically",
    ],
}
print(json.dumps(plan, indent=2, sort_keys=True))
PY
}

check_env() {
  local problems=0
  if [ ! -x "$venv/bin/python" ]; then
    echo "flux-bootstrap-model-env: no environment at $venv" >&2
    problems=1
  else
    echo "flux-bootstrap-model-env: interpreter $("$venv/bin/python" -c 'import platform; print(platform.python_version())' 2>/dev/null || echo unreadable)"
    if ! "$venv/bin/python" - <<'PY'
import importlib

for name in ("torch", "torchvision", "lerobot", "peft", "av", "zmq", "numpy", "safetensors", "transformers", "diffusers"):
    importlib.import_module(name)
PY
    then
      echo "flux-bootstrap-model-env: the environment cannot import the model stack" >&2
      problems=1
    fi
    if [ -d "$lerobot_src/.git" ]; then
      local head
      head="$(git -C "$lerobot_src" rev-parse HEAD 2>/dev/null || echo unknown)"
      echo "flux-bootstrap-model-env: lerobot source HEAD=$head"
      if [ "$head" != "$lerobot_commit" ]; then
        echo "flux-bootstrap-model-env: LeRobot is not at the pinned commit ($lerobot_commit)" >&2
        problems=1
      fi
    else
      echo "flux-bootstrap-model-env: no pinned LeRobot checkout at $lerobot_src" >&2
      problems=1
    fi
    if [ -f "$venv/FLUX_MODEL_ENV.json" ] && [ -f "$lock_file" ]; then
      local recorded
      recorded="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["lock_sha256"])' "$venv/FLUX_MODEL_ENV.json" 2>/dev/null || echo unknown)"
      if [ "$recorded" != "$(lock_sha256)" ]; then
        echo "flux-bootstrap-model-env: the lock file changed since this environment was built" >&2
        problems=1
      fi
    fi
  fi
  if command -v "$uv_bin" >/dev/null 2>&1; then
    echo "flux-bootstrap-model-env: uv $("$uv_bin" --version | awk '{print $2}') (expected $uv_version_expected)"
  else
    echo "flux-bootstrap-model-env: uv not found; install $uv_version_expected to (re)build the environment" >&2
    [ "$action" = check ] && problems=1
  fi
  return "$problems"
}

install_env() {
  command -v "$uv_bin" >/dev/null 2>&1 || {
    echo "error: uv is required (validated with $uv_version_expected): https://docs.astral.sh/uv/" >&2
    exit 2
  }
  [ -f "$lock_file" ] || { echo "error: lock file missing: $lock_file" >&2; exit 2; }

  if [ -e "$venv" ] || [ -d "$lerobot_src" ]; then
    [ "$force" -eq 1 ] || {
      echo "error: environment already exists ($venv, $lerobot_src)" >&2
      echo "       verify it with: $0 --check   (or rebuild with --force, which moves it aside)" >&2
      exit 2
    }
    local stamp
    stamp="$(date +%Y%m%d-%H%M%S)"
    [ -e "$venv" ] && mv "$venv" "$venv.bak-$stamp"
    [ -d "$lerobot_src" ] && mv "$lerobot_src" "$lerobot_src.bak-$stamp"
    echo "flux-bootstrap-model-env: moved the previous environment to *.bak-$stamp" >&2
  fi

  mkdir -p "$(dirname "$venv")" "$(dirname "$lerobot_src")"
  echo "flux-bootstrap-model-env: uv venv ($python_version) -> $venv" >&2
  "$uv_bin" venv --python "$python_version" "$venv"

  echo "flux-bootstrap-model-env: LeRobot @ $lerobot_commit -> $lerobot_src" >&2
  if [ ! -d "$lerobot_src/.git" ]; then
    git init "$lerobot_src" >/dev/null
    git -C "$lerobot_src" remote add origin "$lerobot_repo"
    git -c http.version=HTTP/1.1 -C "$lerobot_src" fetch --depth=1 origin "$lerobot_commit"
    git -C "$lerobot_src" checkout --detach FETCH_HEAD
  fi
  test "$(git -C "$lerobot_src" rev-parse HEAD)" = "$lerobot_commit" || {
    echo "error: could not check out the pinned LeRobot commit" >&2; exit 1; }
  test -z "$(git -C "$lerobot_src" status --porcelain)" || {
    echo "error: the LeRobot checkout is not clean" >&2; exit 1; }

  echo "flux-bootstrap-model-env: installing the pinned lock (torch $torch_index, NATTEN $natten_index)" >&2
  "$uv_bin" pip install --python "$venv/bin/python" \
    --extra-index-url "$torch_index" -f "$natten_index" -r "$lock_file"
  "$uv_bin" pip install --python "$venv/bin/python" -e "$lerobot_src"

  python3 - "$venv" "$lerobot_repo" "$lerobot_commit" "$lock_file" "$uv_bin" <<'PY'
import json
import subprocess
import sys
import time

venv, repo, commit, lock_file, uv_bin = sys.argv[1:6]
torch_version = subprocess.run(
    [venv + "/bin/python", "-c", "import torch; print(torch.__version__)"],
    capture_output=True, text=True, check=True).stdout.strip()
uv_version = subprocess.run([uv_bin, "--version"], capture_output=True, text=True, check=True).stdout.split()[1]
record = {
    "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    "lerobot_repo": repo,
    "lerobot_commit": commit,
    "lock_file": lock_file,
    "lock_sha256": subprocess.run(["sha256sum", lock_file], capture_output=True, text=True,
                                  check=True).stdout.split()[0],
    "uv_version": uv_version,
    "torch_version": torch_version,
}
with open(venv + "/FLUX_MODEL_ENV.json", "w", encoding="utf-8") as stream:
    json.dump(record, stream, indent=2, sort_keys=True)
    stream.write("\n")
print(json.dumps(record, sort_keys=True))
PY
  echo "flux-bootstrap-model-env: done.  The model server picks this environment up automatically;" >&2
  echo "  export FLUX_MODEL_PYTHON=$venv/bin/python   # only needed to override it explicitly" >&2
}

case "$action" in
  plan) plan_json ;;
  check) check_env ;;
  install) install_env ;;
esac
