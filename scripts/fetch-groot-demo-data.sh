#!/usr/bin/env bash
# Fetch the small official GR00T fine-tuning dataset into the persistent data mount.
set -euo pipefail

readonly SOURCE_REPO="${GROOT_DATASET_REPO:-https://github.com/NVIDIA/Isaac-GR00T.git}"
readonly SOURCE_COMMIT="${GROOT_DATASET_REVISION:-1a1837f20538b7d7e21f977a11a5aee14f99803c}"
readonly DATA_ROOT="${HUMANOID_DATA_ROOT:-/data}/datasets/groot"
readonly DATASET_DIR="${GROOT_DATASET_DIR:-$DATA_ROOT/cube_to_bowl_5}"
FORCE=0
STAGING_DIR=""

usage() {
  cat <<'EOF'
usage: fetch-groot-demo-data.sh [--force]

Fetch NVIDIA's pinned Isaac-GR00T demo_data/cube_to_bowl_5 Git LFS content
into the persistent dataset mount. The target is:
  /data/datasets/groot/cube_to_bowl_5

The command refuses to overwrite an existing dataset unless --force is given.
EOF
}

fail() {
  echo "FAIL: $*" >&2
  exit 2
}

contains_lfs_pointer() {
  local path="$1"
  head -c 42 "$path" 2>/dev/null | cmp -s - <(printf '%s' 'version https://git-lfs.github.com/spec/v1')
}

require_non_pointer_assets() {
  local root="$1"
  local pattern="$2"
  local description="$3"
  local first
  local asset

  first="$(find "$root" -type f -name "$pattern" -print -quit)"
  [[ -n "$first" ]] || fail "$description is missing below $root"
  while IFS= read -r asset; do
    contains_lfs_pointer "$asset" && fail "$description is an unresolved Git LFS pointer: $asset"
  done < <(find "$root" -type f -name "$pattern" -print)
  return 0
}

validate_dataset() {
  local root="$1"
  [[ -f "$root/meta/modality.json" ]] || fail "GR00T dataset modality metadata is missing: $root/meta/modality.json"
  require_non_pointer_assets "$root" '*.parquet' 'GR00T demo parquet data'
  require_non_pointer_assets "$root" '*.mp4' 'GR00T demo video data'
}

while (($#)); do
  case "$1" in
    --force) FORCE=1 ;;
    -h|--help) usage; exit 0 ;;
    *) fail "unknown argument: $1" ;;
  esac
  shift
done

command -v git >/dev/null 2>&1 || fail "git is required"
command -v git-lfs >/dev/null 2>&1 || fail "git-lfs is required"
[[ "$SOURCE_COMMIT" =~ ^[0-9a-f]{40}$ ]] || fail "invalid pinned GR00T source revision: $SOURCE_COMMIT"

mkdir -p "$DATA_ROOT"
if [[ -e "$DATASET_DIR" ]]; then
  [[ "$FORCE" == 1 ]] || fail "dataset already exists: $DATASET_DIR (pass --force to replace it)"
fi

STAGING_DIR="$(mktemp -d "$DATA_ROOT/.cube_to_bowl_5.fetch.XXXXXX")"
cleanup() {
  [[ -z "$STAGING_DIR" ]] || rm -rf "$STAGING_DIR"
}
trap cleanup EXIT

checkout="$STAGING_DIR/source"
git init -q "$checkout"
git -C "$checkout" remote add origin "$SOURCE_REPO"
git -C "$checkout" sparse-checkout init --cone
git -C "$checkout" sparse-checkout set demo_data/cube_to_bowl_5
git -C "$checkout" fetch --depth=1 origin "$SOURCE_COMMIT"
git -C "$checkout" checkout --detach -q FETCH_HEAD
[[ "$(git -C "$checkout" rev-parse HEAD)" == "$SOURCE_COMMIT" ]] || fail "fetched GR00T source revision does not match $SOURCE_COMMIT"
git -C "$checkout" lfs pull --include 'demo_data/cube_to_bowl_5/**'

staged_dataset="$checkout/demo_data/cube_to_bowl_5"
validate_dataset "$staged_dataset"
"${PYTHON:-python3}" - "$staged_dataset" "$SOURCE_REPO" "$SOURCE_COMMIT" <<'PY'
import json
import os
import sys
import time

root, repo, revision = sys.argv[1:]
files = []
for directory, _children, names in os.walk(root):
    for name in sorted(names):
        if name == "DATASET_PROVENANCE.json":
            continue
        path = os.path.join(directory, name)
        files.append({"path": os.path.relpath(path, root), "bytes": os.path.getsize(path)})
with open(os.path.join(root, "DATASET_PROVENANCE.json"), "w", encoding="utf-8") as handle:
    json.dump({
        "repo": repo,
        "revision": revision,
        "fetched_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "files": files,
    }, handle, indent=2)
    handle.write("\n")
PY

if [[ -e "$DATASET_DIR" ]]; then
  rm -rf "$DATASET_DIR"
fi
mv "$staged_dataset" "$DATASET_DIR"
rm -rf "$STAGING_DIR"
STAGING_DIR=""
printf 'PASS: fetched pinned GR00T demo dataset: %s\n' "$DATASET_DIR"
