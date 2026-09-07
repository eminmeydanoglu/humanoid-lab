#!/usr/bin/env bash
# Validate the materialized GR00T cube_to_bowl_5 fixture without modifying it.
set -euo pipefail

usage() {
  printf 'usage: %s DATASET_ROOT\n' "${0##*/}" >&2
}

fail() {
  printf 'FAIL: %s\n' "$*" >&2
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
  local asset
  local found=0

  while IFS= read -r -d '' asset; do
    found=1
    contains_lfs_pointer "$asset" && fail "$description is an unresolved Git LFS pointer: $asset"
  done < <(find "$root" -type f -name "$pattern" -print0)
  (( found == 1 )) || fail "$description is missing below $root"
}

(($# == 1)) || {
  usage
  exit 2
}

readonly DATASET_ROOT="$1"
[[ -d "$DATASET_ROOT" ]] || fail "GR00T demo dataset is missing: $DATASET_ROOT"
[[ -f "$DATASET_ROOT/meta/modality.json" ]] || \
  fail "GR00T dataset modality metadata is missing: $DATASET_ROOT/meta/modality.json"
require_non_pointer_assets "$DATASET_ROOT" '*.parquet' 'GR00T demo parquet data'
require_non_pointer_assets "$DATASET_ROOT" '*.mp4' 'GR00T demo video data'
printf 'PASS: valid GR00T demo fixture: %s\n' "$DATASET_ROOT"
