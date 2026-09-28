#!/usr/bin/env bash
# Download the Unitree G1 Dex3 collection (13 LeRobot v3 datasets) into the persistent data mount.
#
# Dataset list and Hugging Face repo ids come from the collection:
#   https://huggingface.co/collections/mindchain/unitree-robotics-g1-dex3-datasets
# Repo ids are pinned in <DATA_ROOT>/datasets/first_tur_ham/unitree-g1-dex3/sources.json, written from the
# collection API, so reruns stay on the same set even if the collection changes.
set -euo pipefail

readonly DATA_ROOT="${HUMANOID_DATA_ROOT:-/data}"
readonly DATASET_ROOT="${G1_DEX3_ROOT:-$DATA_ROOT/datasets/first_tur_ham/unitree-g1-dex3}"
readonly SOURCES_FILE="$DATASET_ROOT/sources.json"
readonly LOG_ROOT="${G1_DEX3_LOG_ROOT:-$DATA_ROOT/diagnostics/g1-dex3-download}"
readonly JOBS="${G1_DEX3_JOBS:-2}"

usage() {
  cat <<'EOF'
usage: fetch-g1-dex3-datasets.sh [--filter substring ...]

Downloads every dataset listed in <DATA_ROOT>/datasets/first_tur_ham/unitree-g1-dex3/sources.json with
`hf download` (resumable; already-complete files are skipped). Logs land in
<DATA_ROOT>/diagnostics/g1-dex3-download/<dataset>.log and a provenance file is
written at <DATA_ROOT>/datasets/first_tur_ham/unitree-g1-dex3/provenance.json.

Options:
  --filter substring   only datasets whose directory name contains substring (repeatable)
EOF
}

fail() {
  echo "FAIL: $*" >&2
  exit 2
}

FILTERS=()
while (($#)); do
  case "$1" in
    --filter) FILTERS+=("$2"); shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) fail "unknown argument: $1" ;;
  esac
done

[[ -f "$SOURCES_FILE" ]] || fail "missing $SOURCES_FILE"
command -v hf >/dev/null 2>&1 || fail "hf (huggingface_hub CLI) is required"
mkdir -p "$DATASET_ROOT" "$LOG_ROOT"

readonly PYTHON_BIN="${PYTHON:-python3}"
python3 - "$SOURCES_FILE" "${FILTERS[@]:-}" > "$LOG_ROOT/.queue" <<'PY'
import json
import sys

sources = json.load(open(sys.argv[1]))
filters = [f for f in sys.argv[2:] if f]
for name, repo in sorted(sources.items()):
    if filters and not any(f in name for f in filters):
        continue
    print(f"{name}\t{repo}")
PY

count=$(wc -l < "$LOG_ROOT/.queue")
[[ "$count" -gt 0 ]] || fail "no datasets selected from $SOURCES_FILE"
echo "downloading $count datasets with $JOBS parallel jobs into $DATASET_ROOT"

# shellcheck disable=SC2329 # invoked through the exported function in the parallel loop below
download_one() {
  local name="$1" repo="$2"
  local log="$LOG_ROOT/$name.log"
  local target="$DATASET_ROOT/$name"
  mkdir -p "$target"
  if HF_HUB_DISABLE_PROGRESS_BARS=1 hf download "$repo" --repo-type dataset --local-dir "$target" >"$log" 2>&1; then
    echo "OK   $name"
  else
    echo "FAIL $name (see $log)"
    return 1
  fi
}
export -f download_one
export DATASET_ROOT LOG_ROOT

status=0
while IFS=$'\t' read -r name repo; do
  while (( $(jobs -pr | wc -l) >= JOBS )); do wait -n || status=1; done
  bash -c 'download_one "$0" "$1"' "$name" "$repo" &
done < "$LOG_ROOT/.queue"
while (( $(jobs -pr | wc -l) )); do wait -n || status=1; done
rm -f "$LOG_ROOT/.queue"

"$PYTHON_BIN" - "$SOURCES_FILE" "$DATASET_ROOT" "$LOG_ROOT" <<'PY'
import json
import os
import sys
import time
import urllib.request

sources_path, dataset_root, log_root = sys.argv[1:]
sources = json.load(open(sources_path))
provenance = {"collection": "mindchain/unitree-robotics-g1-dex3-datasets", "datasets": {}}
for name, repo in sorted(sources.items()):
    target = os.path.join(dataset_root, name)
    if not os.path.isdir(target):
        continue
    try:
        meta = json.load(urllib.request.urlopen(f"https://huggingface.co/api/datasets/{repo}"))
        revision = meta.get("sha") or meta.get("lastModified")
    except Exception:  # noqa: BLE001 - provenance lookup is best effort
        revision = None
    total = 0
    files = 0
    for directory, _children, names in os.walk(target):
        if ".cache" in directory.split(os.sep):
            continue
        for entry in names:
            total += os.path.getsize(os.path.join(directory, entry))
            files += 1
    provenance["datasets"][name] = {"repo": repo, "revision": revision, "files": files, "bytes": total}
provenance["recorded_at_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
with open(os.path.join(dataset_root, "provenance.json"), "w", encoding="utf-8") as handle:
    json.dump(provenance, handle, indent=2, sort_keys=True)
    handle.write("\n")
print("wrote provenance.json")
PY

exit "$status"
