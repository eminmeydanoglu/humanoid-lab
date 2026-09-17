#!/usr/bin/env bash
# Canonical sequential full-corpus conversion. One collection at a time keeps
# ONNX/GPU memory deterministic; each collection itself reuses one encoder session.
# A collection that still fails does not abort the sweep: every collection is
# attempted and the run exits non-zero only at the end, after the aggregate
# summary, so a single bad episode cannot hide the rest of the corpus.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

config="configs/datasets/sonic/pilots.json"
output_root="/data/datasets/unitree-sonic-v1.1-78d"
qc_manifest="$output_root/qc/unitree-production-qc.json"

# Exclusions are resolved by collection directory name, so a key written as
# "unitree-g1-dex3/<Dataset>" and a bare "<Dataset>" both match.  A key that
# resolves to nothing is refused here instead of silently converting the
# episode the config wanted dropped.
python3 - "$config" <<'PY' || exit 1
"""Fail fast on an exclusion that cannot name a declared bulk collection."""
import json
import sys
from pathlib import Path

entry = json.loads(Path(sys.argv[1]).read_text())["unitree"]
declared = {Path(name).name for name in entry["bulk_datasets"]}
problems = []
for name, item in (entry.get("bulk_excluded_episodes") or {}).items():
    item = item if isinstance(item, dict) else {}
    episodes = item.get("episodes") or []
    if Path(name).name not in declared:
        problems.append(f"{name!r} is not a declared bulk dataset")
    if not episodes or any(not isinstance(index, int) or index < 0 for index in episodes):
        problems.append(f"{name!r} has no valid episode list")
    elif not str(item.get("reason", "")).strip():
        problems.append(f"{name!r} has no exclusion reason")
if problems:
    raise SystemExit("invalid bulk_excluded_episodes: " + "; ".join(problems))
PY

mapfile -t declared_names < <(python3 - "$config" <<'PY'
import json
import sys
from pathlib import Path

entry = json.loads(Path(sys.argv[1]).read_text())["unitree"]
for name in entry["bulk_datasets"]:
    print(name)
PY
)

excluded_episodes() {  # declared exclusions of one collection, one index per line
  python3 - "$config" "$1" <<'PY'
import json
import sys
from pathlib import Path

entry = json.loads(Path(sys.argv[1]).read_text())["unitree"]
wanted = Path(sys.argv[2]).name
for name, item in (entry.get("bulk_excluded_episodes") or {}).items():
    if Path(name).name == wanted:
        for episode in item["episodes"]:
            print(int(episode))
PY
}

status=0
for name in "${declared_names[@]}"; do
  dataset="$(basename "$name")"
  echo "[unitree-full] $dataset"
  exclude_args=()
  while IFS= read -r episode; do
    [ -n "$episode" ] && exclude_args+=(--exclude-episode "$episode")
  done < <(excluded_episodes "$name")
  ./dev.sh sonic-convert-unitree \
    --dataset "$dataset" --all-episodes "${exclude_args[@]}" \
    --output-root "$output_root" --qc-manifest "$qc_manifest" \
    --summary "$output_root/$dataset/conversion_summary.json" \
    || { echo "[unitree-full] FAILED $dataset" >&2; status=1; }
done

./dev.sh sonic-convert-unitree-summary --output-root "$output_root" || status=1
echo "[unitree-full] sweep complete status=$status"
exit "$status"
