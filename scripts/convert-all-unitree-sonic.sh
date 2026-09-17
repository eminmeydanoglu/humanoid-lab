#!/usr/bin/env bash
# Canonical sequential full-corpus conversion. One collection at a time keeps
# ONNX/GPU memory deterministic; each collection itself reuses one encoder session.
# A collection that still fails does not abort the sweep: every collection is
# attempted and the run exits non-zero only at the end, after the aggregate
# summary, so a single bad episode cannot hide the rest of the corpus.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

output_root="/data/datasets/unitree-sonic-v1.1-78d"
qc_manifest="$output_root/qc/unitree-production-qc.json"
mapfile -t datasets < <(python3 - <<'PY'
import json
from pathlib import Path
doc = json.loads(Path("configs/datasets/sonic/pilots.json").read_text())
for name in doc["unitree"]["bulk_datasets"]:
    print(Path(name).name)
PY
)

excluded_episodes() {  # config-declared exclusions of one collection, one index per line
  python3 - "$1" <<'PY'
import json
import sys
from pathlib import Path
doc = json.loads(Path("configs/datasets/sonic/pilots.json").read_text())
for episode in doc["unitree"].get("bulk_excluded_episodes", {}).get(sys.argv[1], []):
    print(int(episode))
PY
}

status=0
for dataset in "${datasets[@]}"; do
  echo "[unitree-full] $dataset"
  exclude_args=()
  while IFS= read -r episode; do
    [ -n "$episode" ] && exclude_args+=(--exclude-episode "$episode")
  done < <(excluded_episodes "$dataset")
  ./dev.sh sonic-convert-unitree \
    --dataset "$dataset" --all-episodes "${exclude_args[@]}" \
    --output-root "$output_root" --qc-manifest "$qc_manifest" \
    --summary "$output_root/$dataset/conversion_summary.json" \
    || { echo "[unitree-full] FAILED $dataset" >&2; status=1; }
done

./dev.sh sonic-convert-unitree-summary --output-root "$output_root" || status=1
echo "[unitree-full] sweep complete status=$status"
exit "$status"
