#!/usr/bin/env bash
# Canonical sequential full-corpus conversion. One collection at a time keeps
# ONNX/GPU memory deterministic; each collection itself reuses one encoder session.
set -euo pipefail
cd "$(dirname "$0")/.."

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

for dataset in "${datasets[@]}"; do
  echo "[unitree-full] $dataset"
  ./dev.sh sonic-convert-unitree \
    --dataset "$dataset" --all-episodes \
    --output-root "$output_root" --qc-manifest "$qc_manifest" \
    --summary "$output_root/$dataset/conversion_summary.json"
done

./dev.sh sonic-convert-unitree-summary --output-root "$output_root"
