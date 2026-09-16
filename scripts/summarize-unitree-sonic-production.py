#!/usr/bin/env python3
"""Aggregate per-collection production conversion summaries."""
import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--output-root", type=Path, required=True)
args = parser.parse_args()
summaries = [json.loads(path.read_text()) for path in sorted(args.output_root.glob("G1_Dex3_*_Dataset/conversion_summary.json"))]
if not summaries:
    raise SystemExit("no Unitree conversion summaries found")
keys = ("requested", "converted", "skipped", "failed", "total_frames", "valid_training_frames", "excluded_tail_frames", "output_bytes", "elapsed_s")
report = {key: sum(float(item[key]) for item in summaries) for key in keys}
for key in keys[:-2]:
    report[key] = int(report[key])
report["collections"] = len(summaries)
report["throughput_frames_s"] = report["total_frames"] / report["elapsed_s"] if report["elapsed_s"] else None
report["encoder_sha256"] = summaries[0]["encoder_sha256"]
report["observation_config_sha256"] = summaries[0]["observation_config_sha256"]
report["status"] = "PASS" if report["failed"] == 0 else "FAIL"
path = args.output_root / "conversion_summary.json"
path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
print(json.dumps(report, indent=2))
