#!/usr/bin/env python3
"""Build one SONIC pilot: source mapping -> canonical reference -> QC -> latent.

Read-only on the raw collection; every artifact lands in a new timestamped
directory under ``first_tur_processed/sonic_v1_1/pilots/<pilot>/<UTC>``. The
canonical episode then enters the same production encoder used by bulk conversion.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from humanoid_lab.datasets.sonic.pilot import (  # noqa: E402
    PILOT_KINDS,
    PilotSpec,
    prepare_pilot,
)
from humanoid_lab.datasets.sonic.encoder_observation import load_encoder_layout  # noqa: E402
from humanoid_lab.datasets.sonic.reference import deployment_standing_pose, load_standing_pose  # noqa: E402

DEFAULT_RAW_ROOT = Path("/data/datasets/first_tur_ham")
DEFAULT_PROCESSED_ROOT = Path("/data/datasets/first_tur_processed/sonic_v1_1")
DEFAULT_MODEL_DIR = Path("/data/models/sonic-isaac/sonic_v1_1")
DEFAULT_ENCODER_PYTHON = Path("/opt/venvs/sonic-sim/bin/python")
ENCODE_SCRIPT = ROOT / "scripts/encode-sonic-episode.py"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", choices=PILOT_KINDS, required=True)
    parser.add_argument("--episode", type=int, help="override the episode index from the pilot config")
    parser.add_argument("--dataset", help="Unitree allowlisted collection override for distributed QC")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/datasets/sonic/pilots.json")
    parser.add_argument("--limits", type=Path, default=ROOT / "configs/datasets/sonic/g1_joint_limits.json")
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--processed-root", type=Path, default=DEFAULT_PROCESSED_ROOT)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR,
                        help="pinned SONIC v1.1 directory; its observation_config.yaml must match our layout")
    parser.add_argument("--standing-reference", type=Path,
                        help="captured single standing frame; defaults to the deployment's standing pose")
    parser.add_argument("--assume-unitree-mujoco-body-order", action="store_true",
                        help="AppleToPlate only: accept the documented block-interior order assumption")
    parser.add_argument("--no-video", action="store_true", help="skip cutting the source camera clip")
    parser.add_argument("--prepare-only", action="store_true",
                        help="write canonical/QC artifacts without running the common SONIC encoder")
    parser.add_argument("--encoder-python", type=Path, default=DEFAULT_ENCODER_PYTHON)
    parser.add_argument("--expected-sha256", help="fail closed unless model_encoder.onnx has this checksum")
    args = parser.parse_args()

    layout_path = args.model_dir / "observation_config.yaml"
    if not layout_path.is_file():
        print(f"error: pinned observation config not found: {layout_path}", file=sys.stderr)
        return 2
    load_encoder_layout(layout_path)
    standing = load_standing_pose(args.standing_reference) if args.standing_reference else deployment_standing_pose()
    spec = PilotSpec.from_config(
        args.config,
        args.pilot,
        raw_root=args.raw_root,
        episode_index=args.episode,
        assume_unitree_mujoco_body_order=args.assume_unitree_mujoco_body_order,
    )
    if args.dataset:
        if args.pilot != "unitree":
            parser.error("--dataset is only supported for the Unitree QC pilot")
        declared = PilotSpec.bulk_datasets(args.config, "unitree")
        if args.dataset not in declared and f"unitree-g1-dex3/{args.dataset}" not in declared:
            parser.error(f"Unitree dataset is not allowlisted: {args.dataset}")
        relative = args.dataset if "/" in args.dataset else f"unitree-g1-dex3/{args.dataset}"
        spec = replace(spec, dataset=args.raw_root / relative,
                       pilot_name=f"unitree_{Path(args.dataset).name}_ep{spec.episode_index:03d}")
    try:
        manifest = prepare_pilot(
            spec,
            raw_root=args.raw_root,
            processed_root=args.processed_root,
            limits_path=args.limits,
            standing=standing,
            layout_path=layout_path,
            with_video=not args.no_video,
        )
    except (ValueError, FileNotFoundError, FileExistsError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    encoder = None
    if not args.prepare_only:
        if not args.encoder_python.is_file():
            print(f"error: encoder interpreter not found: {args.encoder_python}", file=sys.stderr)
            return 2
        command = [
            str(args.encoder_python), str(ENCODE_SCRIPT), manifest["processed_dir"],
            "--model-dir", str(args.model_dir),
        ]
        if args.expected_sha256:
            command += ["--expected-sha256", args.expected_sha256]
        completed = subprocess.run(command, capture_output=True, text=True, timeout=1800)
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout or "encoder failed").strip()
            print(f"error: {detail}", file=sys.stderr)
            return completed.returncode
        encoder = json.loads(
            (Path(manifest["processed_dir"]) / "encoder_manifest.json").read_text(encoding="utf-8")
        )

    summary = {key: manifest[key] for key in (
        "pilot", "dataset_name", "episode_index", "frames", "duration_s", "hand_schema_status",
        "final_action_78d", "encoder_orientation_policy", "standing_completion_policy")}
    summary["encoded"] = encoder is not None
    if encoder is not None:
        summary["encoder"] = {
            "pipeline": encoder["pipeline"],
            "motion_token_dim": encoder["motion_token_dim"],
            "action_dim": encoder["action_dim"],
            "sha256": encoder["encoder"]["sha256"],
        }
    print(json.dumps(summary, indent=2))
    print(f"qc: {manifest['qc']['episode_result']} (thresholds {manifest['qc']['threshold_result']})  decisions: "
          + ", ".join(f"{item['metric']}={item['result']}" for item in manifest["qc"]["decisions"]))
    print(f"output: {manifest['processed_dir']}")
    return 1 if manifest["qc"]["episode_result"] == "FAIL" else 0


if __name__ == "__main__":
    raise SystemExit(main())
