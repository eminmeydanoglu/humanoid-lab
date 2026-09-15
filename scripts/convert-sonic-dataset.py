#!/usr/bin/env python3
"""Bulk conversion for the SONIC v1.1 corpus — gated behind human review.

Per episode this command runs the same production path as a single pilot: the
source adapter prepares the canonical reference, ``production.build_encoder_input``
builds the sole 1751D observation, and ``production.encode_prepared_episode``
runs the pinned encoder through the onnxruntime CLI environment. An episode counts as converted
only when both ``action_tokens.npz`` and ``encoder_manifest.json`` exist in its
output directory afterwards; anything else is reported as a failure.  The
encoder subprocess inherits ``--model-dir`` and, when given,
``--expected-sha256``, so a wrong or drifted model cannot silently produce
tokens.

A source can also be marked ``excluded`` in the pilot configuration.  Excluded
sources remain archived with their pilot evidence, but this command refuses to
write them into the processed corpus even if a review file is edited later.

This command refuses to run unless the human review of the pilot for *every*
source kind it would convert says the pilots were accepted.  That is the whole
point of the gate: the pilot review page is what a human reads, and the review
files are what this command checks; no flag can override a missing approval.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from humanoid_lab.datasets.sonic.pilot import (  # noqa: E402
    PILOT_KINDS,
    PilotSpec,
    latest_pilot_dir,
    prepare_pilot,
    review_status,
)
from humanoid_lab.datasets.sonic.provenance import assert_processed_destination  # noqa: E402
from humanoid_lab.datasets.sonic.reference import deployment_standing_pose  # noqa: E402

ACCEPTED_STATUSES = ("accepted", "accepted_with_notes")
ENCODER_ARTIFACTS = ("action_tokens.npz", "encoder_manifest.json")
DEFAULT_RAW_ROOT = Path("/data/datasets/first_tur_ham")
DEFAULT_PROCESSED_ROOT = Path("/data/datasets/first_tur_processed/sonic_v1_1")
DEFAULT_MODEL_DIR = Path("/data/models/sonic-isaac/sonic_v1_1")
#: onnxruntime lives in the simulation environment, not in the conversion one.
DEFAULT_ENCODER_PYTHON = Path("/opt/venvs/sonic-sim/bin/python")
ENCODE_SCRIPT = ROOT / "scripts/encode-sonic-episode.py"


class EncodeError(RuntimeError):
    """The encoder step did not produce its artifacts."""


def encode_episode(
    pilot_dir: Path,
    *,
    model_dir: Path,
    expected_sha256: str | None,
    encoder_python: Path = DEFAULT_ENCODER_PYTHON,
) -> dict[str, object]:
    """Run the pinned encoder over a prepared pilot and verify its artifacts."""
    if not encoder_python.is_file():
        raise EncodeError(f"encoder interpreter not found: {encoder_python}")
    command = [str(encoder_python), str(ENCODE_SCRIPT), str(pilot_dir), "--model-dir", str(model_dir)]
    if expected_sha256:
        command += ["--expected-sha256", expected_sha256]
    completed = subprocess.run(command, capture_output=True, text=True, timeout=1800)
    missing = [name for name in ENCODER_ARTIFACTS if not (pilot_dir / name).is_file()]
    if completed.returncode != 0 or missing:
        detail = (completed.stderr or completed.stdout or "").strip().splitlines()
        raise EncodeError(
            f"encoder step failed for {pilot_dir} (exit {completed.returncode}, missing {missing}): "
            f"{detail[-1] if detail else 'no output'}"
        )
    manifest = json.loads((pilot_dir / "encoder_manifest.json").read_text(encoding="utf-8"))
    return {
        "motion_token_dim": manifest["motion_token_dim"],
        "action_dim": manifest["action_dim"],
        "final_action_78d": manifest["final_action_78d"]["status"],
        "encoder_sha256": manifest["encoder"]["sha256"],
        "repeatable": manifest["repeatability"]["bitwise_identical"],
    }


def parse_range(value: str) -> tuple[int, int]:
    try:
        start, stop = (int(part) for part in value.split(":"))
    except ValueError as error:
        raise argparse.ArgumentTypeError("episode range must look like START:STOP") from error
    if stop <= start:
        raise argparse.ArgumentTypeError("episode range must be non-empty")
    return start, stop


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", choices=PILOT_KINDS, action="append", required=True,
                        help="source kind to convert; repeat for more than one")
    parser.add_argument("--episodes", type=parse_range, required=True,
                        help="explicit episode range START:STOP (bulk conversion is never implicit)")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/datasets/sonic/pilots.json")
    parser.add_argument("--limits", type=Path, default=ROOT / "configs/datasets/sonic/g1_joint_limits.json")
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--processed-root", type=Path, default=DEFAULT_PROCESSED_ROOT)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--expected-sha256", help="encoder checksum every episode must match")
    parser.add_argument("--encoder-python", type=Path, default=DEFAULT_ENCODER_PYTHON)
    parser.add_argument("--assume-unitree-mujoco-body-order", action="store_true")
    args = parser.parse_args()

    assert_processed_destination(args.raw_root, args.processed_root)
    blockers: list[str] = []
    for kind in args.pilot:
        spec = PilotSpec.from_config(args.config, kind, raw_root=args.raw_root)
        if spec.bulk_conversion_status != "eligible":
            reason = spec.bulk_conversion_reason or "dataset is not eligible for processed conversion"
            blockers.append(f"{kind}: bulk conversion status is {spec.bulk_conversion_status!r} ({reason})")
            continue
        try:
            pilot_dir = latest_pilot_dir(args.processed_root, spec.pilot_name)
        except FileNotFoundError as error:
            blockers.append(f"{kind}: no pilot run yet ({error})")
            continue
        status = review_status(pilot_dir)
        if status not in ACCEPTED_STATUSES:
            blockers.append(f"{kind}: pilot review status is {status!r} ({pilot_dir / 'human_review.json'})")
    if blockers:
        print("bulk conversion blocked:", file=sys.stderr)
        for blocker in blockers:
            print(f"  - {blocker}", file=sys.stderr)
        print(
            "Resolve the listed policy/review blockers before conversion. Human review can approve an eligible "
            "pilot, but it cannot override a dataset marked excluded in the pilot configuration.",
            file=sys.stderr,
        )
        return 2

    start, stop = args.episodes
    standing = deployment_standing_pose()
    converted: list[dict[str, object]] = []
    failures: list[str] = []
    for kind in args.pilot:
        for index in range(start, stop):
            spec = PilotSpec.from_config(
                args.config,
                kind,
                raw_root=args.raw_root,
                episode_index=index,
                assume_unitree_mujoco_body_order=args.assume_unitree_mujoco_body_order,
            )
            try:
                manifest = prepare_pilot(
                    spec,
                    raw_root=args.raw_root,
                    processed_root=args.processed_root,
                    limits_path=args.limits,
                    standing=standing,
                    layout_path=args.model_dir / "observation_config.yaml",
                )
                # The manifest owns the output location; never guess it from the CLI.
                pilot_dir = Path(manifest["processed_dir"])
                encoder = encode_episode(
                    pilot_dir,
                    model_dir=args.model_dir,
                    expected_sha256=args.expected_sha256,
                    encoder_python=args.encoder_python,
                )
                converted.append({"pilot": spec.pilot_name, "dir": str(pilot_dir), **encoder})
            except Exception as error:  # noqa: BLE001 - one bad episode must not stop the batch
                failures.append(f"{spec.pilot_name}: {type(error).__name__}: {error}")
    summary = {
        "converted": len(converted),
        "episodes": converted,
        "failures": failures,
        "encoder_artifacts_required": list(ENCODER_ARTIFACTS),
    }
    print(json.dumps(summary, indent=2))
    return 0 if not failures else 1


def _pilot_name(config: Path, kind: str, raw_root: Path) -> str:
    return PilotSpec.from_config(config, kind, raw_root=raw_root).pilot_name


if __name__ == "__main__":
    raise SystemExit(main())
