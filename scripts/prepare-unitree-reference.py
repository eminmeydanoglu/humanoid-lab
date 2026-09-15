#!/usr/bin/env python3
"""Prepare one Unitree Dex3 episode for SONIC direct-reference validation.

Thin wrapper over the shared pilot library: it writes the canonical 50 Hz
reference (official IsaacLab joint order), the encoder observation and the QC
report into ``--output``.  The lower body comes from the static standing
completion policy, not from a captured IDLE time series — see
docs/sonic-pilots.md; use ``./dev.sh sonic-pilot --pilot unitree`` for the
timestamped pilot layout.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from humanoid_lab.datasets.sonic.pilot import (  # noqa: E402
    PilotSpec,
    evaluate_episode_qc,
    extract_source_video,
    map_source_episode,
    write_canonical_reference,
)
from humanoid_lab.datasets.sonic.production import write_encoder_input  # noqa: E402
from humanoid_lab.datasets.sonic.provenance import sha256_file  # noqa: E402
from humanoid_lab.datasets.sonic.reference import deployment_standing_pose, load_standing_pose  # noqa: E402

DEFAULT_LIMITS = ROOT / "configs/datasets/sonic/g1_joint_limits.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--standing-reference", type=Path,
                        help="captured single standing frame; defaults to the deployment's standing pose")
    parser.add_argument("--limits", type=Path, default=DEFAULT_LIMITS)
    parser.add_argument("--no-video", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        print(f"error: {args.output} already exists", file=sys.stderr)
        return 2

    standing = load_standing_pose(args.standing_reference) if args.standing_reference else deployment_standing_pose()
    spec = PilotSpec("unitree", args.dataset, args.episode, f"unitree_{args.dataset.name}_ep{args.episode:03d}")
    build = map_source_episode(spec, standing)
    episode = build.episode
    args.output.mkdir(parents=True)
    write_canonical_reference(args.output, build)
    encoder_input = write_encoder_input(args.output, episode)
    observation = encoder_input.observation
    clamp = encoder_input.future_clamp_fraction
    video = extract_source_video(spec, args.output / "source.mp4") if not args.no_video else None
    qc = evaluate_episode_qc(build, spec, standing, observation=observation, clamp_fraction=clamp,
                     limits_path=args.limits, layout_path=None)
    manifest = {
        "dataset": args.dataset.name,
        "episode_index": args.episode,
        "frames": int(len(episode.timestamps)),
        "duration_s": float(episode.timestamps[-1] - episode.timestamps[0]),
        "lookahead_frames": 46,
        "source_semantics": "same-row desired action",
        "source_video": video,
        "source_video_sha256": None if video is None else sha256_file(args.output / "source.mp4"),
        "qc_result": qc["episode_result"],
        "qc": qc,
        "provenance": build.provenance,
        "human_review": "pending_human_review",
    }
    (args.output / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.output / "human_review.json").write_text(json.dumps({"status": "pending_human_review"}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
