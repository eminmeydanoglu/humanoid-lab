#!/usr/bin/env python3
"""Prepare one Unitree Dex3 episode for SONIC direct-reference validation."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np

from humanoid_lab.controllers.sonic import SONIC_REFERENCE_JOINT_ORDER
from humanoid_lab.datasets.sonic.adapters.unitree_dex3 import load_episode


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--idle-reference", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    episode = load_episode(args.dataset, args.episode, args.idle_reference)

    reference_dir = args.output / "reference"
    reference_dir.mkdir()
    np.savetxt(reference_dir / "timestamps.csv", episode.timestamps, delimiter=",")
    np.savetxt(reference_dir / "joint_pos.csv", episode.joint_pos, delimiter=",")
    np.savetxt(reference_dir / "joint_vel.csv", episode.joint_vel, delimiter=",")
    np.savetxt(reference_dir / "body_pos.csv", episode.body_pos, delimiter=",")
    np.savetxt(reference_dir / "body_quat.csv", episode.body_quat_wxyz, delimiter=",")
    (reference_dir / "metadata.txt").write_text("SONIC v1.1 IDLE trajectory + Unitree absolute arm trajectory\n", encoding="utf-8")
    idle_provenance = json.loads((args.idle_reference / "provenance.json").read_text())
    provenance = {
        "composition": "SONIC IDLE time series with absolute Unitree arm replacement",
        "body_joint_order": list(SONIC_REFERENCE_JOINT_ORDER),
        "body_joint_order_contract": "official SONIC reference / IsaacLab",
        "preserved_idle_leg_waist_names": [
            name for name in SONIC_REFERENCE_JOINT_ORDER
            if "hip" in name or "knee" in name or "ankle" in name or "waist" in name
        ],
        "idle_reference_provenance": idle_provenance,
    }
    (reference_dir / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")

    np.savez_compressed(args.output / "reference.npz", timestamps=episode.timestamps,
        joint_pos=episode.joint_pos, joint_vel=episode.joint_vel,
        body_quat_wxyz=episode.body_quat_wxyz, body_pos=episode.body_pos,
        left_hand_joints=episode.left_hand_joints, right_hand_joints=episode.right_hand_joints)

    import pyarrow as pa
    import pyarrow.parquet as pq
    table = pa.table({"timestamp": episode.timestamps, "frame_index": np.arange(len(episode.timestamps)),
        "joint_pos": episode.joint_pos.tolist(), "joint_vel": episode.joint_vel.tolist(),
        "body_quat_wxyz": episode.body_quat_wxyz.tolist(), "left_hand_joints": episode.left_hand_joints.tolist(),
        "right_hand_joints": episode.right_hand_joints.tolist()})
    pq.write_table(table, args.output / "reference.parquet")

    meta_files = sorted((args.dataset / "meta/episodes").glob("chunk-*/*.parquet"))
    rows = []
    for path in meta_files:
        rows.extend(row for row in pq.read_table(path).to_pylist() if int(row["episode_index"]) == args.episode)
    row = rows[0]
    video = args.dataset / f"videos/observation.images.cam_left_high/chunk-{int(row['videos/observation.images.cam_left_high/chunk_index']):03d}/file-{int(row['videos/observation.images.cam_left_high/file_index']):03d}.mp4"
    subprocess.run(["ffmpeg", "-y", "-ss", str(row["videos/observation.images.cam_left_high/from_timestamp"]),
        "-to", str(row["videos/observation.images.cam_left_high/to_timestamp"]), "-i", str(video),
        "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(args.output / "source.mp4")], check=True,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    manifest = {"dataset": args.dataset.name, "episode_index": args.episode, "source_semantics": "same-row desired action",
        "source_fps": 30.0, "processed_fps": 50.0, "frames": len(episode.timestamps),
        "duration_s": float(episode.timestamps[-1] - episode.timestamps[0]), "lookahead_frames": 46,
        "source_video_sha256": sha256(args.output / "source.mp4"), "human_review": "pending_human_review"}
    (args.output / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (args.output / "human_review.json").write_text(json.dumps({"status": "pending_human_review"}, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
