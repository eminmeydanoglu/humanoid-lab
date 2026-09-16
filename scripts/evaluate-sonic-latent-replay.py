#!/usr/bin/env python3
"""Evaluate the complete A=reference, B=SONIC command, C=robot response path."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from humanoid_lab.controllers.sonic import BODY_JOINT_ORDER  # noqa: E402
from humanoid_lab.datasets.sonic.fidelity import DEFAULT_MODEL_XML, compare_reference_command_response, load_timeline  # noqa: E402
from humanoid_lab.datasets.sonic.tracking import load_tracking  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pilot_dir", type=Path)
    parser.add_argument("--require-pass", action="store_true", help="exit nonzero on incomplete or inaccurate replay")
    args = parser.parse_args()
    directory = args.pilot_dir
    with np.load(directory / "reference.npz") as payload:
        reference = {name: payload[name] for name in payload.files}
    tracking = load_tracking(directory / "sonic_latent_tracking.parquet")
    timeline = load_timeline(directory / "sonic_latent_stream.npz")
    result = compare_reference_command_response(reference, tracking, timeline, model_xml=DEFAULT_MODEL_XML)
    transport_path = directory / "sonic_transport_coverage.json"
    transport = json.loads(transport_path.read_text(encoding="utf-8")) if transport_path.is_file() else None
    result["transport"] = transport
    received = bool(transport and transport.get("complete")
                    and transport.get("expected_frames") == len(reference["joint_pos"]))
    result["decisions"].insert(1, {"metric": "coverage.controller_received_all_frames",
                                   "value": float(received), "limit": 1.0,
                                   "result": "PASS" if received else "FAIL"})
    result["result"] = "PASS" if all(item["result"] == "PASS" for item in result["decisions"]) else "FAIL"
    plots = result.pop("plot_data")

    fig, axes = plt.subplots(4, 1, figsize=(13, 10), sharex=True)
    focus = ("left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_joint", "left_wrist_yaw_joint")
    for axis, name in zip(axes, focus):
        joint = BODY_JOINT_ORDER.index(name)
        for label, key in (("A reference", "reference"), ("B SONIC command", "command"), ("C robot response", "response")):
            axis.plot(plots["elapsed_s"], plots[key][:, joint], label=label, linewidth=1)
        axis.set_ylabel(name.removesuffix("_joint") + " (rad)")
        axis.grid(alpha=0.2)
    axes[0].legend(ncol=3)
    axes[-1].set_xlabel("wall time since first token (s)")
    fig.suptitle("Left arm: reference → SONIC command → measured robot")
    fig.tight_layout()
    fig.savefig(directory / "sonic_left_arm_abc.png", dpi=140)
    plt.close(fig)

    fig, axes = plt.subplots(7, 1, figsize=(13, 13), sharex=True)
    for joint, axis in enumerate(axes):
        for label, key in (("A reference", "reference_left_hand"), ("B SONIC command", "command_left_hand"),
                           ("C robot response", "response_left_hand")):
            axis.plot(plots["elapsed_s"], plots[key][:, joint], label=label, linewidth=1)
        axis.set_ylabel(f"left hand {joint} (rad)")
        axis.grid(alpha=0.2)
    axes[0].legend(ncol=3)
    axes[-1].set_xlabel("wall time since first token (s)")
    fig.suptitle("Left Dex3 hand: reference → SONIC command → measured robot")
    fig.tight_layout()
    fig.savefig(directory / "sonic_left_hand_abc.png", dpi=140)
    plt.close(fig)

    fig, axes = plt.subplots(3, 1, figsize=(13, 8), sharex=True)
    for coordinate, axis in enumerate(axes):
        for label, key in (("A reference", "reference_wrist"), ("B SONIC command", "command_wrist"),
                           ("C robot response", "response_wrist")):
            axis.plot(plots["elapsed_s"], plots[key][:, coordinate], label=label, linewidth=1)
        axis.set_ylabel(f"left wrist {'xyz'[coordinate]} (m)")
        axis.grid(alpha=0.2)
    axes[0].legend(ncol=3)
    axes[-1].set_xlabel("wall time since first token (s)")
    fig.suptitle("Pelvis-local left wrist path (pinned G1 MuJoCo forward kinematics)")
    fig.tight_layout()
    fig.savefig(directory / "sonic_left_wrist_abc.png", dpi=140)
    plt.close(fig)

    fig, axes = plt.subplots(3, 1, figsize=(13, 8), sharex=True)
    for coordinate, axis in enumerate(axes):
        for label, key in (("A reference", "reference_world_wrist"), ("B SONIC command", "command_world_wrist"),
                           ("C robot response", "response_world_wrist")):
            axis.plot(plots["elapsed_s"], plots[key][:, coordinate], label=label, linewidth=1)
        axis.set_ylabel(f"world wrist {'xyz'[coordinate]} (m)")
        axis.grid(alpha=0.2)
    axes[0].legend(ncol=3)
    axes[-1].set_xlabel("simulation time since first token (s)")
    fig.suptitle("World-frame left wrist path, including measured robot root")
    fig.tight_layout()
    fig.savefig(directory / "sonic_left_wrist_world_abc.png", dpi=140)
    plt.close(fig)

    (directory / "sonic_fidelity.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"result": result["result"], "coverage": result["coverage"],
                      "decisions": result["decisions"]}, indent=2))
    return 1 if args.require_pass and result["result"] != "PASS" else 0


if __name__ == "__main__":
    raise SystemExit(main())
