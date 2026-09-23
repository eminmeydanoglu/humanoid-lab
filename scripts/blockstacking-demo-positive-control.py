#!/usr/bin/env python3
"""Replay one real BlockStacking demonstration through the live SONIC decoder.

The preparation stage extracts the exact stored 64-D SONIC token and named Dex3
hand targets from one converted demonstration, then prepares a nominal 50 Hz
message stream.  The run stage opts into Isaac's simulation clock, so every
source frame is released from elapsed physics time and a paused simulator freezes
the replay.  Recorded tracking independently verifies that the delivered source
timeline spans the same Isaac simulation interval.  The run stage launches the normal free-base
BlockStacking evaluation, selects GR00T only to obtain the existing warm-start
router ownership, resets, and never presses Start.  The prepared stream therefore
owns the action port for the whole diagnostic and no learned policy action is sent.

This is a positive-control diagnostic, not a task replay claim.  The dataset has
no object poses or demo pelvis-to-scene transform, so failure to touch the Isaac
cubes is reported as scene alignment indeterminate.  A lift is evidence only when
the live scene probe records cube motion; geometric proximity is kept separately
as a contact proxy.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
_SRC = ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from humanoid_lab.controllers.sonic import hand_joint_names  # noqa: E402

DEFAULT_OUT = ROOT / "data/outputs/blockstacking-debug/experiments/13-demo-task-positive-control"
DEFAULT_DATA_ROOT = ROOT / "data/datasets"
DEFAULT_GROOT_ROOT = DEFAULT_DATA_ROOT / "groot/unitree-dex3-sonic-v1/train"
DEFAULT_EPISODE = 21
CONTROL_HZ = 50.0
RESET_GUARD_S = 2.0
END_HOLD_S = 1.0
LIFT_THRESHOLD_M = 0.03
CONTACT_PROXY_M = 0.075


class DiagnosticError(RuntimeError):
    """The diagnostic cannot proceed without guessing a contract."""


def load_module(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise DiagnosticError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def source_indices(timestamp: np.ndarray, ticks: np.ndarray) -> np.ndarray:
    """Zero-order hold from the 30 Hz dataset timestamps onto SONIC's grid."""
    return np.clip(np.searchsorted(timestamp, ticks, side="right") - 1, 0, len(timestamp) - 1)


def prepare_stream(*, data_root: Path, episode: int, out: Path) -> dict[str, Any]:
    replay = load_module(ROOT / "scripts/demo-token-decoder-replay.py", "positive_control_demo_replay")
    demo = replay.read_demo_episode(Path(data_root), int(episode), "positive_control", None)
    valid_frames = int(demo.frames_valid)
    if valid_frames < 2:
        raise DiagnosticError(f"episode {episode} has only {valid_frames} valid frames")
    # The PSI copy is a 50->30 Hz nearest-timestamp reduction.  The live SONIC
    # replay must use the sibling GR00T corpus's original 50 Hz rows directly;
    # re-expanding PSI would duplicate rows and lose roughly 20% of token changes.
    import pyarrow.parquet as pq

    groot_path = DEFAULT_GROOT_ROOT / "data" / f"chunk-{episode // 1000:03d}" / f"episode_{episode:06d}.parquet"
    if not groot_path.is_file():
        raise DiagnosticError(f"matching GR00T 50 Hz episode is missing: {groot_path}")
    groot = pq.read_table(
        groot_path,
        columns=["action.motion_token", "teleop.left_hand_joints", "teleop.right_hand_joints", "timestamp"],
    ).to_pydict()
    ticks = np.asarray(groot["timestamp"], dtype=np.float64)
    tokens = np.stack(groot["action.motion_token"]).astype(np.float64)
    left_groot = np.stack(groot["teleop.left_hand_joints"]).astype(np.float64)
    right = np.stack(groot["teleop.right_hand_joints"]).astype(np.float64)
    # GR00T action metadata is thumb/index/middle on both hands.  SONIC's named
    # Dex3 boundary is asymmetric: the left hand is thumb/middle/index.
    left = left_groot[:, [0, 1, 2, 5, 6, 3, 4]]
    hands = np.concatenate([left, right], axis=1)
    if tokens.shape != (len(ticks), 64) or hands.shape != (len(ticks), 14):
        raise DiagnosticError(f"GR00T episode shapes are tokens={tokens.shape}, hands={hands.shape}")
    if len(ticks) < 2 or not np.all(np.diff(ticks) > 0):
        raise DiagnosticError("GR00T episode timestamps are not strictly increasing")
    last_time = float(ticks[-1] - ticks[0])

    stream_path = out / "demo-stream.json"
    payload = {
        "control_hz": CONTROL_HZ,
        "source": {
            "kind": "full_valid_blockstacking_demonstration",
            "episode_index": int(episode),
            "source_episode_index": int(demo.source_episode),
            "source_path": str(groot_path),
            "source_sha256": sha256(groot_path),
            "raw_source_episode_index": int(demo.source_episode),
            "dataset_timeline_hz": 50.0,
            "resampling": "none_exact_groot_corpus_rows",
            "hand_mapping": "GR00T left thumb/index/middle permuted to SONIC left thumb/middle/index; right unchanged",
            "first_timestamp_s": float(ticks[0]),
            "last_valid_timestamp_s": float(ticks[-1]),
            "valid_source_frames": int(len(ticks)),
            "object_pose_available": False,
            "pelvis_to_object_transform_available": False,
            "left_hand_joint_names": list(hand_joint_names("left")),
            "right_hand_joint_names": list(hand_joint_names("right")),
        },
        "tokens": tokens.tolist(),
        "left_hand_joints": hands[:, :7].tolist(),
        "right_hand_joints": hands[:, 7:].tolist(),
    }
    write_json(stream_path, payload)

    # The converted demo records palms only through body/hand state; no object
    # poses exist, so this is a reach/closure description, never an alignment.
    q = np.asarray(demo.q_measured[:valid_frames, :29], dtype=np.float64)
    compare = replay.load_compare_module()
    urdf = compare.Urdf(replay.DEFAULT_URDF)
    palms = urdf.palms(q)
    closure = np.max(np.abs(hands), axis=1)
    description = {
        "episode_index": int(episode),
        "stream": str(stream_path),
        "stream_sha256": sha256(stream_path),
        "ticks": int(len(ticks)),
        "duration_s": float(len(ticks) / CONTROL_HZ),
        "valid_source_frames": int(len(ticks)),
        "source_duration_s": last_time,
        "stream_source": "exact_groot_50hz_corpus",
        "psi_reference_frames": valid_frames,
        "token_unique_rows": int(len(np.unique(tokens, axis=0))),
        "hand_target_abs_max_rad": float(np.max(np.abs(hands))),
        "hand_active_fraction": float(np.mean(closure > 0.25)),
        "demo_palm_z_pelvis_m": {
            side: {
                "start": float(values[0, 2]),
                "min": float(np.min(values[:, 2])),
                "max": float(np.max(values[:, 2])),
                "end": float(values[-1, 2]),
            }
            for side, values in palms.items()
        },
        "geometry_contract": {
            "demo_object_pose_available": False,
            "demo_pelvis_to_object_transform_available": False,
            "consequence": "cube contact/lift failure cannot diagnose the plant unless live replay first establishes scene alignment",
        },
        "hand_contract": {
            "source": "GR00T teleop hand actions reordered by declared joint names into canonical asymmetric Dex3 order",
            "left_joint_names": list(hand_joint_names("left")),
            "right_joint_names": list(hand_joint_names("right")),
            "name_mapping_required_at_sonic_boundary": True,
        },
    }
    write_json(out / "preparation.json", description)
    return description


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def read_tracking(path: Path) -> list[dict[str, Any]]:
    import pyarrow.parquet as pq

    if not path.is_file():
        return []
    try:
        return pq.read_table(path).to_pylist()
    except Exception:  # JSONL fixtures and legacy captures remain readable.
        return read_jsonl(path)


def analyse_run(out: Path) -> dict[str, Any]:
    preparation = json.loads((out / "preparation.json").read_text(encoding="utf-8"))
    samples = read_jsonl(out / "raw/isaac.samples.jsonl")
    tracking = read_tracking(out / "raw/isaac.tracking.jsonl")
    telemetry = read_jsonl(out / "raw/telemetry/bridge-telemetry.jsonl")

    cube_rows: list[dict[str, Any]] = []
    palm_rows: list[dict[str, Any]] = []
    palm_cube_distances: list[dict[str, Any]] = []
    for row in samples:
        scene = row.get("scene") or {}
        cubes = scene.get("cubes") or {}
        if cubes:
            cube_rows.append({"wall_time_ns": row.get("wall_time_ns"), "cubes": cubes})
        palm_poses = row.get("palm_pose_w")
        if isinstance(palm_poses, dict):
            palm_rows.append({"wall_time_ns": row.get("wall_time_ns"), "palm_pose_w": palm_poses})
            for palm_name, pose in palm_poses.items():
                if not isinstance(pose, list) or len(pose) < 3:
                    continue
                palm_point = np.asarray(pose[:3], dtype=float)
                for cube_name, entry in cubes.items():
                    cube_point = entry.get("live_center_xyz_m")
                    if cube_point is None:
                        continue
                    palm_cube_distances.append({
                        "palm": palm_name,
                        "cube": cube_name,
                        "distance_m": float(np.linalg.norm(palm_point - np.asarray(cube_point, dtype=float))),
                        "wall_time_ns": row.get("wall_time_ns"),
                    })

    initial_cubes: dict[str, np.ndarray] = {}
    max_lift: dict[str, float] = {}
    max_xy: dict[str, float] = {}
    for sample in cube_rows:
        for name, entry in sample["cubes"].items():
            position = (
                entry.get("live_center_xyz_m") or entry.get("live_position_m")
                or entry.get("position_m") or entry.get("live_center_m")
            )
            if position is None:
                z = entry.get("live_center_z_m")
                pose = entry.get("pose_w")
                position = pose[:3] if isinstance(pose, list) and len(pose) >= 3 else None
                if position is None and z is not None:
                    position = [0.0, 0.0, z]
            if position is None:
                continue
            point = np.asarray(position, dtype=float)[:3]
            initial_cubes.setdefault(name, point)
            delta = point - initial_cubes[name]
            max_lift[name] = max(max_lift.get(name, 0.0), float(delta[2]))
            max_xy[name] = max(max_xy.get(name, 0.0), float(np.linalg.norm(delta[:2])))

    target = [np.asarray(row["body_target"], dtype=float) for row in tracking if row.get("body_target") is not None]
    measured = [np.asarray(row["body_measured"], dtype=float) for row in tracking if row.get("body_measured") is not None]
    alignment = None
    palm_tracking = None
    if target and measured and len(target) == len(measured):
        target_array = np.asarray(target)
        measured_array = np.asarray(measured)
        residual = target_array - measured_array
        alignment = {
            "rows": len(target),
            "body_mae_rad": float(np.mean(np.abs(residual))),
            "body_p95_abs_rad": float(np.quantile(np.abs(residual), 0.95)),
            "body_max_abs_rad": float(np.max(np.abs(residual))),
        }
        replay = load_module(ROOT / "scripts/demo-token-decoder-replay.py", "positive_control_analysis_replay")
        urdf = replay.load_compare_module().Urdf(replay.DEFAULT_URDF)
        target_palms = urdf.palms(target_array[:, :29])
        measured_palms = urdf.palms(measured_array[:, :29])
        palm_tracking = {
            side: {
                "target_z_min_m": float(np.min(target_palms[side][:, 2])),
                "target_z_max_m": float(np.max(target_palms[side][:, 2])),
                "measured_z_min_m": float(np.min(measured_palms[side][:, 2])),
                "measured_z_max_m": float(np.max(measured_palms[side][:, 2])),
                "target_vs_measured_position_mae_m": float(np.mean(np.abs(target_palms[side] - measured_palms[side]))),
            }
            for side in ("left", "right")
        }

    applied = [
        row for row in telemetry
        if (row.get("kind") or row.get("event")) == "applied_action"
        and row.get("source") == "warmstart"
    ]

    def frame_index(row: dict[str, Any]) -> int:
        value: Any = (row.get("fields") or {}).get("frame_index", -1)
        while isinstance(value, list) and value:
            value = value[0]
        return int(value)

    sent_indices = [frame_index(row) for row in applied]
    expected_ticks = int(preparation["ticks"])
    expected_indices = list(range(expected_ticks))
    applied_by_index = {
        frame_index(row): row for row in applied
        if 0 <= frame_index(row) < expected_ticks
    }
    complete_indices = sorted(applied_by_index) == expected_indices
    stream_path = out / "demo-stream.json"
    component_errors: dict[str, float | None] = {}
    if stream_path.is_file():
        stream_payload = json.loads(stream_path.read_text(encoding="utf-8"))
        for field, source_key in (
            ("token_state", "tokens"),
            ("left_hand_joints", "left_hand_joints"),
            ("right_hand_joints", "right_hand_joints"),
        ):
            errors = []
            expected = np.asarray(stream_payload[source_key], dtype=float)
            for index, row in applied_by_index.items():
                actual = np.asarray((row.get("fields") or {}).get(field), dtype=float).reshape(-1)
                if actual.shape != expected[index].shape:
                    errors.append(float("inf"))
                else:
                    errors.append(float(np.max(np.abs(actual - expected[index]))))
            component_errors[field] = max(errors) if errors else None
        values_match = complete_indices and all(
            value is not None and np.isfinite(value) and value <= 1e-6
            for value in component_errors.values()
        )
    else:
        values_match = complete_indices
    cadence = None
    times = [float(row["wall_time_ns"]) / 1e9 for row in applied if row.get("wall_time_ns") is not None]
    if len(times) > 1 and times[-1] > times[0]:
        cadence = (len(times) - 1) / (times[-1] - times[0])

    any_lift = any(value >= LIFT_THRESHOLD_M for value in max_lift.values())
    min_palm_cube = min((row["distance_m"] for row in palm_cube_distances), default=None)
    root_rows = [np.asarray(row["root_position"], dtype=float) for row in tracking if row.get("root_position") is not None]
    initial_root = root_rows[0] if root_rows else None
    cube_in_initial_pelvis = {
        name: (point - initial_root).tolist() for name, point in initial_cubes.items()
    } if initial_root is not None else {}
    # Align the first and final source-frame sends to tracking and independently
    # verify the physics-clock interval that paced the stream.
    temporal: dict[str, Any] = {
        "clock": "isaac_simulation_clock_verified_against_tracking_sim_s",
        "source_timeline_s": float(preparation["source_duration_s"]),
        "valid": False,
        "reason": "tracking or applied-action timestamps unavailable",
    }
    tracking_wall = np.asarray([row.get("wall_time_ns", -1) for row in tracking], dtype=np.int64)
    tracking_sim = np.asarray([row.get("sim_s", np.nan) for row in tracking], dtype=float)
    distinct = [(frame_index(row), int(row["wall_time_ns"])) for row in applied if row.get("wall_time_ns") is not None]
    final_source_index = int(preparation["ticks"]) - 1
    start_send = next((wall for index, wall in distinct if index == 0), None)
    final_send = next((wall for index, wall in distinct if index >= final_source_index), None)
    if start_send is not None and final_send is not None and tracking_wall.size and np.isfinite(tracking_sim).all():
        start_row = int(np.argmin(np.abs(tracking_wall - start_send)))
        final_row = int(np.argmin(np.abs(tracking_wall - final_send)))
        sim_elapsed = float(tracking_sim[final_row] - tracking_sim[start_row])
        source_elapsed = float(preparation["source_duration_s"])
        ratio = sim_elapsed / source_elapsed if source_elapsed > 0 else 0.0
        valid = ratio >= 0.98 and sim_elapsed + 0.05 >= source_elapsed
        temporal.update({
            "valid": bool(valid),
            "source_timeline_s": source_elapsed,
            "isaac_sim_elapsed_s": sim_elapsed,
            "coverage_ratio": ratio,
            "start_sim_s": float(tracking_sim[start_row]),
            "final_source_frame_sim_s": float(tracking_sim[final_row]),
            "start_wall_time_ns": start_send,
            "final_source_frame_wall_time_ns": final_send,
            "reason": None if valid else "delivered source frames did not span the required Isaac simulation timeline",
        })

    geometry_valid = False  # dataset contains no object transform by construction
    warm_walls = [int(row["wall_time_ns"]) for row in applied if row.get("wall_time_ns") is not None]
    replay_start_wall = min(warm_walls) if warm_walls else None
    replay_end_wall = max(warm_walls) if warm_walls else None
    learned_during_replay = [
        row for row in telemetry
        if (row.get("kind") or row.get("event")) == "applied_action"
        and row.get("source") in {"psi", "groot"}
        and replay_start_wall is not None
        and int(row.get("wall_time_ns", -1)) >= replay_start_wall
    ]
    result = {
        "question": "can the current free-base Isaac scene execute an exact real demo SONIC-token and named Dex3 replay?",
        "preparation": preparation,
        "reset": json.loads((out / "reset-diagnostic.json").read_text(encoding="utf-8")) if (out / "reset-diagnostic.json").is_file() else None,
        "stream_delivery": {
            "applied_messages": len(applied),
            "applied_source_frames": len(applied_by_index),
            "expected_source_frames": expected_ticks,
            "first_frame_index": min(sent_indices) if sent_indices else None,
            "last_frame_index": max(sent_indices) if sent_indices else None,
            "all_source_indices_applied": complete_indices,
            "token_and_hand_max_abs_error": component_errors,
            "token_and_hand_values_match": values_match,
            "wall_cadence_hz": cadence,
        },
        "cube_max_lift_m": max_lift,
        "cube_max_xy_displacement_m": max_xy,
        "any_cube_lifted": any_lift,
        "lift_threshold_m": LIFT_THRESHOLD_M,
        "decoded_target_vs_measured": alignment,
        "decoded_and_measured_palms": palm_tracking,
        "temporal_alignment": temporal,
        "contact": {
            "force_or_pair_contact_recorded": False,
            "geometric_proxy_threshold_m": CONTACT_PROXY_M,
            "geometric_proxy_computable": bool(palm_cube_distances),
            "minimum_palm_origin_to_cube_center_m": min_palm_cube,
            "geometric_proxy_reached": (
                None if min_palm_cube is None else min_palm_cube <= CONTACT_PROXY_M
            ),
            "samples": palm_cube_distances,
            "reason": "palm-centre proximity is recorded, but current telemetry has no fingertip/contact-pair force stream; proxy contact is not physical contact evidence",
        },
        "geometry_alignment": {
            "valid": geometry_valid,
            "demo_object_pose_available": False,
            "demo_pelvis_to_object_transform_available": False,
            "live_initial_cube_world_m": {name: point.tolist() for name, point in initial_cubes.items()},
            "live_initial_pelvis_world_m": None if initial_root is None else initial_root.tolist(),
            "live_cube_centres_in_initial_pelvis_translation_m": cube_in_initial_pelvis,
            "demo_cube_centres_in_pelvis_m": None,
            "interpretation": "the live transform is recorded, but no demo object transform exists to align against; a miss is non-diagnostic for plant capability and a lift remains positive evidence",
        },
        "validity_gates": {
            "fresh_reset": bool((json.loads((out / "reset-diagnostic.json").read_text()) if (out / "reset-diagnostic.json").is_file() else {}).get("passed")),
            "exact_stream_delivered": bool(complete_indices and values_match),
            "learned_policy_disabled": not learned_during_replay,
            "source_timeline_covered_in_isaac_sim_time": bool(temporal["valid"]),
            "absolute_scene_alignment_known": geometry_valid,
            "physical_contact_observed": False,
            "cube_lift_observed": any_lift,
        },
        "decision": (
            "DELIVERY_INVALID" if not (complete_indices and values_match) or learned_during_replay else
            "TEMPORALLY_INVALID" if not temporal["valid"] else
            "POSITIVE_CONTROL_PASS" if any_lift else
            "INDETERMINATE_SCENE_ALIGNMENT" if not geometry_valid else
            "NO_LIFT"
        ),
        "negative_outcome_interpretable": bool(temporal["valid"] and geometry_valid),
    }
    write_json(out / "summary.json", result)
    return result


def container_path(path: Path) -> str:
    data = (ROOT / "data").resolve()
    outputs = (data / "outputs").resolve()
    resolved = path.resolve()
    if resolved == outputs or outputs in resolved.parents:
        return "/outputs/" + str(resolved.relative_to(outputs))
    if data in resolved.parents:
        return "/data/" + str(resolved.relative_to(data))
    if ROOT in resolved.parents:
        return "/workspace/humanoid-lab/" + str(resolved.relative_to(ROOT))
    return str(resolved)


def control_status(container: str, endpoint: str = "tcp://127.0.0.1:5559") -> dict[str, Any]:
    code = (
        "import json,zmq; c=zmq.Context(); s=c.socket(zmq.REQ); s.setsockopt(zmq.LINGER,0);"
        f"s.connect({endpoint!r}); s.send(b'status');"
        "print(s.recv().decode() if s.poll(5000) else '{}'); s.close(); c.term()"
    )
    completed = subprocess.run(
        ["docker", "exec", container, "bash", "-lc",
         "source /opt/humanoid-lab/entrypoint.sh && use-isaac-sonic && python -c " + repr(code)],
        capture_output=True, text=True, check=False,
    )
    if completed.returncode != 0:
        raise DiagnosticError(f"cannot query Isaac status: {completed.stderr.strip()}")
    return json.loads(completed.stdout.strip() or "{}")


def wait_reset_complete(container: str, before: dict[str, Any], timeout_s: float = 10.0) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_s
    last: dict[str, Any] = {}
    while time.monotonic() < deadline:
        last = control_status(container)
        camera = last.get("camera") or {}
        if (
            int(last.get("episode_id", -1)) > int(before.get("episode_id", -1))
            and camera.get("has_frame")
            and int(camera.get("episode_id", -2)) == int(last.get("episode_id", -1))
            and int(last.get("physics_tick", 0)) > 0
        ):
            return {"passed": True, "before": before, "after": last}
        time.sleep(0.2)
    return {"passed": False, "before": before, "after": last, "reason": "no new episode with a fresh matching camera frame"}


def wait_for_sim_replay(ui: Any, ticks: int, hold_s: float, timeout_s: float) -> dict[str, Any]:
    required = int(ticks + np.ceil(hold_s * CONTROL_HZ))
    deadline = time.monotonic() + timeout_s
    last: dict[str, Any] = {}
    while time.monotonic() < deadline:
        last = ui.status()
        warm = last.get("warmstart") or {}
        if warm.get("error"):
            raise DiagnosticError(f"simulation-clock replay failed: {warm['error']}")
        if int(warm.get("sent", 0)) >= required:
            return warm
        time.sleep(0.2)
    raise DiagnosticError(
        f"simulation-clock replay did not deliver {required} messages within {timeout_s:.1f}s; "
        f"last status: {json.dumps(last)[:400]}"
    )


def run_diagnostic(args: argparse.Namespace) -> dict[str, Any]:
    out = args.out.resolve()
    preparation = json.loads((out / "preparation.json").read_text(encoding="utf-8"))
    stream = out / "demo-stream.json"
    if not stream.is_file():
        raise DiagnosticError(f"prepared stream missing: {stream}")

    rollout = load_module(ROOT / "scripts/blockstacking-rollout.py", "positive_control_rollout")
    raw = out / "raw"
    logs = out / "logs"
    raw.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    paths = {
        "telemetry": container_path(raw / "telemetry"),
        "video_raw": container_path(raw / "video.raw.mp4"),
        "video_timestamps": container_path(raw / "video.timestamps.jsonl"),
        "isaac_samples": container_path(raw / "isaac.samples.jsonl"),
        "isaac_tracking": container_path(raw / "isaac.tracking.jsonl"),
        "isaac_metrics": container_path(raw / "isaac.metrics.json"),
    }
    duration = float(preparation["duration_s"])
    clock_path = raw / "isaac.sim-clock.txt"
    clock_path.unlink(missing_ok=True)
    launch_args = SimpleNamespace(
        psi_checkpoint_dir=args.psi_checkpoint_dir,
        checkpoint_step=args.checkpoint_step,
        groot_checkpoint_dir=args.groot_checkpoint_dir,
        headless=args.headless,
        gui=args.gui,
        reset_pose_file=None,
        reset_pose_hold=False,
        warmstart_tokens=stream,
        warmstart_sim_clock=clock_path,
        warmstart_clock_timeout_s=args.clock_timeout_s,
        warmstart_window_seconds=duration + END_HOLD_S,
        settle_seconds=RESET_GUARD_S + duration + END_HOLD_S,
        initial_pose_handshake=False,
        scene_profile=None,
        gravity_feedforward=args.gravity_feedforward,
        policy_clock="wall",
        policy_clock_timeout_s=args.clock_timeout_s,
        groot_capture_dir=None,
        groot_capture_max_requests=32,
        groot_left_hand_contract="compatibility",  # explicit legacy demo replay, not learned-policy default
        psi0_neck_policy="error",
        psi0_rtc_off=False,
    )
    rollout.preflight(launch_args, rollout.data_root())
    command = rollout.build_launcher_command(launch_args, paths)
    log_path = out / "launcher.log"
    with log_path.open("wb") as handle:
        launcher = subprocess.Popen(command, cwd=str(ROOT), stdout=handle, stderr=subprocess.STDOUT,
                                    stdin=subprocess.DEVNULL, start_new_session=True)
    ui = rollout.Ui(args.ui_port)
    session: dict[str, Any] = {"command": command, "duration_s": duration, "gravity_feedforward": args.gravity_feedforward}
    try:
        session["meta"] = rollout.wait_for_ui(ui, launcher, log_path, timeout_s=args.startup_timeout_s)
        ui.select_model("groot")
        rollout.wait_for_model(ui, "groot", timeout_s=args.startup_timeout_s)
        container = rollout.container_name()
        before = control_status(container)
        reset_wall = time.time()
        session["reset_response"] = ui.reset()
        reset = wait_reset_complete(container, before)
        reset["queued_wall_time"] = reset_wall
        write_json(out / "reset-diagnostic.json", reset)
        if not reset["passed"]:
            raise DiagnosticError(reset["reason"])
        # Start is never called, so PSI/GR00T learned output remains gated off.
        # Completion is counted from messages released by Isaac simulation time;
        # wall time only bounds a failed or stalled run.
        session["warmstart_completion"] = wait_for_sim_replay(
            ui, int(preparation["ticks"]), END_HOLD_S, args.run_timeout_s,
        )
        session["final_ui_status"] = ui.status()
    finally:
        session["isaac_shutdown"] = rollout.request_isaac_shutdown()
        rollout.wait_for_artifacts([raw / "isaac.metrics.json", raw / "isaac.tracking.jsonl"], rollout.ISAAC_FINALIZE_TIMEOUT_S)
        session["launcher_exit_code"] = rollout.terminate_launcher(launcher, log_path, grace_s=30.0)
        write_json(out / "session.json", session)
    return analyse_run(out)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="stage", required=True)
    prepare = sub.add_parser("prepare", help="CPU-only: extract and validate one exact demo stream")
    prepare.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    prepare.add_argument("--episode", type=int, default=DEFAULT_EPISODE)
    prepare.add_argument("--out", type=Path, default=DEFAULT_OUT)

    run = sub.add_parser("run", help="GPU: replay the prepared stream in current free-base Isaac")
    run.add_argument("--out", type=Path, default=DEFAULT_OUT)
    run.add_argument("--psi-checkpoint-dir", type=Path, required=True)
    run.add_argument("--checkpoint-step", type=int, required=True)
    run.add_argument("--groot-checkpoint-dir", type=Path, required=True)
    run.add_argument("--ui-port", type=int, default=8015)
    run.add_argument("--headless", action="store_true")
    run.add_argument("--gui", action="store_true")
    run.add_argument("--gravity-feedforward", action="store_true")
    run.add_argument("--clock-timeout-s", type=float, default=5.0)
    run.add_argument("--startup-timeout-s", type=float, default=300.0)
    run.add_argument("--run-timeout-s", type=float, default=180.0)

    analyse = sub.add_parser("analyse", help="CPU-only: regenerate summary from a completed run")
    analyse.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.stage == "prepare":
            result = prepare_stream(data_root=args.data_root, episode=args.episode, out=args.out)
        elif args.stage == "run":
            if args.headless and args.gui:
                raise DiagnosticError("--headless and --gui are mutually exclusive")
            result = run_diagnostic(args)
        else:
            result = analyse_run(args.out)
    except (DiagnosticError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
