#!/usr/bin/env python3
"""Experiment 07 -- GR00T demonstration-token warm start A/B.

One question: does warm-starting the SONIC controller with a real BlockStacking
demonstration's start-phase token stream -- sent to the *existing* decoder over
the settle, with the live measured state, so the deployment's own history and
its ``last_actions`` evolve naturally -- and then handing over to GR00T at
``Start`` without resetting anything, reduce the high palm target GR00T asks
for at the beginning and later?

Nothing is teleported and nothing is held: the intervention is on the token
stream the controller is *commanded* with, which is the one route experiment 06
left open (its own reset-state injection was swept by the deployment's closed
loop within 20 ms).

The stages:

* ``select``  the task-0 demonstrations experiment 01's replay validated ->
  each one's start phase (the quiet window before its first sustained arm
  motion), its support, its palm/scene clearance -> the chosen segment as a
  prepared 50 Hz token stream the bridge consumes, plus the selection record.
* ``analyse`` the A/W rollout directories -> validity gates (evaluated *before*
  any policy-output comparison), the policy-output and measured metrics, the
  paired effects, and the run's decision input.
* ``figures`` at most three figures from the analysed tables.
* ``manifest`` script/input hashes and the exact commands.

The demonstrations are read through experiment 01's own episode reader (tokens,
hand command, the 30 Hz -> 50 Hz hold resampling) and the support verdict
through experiment 06's calibration object, so neither is re-implemented here.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

SCHEMA_VERSION = 1
SCRIPT_VERSION = "groot-token-warmstart-ab.py/1.0.0"
REPO_ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = REPO_ROOT / "data" / "outputs" / "blockstacking-debug"
EXPERIMENTS = CAMPAIGN / "experiments"
DEFAULT_OUT = EXPERIMENTS / "07-groot-token-warmstart-ab"
EXP01 = EXPERIMENTS / "01-demo-token-decoder"
EXP05 = EXPERIMENTS / "05-policy-token-support"
EXP06 = EXPERIMENTS / "06-groot-reset-pose-ab"
SUPPORT_DIR = EXPERIMENTS / "02-state-support"
BASELINE_SERIES = CAMPAIGN / "groot-rollout-1"
DATASET_ROOT = REPO_ROOT / "data" / "datasets" / "psi0-unitree-dex3-sonic-v1" / "train"
#: The dataset root experiment 01's reader expects (it appends the psi0 train
#: copy's own relative path itself).
DATASET_COPY_ROOT = DATASET_ROOT.parents[1]
PROFILE = REPO_ROOT / "configs" / "profiles" / "isaac-g1-sonic-blockstacking-dex3.json"

#: The two cells: ``A`` is the canonical settle, ``W`` adds only the demo token
#: stream over the settle's tail.
CELLS = ("A", "W")
#: Policy-output windows, in seconds after ``Start``.
WINDOWS_S = {"first_1s": (0.0, 1.0), "first_5s": (0.0, 5.0), "active": (0.0, None)}
#: The settle tail the warm start occupies (the campaign driver's default).
WARMSTART_WINDOW_S = 6.0
#: How far back of a Start the session tracking stream is kept.
SETTLE_LOOKBACK_S = 40.0
#: The last part of that window in which the robot must already be calm.
CALM_TAIL_S = 1.0
#: Pre-declared acceptance for the treatment, evaluated only on runs that pass
#: every validity gate: a >=5 cm lower target palm or a >=30% shorter
#: target-to-cube distance, in the same direction in every paired repeat.
ACCEPTANCE = {
    "target_palm_z_drop_m": 0.05,
    "target_palm_cube_min_distance_reduction": 0.30,
    "measured_droop_alone_counts": False,
}
#: Pre-declared validity gates (numbers, not prose).
GATE_LIMITS = {
    # G1: the arms at Policy Start sit inside the demonstration p95 (exp-02/06
    # calibration: arms LOEO Mahalanobis p95 = 5.39).
    "start_arms_mahalanobis_p95": 5.39,
    # G2: the limb is calm at the end of the warm start (rad/s, m/s).
    "calm_joint_speed_p95_rad_s": 0.25,
    "calm_joint_speed_max_rad_s": 1.0,
    "calm_palm_speed_p95_m_s": 0.10,
    # G3: the pelvis is where the canonical settle leaves it (m).
    "pelvis_height_drift_max_m": 0.02,
    "pelvis_height_min_m": 0.70,
    "pelvis_xy_drift_max_m": 0.10,
    # G4: no cube moved and nothing in the scene was corrupted (m).
    "cube_displacement_max_m": 0.01,
    # G5: the hand-off to the policy is bounded (rad) and the sim stays finite.
    "transition_target_jump_ceiling_rad": 1.0,
}
#: The demonstration-token support metric is experiment 05's own nearest-neighbour
#: value L1 distance to the balanced task-0 token cloud (seed and density pinned
#: there); its thresholds come from that experiment's calibration.
TOKEN_SUPPORT_METRIC = "value_l1"
TOKEN_CLOUD_FRAMES_PER_EPISODE = 40
TOKEN_CLOUD_SEED = 20260921

ARM_HAND = slice(15, 43)
ARMS = slice(15, 29)
ARM_SLICE = slice(0, 14)


class AbError(RuntimeError):
    """The stage cannot run; the message is meant for the operator."""


# ---------------------------------------------------------------- plumbing


def repo_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (REPO_ROOT / path)


def rel(path: Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_jsonable) + "\n",
                    encoding="utf-8")


def _jsonable(value: Any) -> Any:
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"not JSON serialisable: {type(value).__name__}")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not Path(path).is_file():
        return rows
    with Path(path).open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return rows


def write_csv(path: Path, rows: Sequence[dict[str, Any]],
              columns: Sequence[str] | None = None) -> None:
    if not rows and not columns:
        return
    columns = list(columns or sorted({key for row in rows for key in row}))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in columns})


def sha256(path: Path) -> str | None:
    path = Path(path)
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def module_from(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise AbError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    # Registered before execution: a dataclass in the loaded module resolves its
    # own annotations through sys.modules, and fails without this.
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_CACHE: dict[str, Any] = {}


def exp06_module():
    if "exp06" not in _CACHE:
        _CACHE["exp06"] = module_from(REPO_ROOT / "scripts" / "groot-reset-pose-ab.py", "reset_pose_ab")
    return _CACHE["exp06"]


def exp01_module():
    if "exp01" not in _CACHE:
        _CACHE["exp01"] = module_from(REPO_ROOT / "scripts" / "demo-token-decoder-replay.py",
                                      "demo_token_replay")
    return _CACHE["exp01"]


def exp05_module():
    if "exp05" not in _CACHE:
        _CACHE["exp05"] = module_from(REPO_ROOT / "scripts" / "policy-token-support.py",
                                      "policy_token_support")
    return _CACHE["exp05"]


def calibration():
    if "calibration" not in _CACHE:
        _CACHE["calibration"] = exp06_module().Calibration()
    return _CACHE["calibration"]


def urdf():
    if "urdf" not in _CACHE:
        compare = exp06_module().comparison_module()
        _CACHE["urdf"] = compare.Urdf(REPO_ROOT / "third_party" / "Psi0" / "real" / "assets" / "g1"
                                      / "g1_body29_hand14.urdf")
    return _CACHE["urdf"]


def quaternion_wxyz_to_matrix(quaternion: Sequence[float]) -> np.ndarray:
    return exp06_module().comparison_module().quaternion_wxyz_to_matrix(quaternion)


def palms(q_body: np.ndarray) -> dict[str, np.ndarray]:
    """Palm positions in the pelvis frame, from canonical body joint values."""
    return urdf().palms(np.atleast_2d(np.asarray(q_body, dtype=float)))


def base_transforms(root_quat: np.ndarray, root: np.ndarray) -> np.ndarray:
    base = np.broadcast_to(np.eye(4), (len(root), 4, 4)).copy()
    for index, quaternion in enumerate(root_quat):
        base[index, :3, :3] = quaternion_wxyz_to_matrix(quaternion)
    base[:, :3, 3] = root
    return base


def scene_constants() -> dict[str, Any]:
    profile = read_json(PROFILE)
    scene = profile["scene"]
    return {
        "table_surface_m": float(scene["table"]["surface_height_m"]),
        "table_position_m": list(scene["table"]["position_m"]),
        "cube_size_m": float(scene["cubes"][0]["size_m"][0]),
        "cubes": {cube["color"]: list(cube["position_m"]) for cube in scene["cubes"]},
        "initial_root_z_m": float(profile["robot"]["initial_position_m"][2]),
    }


# ------------------------------------------------------------ start phase


def start_phase_window(episode: int) -> dict[str, Any]:
    """One episode's quiet start window, by experiment 06's own rule.

    The window is the frames before the first 5 consecutive 30 Hz samples whose
    largest arm+hand joint speed exceeds 0.30 rad/s, capped at 1.0 s and floored
    at 3 frames; the episode's start state is the median over it.
    """
    module = exp06_module()
    cal = calibration()
    series = np.asarray(
        cal.read_parquet(cal.episode_path(DATASET_ROOT, episode), ["observation.state"])["observation.state"],
        dtype=float,
    )
    dt = 1.0 / float(read_json(DATASET_ROOT / "meta" / "info.json")["fps"])
    speed = np.zeros(series.shape[0])
    speed[1:] = np.abs(np.diff(series[:, module.ARM_HAND], axis=0)).max(axis=1) / dt
    onset = None
    need = module.START_MOTION_SUSTAIN
    for index in range(series.shape[0] - need):
        if (speed[index:index + need] > module.START_MOTION_RAD_S).all():
            onset = index
            break
    cap = max(module.START_WINDOW_MIN_FRAMES,
              min(int(round(module.START_WINDOW_MAX_S / dt)), series.shape[0]))
    stop = series.shape[0] if onset is None else min(max(onset, module.START_WINDOW_MIN_FRAMES), cap)
    quiet = series[:stop]
    state = np.median(quiet, axis=0)
    return {
        "episode_index": int(episode),
        "frames_total": int(series.shape[0]),
        "onset_frame": None if onset is None else int(onset),
        "window_frames": int(quiet.shape[0]),
        "window_seconds": round(quiet.shape[0] * dt, 4),
        "max_arm_hand_speed_rad_s": float(speed[:stop].max()),
        "start_state": state,
    }


def scene_clearance(state: np.ndarray, depth: float = 0.0) -> dict[str, Any]:
    """Clearance of the demonstrated start pose against this scene.

    The demonstration records no base pose, so the comparison is made in the
    robot's own base frame against the profile's declared worktop and cubes: the
    table surface and the cube tops are shifted by the robot's initial height and
    the palm links are read from the same URDF the palm metrics use.  It is a
    lower bound on the real clearance -- only the palm link is measured -- and it
    is reported as such, not as a collision test.
    """
    scene = scene_constants()
    root_z = scene["initial_root_z_m"]
    palm = palms(state[:29])
    out: dict[str, Any] = {"table_surface_base_frame_m": scene["table_surface_m"] - root_z,
                           "cube_top_base_frame_m": scene["table_surface_m"] + scene["cube_size_m"] - root_z,
                           "palms_base_frame_m": {}}
    for side, position in palm.items():
        point = np.asarray(position[0], dtype=float)
        out["palms_base_frame_m"][side] = [float(v) for v in point]
        out[f"palm_{side}_above_table_top_m"] = float(
            point[2] - (scene["table_surface_m"] - root_z))
        out[f"palm_{side}_above_cube_top_m"] = float(
            point[2] - (scene["table_surface_m"] + scene["cube_size_m"] - root_z))
        over_table = (scene["table_position_m"][0] - 0.39 <= point[0] <= 1.0006) and abs(point[1]) <= 1.2368
        out[f"palm_{side}_over_table_footprint"] = bool(over_table)
    return out


def build_token_stream(episode: int, window: dict[str, Any]) -> dict[str, Any]:
    """The prepared stream: the window's tokens and hand command at 50 Hz.

    Experiment 01's episode reader owns the conversion (30 Hz demonstration ->
    the deployment's 50 Hz control grid by holding each token, which is what the
    live bridge does) and already checks the tokens are on the WBC FSQ grid and
    that the episode is task-0.
    """
    replay = exp01_module()
    # Experiment 01's reader takes the dataset root that *contains* the psi0
    # train copy, not the copy itself.
    demo = replay.read_demo_episode(DATASET_COPY_ROOT, episode, f"episode{episode}", raw_root=None)
    frames = int(window["window_frames"])
    ticks = demo.ticks[demo.ticks <= (frames - 1) / 30.0 + 1e-9]
    if ticks.size == 0:
        raise AbError(f"episode {episode}: the start window has no control ticks")
    index = np.clip(np.searchsorted(demo.timestamp, ticks, side="right") - 1, 0,
                    demo.tokens.shape[0] - 1)
    tokens = demo.tokens[index]
    hands = demo.hand_action[index]
    left = np.asarray(hands[:, 0:7], dtype=float)
    right = np.asarray(hands[:, 7:14], dtype=float)
    if not np.allclose(tokens, replay.fsq_quantize(tokens), atol=1e-6):
        raise AbError(f"episode {episode}: the selected tokens are off the FSQ grid")
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "groot-token-warmstart",
        "control_hz": float(replay.CONTROL_HZ),
        "source": {
            "dataset": rel(DATASET_ROOT),
            "episode_index": int(episode),
            "source_episode_index": int(demo.source_episode),
            "start_window_frames_30hz": [0, frames],
            "start_window_seconds": float(window["window_seconds"]),
            "resampling": "hold: each 30 Hz token spans the ticks the deployment's 50 Hz grid "
                          "gives it (experiment 01's own conversion)",
            "episode_sha256": demo.source_sha256,
            "validated_by": "experiments/01-demo-token-decoder (0.05 m palm-z gate, on plant)",
        },
        "tokens": np.asarray(tokens, dtype=float).tolist(),
        "left_hand_joints": left.tolist(),
        "right_hand_joints": right.tolist(),
    }


# ---------------------------------------------------------------- select


def stage_select(args: argparse.Namespace) -> dict[str, Any]:
    out = repo_path(args.output_dir) if args.output_dir else DEFAULT_OUT
    (out / "tables").mkdir(parents=True, exist_ok=True)
    summary01 = read_json(EXP01 / "summary.json")
    validated = sorted(set(summary01["reproduced_episodes_plant_projected"])
                       & set(summary01["moving_episodes"]))
    if not validated:
        raise AbError("experiment 01 reported no validated moving demonstration")
    candidates: list[dict[str, Any]] = []
    for episode in sorted(summary01["episodes"]):
        episode_index = int(episode)
        window = start_phase_window(episode_index)
        state = np.asarray(window.pop("start_state"), dtype=float)
        support = {row["group"]: row for row in calibration().support(state[None, :])}
        palm = palms(state[:29])
        clearance = scene_clearance(state)
        record = {
            "episode_index": episode_index,
            "label": summary01["episodes"][episode].get("label"),
            "frames_valid": int(summary01["episodes"][episode]["frames_valid"]),
            "validated_by_exp01": bool(episode_index in validated),
            "exp01_plant_palm_z_mae_left_m": float(
                summary01["episodes"][episode]["primary_detail"]["palms"]["left"]
                ["palm_z_mae_m_at_optimal_lag"]),
            "exp01_plant_palm_z_mae_right_m": float(
                summary01["episodes"][episode]["primary_detail"]["palms"]["right"]
                ["palm_z_mae_m_at_optimal_lag"]),
            "exp01_gate_m": float(summary01["gate_m"]),
            **window,
            "arms_mahalanobis": float(support["arms"]["mahalanobis"]),
            "arms_mahalanobis_loeo_p95": float(support["arms"]["mahalanobis_loeo_p95"]),
            "arms_inside_p95": bool(support["arms"]["inside_p95_mahalanobis"]),
            "upper_body_mahalanobis": float(support["upper_body"]["mahalanobis"]),
            "hands_mahalanobis": float(support["hands"]["mahalanobis"]),
            "channels_inside_envelope": int(calibration().envelope_check(state)["channels_inside_p1_p99"]),
            "channels_total": int(calibration().envelope_check(state)["channels_total"]),
            "palm_left_z_m": float(palm["left"][0][2]),
            "palm_right_z_m": float(palm["right"][0][2]),
            **{key: value for key, value in clearance.items()
               if key.startswith("palm_") and key.endswith(("table_top_m", "cube_top_m"))},
        }
        candidates.append(record)

    # The rule, fixed before the choice: among the demonstrations experiment 01's
    # replay validated (the 0.05 m palm-z gate on the moving set), take the best
    # agreement; a tie is broken by the quieter start window, then by the smaller
    # arm Mahalanobis distance.
    pool = [row for row in candidates if row["validated_by_exp01"]]
    if not pool:
        raise AbError("no candidate is validated by experiment 01")
    pool.sort(key=lambda row: (
        max(row["exp01_plant_palm_z_mae_left_m"], row["exp01_plant_palm_z_mae_right_m"]),
        row["max_arm_hand_speed_rad_s"], row["arms_mahalanobis"]))
    chosen = pool[0]
    if not chosen["arms_inside_p95"]:
        raise AbError(f"the chosen segment's start state is not inside the demonstration p95: {chosen}")
    stream = build_token_stream(chosen["episode_index"], chosen)
    stream_path = out / "tokens.json"
    write_json(stream_path, stream)

    selection = {
        "question": "which task-0 demonstration start phase seeds the SONIC warm start?",
        "rule": ("among the demonstrations experiment 01's replay validated (gate %.2f m on the "
                 "plant-projected palm-z reading of the moving set), the smallest worst-hand "
                 "palm-z MAE; ties by the quieter start window, then by the smaller arm "
                 "Mahalanobis distance") % float(summary01["gate_m"]),
        "exp01_gate_m": float(summary01["gate_m"]),
        "exp01_validated_episodes": validated,
        "chosen_episode_index": int(chosen["episode_index"]),
        "chosen_label": chosen["label"],
        "chosen_start_window_frames_30hz": [0, int(chosen["window_frames"])],
        "chosen_start_window_seconds": float(chosen["window_seconds"]),
        "chosen_max_arm_hand_speed_rad_s": float(chosen["max_arm_hand_speed_rad_s"]),
        "chosen_arms_mahalanobis": float(chosen["arms_mahalanobis"]),
        "chosen_palm_z_m": {"left": float(chosen["palm_left_z_m"]),
                            "right": float(chosen["palm_right_z_m"])},
        "stream": {
            "path": rel(stream_path),
            "sha256": sha256(stream_path),
            "ticks": int(len(stream["tokens"])),
            "control_hz": float(stream["control_hz"]),
            "duration_s": float(len(stream["tokens"]) / stream["control_hz"]),
        },
        "candidates": [{key: value for key, value in row.items()} for row in candidates],
    }
    write_json(out / "tables" / "segment_selection.json", selection)
    write_csv(out / "tables" / "segment_candidates.csv", candidates)
    print(f"[select] chose episode {chosen['episode_index']} ({chosen['label']}): "
          f"{selection['stream']['ticks']} ticks at {selection['stream']['control_hz']:g} Hz "
          f"({selection['stream']['duration_s']:.2f}s), arms Mahalanobis "
          f"{chosen['arms_mahalanobis']:.2f} < p95 {chosen['arms_mahalanobis_loeo_p95']:.2f}")
    print(f"[select] token stream: {rel(stream_path)} sha256 {selection['stream']['sha256'][:16]}")
    return selection


# ------------------------------------------------------------------ runs


def rollout_dirs(cell: str, out: Path) -> list[Path]:
    directory = out / "runs" / cell / "rollouts"
    return sorted(directory.glob("rollout-*")) if directory.is_dir() else []


def session_dir(cell: str, out: Path) -> Path:
    return out / "runs" / cell


def read_scene_cubes(session: Path, rollout: Path) -> dict[str, tuple[float, float, float]]:
    samples = load_jsonl(rollout / "isaac.samples.jsonl") or load_jsonl(session / "raw" / "isaac.samples.jsonl")
    for row in samples:
        cubes = (row.get("scene") or {}).get("cubes") or {}
        if cubes:
            return {name: tuple(float(value) for value in entry["live_center_xyz_m"])
                    for name, entry in cubes.items()}
    return {}


def scene_cube_samples(path: Path) -> list[dict[str, Any]]:
    """The live scene probe, one entry per sampled ``palm_height`` event.

    Every sample event carries the live cube centres, so a cube that moves during
    the pre-policy settle is visible without waiting for the campaign crop (whose
    stack block only covers the policy window).
    """
    rows: list[dict[str, Any]] = []
    for row in load_jsonl(path):
        cubes = (row.get("scene") or {}).get("cubes") or {}
        if cubes:
            rows.append({"wall_time_ns": int(row.get("wall_time_ns", 0)),
                         "cubes": {name: list(entry["live_center_xyz_m"])
                                   for name, entry in cubes.items()}})
    rows.sort(key=lambda entry: entry["wall_time_ns"])
    return rows


def load_tracking_series(rows: Sequence[dict[str, Any]], start_wall: float) -> dict[str, np.ndarray]:
    """The arrays every window metric needs, from one tracking record stream.

    Used twice per rollout: once for the campaign's crop (the policy window) and
    once for the session's own stream, which is the only record that contains the
    whole pre-Start settle the warm start runs inside.
    """
    wall = np.array([row["wall_time_ns"] / 1e9 for row in rows])
    target = np.array([row["body_target"] for row in rows], dtype=float)
    measured = np.array([row["body_measured"] for row in rows], dtype=float)
    series = {
        "wall": wall,
        "sim": np.array([row["sim_s"] for row in rows]),
        "target": target,
        "measured": measured,
        "measured_velocity": np.array([row.get("body_measured_velocity") or [0.0] * 29 for row in rows],
                                      dtype=float),
        "hand_target": np.array([list(row["left_hand_target"]) + list(row["right_hand_target"])
                                 for row in rows], dtype=float),
        "hand_measured": np.array([list(row["left_hand_measured"]) + list(row["right_hand_measured"])
                                   for row in rows], dtype=float),
        "hand_measured_velocity": np.array(
            [list(row.get("left_hand_measured_velocity") or [0.0] * 7)
             + list(row.get("right_hand_measured_velocity") or [0.0] * 7) for row in rows], dtype=float),
        "root": np.array([row["root_position"] for row in rows], dtype=float),
        "root_quat": np.array([row["root_quaternion_wxyz"] for row in rows], dtype=float),
    }
    series["relative"] = series["wall"] - start_wall
    series["target_palm"] = palms(series["target"])
    series["measured_palm"] = palms(series["measured"])
    series["state_43d"] = np.hstack([series["measured"], series["hand_measured"]])
    return series


class Rollout:
    """One recorded rollout, read at t=0 == the campaign's ``Start`` call."""

    def __init__(self, cell: str, out: Path, directory: Path) -> None:
        self.cell = cell
        self.out = out
        self.directory = directory
        self.index = int(directory.name.split("-")[-1])
        self.label = f"{cell}{self.index}"
        self.session = session_dir(cell, out)
        self.meta = read_json(directory / "rollout.json")
        self.session_json = read_json(self.session / "session.json")
        entry = self.session_json["rollouts"][self.index - 1]
        self.start_wall = float(entry["start_wall_ns"]) / 1e9
        self.reset_wall = float(entry.get("reset_wall_ns", entry["start_wall_ns"])) / 1e9
        self.stop_wall = float(entry.get("stop_wall_ns", entry["start_wall_ns"])) / 1e9
        self.rows = load_jsonl(directory / "tracking.jsonl")
        if not self.rows:
            raise AbError(f"{directory} has no tracking rows")
        # The campaign's crop (the policy window) and the session's own stream
        # (which also holds the pre-Start settle the warm start runs inside).
        self.crop = load_tracking_series(self.rows, self.start_wall)
        session_rows = load_jsonl(self.session / "raw" / "isaac.tracking.jsonl")
        if not session_rows:
            raise AbError(f"{self.session} has no session tracking stream")
        self.settle = load_tracking_series(
            [row for row in session_rows
             if self.start_wall - SETTLE_LOOKBACK_S <= row["wall_time_ns"] / 1e9 <= self.stop_wall + 0.5],
            self.start_wall,
        )
        # The state at Policy Start itself: the last session frame at or before
        # the campaign's Start call (the crop starts a frame or two after it).
        self.start_settle_index = int(
            np.searchsorted(self.settle["wall"], self.start_wall, side="right") - 1)
        self.start_state_43d = self.settle["state_43d"][self.start_settle_index]
        self.wall = self.crop["wall"]
        self.sim = self.crop["sim"]
        self.target = self.crop["target"]
        self.measured = self.crop["measured"]
        self.measured_velocity = self.crop["measured_velocity"]
        self.hand_target = self.crop["hand_target"]
        self.hand_measured = self.crop["hand_measured"]
        self.hand_measured_velocity = self.crop["hand_measured_velocity"]
        self.root = self.crop["root"]
        self.root_quat = self.crop["root_quat"]
        self.relative = self.crop["relative"]
        self.start_index = int(np.searchsorted(self.wall, self.start_wall, side="right") - 1)
        self.onset_wall = float(self.meta["onset"]["wall_time"])
        self.onset_index = int(np.argmax(self.wall >= self.onset_wall))
        self.telemetry = load_jsonl(directory / "bridge-telemetry.jsonl")
        self.session_telemetry = load_jsonl(self.session / "raw" / "telemetry" / "bridge-telemetry.jsonl")
        self.cubes = read_scene_cubes(self.session, directory)
        self.cube_samples = [row for row in scene_cube_samples(self.session / "raw" / "isaac.samples.jsonl")
                             if row["wall_time_ns"] / 1e9 <= self.stop_wall + 0.5]
        self.urdf = urdf()
        self.base = base_transforms(self.root_quat, self.root)
        self.target_palm = self.crop["target_palm"]
        self.measured_palm = self.crop["measured_palm"]
        self.state_43d = self.crop["state_43d"]
        self.target_43d = np.hstack([self.target, self.hand_target])
        self.support = calibration().support(self.state_43d)
        self.warmstart = self._warmstart_events()

    # -- windows -----------------------------------------------------------

    def _warmstart_events(self) -> dict[str, Any]:
        events = [row for row in self.session_telemetry
                  if row.get("kind") == "warmstart"
                  and self.reset_wall - 1.0 <= row["wall_time_ns"] / 1e9 <= self.start_wall + 1.0]
        return {
            "events": [
                {"state": row.get("state"), "wall_time": row["wall_time_ns"] / 1e9,
                 "ticks": row.get("ticks"), "sent": row.get("sent"),
                 "delay_s": row.get("delay_s"), "sha256": row.get("sha256"),
                 "episode_index": (row.get("source") or {}).get("episode_index")}
                for row in events
            ],
            "started_wall": next((row["wall_time_ns"] / 1e9 for row in events
                                  if row.get("state") == "started"), None),
            "halted_wall": next((row["wall_time_ns"] / 1e9 for row in reversed(events)
                                 if row.get("state") == "halted"), None),
        }

    def window(self, name: str) -> np.ndarray:
        low, high = WINDOWS_S[name]
        mask = self.relative >= low
        if high is not None:
            mask &= self.relative < high
        if not mask.any():
            mask = np.zeros_like(self.relative, dtype=bool)
            mask[min(self.start_index, mask.size - 1)] = True
        return mask

    def settle_tail(self, seconds: float = WARMSTART_WINDOW_S) -> dict[str, Any]:
        """The window the treatment occupies: the last ``seconds`` before Start.

        Read from the session's own tracking stream, because that is the only
        record that contains the whole window.  The flat series are returned
        sliced; the two per-side palm dictionaries are sliced per side.
        """
        mask = (self.settle["relative"] >= -seconds) & (self.settle["relative"] < 0.0)
        out: dict[str, Any] = {"mask": mask}
        for key, value in self.settle.items():
            if key in ("target_palm", "measured_palm"):
                out[key] = {side: positions[mask] for side, positions in value.items()}
            else:
                out[key] = value[mask]
        return out

    def calm_tail(self, seconds: float = CALM_TAIL_S) -> np.ndarray:
        return (self.relative >= -seconds) & (self.relative < 0.0)

    # -- derived series ----------------------------------------------------

    def cube_distance(self, side: str, mask: np.ndarray) -> np.ndarray:
        palm = self.target_palm[side][mask]
        out = None
        for centre in self.cubes.values():
            rotated = np.array([
                quaternion_wxyz_to_matrix(quaternion).T @ (np.asarray(centre) - position)
                for quaternion, position in zip(self.root_quat[mask], self.root[mask])
            ])
            distance = np.linalg.norm(palm - rotated, axis=1)
            out = distance if out is None else np.minimum(out, distance)
        return out if out is not None else np.zeros(int(mask.sum()))

    def measured_cube_distance(self, side: str, mask: np.ndarray) -> np.ndarray:
        palm = self.measured_palm[side][mask]
        out = None
        for centre in self.cubes.values():
            rotated = np.array([
                quaternion_wxyz_to_matrix(quaternion).T @ (np.asarray(centre) - position)
                for quaternion, position in zip(self.root_quat[mask], self.root[mask])
            ])
            distance = np.linalg.norm(palm - rotated, axis=1)
            out = distance if out is None else np.minimum(out, distance)
        return out if out is not None else np.zeros(int(mask.sum()))

    def tokens(self, source: str | None = None, *, low: float | None = None,
               high: float | None = None, stream: str = "session") -> list[dict[str, Any]]:
        """Applied ``pose`` messages, filtered by source and by wall time.

        The settle the warm start runs inside is only in the *session's* stream
        (the per-rollout telemetry is the campaign's crop, which starts one second
        before the onset), so the session stream is the default and the crop is
        available for the cross-check that the policy's own tokens are the ones
        the campaign recorded.
        """
        out: list[dict[str, Any]] = []
        rows = self.session_telemetry if stream == "session" else self.telemetry
        for row in rows:
            if row.get("kind") != "applied_action":
                continue
            if source is not None and row.get("source") != source:
                continue
            wall = row["wall_time_ns"] / 1e9
            if low is not None and wall < low:
                continue
            if high is not None and wall >= high:
                continue
            fields = row.get("fields") or {}
            token = fields.get("token_state")
            if not token:
                continue
            index = fields.get("frame_index")
            index = int(index[0]) if isinstance(index, (list, tuple)) and index else int(index or 0)
            out.append({"wall_time": wall, "relative": wall - self.start_wall,
                        "token": [float(value) for value in token[0]], "frame_index": index,
                        "source": row.get("source")})
        return out

    def group_support(self, mask: np.ndarray, group: str) -> dict[str, Any]:
        rows = [row for row in np.asarray(self.support, dtype=object)]
        values = np.array([row["mahalanobis"] for row in rows if row["group"] == group])
        p95 = np.array([row["mahalanobis_loeo_p95"] for row in rows if row["group"] == group])
        p99 = np.array([row["mahalanobis_loeo_p99"] for row in rows if row["group"] == group])
        percentiles = np.array([row["mahalanobis_loeo_percentile"] for row in rows if row["group"] == group])
        values, p95, p99, percentiles = values[mask], p95[mask], p99[mask], percentiles[mask]
        return {
            "frames": int(values.size),
            "mahalanobis_median": float(np.median(values)),
            "mahalanobis_loeo_percentile_median": float(np.median(percentiles)),
            "in_support_at_p95_frac": float(np.mean(values <= p95)),
            "in_support_at_p99_frac": float(np.mean(values <= p99)),
            "second_support_p99": float(np.median(p99)),
        }

    def support_of(self, state_43d: np.ndarray) -> dict[str, dict[str, Any]]:
        """Per-group support of one 43D state (the start state, at Policy Start)."""
        return {row["group"]: row for row in calibration().support(np.asarray(state_43d)[None, :])}

    def start_support(self) -> dict[str, dict[str, Any]]:
        return self.support_of(self.start_state_43d)


# ----------------------------------------------------------------- gates


def rollout_gates(rollout: Rollout, *, baseline_jump: float | None = None) -> dict[str, Any]:
    """The pre-declared validity gates, each with the value it was compared against.

    The settle-side gates read the *session's* tracking stream, which is the only
    record that contains the whole pre-Start window the warm start runs inside;
    the policy-side gates read the campaign's own crop.
    """
    limits = GATE_LIMITS
    start_support = rollout.start_support()["arms"]
    settle = rollout.settle
    relative = settle["relative"]
    calm = (relative >= -CALM_TAIL_S) & (relative < 0.0)
    joint_speed = np.abs(settle["measured_velocity"][calm][:, ARM_HAND]).max(axis=1) \
        if calm.any() else np.zeros(0)
    palm_speed = np.concatenate([
        _finite_difference_speed(settle["measured_palm"][side], settle["wall"])[calm]
        for side in ("left", "right")
    ]) if calm.any() else np.zeros(0)
    pelvis = settle["root"][calm]
    calm_indices = np.flatnonzero(calm)
    reference_index = max(0, int(calm_indices[0]) - 1) if calm_indices.size else 0
    pelvis_reference = settle["root"][reference_index]
    pelvis_drift = float(np.abs(pelvis[:, 2] - pelvis_reference[2]).max()) if calm.any() else float("nan")
    pelvis_xy = float(np.linalg.norm(pelvis[:, :2] - pelvis_reference[:2], axis=1).max()) if calm.any() else float("nan")

    # Cube displacement and scene sanity.  The window the treatment occupies is
    # the settle tail; the whole session is checked for a cube that fell through
    # the worktop or was corrupted by the intervention.
    window_cubes = [row for row in rollout.cube_samples
                    if -WARMSTART_WINDOW_S <= (row["wall_time_ns"] / 1e9 - rollout.start_wall) < 0.0]
    table = scene_constants()["table_surface_m"]
    first = window_cubes[0]["cubes"] if window_cubes else {}
    last = window_cubes[-1]["cubes"] if window_cubes else {}
    displacement = {name: float(np.linalg.norm(np.asarray(last[name])[:2] - np.asarray(first[name])[:2]))
                    for name in first if name in last}
    minimum_cube_z = min((min(np.asarray(entry)[2] for entry in row["cubes"].values())
                          for row in rollout.cube_samples), default=float("nan"))

    # The hand-off: how far the policy's own first commanded targets move away
    # from the target the settle left the controller holding.  Both sides come
    # from the session stream -- the campaign's crop starts a frame or two after
    # Start, so it cannot show the crossing.
    pre = rollout.start_settle_index
    held = settle["target"][pre]
    transition = (relative > 0.0) & (relative < 0.5)
    target_jump = float(np.abs(settle["target"][transition] - held[None, :]).max()) \
        if transition.any() else 0.0
    finite = bool(np.isfinite(rollout.target).all() and np.isfinite(rollout.measured).all()
                  and np.isfinite(settle["target"]).all() and np.isfinite(settle["measured"]).all())
    fall = bool(rollout.meta.get("fall", {}).get("detected"))

    gates = {
        "G1_start_arms_in_demo_p95": {
            "value": float(start_support["mahalanobis"]),
            "limit": float(limits["start_arms_mahalanobis_p95"]),
            "passed": bool(start_support["mahalanobis"] <= limits["start_arms_mahalanobis_p95"]),
            "loeo_percentile": float(start_support["mahalanobis_loeo_percentile"]),
            "loeo_p99": float(start_support["mahalanobis_loeo_p99"]),
        },
        "G2_calm_at_warmstart_end": {
            "frames": int(calm.sum()),
            "joint_speed_p95_rad_s": float(np.percentile(joint_speed, 95)) if joint_speed.size else float("nan"),
            "joint_speed_max_rad_s": float(joint_speed.max()) if joint_speed.size else float("nan"),
            "palm_speed_p95_m_s": float(np.percentile(palm_speed, 95)) if palm_speed.size else float("nan"),
            "limits": {"joint_speed_p95_rad_s": limits["calm_joint_speed_p95_rad_s"],
                       "joint_speed_max_rad_s": limits["calm_joint_speed_max_rad_s"],
                       "palm_speed_p95_m_s": limits["calm_palm_speed_p95_m_s"]},
            "passed": bool(
                joint_speed.size
                and np.percentile(joint_speed, 95) <= limits["calm_joint_speed_p95_rad_s"]
                and joint_speed.max() <= limits["calm_joint_speed_max_rad_s"]
                and np.percentile(palm_speed, 95) <= limits["calm_palm_speed_p95_m_s"]
            ),
        },
        "G3_pelvis_stable": {
            "height_drift_m": pelvis_drift,
            "height_min_m": float(pelvis[:, 2].min()) if calm.any() else float("nan"),
            "xy_drift_m": pelvis_xy,
            "limits": {"height_drift_m": limits["pelvis_height_drift_max_m"],
                       "height_min_m": limits["pelvis_height_min_m"],
                       "xy_drift_m": limits["pelvis_xy_drift_max_m"]},
            "passed": bool(
                calm.any() and pelvis_drift <= limits["pelvis_height_drift_max_m"]
                and float(pelvis[:, 2].min()) >= limits["pelvis_height_min_m"]
                and pelvis_xy <= limits["pelvis_xy_drift_max_m"]
            ),
        },
        "G4_scene_intact": {
            "cube_displacement_m": displacement,
            "max_cube_displacement_m": float(max(displacement.values())) if displacement else float("nan"),
            "minimum_cube_centre_z_m": minimum_cube_z,
            "table_surface_m": table,
            "limit": limits["cube_displacement_max_m"],
            "cube_probe_samples_in_window": len(window_cubes),
            "passed": bool(displacement and max(displacement.values()) <= limits["cube_displacement_max_m"]
                           and minimum_cube_z >= table - 0.01),
        },
        "G5_handoff_bounded": {
            "target_jump_rad": target_jump,
            "baseline_max_jump_rad": baseline_jump,
            "limit_rad": limits["transition_target_jump_ceiling_rad"],
            "sim_finite": finite,
            "fall": fall,
            "passed": bool(target_jump <= limits["transition_target_jump_ceiling_rad"] and finite and not fall
                           and (baseline_jump is None or target_jump <= baseline_jump + 0.25)),
        },
        "G6_token_hygiene": {
            "warmstart_tokens_in_settle": int(len(rollout.tokens(
                "warmstart", low=rollout.start_wall - WARMSTART_WINDOW_S, high=rollout.start_wall))),
            "policy_tokens_after_start": int(len(rollout.tokens(
                "groot", low=rollout.start_wall, high=rollout.stop_wall))),
            # Everything before the policy's own first token is still the token the
            # deployment was left holding; from that first policy token onwards no
            # other source may appear, or the policy output would be mixed.
            "policy_token_relative_s": policy_first_relative(rollout),
            "non_policy_tokens_after_policy_start": int(len([
                row for row in rollout.tokens(
                    low=(rollout.start_wall + policy_first_relative(rollout))
                    if policy_first_relative(rollout) is not None else rollout.start_wall,
                    high=rollout.stop_wall)
                if row["source"] != "groot"])),
            "other_source_tokens_in_settle": int(len([
                row for row in rollout.tokens(low=rollout.start_wall - WARMSTART_WINDOW_S,
                                              high=rollout.start_wall)
                if row["source"] != "warmstart"])),
            "passed": bool(
                int(len(rollout.tokens("groot", low=rollout.start_wall, high=rollout.stop_wall))) > 0
                and not [
                    row for row in rollout.tokens(
                        low=(rollout.start_wall + (policy_first_relative(rollout) or 0.0)),
                        high=rollout.stop_wall)
                    if row["source"] != "groot"]
                and not [row for row in rollout.tokens(
                    low=rollout.start_wall - WARMSTART_WINDOW_S, high=rollout.start_wall)
                    if row["source"] != "warmstart"]
            ),
        },
    }
    gates["integrity_passed"] = all(gates[key]["passed"] for key in
                                    ("G2_calm_at_warmstart_end", "G3_pelvis_stable", "G4_scene_intact",
                                     "G5_handoff_bounded", "G6_token_hygiene"))
    gates["treatment_state_fixed"] = bool(gates["G1_start_arms_in_demo_p95"]["passed"])
    gates["passed"] = bool(gates["integrity_passed"] and gates["treatment_state_fixed"])
    return gates


def policy_first_relative(rollout: "Rollout") -> float | None:
    """Seconds from Start to the first GR00T-sourced applied token."""
    tokens = rollout.tokens("groot", low=rollout.start_wall, high=rollout.stop_wall)
    return float(tokens[0]["relative"]) if tokens else None


def _finite_difference_speed(values: np.ndarray, wall: np.ndarray) -> np.ndarray:
    speed = np.zeros(len(values))
    if len(values) > 1:
        dt = np.diff(wall)
        dt[dt <= 0] = 1e-3
        speed[1:] = np.linalg.norm(np.diff(values, axis=0), axis=1) / dt
    return speed


# ---------------------------------------------------------------- metrics


def token_support_distances(tokens: Sequence[Sequence[float]]) -> dict[str, Any]:
    """Experiment 05's nearest-neighbour value L1 distance to the task-0 cloud."""
    if not tokens:
        return {"frames": 0}
    module = exp05_module()
    cloud = token_cloud()
    values = module._token_distance(np.asarray(tokens, dtype=np.float32), cloud,
                                    TOKEN_SUPPORT_METRIC)
    thresholds = token_support_thresholds()
    return {
        "frames": int(values.size),
        "metric": TOKEN_SUPPORT_METRIC,
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "loeo_median": thresholds["loeo_median"],
        "loeo_p95": thresholds["loeo_p95"],
        "loeo_p99": thresholds["loeo_p99"],
        "heldout_median": thresholds["heldout_median"],
        "ratio_to_loeo_p99": float(np.median(values) / thresholds["loeo_p99"]),
    }


def token_cloud() -> np.ndarray:
    if "token_cloud" not in _CACHE:
        module = exp05_module()
        episodes = module.blockstacking_episodes(DATASET_ROOT)
        tokens, owners, _ = module.load_demo_pool(DATASET_ROOT, episodes)
        cloud, _ = module.balanced_cloud(tokens, owners, [int(entry["episode_index"]) for entry in episodes],
                                         TOKEN_CLOUD_FRAMES_PER_EPISODE, TOKEN_CLOUD_SEED)
        _CACHE["token_cloud"] = cloud
    return _CACHE["token_cloud"]


def token_support_thresholds() -> dict[str, Any]:
    """Experiment 05's own reference numbers for the token distance metric.

    ``thresholds.q95/q99`` are that experiment's leave-one-episode-out quantiles
    of the demonstration cloud's own frames; ``heldout_baseline`` is the same
    metric measured on the val copy's episodes, which are in neither the cloud
    nor the calibration.
    """
    if "token_thresholds" not in _CACHE:
        calibration_json = read_json(EXP05 / "tables" / "loeo_calibration.json")
        loeo = calibration_json["knn_loeo"][TOKEN_SUPPORT_METRIC]
        heldout = read_json(EXP05 / "tables" / "heldout_baseline.json").get(
            TOKEN_SUPPORT_METRIC, {})
        _CACHE["token_thresholds"] = {
            "loeo_median": float(loeo["median"]),
            "loeo_p95": float(calibration_json["thresholds"]["q95"][TOKEN_SUPPORT_METRIC]),
            "loeo_p99": float(calibration_json["thresholds"]["q99"][TOKEN_SUPPORT_METRIC]),
            "heldout_median": None if not heldout else float(heldout["median"]),
            "heldout_p99": None if not heldout else float(heldout["p99"]),
            "source": rel(EXP05 / "tables" / "loeo_calibration.json"),
        }
    return _CACHE["token_thresholds"]


def rollout_metrics(rollout: Rollout) -> dict[str, Any]:
    """Policy-output and measured metrics, kept apart, per window."""
    record: dict[str, Any] = {
        "label": rollout.label,
        "cell": rollout.cell,
        "rollout": rollout.index,
        "directory": rel(rollout.directory),
        "session": rel(rollout.session),
        "start_wall_time": rollout.start_wall,
        "reset_wall_time": rollout.reset_wall,
        "onset_seconds_after_start": float(rollout.meta["onset"]["wall_time"] - rollout.start_wall),
        "onset_source": rollout.meta["onset"].get("chosen_source"),
        "rollout_seconds": float(rollout.wall[-1] - rollout.start_wall),
        "applied_actions": int(rollout.meta.get("applied_actions") or 0),
        "fall": bool(rollout.meta.get("fall", {}).get("detected")),
        "cubes_lifted": bool(rollout.meta.get("stack", {}).get("any_cube_lifted")),
        "cube_displacement_xy_m": max(
            rollout.meta.get("stack", {}).get("cube_pose_w", {}).get(colour, {}).get("displacement_xy_m", 0.0)
            for colour in ("red", "yellow", "blue")),
        "warmstart": rollout.warmstart,
        "windows": {},
    }
    for name in WINDOWS_S:
        mask = rollout.window(name)
        entry: dict[str, Any] = {
            "frames": int(mask.sum()),
            "seconds": float(WINDOWS_S[name][1] if WINDOWS_S[name][1] else
                             max(0.0, rollout.relative[-1])),
        }
        for side in ("left", "right"):
            z = rollout.target_palm[side][mask][:, 2]
            distance = rollout.cube_distance(side, mask)
            measured_z = rollout.measured_palm[side][mask][:, 2]
            measured_distance = rollout.measured_cube_distance(side, mask)
            entry[f"target_palm_z_{side}_median_m"] = float(np.median(z))
            entry[f"target_palm_z_{side}_mean_m"] = float(z.mean())
            entry[f"target_palm_z_{side}_min_m"] = float(z.min())
            entry[f"target_palm_z_{side}_first_m"] = float(z[0]) if z.size else float("nan")
            entry[f"target_palm_cube_min_distance_{side}_m"] = float(distance.min())
            entry[f"target_palm_cube_distance_{side}_median_m"] = float(np.median(distance))
            entry[f"measured_palm_z_{side}_median_m"] = float(np.median(measured_z))
            entry[f"measured_palm_cube_min_distance_{side}_m"] = float(measured_distance.min())
        for group in ("arms", "left_arm", "right_arm", "hands", "upper_body"):
            support = rollout.group_support(mask, group)
            entry[f"measured_{group}_mahalanobis_median"] = support["mahalanobis_median"]
            entry[f"measured_{group}_in_support_p99_frac"] = support["in_support_at_p99_frac"]
        record["windows"][name] = entry

    settle = rollout.settle_tail()
    settle_speed = np.abs(settle["measured_velocity"][:, ARM_HAND]).max(axis=1) \
        if settle["relative"].size else np.zeros(0)
    record["settle_tail"] = {
        "frames": int(settle["relative"].size),
        "joint_speed_p95_rad_s": float(np.percentile(settle_speed, 95)) if settle_speed.size else None,
        "joint_speed_max_rad_s": float(settle_speed.max()) if settle_speed.size else None,
        "target_palm_z_left_median_m": float(np.median(settle["target_palm"]["left"][:, 2])) if settle["relative"].size else None,
        "target_palm_z_right_median_m": float(np.median(settle["target_palm"]["right"][:, 2])) if settle["relative"].size else None,
        "measured_palm_z_left_median_m": float(np.median(settle["measured_palm"]["left"][:, 2])) if settle["relative"].size else None,
        "measured_palm_z_right_median_m": float(np.median(settle["measured_palm"]["right"][:, 2])) if settle["relative"].size else None,
        "measured_palm_z_left_first_m": float(settle["measured_palm"]["left"][0, 2]) if settle["relative"].size else None,
        "measured_palm_z_right_first_m": float(settle["measured_palm"]["right"][0, 2]) if settle["relative"].size else None,
    }
    record["start_state"] = {
        "row_index": int(rollout.start_settle_index),
        "wall_offset_s": float(rollout.settle["relative"][rollout.start_settle_index]),
        "sim_s": float(rollout.settle["sim"][rollout.start_settle_index]),
        "state_43d": [float(value) for value in rollout.start_state_43d],
        "measured_palm_z_m": {
            side: float(rollout.settle["measured_palm"][side][rollout.start_settle_index][2])
            for side in ("left", "right")},
        "target_palm_z_m": {
            side: float(rollout.settle["target_palm"][side][rollout.start_settle_index][2])
            for side in ("left", "right")},
        "support": {group: rollout.start_support()[group]
                    for group in ("arms", "left_arm", "right_arm", "hands", "upper_body")},
    }
    # Policy tokens: only the GR00T-sourced ones after Start count as the
    # policy's own output; the warm start's tokens are measured separately.
    for window, (low, high) in WINDOWS_S.items():
        bound_low = rollout.start_wall + low
        bound_high = None if high is None else rollout.start_wall + high
        policy = rollout.tokens("groot", low=bound_low, high=bound_high)
        warm = rollout.tokens("warmstart", low=rollout.start_wall - WARMSTART_WINDOW_S - 0.5)
        if high is not None:
            warm = [row for row in warm if row["relative"] < high]
        entry = {
            "groot_tokens": len(policy),
            "warmstart_tokens": len(warm),
            "support": token_support_distances([row["token"] for row in policy]),
        }
        if policy:
            first = policy[0]
            entry["first_policy_token_relative_s"] = float(first["relative"])
            entry["first_policy_token_frame_index"] = int(first["frame_index"])
            entry["first_policy_token"] = first["token"]
            entry["first_policy_token_support"] = token_support_distances([first["token"]])
        if warm:
            entry["last_warmstart_token_relative_s"] = float(warm[-1]["relative"])
            entry["last_warmstart_token"] = warm[-1]["token"]
        if policy and warm:
            entry["token_step_l1_warm_to_policy"] = float(
                np.abs(np.asarray(warm[-1]["token"]) - np.asarray(policy[0]["token"])).sum())
        record[f"tokens_{window}"] = entry
    return record


def pair_effect(treatment: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    """The paired W-vs-A effect, metric by metric, with the direction kept."""
    out: dict[str, Any] = {"pair": f"{baseline['label']}/{treatment['label']}",
                           "baseline": baseline["label"], "treatment": treatment["label"]}
    for window in WINDOWS_S:
        a, w = baseline["windows"][window], treatment["windows"][window]
        for side in ("left", "right"):
            for metric, key in (("target_palm_z", f"target_palm_z_{side}_median_m"),
                                ("target_palm_cube_min_distance",
                                 f"target_palm_cube_min_distance_{side}_m")):
                value_a, value_w = a[key], w[key]
                out[f"{window}_{metric}_{side}_baseline_m"] = value_a
                out[f"{window}_{metric}_{side}_treatment_m"] = value_w
                out[f"{window}_{metric}_{side}_delta_m"] = value_w - value_a
                if metric == "target_palm_cube_min_distance" and value_a:
                    out[f"{window}_{metric}_{side}_relative_delta"] = (value_w - value_a) / value_a
        out[f"{window}_target_palm_z_baseline_m"] = 0.5 * (
            a["target_palm_z_left_median_m"] + a["target_palm_z_right_median_m"])
        out[f"{window}_target_palm_z_treatment_m"] = 0.5 * (
            w["target_palm_z_left_median_m"] + w["target_palm_z_right_median_m"])
        out[f"{window}_target_palm_z_delta_m"] = (
            out[f"{window}_target_palm_z_treatment_m"] - out[f"{window}_target_palm_z_baseline_m"])
    for group in ("arms", "upper_body"):
        out[f"start_state_{group}_mahalanobis_baseline"] = baseline["start_state"]["support"][group]["mahalanobis"]
        out[f"start_state_{group}_mahalanobis_treatment"] = treatment["start_state"]["support"][group]["mahalanobis"]
    out["start_arms_in_p95_baseline"] = bool(
        baseline["start_state"]["support"]["arms"]["inside_p95_mahalanobis"])
    out["start_arms_in_p95_treatment"] = bool(
        treatment["start_state"]["support"]["arms"]["inside_p95_mahalanobis"])
    support_a = baseline["tokens_first_5s"]["support"]
    support_w = treatment["tokens_first_5s"]["support"]
    if support_a.get("frames") and support_w.get("frames"):
        out["first_5s_token_support_median_baseline"] = support_a["median"]
        out["first_5s_token_support_median_treatment"] = support_w["median"]
        out["first_5s_token_support_delta"] = support_w["median"] - support_a["median"]
    step_a = baseline["tokens_first_5s"].get("token_step_l1_warm_to_policy")
    step_w = treatment["tokens_first_5s"].get("token_step_l1_warm_to_policy")
    if step_a is not None and step_w is not None:
        out["first_5s_token_step_l1_baseline"] = step_a
        out["first_5s_token_step_l1_treatment"] = step_w
    return out


def decide(records: Sequence[dict[str, Any]], effects: Sequence[dict[str, Any]],
           gates: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """The single A/B/C/D decision, from the pre-declared gates and acceptance.

    A run is *valid* when its integrity gates hold (calm limb, stable pelvis,
    intact scene, bounded hand-off, no token from another source inside the
    policy's own stream) and, for the treatment, when the warm start actually
    produced a demonstration-supported start state (the gate the baseline is
    expected to fail).  Invalid runs never contribute a target metric.
    """
    validity = {row["label"]: bool(row["gates"]["integrity_passed"]
                                   and (row["cell"] != "W" or row["gates"]["treatment_state_fixed"]))
                for row in gates}
    pairs = [effect for effect in effects if effect["treatment"].startswith("W")]
    valid_pairs = [effect for effect in pairs
                   if validity.get(effect["baseline"]) and validity.get(effect["treatment"])]
    if not valid_pairs:
        return {
            "decision": "D",
            "reason": "no paired repeat produced a valid warm start under the pre-declared gates; "
                      "target metrics from invalid runs are not causal evidence",
            "validity": validity,
            "paired_repeats": 0,
            "pairs_dropped_for_validity": [effect["pair"] for effect in pairs],
            "acceptance": ACCEPTANCE,
        }

    drops = [effect["first_5s_target_palm_z_delta_m"] for effect in valid_pairs]
    relative = [effect.get("first_5s_target_palm_cube_min_distance_right_relative_delta")
                for effect in valid_pairs]
    drop_ok = all(value <= -ACCEPTANCE["target_palm_z_drop_m"] for value in drops)
    distance_ok = all(value is not None and value <= -ACCEPTANCE["target_palm_cube_min_distance_reduction"]
                      for value in relative)
    same_direction = len({value < 0 for value in drops}) == 1
    state_fixed = all(effect["start_arms_in_p95_treatment"] for effect in valid_pairs) and not all(
        effect["start_arms_in_p95_baseline"] for effect in valid_pairs)

    # Is the paired change distinguishable from the baseline's own repeat noise?
    mean_drop = float(np.mean(drops))
    sem = float(np.std(drops, ddof=1) / math.sqrt(len(drops))) if len(drops) > 1 else float("nan")
    detectable = bool(len(drops) > 1 and math.isfinite(sem) and abs(mean_drop) > 2.0 * sem)
    baseline_spread = float(np.std(
        [effect["first_5s_target_palm_z_baseline_m"] for effect in valid_pairs], ddof=1)) \
        if len(valid_pairs) > 1 else float("nan")

    if (drop_ok or distance_ok) and same_direction:
        decision = "A"
        reason = ("the valid warm start moved the policy's own target: every paired repeat is at least "
                  "%.0f cm lower (or %.0f%% closer to a cube) in the same direction"
                  % (100 * ACCEPTANCE["target_palm_z_drop_m"],
                     100 * ACCEPTANCE["target_palm_cube_min_distance_reduction"]))
    elif state_fixed and not same_direction:
        decision = "C"
        reason = ("a valid warm start was produced and it fixed the start state in every repeat, but the "
                  "pairs do not agree on the policy's own target (changes %s m; mean %+.3f, sem %.3f "
                  "against the baseline's own repeat sd %.3f): the pre-declared %.0f cm / %.0f%% effect "
                  "is not reproduced in the same direction"
                  % ("/".join("%+.3f" % value for value in drops), mean_drop, sem, baseline_spread,
                     100 * ACCEPTANCE["target_palm_z_drop_m"],
                     100 * ACCEPTANCE["target_palm_cube_min_distance_reduction"]))
    elif state_fixed:
        decision = "B"
        reason = ("a valid warm start was produced and it fixed the start state in every repeat, but the "
                  "policy's own target did not change: the paired first-5 s changes (%s m) are all below "
                  "the pre-declared %.0f cm and inside the baseline's own repeat spread (%.3f m)"
                  % ("/".join("%+.3f" % value for value in drops),
                     100 * ACCEPTANCE["target_palm_z_drop_m"], baseline_spread))
    else:
        decision = "C"
        reason = "the paired effect is not consistent across the repeats that passed the gates"
    return {"decision": decision, "reason": reason, "validity": validity,
            "paired_repeats": len(valid_pairs),
            "first_5s_target_palm_z_delta_m": drops,
            "first_5s_target_palm_z_delta_mean_m": mean_drop,
            "first_5s_target_palm_z_delta_sem_m": sem,
            "first_5s_target_palm_z_baseline_repeat_sd_m": baseline_spread,
            "first_5s_target_palm_z_detection_floor_m": 2.0 * sem if math.isfinite(sem) else None,
            "paired_change_exceeds_predeclared_size": bool(any(
                abs(value) >= ACCEPTANCE["target_palm_z_drop_m"] for value in drops)),
            "first_5s_target_palm_cube_relative_delta_right": relative,
            "acceptance": ACCEPTANCE}


# ---------------------------------------------------------------- analyse


def stage_analyse(args: argparse.Namespace) -> dict[str, Any]:
    out = repo_path(args.output_dir) if args.output_dir else DEFAULT_OUT
    tables = out / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    gate_rows: list[dict[str, Any]] = []
    cell_rollouts: list[Rollout] = []
    rejected: list[str] = []
    for cell in CELLS:
        for directory in rollout_dirs(cell, out):
            try:
                rollout = Rollout(cell, out, directory)
            except AbError as exc:
                rejected.append(f"{cell}/{directory.name}: {exc}")
                continue
            cell_rollouts.append(rollout)
            records.append(rollout_metrics(rollout))
            gate_rows.append({"label": rollout.label, "cell": cell, "rollout": rollout.index,
                              "directory": rel(directory), "gates": rollout_gates(rollout)})

    # The baseline's own canonical hand-off jump, used as G5's reference.
    baseline_jumps = [row["gates"]["G5_handoff_bounded"]["target_jump_rad"] for row in gate_rows
                      if row["cell"] == "A"]
    baseline_jump = max(baseline_jumps) if baseline_jumps else None
    for row in gate_rows:
        if row["cell"] != "A" and baseline_jump is not None:
            gate = row["gates"]["G5_handoff_bounded"]
            gate["baseline_max_jump_rad"] = baseline_jump
            gate["passed"] = bool(gate["target_jump_rad"] <= GATE_LIMITS["transition_target_jump_ceiling_rad"]
                                  and gate["sim_finite"] and not gate["fall"]
                                  and gate["target_jump_rad"] <= baseline_jump + 0.25)
        gates = row["gates"]
        gates["integrity_passed"] = all(gates[key]["passed"] for key in
                                        ("G2_calm_at_warmstart_end", "G3_pelvis_stable",
                                         "G4_scene_intact", "G5_handoff_bounded", "G6_token_hygiene"))
        gates["treatment_state_fixed"] = bool(gates["G1_start_arms_in_demo_p95"]["passed"])
        gates["passed"] = bool(gates["integrity_passed"] and gates["treatment_state_fixed"])
        gates["valid"] = bool(gates["integrity_passed"]
                              and (row["cell"] != "W" or gates["treatment_state_fixed"]))

    by_label = {row["label"]: row for row in records}
    effects: list[dict[str, Any]] = []
    for treatment in [row for row in records if row["cell"] == "W"]:
        baseline = by_label.get(f"A{treatment['rollout']}")
        if baseline is not None:
            effects.append(pair_effect(treatment, baseline))

    decision = decide(records, effects, gate_rows)

    def series(cell: str, key: str) -> list[float]:
        values: list[float] = []
        for row in records:
            if row["cell"] != cell:
                continue
            for window in WINDOWS_S:
                value = row["windows"][window].get(key)
                if value is not None:
                    values.append(float(value))
        return values

    def spread(cell: str, key: str, window: str) -> dict[str, Any] | None:
        values = [float(row["windows"][window][key]) for row in records
                  if row["cell"] == cell and key in row["windows"][window]]
        if not values:
            return None
        return {"values": values, "median": float(np.median(values)),
                "sd": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0}

    summary = {
        "schema_version": SCHEMA_VERSION,
        "script": SCRIPT_VERSION,
        "question": ("Does warm-starting SONIC with a real task-0 demonstration's start-phase token "
                     "stream -- live measured state, no teleport, no hold -- and handing over to "
                     "GR00T without resetting the deployment reduce GR00T's first and continuing "
                     "high palm targets?"),
        "decision": decision,
        "acceptance": ACCEPTANCE,
        "gate_limits": GATE_LIMITS,
        "cells": {
            cell: {
                "rollouts": len([row for row in records if row["cell"] == cell]),
                "valid": len([row for row in gate_rows
                              if row["cell"] == cell and row["gates"]["valid"]]),
                "start_arms_mahalanobis": spread(cell, "measured_arms_mahalanobis_median", "first_1s"),
                "target_palm_z_right_first_5s": spread(cell, "target_palm_z_right_median_m", "first_5s"),
                "target_palm_z_left_first_5s": spread(cell, "target_palm_z_left_median_m", "first_5s"),
                "target_palm_cube_min_distance_right_first_5s": spread(
                    cell, "target_palm_cube_min_distance_right_m", "first_5s"),
            }
            for cell in CELLS
        },
        "repeat_spread": {
            cell: {
                "target_palm_z_right_first_5s_m": spread(cell, "target_palm_z_right_median_m", "first_5s"),
                "target_palm_cube_min_distance_right_first_5s_m": spread(
                    cell, "target_palm_cube_min_distance_right_m", "first_5s"),
            }
            for cell in CELLS
        },
        "paired_effects": effects,
        "gates": gate_rows,
        "records": records,
        "rejected": rejected,
        "windows": {name: {"low_s": low, "high_s": high} for name, (low, high) in WINDOWS_S.items()},
    }
    write_json(out / "summary.json", summary)
    write_csv(tables / "primary_metrics.csv", flatten_metrics(records))
    write_csv(tables / "paired_effects.csv", effects)
    write_csv(tables / "validity_gates.csv", flatten_gates(gate_rows))
    write_json(tables / "rollouts.json", records)
    write_json(tables / "gates.json", gate_rows)
    series_written = write_series(cell_rollouts)
    print(f"[analyse] cells: " + ", ".join(
        f"{cell}={len([row for row in records if row['cell'] == cell])} rollouts, "
        f"{len([row for row in gate_rows if row['cell'] == cell and row['gates']['valid']])} valid"
        for cell in CELLS))
    print(f"[analyse] decision: {decision['decision']} -- {decision['reason']}")
    return summary


def write_series(rollouts: Sequence[Rollout]) -> list[str]:
    """Per-rollout time series behind the figures, as one npz per rollout.

    Written by the analysis stage (which already holds the session streams and the
    calibration) so the figures stage needs neither pyarrow nor the demo cloud:
    it only plots what was measured.
    """
    written: list[str] = []
    tables = None
    for rollout in rollouts:
        tables = rollout.out / "tables"
        settle_mask = (rollout.settle["relative"] >= -WARMSTART_WINDOW_S - 2.0) \
            & (rollout.settle["relative"] < 0.0)
        crop_mask = (rollout.relative >= 0.0) & (rollout.relative < 6.0)
        payload = {
            "settle_seconds": rollout.settle["relative"][settle_mask],
            "policy_seconds": rollout.relative[crop_mask],
            "settle_joint_speed": np.abs(
                rollout.settle["measured_velocity"][settle_mask][:, ARM_HAND]).max(axis=1),
            "crop_joint_speed": np.abs(rollout.measured_velocity[crop_mask][:, ARM_HAND]).max(axis=1),
            "settle_root_z": rollout.settle["root"][settle_mask][:, 2],
            "crop_root_z": rollout.root[crop_mask][:, 2],
        }
        for side in ("left", "right"):
            payload[f"settle_measured_palm_z_{side}"] = rollout.settle["measured_palm"][side][settle_mask][:, 2]
            payload[f"settle_target_palm_z_{side}"] = rollout.settle["target_palm"][side][settle_mask][:, 2]
            payload[f"crop_measured_palm_z_{side}"] = rollout.measured_palm[side][crop_mask][:, 2]
            payload[f"crop_target_palm_z_{side}"] = rollout.target_palm[side][crop_mask][:, 2]
        path = tables / f"series_{rollout.label}.npz"
        np.savez_compressed(path, **payload)
        written.append(rel(path))
    return written


def flatten_metrics(records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        for window, entry in record["windows"].items():
            rows.append({"label": record["label"], "cell": record["cell"], "window": window, **entry})
    return rows


def flatten_gates(gate_rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in gate_rows:
        flat: dict[str, Any] = {"label": row["label"], "cell": row["cell"],
                                "valid": row["gates"]["valid"],
                                "integrity_passed": row["gates"]["integrity_passed"],
                                "treatment_state_fixed": row["gates"]["treatment_state_fixed"]}
        for name, gate in row["gates"].items():
            if isinstance(gate, dict):
                for key, value in gate.items():
                    if isinstance(value, (int, float, bool)) or value is None:
                        flat[f"{name}.{key}"] = value
            else:
                flat[name] = gate
        rows.append(flat)
    return rows


# ---------------------------------------------------------------- figures


def stage_figures(args: argparse.Namespace) -> dict[str, Any]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = repo_path(args.output_dir) if args.output_dir else DEFAULT_OUT
    figures = out / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    summary = read_json(out / "summary.json")
    records = summary["records"]
    written: list[str] = []

    # 1. The intervention window: the measured and commanded right palm through
    #    the settle tail and the first seconds of the policy.
    figure, axes = plt.subplots(1, 2, figsize=(13.0, 4.4))
    for record in records:
        series_path = out / "tables" / f"series_{record['label']}.npz"
        if not series_path.is_file():
            continue
        series = np.load(series_path)
        style = "-" if record["cell"] == "W" else "--"
        colour = "#d62728" if record["cell"] == "W" else "#1f77b4"
        axes[0].plot(series["settle_seconds"], series["settle_measured_palm_z_right"],
                     style, color=colour, linewidth=1.2, label=f"{record['label']} measured")
        axes[0].plot(series["settle_seconds"], series["settle_target_palm_z_right"],
                     style, color=colour, linewidth=0.8, alpha=0.55)
        axes[0].plot(series["policy_seconds"], series["crop_measured_palm_z_right"],
                     style, color=colour, linewidth=1.2)
        axes[0].plot(series["policy_seconds"], series["crop_target_palm_z_right"],
                     style, color=colour, linewidth=0.8, alpha=0.55)
    axes[0].axvline(0.0, color="black", linewidth=0.8)
    axes[0].axvspan(-WARMSTART_WINDOW_S, 0.0, color="#ffd9a0", alpha=0.25)
    axes[0].set_title("right palm z (pelvis frame): solid measured, faint target")
    axes[0].set_xlabel("seconds relative to Start (shaded: the settle tail)")
    axes[0].set_ylabel("m")
    axes[0].legend(fontsize=7)
    for cell, colour in (("A", "#1f77b4"), ("W", "#d62728")):
        rows = [row for row in records if row["cell"] == cell]
        values = [row["windows"]["first_5s"]["measured_arms_mahalanobis_median"] for row in rows]
        axes[1].scatter([cell] * len(values), values, color=colour, s=28)
    axes[1].axhline(GATE_LIMITS["start_arms_mahalanobis_p95"], color="black", linestyle=":",
                    label="demonstration p95 (5.39)")
    axes[1].set_title("measured arms Mahalanobis, first 5 s")
    axes[1].legend(fontsize=8)
    figure.tight_layout()
    made = figures / "fig01_warmstart_window.png"
    figure.savefig(made, dpi=130)
    plt.close(figure)
    written.append(rel(made))

    # 2. The paired effect against the pre-declared acceptance.
    figure, axes = plt.subplots(1, 2, figsize=(12.0, 4.2))
    effects = summary["paired_effects"]
    labels = [effect["pair"] for effect in effects]
    drops = [effect["first_5s_target_palm_z_delta_m"] for effect in effects]
    relative = [effect.get("first_5s_target_palm_cube_min_distance_right_relative_delta", float("nan"))
                for effect in effects]
    axes[0].bar(labels, drops, color="#d62728")
    axes[0].axhline(-ACCEPTANCE["target_palm_z_drop_m"], color="black", linestyle=":")
    axes[0].set_title("paired first-5 s target palm z change (m)")
    axes[1].bar(labels, [100.0 * value for value in relative], color="#2ca02c")
    axes[1].axhline(-100.0 * ACCEPTANCE["target_palm_cube_min_distance_reduction"], color="black",
                    linestyle=":")
    axes[1].set_title("paired target-to-cube minimum distance change (%)")
    figure.tight_layout()
    made = figures / "fig02_paired_effect.png"
    figure.savefig(made, dpi=130)
    plt.close(figure)
    written.append(rel(made))

    # 3. The warm start's own event: token support distance and the state.
    figure, axes = plt.subplots(1, 2, figsize=(12.0, 4.2))
    for cell, colour in (("A", "#1f77b4"), ("W", "#d62728")):
        for record in [row for row in records if row["cell"] == cell]:
            support = record["tokens_first_5s"]["support"]
            if support.get("frames"):
                axes[0].scatter([record["label"]], [support["median"]], color=colour, s=30)
    thresholds = token_support_thresholds()
    axes[0].axhline(thresholds["loeo_p99"], color="black", linestyle=":",
                    label=f"demonstration LOEO p99 ({thresholds['loeo_p99']:.2f})")
    axes[0].set_title("policy token support distance (value L1), first 5 s")
    axes[0].legend(fontsize=8)
    for cell, colour in (("A", "#1f77b4"), ("W", "#d62728")):
        for record in [row for row in records if row["cell"] == cell]:
            start = record["start_state"]["support"]["arms"]
            axes[1].scatter([record["label"]], [start["mahalanobis"]], color=colour, s=30)
            axes[1].annotate(f"{start['mahalanobis']:.1f}", (record["label"], start["mahalanobis"]),
                             fontsize=7, xytext=(4, 4), textcoords="offset points")
    axes[1].axhline(GATE_LIMITS["start_arms_mahalanobis_p95"], color="black", linestyle=":",
                    label="arms p95")
    axes[1].set_title("arms Mahalanobis at Policy Start")
    axes[1].legend(fontsize=8)
    figure.tight_layout()
    made = figures / "fig03_token_support_and_start_state.png"
    figure.savefig(made, dpi=130)
    plt.close(figure)
    written.append(rel(made))

    write_json(figures / "figures.json", {"figures": written, "script": SCRIPT_VERSION})
    print(f"[figures] wrote {len(written)} figures")
    return {"figures": written}


# ---------------------------------------------------------------- manifest


def stage_manifest(args: argparse.Namespace) -> dict[str, Any]:
    out = repo_path(args.output_dir) if args.output_dir else DEFAULT_OUT
    inputs: dict[str, Any] = {}
    for key, path in {
        "dataset_info": DATASET_ROOT / "meta" / "info.json",
        "dataset_episodes": DATASET_ROOT / "meta" / "episodes.jsonl",
        "profile": PROFILE,
        "exp01_summary": EXP01 / "summary.json",
        "exp02_channel_calibration": SUPPORT_DIR / "tables" / "channel_calibration.csv",
        "exp02_support_calibration": SUPPORT_DIR / "tables" / "support_calibration.npz",
        "exp05_loeo_calibration": EXP05 / "tables" / "loeo_calibration.json",
        "exp05_heldout_baseline": EXP05 / "tables" / "heldout_baseline.json",
        "exp06_selection": EXP06 / "tables" / "reset_pose.json",
        "baseline_session": BASELINE_SERIES / "session.json",
    }.items():
        inputs[key] = {"path": rel(path), "sha256": sha256(path)}
    for key, path in {
        "baseline_rollout_01": BASELINE_SERIES / "rollouts" / "rollout-01" / "tracking.jsonl",
        "baseline_rollout_02": BASELINE_SERIES / "rollouts" / "rollout-02" / "tracking.jsonl",
    }.items():
        inputs[key] = {"path": rel(path), "sha256": sha256(path), "bytes": path.stat().st_size
                       if path.is_file() else None}
    selection = read_json(out / "tables" / "segment_selection.json") if (out / "tables" / "segment_selection.json").is_file() else None
    runs: dict[str, Any] = {}
    for cell in CELLS:
        session = session_dir(cell, out)
        if not (session / "session.json").is_file():
            continue
        session_json = read_json(session / "session.json")
        runs[cell] = {
            "session": rel(session),
            "command": session_json.get("command"),
            "settle_seconds": session_json.get("settle_seconds"),
            "rollout_seconds": session_json.get("rollout_seconds"),
            "warmstart": session_json.get("warmstart"),
            "checkpoints": session_json.get("checkpoints"),
            "prompt": session_json.get("prompt"),
            "session_sha256": sha256(session / "session.json"),
            "rollouts": [rel(path) for path in rollout_dirs(cell, out)],
        }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "script": {"path": rel(Path(__file__)), "sha256": sha256(Path(__file__)),
                   "version": SCRIPT_VERSION},
        "inputs": inputs,
        "selection": selection,
        "runs": runs,
        "constants": {
            "cells": list(CELLS),
            "windows_s": {name: list(bounds) for name, bounds in WINDOWS_S.items()},
            "warmstart_window_s": WARMSTART_WINDOW_S,
            "gate_limits": GATE_LIMITS,
            "acceptance": ACCEPTANCE,
            "token_support": {"metric": TOKEN_SUPPORT_METRIC,
                              "cloud_frames_per_episode": TOKEN_CLOUD_FRAMES_PER_EPISODE,
                              "seed": TOKEN_CLOUD_SEED},
        },
    }
    write_json(out / "manifest.json", manifest)
    print(f"[manifest] wrote {rel(out / 'manifest.json')}")
    return manifest


# ---------------------------------------------------------------- cli


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GR00T demonstration-token warm start A/B")
    parser.add_argument("stage", choices=("select", "analyse", "figures", "manifest", "all"))
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.stage in ("select", "all"):
        stage_select(args)
    if args.stage in ("analyse", "all"):
        stage_analyse(args)
    if args.stage in ("figures", "all"):
        stage_figures(args)
    if args.stage in ("manifest", "all"):
        stage_manifest(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
