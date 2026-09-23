#!/usr/bin/env python3
"""Experiment 06 -- GR00T Dataset-Supported Arm Reset Pose A/B.

One question: does starting GR00T from the arm+hand start pose the task-0
demonstrations actually support reduce the high arm/palm target and the
away-from-the-cubes behaviour?

The stages:

* ``select-pose``  task-0 train demonstrations -> the demonstrated *start phase*
  (the quiet window before each episode's first sustained arm motion), a
  fixed-seed robust medoid of the 286 episode start states, and the support
  verdict for that pose under experiment 02's own calibration (LOEO quantiles,
  robust Mahalanobis / k-NN, per-channel envelope).  Writes the pose that the
  opt-in simulator flag consumes as JSON.
* ``analyse``      the A/B rollout directories -> the paired primary metrics
  (policy output and measured outcome kept apart).
* ``figures``      at most three figures from the analysed tables.
* ``manifest``     script/input hashes and the exact commands.

The demonstrations are read through the same module experiment 02 used, so the
channel order, the layout asserts and the support calibration are the same code,
not a re-implementation.
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
SCRIPT_VERSION = "groot-reset-pose-ab.py/1.0.0"
REPO_ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = REPO_ROOT / "data" / "outputs" / "blockstacking-debug"
SUPPORT_DIR = CAMPAIGN / "experiments" / "02-state-support"
DATASET_ROOT = REPO_ROOT / "data" / "datasets" / "psi0-unitree-dex3-sonic-v1" / "train"
DEFAULT_OUT = CAMPAIGN / "experiments" / "06-groot-reset-pose-ab"
URDF = REPO_ROOT / "third_party" / "Psi0" / "real" / "assets" / "g1" / "g1_body29_hand14.urdf"

#: Fixed seed for every stochastic step of the pose selection.
SELECT_SEED = 20260921
#: Arm+hand joint velocity (rad/s) that counts as movement when cutting an
#: episode's quiet start window off the moving rest.
START_MOTION_RAD_S = 0.30
#: The movement must persist this many samples (30 Hz) before it is an onset.
START_MOTION_SUSTAIN = 5
#: The start window never reaches further than this into an episode (seconds).
START_WINDOW_MAX_S = 1.0
#: The start window never holds fewer than this many samples.
START_WINDOW_MIN_FRAMES = 3

ARM_HAND = slice(15, 43)
#: The two baseline pre-policy arm states that were measured in the previous
#: campaign (groot-rollout-1, last tracking row before each Start).
BASELINE_SERIES = "groot-rollout-1"
#: Scene constants used to name the palm target in the same terms as the
#: primary metrics: worktop height and the cube centre height above it.
TABLE_SURFACE_M = 0.8357609
CUBE_Z_M = 0.8607609


class AbError(RuntimeError):
    """The stage cannot run; the message is meant for the operator."""


def repo_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (REPO_ROOT / path)


def support_module():
    """Experiment 02's analyser, imported as a module (it owns the calibration)."""
    path = REPO_ROOT / "scripts" / "analyze-blockstacking-state-support.py"
    spec = importlib.util.spec_from_file_location("state_support", path)
    if spec is None or spec.loader is None:
        raise AbError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def comparison_module():
    path = REPO_ROOT / "scripts" / "compare-blockstacking-dataset.py"
    spec = importlib.util.spec_from_file_location("dataset_compare", path)
    if spec is None or spec.loader is None:
        raise AbError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def rel(path: Path) -> str:
    """Repository-relative spelling when the path is inside it, else as given."""
    try:
        return str(Path(path).resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path: Path, rows: Sequence[dict[str, Any]], columns: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    names = list(columns) if columns else list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=names)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.is_file():
        return rows
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


# ------------------------------------------------------------------ calibration


class Calibration:
    """Experiment 02's robust scaling, LOEO thresholds and envelope."""

    def __init__(self) -> None:
        module = support_module()
        self.module = module
        self.default_dataset = DATASET_ROOT
        self.channels = tuple(module.CHANNELS)
        rows = read_csv(SUPPORT_DIR / "tables" / "channel_calibration.csv")
        if [row["channel"] for row in rows] != list(self.channels):
            raise AbError("channel_calibration.csv is not in the canonical channel order")
        self.median = np.array([float(row["weighted_median_rad"]) for row in rows])
        self.scale = np.array([float(row["robust_scale_rad"]) for row in rows])
        self.degenerate = np.array([row["degenerate"] == "true" for row in rows])
        self.envelope_low = np.array([float(row["cloud_p01_rad"]) for row in rows])
        self.envelope_high = np.array([float(row["cloud_p99_rad"]) for row in rows])
        self.hull_low = np.array([float(row["cloud_min_rad"]) for row in rows])
        self.hull_high = np.array([float(row["cloud_max_rad"]) for row in rows])
        self.loeo = dict(np.load(SUPPORT_DIR / "tables" / "support_calibration.npz"))
        self.episode_path = module.episode_path
        self.read_parquet = module.read_parquet

    def group_name(self, block: slice) -> str:
        for name, other in self.module.GROUPS.items():
            if name == "all_43d":
                continue
            if other.start == block.start and other.stop == block.stop:
                return name
        raise AbError(f"no group for slice {block}")

    def scaled(self, states: np.ndarray) -> np.ndarray:
        """Robust-scaled arm+hand subspace (28 informational channels)."""
        return (np.asarray(states, dtype=float)[:, ARM_HAND] - self.median[ARM_HAND]) / self.scale[ARM_HAND]

    def scaled_cloud(self) -> np.ndarray:
        """The demonstration cloud in robust-scaled units (all 43 channels)."""
        cache = getattr(self, "_scaled_cloud", None)
        if cache is None:
            cloud = self._cloud()
            cache = (cloud - self.median[None, :]) / self.scale[None, :]
            self._scaled_cloud = cache
        return cache

    def support(self, states: np.ndarray) -> list[dict[str, Any]]:
        """Support of one or more 43D states, per channel group, against LOEO."""
        states = np.atleast_2d(np.asarray(states, dtype=float))
        cloud = self.scaled_cloud()
        out: list[dict[str, Any]] = []
        for block in (slice(15, 22), slice(22, 29), slice(15, 29), slice(29, 36),
                      slice(36, 43), slice(29, 43), slice(15, 43)):
            name = self.group_name(block)
            index = np.arange(block.start - 15, block.stop - 15)
            scaled = (states[:, block] - self.median[block]) / self.scale[block]
            group_cloud = cloud[:, block]
            model = self.module.MahalanobisModel(group_cloud)
            maha = model.distance(scaled)
            knn = self.module.knn_distances(scaled, group_cloud, np.arange(scaled.shape[1]))[1]
            loeo_maha = np.sort(self.loeo[f"loeo_{name}_mahalanobis"])
            loeo_knn = np.sort(self.loeo[f"loeo_{name}_knn_k1"])
            for row in range(states.shape[0]):
                out.append({
                    "group": name,
                    "mahalanobis": float(maha[row]),
                    "mahalanobis_loeo_median": float(np.median(loeo_maha)),
                    "mahalanobis_loeo_p95": float(np.percentile(loeo_maha, 95)),
                    "mahalanobis_loeo_p99": float(np.percentile(loeo_maha, 99)),
                    "mahalanobis_loeo_percentile": float(np.searchsorted(loeo_maha, maha[row]) / loeo_maha.size),
                    "knn_k1": float(knn[row]),
                    "knn_loeo_p95": float(np.percentile(loeo_knn, 95)),
                    "knn_loeo_p99": float(np.percentile(loeo_knn, 99)),
                    "knn_loeo_percentile": float(np.searchsorted(loeo_knn, knn[row]) / loeo_knn.size),
                    "inside_p95_mahalanobis": bool(maha[row] <= np.percentile(loeo_maha, 95)),
                    "inside_p95_knn": bool(knn[row] <= np.percentile(loeo_knn, 95)),
                })
        return out

    def _cloud(self) -> np.ndarray:
        cache = getattr(self, "_cloud_cache", None)
        if cache is not None:
            return cache
        module = self.module
        episodes = module.blockstacking_episodes(DATASET_ROOT)
        states, _, owners = module.load_all_valid_states(DATASET_ROOT, episodes)
        cloud, _ = module.balanced_cloud(
            states, owners, [int(e["episode_index"]) for e in episodes],
            module.FRAMES_PER_EPISODE, module.SEED,
        )
        self._cloud_cache = cloud
        return cloud

    def envelope_check(self, state: np.ndarray) -> dict[str, Any]:
        inside_p = (state >= self.envelope_low) & (state <= self.envelope_high)
        inside_hull = (state >= self.hull_low) & (state <= self.hull_high)
        upper = np.arange(15, 43)
        return {
            "channels_inside_p1_p99": int(inside_p[upper].sum()),
            "channels_total": int(upper.size),
            "channels_inside_min_max": int(inside_hull[upper].sum()),
            "outside_p1_p99": [self.channels[i] for i in upper[~inside_p[upper]]],
            "strictest_loeo_reference": 0.986,
            "frame_all_channels_p1_p99_loeo_reference": 0.624,
        }


# --------------------------------------------------------------- start phase


def episode_start_states(calibration: Calibration) -> tuple[list[dict[str, Any]], np.ndarray]:
    """One start state per demonstration episode plus the window it came from."""
    module = calibration.module
    episodes = module.blockstacking_episodes(DATASET_ROOT)
    if len(episodes) != 286:
        raise AbError(f"expected 286 task-0 training episodes, found {len(episodes)}")
    rows: list[dict[str, Any]] = []
    states: list[np.ndarray] = []
    dt = 1.0 / float(read_json(DATASET_ROOT / "meta" / "info.json")["fps"])
    for entry in episodes:
        episode = int(entry["episode_index"])
        series = np.asarray(
            calibration.read_parquet(calibration.episode_path(DATASET_ROOT, episode),
                                     ["observation.state"])["observation.state"], dtype=float)
        valid = min(int(entry.get("frames_valid", series.shape[0])), series.shape[0])
        window = series[:valid]
        speed = np.zeros(window.shape[0])
        if window.shape[0] > 1:
            speed[1:] = np.abs(np.diff(window[:, ARM_HAND], axis=0)).max(axis=1) / dt
        onset = None
        need = START_MOTION_SUSTAIN
        for index in range(window.shape[0] - need):
            if (speed[index:index + need] > START_MOTION_RAD_S).all():
                onset = index
                break
        cap = max(START_WINDOW_MIN_FRAMES, min(int(round(START_WINDOW_MAX_S / dt)), window.shape[0]))
        stop = window.shape[0] if onset is None else min(max(onset, START_WINDOW_MIN_FRAMES), cap)
        quiet = window[:stop]
        state = np.median(quiet, axis=0)
        rows.append({
            "episode_index": episode,
            "frames_total": int(window.shape[0]),
            "onset_frame": None if onset is None else int(onset),
            "onset_s": None if onset is None else round(onset * dt, 4),
            "window_frames": int(quiet.shape[0]),
            "window_s": round(quiet.shape[0] * dt, 4),
            "max_speed_in_window_rad_s": float(speed[:stop].max()),
            "source_data_file": entry.get("source_data_file"),
        })
        states.append(state)
    return rows, np.asarray(states, dtype=float)


def medoid_and_median(calibration: Calibration, states: np.ndarray) -> dict[str, Any]:
    scaled = calibration.scaled(states)
    gram = scaled @ scaled.T
    square = np.diag(gram)
    distances = np.sqrt(np.maximum(square[:, None] + square[None, :] - 2.0 * gram, 0.0))
    total = distances.sum(axis=1)
    medoid = int(np.argmin(total))
    return {
        "medoid_index": medoid,
        "medoid_total_distance": float(total[medoid]),
        "medoid_median_total_distance": float(np.median(total)),
        "median_state": np.median(states, axis=0),
        "medoid_state": states[medoid],
        "start_state_p50_distance": float(np.median(distances[medoid])),
        "start_state_max_distance": float(distances[medoid].max()),
    }


# ------------------------------------------------------------------- forearm


def palm_positions(state: np.ndarray) -> dict[str, list[float]]:
    module = comparison_module()
    urdf = module.Urdf(URDF)
    body = np.asarray(state[:29], dtype=float)[None, :]
    palms = urdf.palms(body)
    return {side: [float(v) for v in position[0]] for side, position in palms.items()}


# ------------------------------------------------------------------ select


def stage_select_pose(args: argparse.Namespace) -> dict[str, Any]:
    out = repo_path(args.output_dir) if args.output_dir else DEFAULT_OUT
    out.mkdir(parents=True, exist_ok=True)
    calibration = Calibration()
    rows, states = episode_start_states(calibration)
    pick = medoid_and_median(calibration, states)
    medoid = np.asarray(pick["medoid_state"], dtype=float)
    median = np.asarray(pick["median_state"], dtype=float)

    # The baseline the pose will be compared against: the measured pre-Start arm
    # state of the existing GR00T rollouts (last tracking row before each Start).
    baseline_states: list[np.ndarray] = []
    baseline_rows: list[dict[str, Any]] = []
    session = read_json(CAMPAIGN / BASELINE_SERIES / "session.json")
    tracking = load_jsonl(CAMPAIGN / BASELINE_SERIES / "raw" / "isaac.tracking.jsonl")
    for rollout in session["rollouts"]:
        before = [r for r in tracking if r["wall_time_ns"] <= rollout["start_wall_ns"]]
        row = before[-1]
        state = np.array(row["body_measured"][:29] + row["left_hand_measured"] + row["right_hand_measured"],
                         dtype=float)
        baseline_states.append(state)
        baseline_rows.append({"rollout": int(rollout["index"]), "wall_time_ns": int(row["wall_time_ns"]),
                              "sim_s": float(row["sim_s"]), "state": [float(v) for v in state]})

    # The declared reset pose the baseline starts from: the profile's
    # ``sonic_standing`` pose for the 29 body joints and the asset's own hand
    # defaults (zero) for the two Dex3 blocks.
    declared = np.zeros(43)
    declared[0:12] = [-0.312, 0.0, 0.0, 0.669, -0.363, 0.0] * 2
    declared[15:22] = [0.2, 0.2, 0.0, 0.6, 0.0, 0.0, 0.0]
    declared[22:29] = [0.2, -0.2, 0.0, 0.6, 0.0, 0.0, 0.0]

    candidates = {
        "demonstrated_medoid": medoid,
        "demonstrated_median": median,
        "baseline_groot_rollout1_r1": baseline_states[0],
        "baseline_groot_rollout1_r2": baseline_states[1],
        "declared_reset_pose": declared,
    }
    # The reset pose only names the 14 arm joints; the hands stay at the asset
    # default, and the site the policy starts from is measured, not assumed.
    support_rows: list[dict[str, Any]] = []
    envelope: dict[str, Any] = {}
    for name, state in candidates.items():
        for row in calibration.support(np.asarray(state, dtype=float)):
            support_rows.append({"candidate": name, **row})
        envelope[name] = calibration.envelope_check(np.asarray(state, dtype=float))

    # The whole start-state population, as the distribution the pose is drawn from.
    population = calibration.support(states)
    population_by_group: dict[str, dict[str, float]] = {}
    for group in sorted({row["group"] for row in population}):
        values = np.array([row["mahalanobis_loeo_percentile"] for row in population if row["group"] == group])
        population_by_group[group] = {
            "episodes": int(values.size),
            "percentile_median": float(np.median(values)),
            "percentile_p95": float(np.percentile(values, 95)),
            "fraction_above_loeo_p95": float((values > 0.95).mean()),
        }

    # Palm (pelvis frame) of the candidate poses against every demonstrated frame.
    demo_palms = {"left": [], "right": []}
    for state in states:
        palms = palm_positions(state)
        for side in demo_palms:
            demo_palms[side].append(palms[side][2])
    palm_table: dict[str, Any] = {}
    for side in ("left", "right"):
        values = np.asarray(demo_palms[side])
        palm_table[side] = {
            "demostart_median_m": float(np.median(values)),
            "demostart_p05_m": float(np.percentile(values, 5)),
            "demostart_p95_m": float(np.percentile(values, 95)),
        }
        for name, state in candidates.items():
            if name == "declared_reset_pose":
                continue
            palm_table[side][f"{name}_m"] = float(palm_positions(state)[side][2])

    pose_payload = {
        "schema_version": SCHEMA_VERSION,
        "source": {
            "dataset": str(DATASET_ROOT.relative_to(REPO_ROOT)),
            "collection": calibration.module.COLLECTION,
            "episodes": len(rows),
            "start_window_rule": (
                f"per episode: median over the frames before the first {START_MOTION_SUSTAIN} "
                f"consecutive samples with an arm+hand joint speed above {START_MOTION_RAD_S} rad/s, "
                f"capped at {START_WINDOW_MAX_S}s and at least {START_WINDOW_MIN_FRAMES} frames"
            ),
            "seed": SELECT_SEED,
        },
        "selected": "demonstrated_medoid",
        "selection_metric": "robust-scaled (MAD) Euclidean medoid over the 28 arm+hand channels",
        "selection": {
            "medoid_episode_index": rows[pick["medoid_index"]]["episode_index"],
            "medoid_total_distance": pick["medoid_total_distance"],
            "medoid_median_total_distance": pick["medoid_median_total_distance"],
            "start_state_distance_p50": pick["start_state_p50_distance"],
            "start_state_distance_max": pick["start_state_max_distance"],
        },
        "pose": {
            "left_arm": [float(v) for v in medoid[15:22]],
            "right_arm": [float(v) for v in medoid[22:29]],
            "left_hand": [float(v) for v in medoid[29:36]],
            "right_hand": [float(v) for v in medoid[36:43]],
        },
        "support": [row for row in support_rows if row["candidate"] == "demonstrated_medoid"],
        "envelope": envelope["demonstrated_medoid"],
    }
    write_json(out / "tables" / "reset_pose.json", pose_payload)
    # The file the simulator's opt-in ``--reset-pose-file`` consumes: the four
    # joint groups only, no metadata, so the runtime reads exactly the pose.
    write_json(out / "tables" / "reset_pose_runtime.json", {
        key: pose_payload["pose"][key]
        for key in ("left_arm", "right_arm", "left_hand", "right_hand")
    })
    write_json(out / "tables" / "pose_candidates.json", {
        "candidates": {name: [float(v) for v in state] for name, state in candidates.items()},
        "channels": list(calibration.channels),
        "baseline_rows": baseline_rows,
    })
    write_csv(out / "tables" / "support_vs_candidates.csv", support_rows)
    write_json(out / "tables" / "start_state_population.json", {
        "by_group": population_by_group,
        "candidate_envelope": envelope,
        "palm_z_pelvis_frame_m": palm_table,
        "episode_starts": rows,
    })
    return {
        "selected_episode": rows[pick["medoid_index"]]["episode_index"],
        "pose": pose_payload["pose"],
        "support": pose_payload["support"],
        "envelope": pose_payload["envelope"],
        "population": population_by_group,
        "palm": palm_table,
    }


# --------------------------------------------------------------- run stages

#: The A/B cells.  ``A`` is the unmodified reset the campaign already ran under;
#: ``B`` adds nothing but the opt-in upper-limb reset pose (and, when the settle
#: proves to sweep it, its hold).  ``R`` is the reset-pose-only control.
CELLS = ("A", "B", "R")
#: Primary windows, in seconds after the measured motion onset.
WINDOWS_S = {"first_1s": (0.0, 1.0), "first_5s": (0.0, 5.0), "active": (0.0, None)}
#: The acceptance rule, fixed before the runs: B must start in the demonstrated
#: support and must move the *policy's own target*, not just the realised pose.
ACCEPTANCE = {
    "start_arm_state_inside_demo_p95": True,
    "target_palm_z_drop_m": 0.05,
    "target_palm_cube_min_distance_reduction": 0.30,
    "measured_droop_alone_counts": False,
}


def rollout_dirs(cell: str, out: Path) -> list[Path]:
    directory = out / "runs" / cell / "rollouts"
    if not directory.is_dir():
        return []
    return sorted(directory.glob("rollout-*"))


def session_dir(cell: str, out: Path) -> Path:
    return out / "runs" / cell


def read_scene_cubes(session: Path, rollout: Path) -> dict[str, tuple[float, float, float]]:
    """Cube centres, from the rollout's own samples (the live scene probe)."""
    samples = load_jsonl(rollout / "isaac.samples.jsonl")
    if not samples:
        samples = load_jsonl(session / "raw" / "isaac.samples.jsonl")
    for row in samples:
        scene = row.get("scene") or {}
        cubes = scene.get("cubes") or {}
        if cubes:
            return {name: tuple(float(v) for v in entry["live_center_xyz_m"])
                    for name, entry in cubes.items()}
    return {}


def token_series(telemetry: Sequence[dict[str, Any]]) -> list[tuple[float, list[float], int]]:
    """(wall seconds, 64-D motion token, frame index) of every applied action."""
    out: list[tuple[float, list[float], int]] = []
    for row in telemetry:
        if row.get("kind") != "applied_action":
            continue
        fields = row.get("fields") or {}
        token = fields.get("token_state")
        if not token:
            continue
        index = fields.get("frame_index")
        index = int(index[0]) if isinstance(index, (list, tuple)) and index else int(index or 0)
        out.append((float(row["wall_time_ns"]) / 1e9, [float(v) for v in token[0]], index))
    return out


class RolloutMetrics:
    """Everything one rollout contributes to the A/B comparison."""

    def __init__(self, cell: str, out: Path, directory: Path, calibration: Calibration) -> None:
        module = comparison_module()
        self.cell = cell
        self.directory = directory
        self.label = f"{cell}{int(directory.name.split('-')[-1])}"
        self.meta = read_json(directory / "rollout.json")
        session = session_dir(cell, out)
        self.session = session
        rows = load_jsonl(directory / "tracking.jsonl")
        if not rows:
            raise AbError(f"{directory} has no tracking rows")
        self.rows = rows
        self.wall = np.array([r["wall_time_ns"] / 1e9 for r in rows])
        self.sim = np.array([r["sim_s"] for r in rows])
        self.target = np.array([r["body_target"] for r in rows])
        self.measured = np.array([r["body_measured"] for r in rows])
        self.hand_target = np.array([
            list(r["left_hand_target"]) + list(r["right_hand_target"]) for r in rows
        ], dtype=float)
        self.hand_measured = np.array([
            list(r["left_hand_measured"]) + list(r["right_hand_measured"]) for r in rows
        ], dtype=float)
        self.hold = np.array([bool(r.get("reset_pose_hold")) for r in rows])
        self.root = np.array([r["root_position"] for r in rows])
        self.root_quat = np.array([r["root_quaternion_wxyz"] for r in rows])
        self.onset_wall = float(self.meta["onset"]["wall_time"])
        # The sim clock of the onset: the analysis supplies it for a velocity
        # onset; a rollout whose limb never moves (a held pose) has no velocity
        # onset, so the action onset's simulation time is interpolated here.
        declared_sim = self.meta["onset"].get("sim_s")
        self.onset_index = int(np.argmax(self.wall >= self.onset_wall))
        self.onset_sim = (
            float(declared_sim) if declared_sim is not None
            else float(self.sim[min(self.onset_index, self.sim.size - 1)])
        )
        self.onset_source = self.meta["onset"].get("chosen_source")
        # The state the policy's first observation sees: the last frame before
        # the measured onset (the onset itself is already the policy moving).
        self.start_index = max(0, self.onset_index - 1)
        self.state_43d = np.hstack([self.measured, self.hand_measured])
        self.target_43d = np.hstack([self.target, self.hand_target])
        self.urdf = module.Urdf(URDF)
        base = np.broadcast_to(np.eye(4), (len(rows), 4, 4)).copy()
        for index, quaternion in enumerate(self.root_quat):
            base[index, :3, :3] = module.quaternion_wxyz_to_matrix(quaternion)
        base[:, :3, 3] = self.root
        self.base = base
        self.target_palm = self.urdf.palms(self.target)
        self.measured_palm = self.urdf.palms(self.measured)
        self.cubes = read_scene_cubes(session, directory)
        self.tokens = token_series(load_jsonl(directory / "bridge-telemetry.jsonl"))
        self.support = calibration.support(self.state_43d) if calibration is not None else []

    def window(self, name: str) -> np.ndarray:
        low, high = WINDOWS_S[name]
        seconds = self.wall - self.onset_wall
        mask = seconds >= low
        if high is not None:
            mask &= seconds < high
        if not mask.any():
            mask = np.zeros_like(seconds, dtype=bool)
            mask[min(self.onset_index, mask.size - 1)] = True
        return mask

    def cube_distance(self, side: str, mask: np.ndarray) -> np.ndarray:
        """Target palm to nearest cube centre, both in the pelvis frame (m)."""
        module = comparison_module()
        palm = self.target_palm[side][mask]
        out = None
        for centre in self.cubes.values():
            rotated = np.array([
                module.quaternion_wxyz_to_matrix(quaternion).T @ (np.asarray(centre) - position)
                for quaternion, position in zip(self.root_quat[mask], self.root[mask])
            ])
            distance = np.linalg.norm(palm - rotated, axis=1)
            out = distance if out is None else np.minimum(out, distance)
        return out if out is not None else np.zeros(int(mask.sum()))

    def group_support(self, mask: np.ndarray, group: str) -> dict[str, Any]:
        rows = [row for row in self.support if row["group"] == group]
        values = np.array([row["mahalanobis"] for row in rows])
        return {
            "frames": int(values.size),
            "mahalanobis_median": float(np.median(values)),
            "mahalanobis_loeo_percentile_median": float(np.median(
                [row["mahalanobis_loeo_percentile"] for row in rows])),
            "in_support_at_p99_frac": float(np.mean([
                row["mahalanobis"] <= row["mahalanobis_loeo_p99"] for row in rows])),
            "second_support_p99": float(np.percentile(
                [row["mahalanobis_loeo_p99"] for row in rows], 50)),
            "first_out_of_support_index": int(mask.nonzero()[0][
                next((i for i, row in enumerate(rows)
                      if row["mahalanobis"] > row["mahalanobis_loeo_p99"]), len(rows) - 1)
            ]) if any(row["mahalanobis"] > row["mahalanobis_loeo_p99"] for row in rows) else None,
        }

    def write_series(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        mask = self.window("active")
        seconds = self.wall[mask] - self.onset_wall
        payload = {
            "seconds": seconds,
            "hold_active": self.hold[mask].astype(np.int8),
            "measured_arm": self.measured[mask][:, 15:29],
            "target_arm": self.target[mask][:, 15:29],
        }
        for side in ("left", "right"):
            payload[f"target_palm_z_{side}"] = self.target_palm[side][mask][:, 2]
            payload[f"measured_palm_z_{side}"] = self.measured_palm[side][mask][:, 2]
            payload[f"target_cube_distance_{side}"] = self.cube_distance(side, mask)
        np.savez_compressed(path, **payload)

    def record(self, calibration: Calibration) -> dict[str, Any]:
        record: dict[str, Any] = {
            "cell": self.cell,
            "label": self.label,
            "rollout": int(self.meta["index"]),
            "session": rel(self.session),
            "directory": rel(self.directory),
            "onset_seconds_after_start": self.meta["onset"]["seconds_after_start"],
            "onset_source": self.onset_source,
            "onset_wall_time": self.onset_wall,
            "onset_sim_s": self.onset_sim,
            "active_seconds": float(self.sim[-1] - self.onset_sim),
            "applied_actions": int(self.meta.get("applied_actions") or 0),
            "target_actions": int(self.meta.get("target_actions") or 0),
            "fall": bool(self.meta.get("fall", {}).get("detected")),
            "robot_down_root_z_m": self.meta.get("fall", {}).get("minimum_root_z_m"),
            "cubes_lifted": bool(self.meta.get("stack", {}).get("any_cube_lifted")),
            "max_cube_lift_m": max(self.meta.get("stack", {}).get("cube_lift_above_surface_m", {}).values() or [0.0]),
            "cube_displacement_xy_m": max(
                self.meta.get("stack", {}).get("cube_pose_w", {}).get(side, {})
                .get("displacement_xy_m", 0.0) for side in ("red", "yellow", "blue")),
            "tracking_rows": len(self.rows),
            "held_frames": int(self.hold.sum()),
            "held_fraction": float(self.hold.mean()) if self.hold.size else 0.0,
            "hold_at_first_observation": bool(self.hold[self.start_index]) if self.hold.size else False,
            "windows": {},
        }
        for name in WINDOWS_S:
            mask = self.window(name)
            entry: dict[str, Any] = {"frames": int(mask.sum()),
                                     "seconds": float(WINDOWS_S[name][1] if WINDOWS_S[name][1] else
                                                      self.sim[-1] - self.onset_sim)}
            for side in ("left", "right"):
                z = self.target_palm[side][mask][:, 2]
                distance = self.cube_distance(side, mask)
                measured_z = self.measured_palm[side][mask][:, 2]
                entry[f"target_palm_z_{side}_median_m"] = float(np.median(z))
                entry[f"target_palm_z_{side}_mean_m"] = float(z.mean())
                entry[f"target_palm_z_{side}_min_m"] = float(z.min())
                entry[f"target_palm_cube_min_distance_{side}_m"] = float(distance.min())
                entry[f"target_palm_cube_distance_{side}_median_m"] = float(np.median(distance))
                entry[f"measured_palm_z_{side}_median_m"] = float(np.median(measured_z))
                entry[f"measured_minus_target_palm_z_{side}_median_m"] = float(
                    np.median(measured_z - z))
            record["windows"][name] = entry
        # Arm-state support, on the measured state the policy conditions on.
        mask = self.window("active")
        for group in ("arms", "right_arm", "left_arm", "hands", "upper_body"):
            record[f"support_{group}"] = self.group_support(mask, group)
        start = self.state_43d[self.start_index]
        start_support = calibration.support(start[None, :])
        record["start_state"] = {
            "sim_s": float(self.sim[self.start_index]),
            "wall_time": float(self.wall[self.start_index]),
            "value": [float(v) for v in start],
            "support": start_support,
            "inside_p95_mahalanobis_arms": next(
                row["inside_p95_mahalanobis"] for row in start_support if row["group"] == "arms"),
            "inside_p95_knn_arms": next(
                row["inside_p95_knn"] for row in start_support if row["group"] == "arms"),
            "arm_mahalanobis": next(row["mahalanobis"] for row in start_support
                                    if row["group"] == "arms"),
            "arm_mahalanobis_loeo_p95": next(row["mahalanobis_loeo_p95"] for row in start_support
                                             if row["group"] == "arms"),
            "arm_mahalanobis_loeo_percentile": next(
                row["mahalanobis_loeo_percentile"] for row in start_support if row["group"] == "arms"),
        }
        # Token-level: what the policy asked for, before any controller executed it.
        tokens = [(wall, token, index) for wall, token, index in self.tokens
                  if wall >= self.onset_wall]
        record["tokens"] = {
            "count": len(tokens),
            "first": None if not tokens else tokens[0][1],
            "first_frame_index": None if not tokens else tokens[0][2],
            "first_wall_offset_s": None if not tokens else tokens[0][0] - self.onset_wall,
            "first_5s": [token for wall, token, _ in tokens if wall < self.onset_wall + 5.0],
            "first_1s_count": int(sum(1 for wall, _, _ in tokens if wall < self.onset_wall + 1.0)),
        }
        return record


def stage_analyse(args: argparse.Namespace) -> dict[str, Any]:
    out = repo_path(args.output_dir) if args.output_dir else DEFAULT_OUT
    calibration = Calibration()
    records: list[dict[str, Any]] = []
    for cell in CELLS:
        for directory in rollout_dirs(cell, out):
            metrics = RolloutMetrics(cell, out, directory, calibration)
            metrics.write_series(out / "tables" / f"series_{metrics.label}.npz")
            records.append(metrics.record(calibration))
    if not records:
        raise AbError(f"no rollouts under {out / 'runs'}")

    effects: list[dict[str, Any]] = []
    by_label = {record["label"]: record for record in records}
    for record in records:
        if record["cell"] != "A":
            continue
        partner = by_label.get(f"B{record['rollout']}")
        if partner is None:
            continue
        entry = pair_effect(record, partner, treatment="B")
        effects.append(entry)
    reset_only: list[dict[str, Any]] = []
    for record in records:
        if record["cell"] != "A":
            continue
        partner = by_label.get(f"R{record['rollout']}")
        if partner is None:
            continue
        reset_only.append(pair_effect(record, partner, treatment="R"))

    summary = build_summary(records, effects, reset_only)
    write_json(out / "summary.json", summary)
    write_json(out / "tables" / "rollouts.json", records)
    write_csv(out / "tables" / "primary_metrics.csv", flatten_records(records))
    write_csv(out / "tables" / "paired_effects.csv", effects)
    write_csv(out / "tables" / "paired_effects_reset_only.csv", reset_only)
    return summary


def pair_effect(record: dict[str, Any], partner: dict[str, Any], *, treatment: str) -> dict[str, Any]:
    """One A/B (or A/reset-only) pair, on the primary metrics only."""
    entry: dict[str, Any] = {"pair": f"A{record['rollout']}/{treatment}{partner['rollout']}",
                             "treatment": treatment}
    for window in WINDOWS_S:
        for side in ("left", "right"):
            base = record["windows"][window][f"target_palm_z_{side}_median_m"]
            other = partner["windows"][window][f"target_palm_z_{side}_median_m"]
            entry[f"target_palm_z_{side}_{window}_A_m"] = base
            entry[f"target_palm_z_{side}_{window}_{treatment}_m"] = other
            entry[f"target_palm_z_{side}_{window}_delta_m"] = other - base
            base_d = record["windows"][window][f"target_palm_cube_min_distance_{side}_m"]
            other_d = partner["windows"][window][f"target_palm_cube_min_distance_{side}_m"]
            entry[f"target_palm_cube_min_distance_{side}_{window}_A_m"] = base_d
            entry[f"target_palm_cube_min_distance_{side}_{window}_{treatment}_m"] = other_d
            entry[f"target_palm_cube_min_distance_{side}_{window}_relative_delta"] = (
                (other_d - base_d) / base_d if base_d else None)
    entry["start_arm_mahalanobis_A"] = record["start_state"]["arm_mahalanobis"]
    entry[f"start_arm_mahalanobis_{treatment}"] = partner["start_state"]["arm_mahalanobis"]
    entry[f"start_arm_inside_p95_{treatment}"] = partner["start_state"]["inside_p95_mahalanobis_arms"]
    token_a = record["tokens"]["first"]
    token_b = partner["tokens"]["first"]
    entry["first_token_l2_distance"] = (
        None if token_a is None or token_b is None
        else float(np.linalg.norm(np.asarray(token_a) - np.asarray(token_b))))
    seq_a = record["tokens"]["first_5s"]
    seq_b = partner["tokens"]["first_5s"]
    shared = min(len(seq_a), len(seq_b))
    entry["first_5s_tokens_A"] = len(seq_a)
    entry[f"first_5s_tokens_{treatment}"] = len(seq_b)
    entry["first_5s_token_mean_l2_distance"] = (
        None if not shared else float(np.mean([
            np.linalg.norm(np.asarray(seq_a[i]) - np.asarray(seq_b[i])) for i in range(shared)])))
    entry["first_5s_token_max_l2_distance"] = (
        None if not shared else float(np.max([
            np.linalg.norm(np.asarray(seq_a[i]) - np.asarray(seq_b[i])) for i in range(shared)])))
    return entry


def flatten_records(records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        row = {k: v for k, v in record.items()
               if not isinstance(v, (dict, list)) and k != "windows"}
        for window, entry in record["windows"].items():
            for key, value in entry.items():
                row[f"{window}_{key}"] = value
        for group in ("arms", "right_arm", "left_arm", "hands", "upper_body"):
            for key, value in record[f"support_{group}"].items():
                row[f"support_{group}_{key}"] = value
        for key in ("arm_mahalanobis", "arm_mahalanobis_loeo_percentile", "inside_p95_mahalanobis_arms"):
            row[f"start_{key}"] = record["start_state"][key]
        rows.append(row)
    return rows


def build_summary(records: Sequence[dict[str, Any]], effects: Sequence[dict[str, Any]],
                  reset_only: Sequence[dict[str, Any]] = ()) -> dict[str, Any]:
    """The paired effect, the repeat spread and the pre-declared decision."""
    def values(cell: str, key: str) -> list[float]:
        out: list[float] = []
        for record in records:
            if record["cell"] != cell:
                continue
            for window in WINDOWS_S:
                value = record["windows"][window].get(key)
                if value is not None:
                    out.append(float(value))
        return out

    def stats(cell: str, key: str) -> dict[str, float] | None:
        values_ = values(cell, key)
        if not values_:
            return None
        return {"n": len(values_), "mean": float(np.mean(values_)),
                "median": float(np.median(values_)), "min": float(np.min(values_)),
                "max": float(np.max(values_)), "sd": float(np.std(values_, ddof=0))}

    summary: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "question": ("Does starting GR00T from a task-0-demonstration-supported arm+hand "
                     "start pose reduce its high arm/palm target and its away-from-the-cubes "
                     "behaviour?"),
        "acceptance": ACCEPTANCE,
        "windows": {name: {"low_s": low, "high_s": high} for name, (low, high) in WINDOWS_S.items()},
        "cells": {},
        "paired_effects": list(effects),
        "paired_effects_reset_only": list(reset_only),
        "repeat_spread": {},
    }
    for cell in CELLS:
        selected = [record for record in records if record["cell"] == cell]
        if not selected:
            continue
        summary["cells"][cell] = {
            "rollouts": len(selected),
            "start_arm_mahalanobis": stats(cell, "start_arm_mahalanobis"),
            "start_inside_p95_arms": [bool(record["start_state"]["inside_p95_mahalanobis_arms"])
                                       for record in selected],
            "held_fraction": [record["held_fraction"] for record in selected],
            "hold_at_first_observation": [record["hold_at_first_observation"] for record in selected],
            "falls": [record["fall"] for record in selected],
            "cubes_lifted": [record["cubes_lifted"] for record in selected],
            "target_palm_z_left_first_5s": stats(cell, "target_palm_z_left_median_m"),
            "target_palm_cube_min_distance_left_first_5s": stats(
                cell, "target_palm_cube_min_distance_left_m"),
        }
        group = np.array([record["windows"]["first_5s"]["target_palm_z_right_median_m"]
                          for record in selected])
        summary["repeat_spread"][cell] = {
            "target_palm_z_right_first_5s_sd_m": float(group.std(ddof=0)),
            "target_palm_z_right_first_5s_values_m": [float(v) for v in group],
        }
    # The decision rule, applied to the paired deltas only.
    deltas_z = [entry["target_palm_z_right_first_5s_delta_m"] for entry in effects]
    deltas_d = [entry["target_palm_cube_min_distance_right_first_5s_relative_delta"]
                for entry in effects]
    start_inside = [entry["start_arm_inside_p95_B"] for entry in effects]
    summary["decision_input"] = {
        "paired_target_palm_z_right_first_5s_delta_m": deltas_z,
        "paired_target_palm_cube_distance_right_first_5s_relative_delta": deltas_d,
        "start_arm_inside_p95_B": start_inside,
        "paired_repeats": len(effects),
    }
    return summary



# ------------------------------------------------------------------ figures

#: The demonstrated *start* palm height band in the pelvis frame, used as the
#: reference line under the policy's own target (experiment 06 select-pose).
DEMO_START_PALM_P05_P95_M = {"left": (0.1277, 0.1974), "right": (0.1336, 0.2040)}
#: Offsets (seconds after the measured onset) of the frames a contact sheet shows.
SHEET_OFFSETS = (-0.5, 0.5, 2.0, 5.0, 10.0)


def head_camera_frames(session: Path) -> list[tuple[float, Path]]:
    """(wall seconds, JPEG path) of every head-camera frame the bridge recorded."""
    telemetry = load_jsonl(session / "raw" / "telemetry" / "bridge-telemetry.jsonl")
    out: list[tuple[float, Path]] = []
    for row in telemetry:
        if row.get("kind") != "frame" or not row.get("camera_jpeg"):
            continue
        name = Path(str(row["camera_jpeg"])).name
        path = session / "raw" / "telemetry" / "head_camera" / name
        if path.is_file():
            out.append((float(row["wall_time_ns"]) / 1e9, path))
    return sorted(out)


def head_camera_sheet(out: Path, cells: Sequence[str], index: int) -> dict[str, Any]:
    """One contact sheet per cell: the same offsets after the onset, row by row."""
    from PIL import Image, ImageDraw

    rows: list[tuple[str, list[Image.Image | None]]] = []
    for cell in cells:
        session = session_dir(cell, out)
        frames = head_camera_frames(session)
        rollout = session / "rollouts" / f"rollout-{index:02d}" / "rollout.json"
        if not frames or not rollout.is_file():
            return {"written": False, "reason": f"{cell}: no head-camera frames or rollouts"}
        onset = float(read_json(rollout)["onset"]["wall_time"])
        picks: list[Image.Image | None] = []
        for offset in SHEET_OFFSETS:
            if not frames:
                picks.append(None)
                continue
            wall, path = min(frames, key=lambda item: abs(item[0] - (onset + offset)))
            picks.append(Image.open(path).convert("RGB").resize((320, 240)))
        rows.append((cell, picks))
    sheet = Image.new("RGB", (320 * len(SHEET_OFFSETS), 240 * len(rows) + 20), "white")
    draw = ImageDraw.Draw(sheet)
    for row, (cell, picks) in enumerate(rows):
        for column, image in enumerate(picks):
            if image is not None:
                sheet.paste(image, (320 * column, 240 * row))
        draw.text((4, 240 * (row + 1) - 14), f"{cell}: onset {SHEET_OFFSETS[0]:+.1f}s ... "
                                             f"{SHEET_OFFSETS[-1]:+.1f}s", fill="black")
    directory = out / "videos"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"headcam_rollout-{index:02d}_{''.join(cells)}.jpg"
    sheet.save(path, quality=88)
    return {"written": True, "path": str(path.relative_to(REPO_ROOT)),
            "cells": list(cells), "offsets_s": list(SHEET_OFFSETS)}


def stage_figures(args: argparse.Namespace) -> dict[str, Any]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = repo_path(args.output_dir) if args.output_dir else DEFAULT_OUT
    figures = out / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    records = read_json(out / "tables" / "rollouts.json")
    effects = read_csv(out / "tables" / "paired_effects.csv")
    calibration = Calibration()
    candidates = read_json(out / "tables" / "pose_candidates.json")["candidates"]
    produced: dict[str, Any] = {}

    # Fig 1 -- the 28 arm+hand channels: demonstrated envelope against the states
    # the policy actually starts from.
    figure, axis = plt.subplots(figsize=(13, 6))
    channels = list(calibration.channels)[15:43]
    x = np.arange(len(channels))
    axis.fill_between(x, calibration.envelope_low[15:43], calibration.envelope_high[15:43],
                      color="0.85", label="demonstration p1-p99")
    axis.plot(x, calibration.median[15:43], color="black", lw=1.0, label="demonstration median")
    styles = {"A": "tab:red", "B": "tab:green", "R": "tab:orange"}
    for record in records:
        axis.plot(x, record["start_state"]["value"][15:43], marker="o", ls="none", ms=5,
                  color=styles.get(record["cell"], "tab:blue"), alpha=0.85,
                  label=f"{record['label']} start (LOEO p"
                        f"{100 * record['start_state']['arm_mahalanobis_loeo_percentile']:.0f})")
    axis.plot(x, candidates["demonstrated_medoid"][15:43], color="tab:blue", lw=1.6,
              label="selected demo-supported pose")
    axis.set_xticks(x)
    axis.set_xticklabels(channels, rotation=90, fontsize=7)
    axis.set_ylabel("joint angle (rad)")
    axis.set_title("Upper limb at the policy's first observation against the demonstrated envelope")
    axis.legend(fontsize=7, ncol=3)
    figure.tight_layout()
    figure.savefig(figures / "fig01_start_state_channels.png", dpi=140)
    plt.close(figure)
    produced["fig01_start_state_channels.png"] = {"channels": channels}

    # Fig 2 -- the policy's own target, which is what the question is about.
    figure, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    for record in records:
        series_path = out / "tables" / f"series_{record['label']}.npz"
        if not series_path.is_file():
            continue
        with np.load(series_path) as series:
            colour = styles.get(record["cell"], "tab:blue")
            axes[0].plot(series["seconds"], series["target_palm_z_right"], lw=1.2, color=colour,
                         alpha=0.8, label=f"{record['label']} target")
            axes[0].plot(series["seconds"], series["measured_palm_z_right"], lw=0.8, color=colour,
                         alpha=0.35, ls=":")
            axes[1].plot(series["seconds"], series["target_cube_distance_right"], lw=1.2,
                         color=colour, alpha=0.8, label=f"{record['label']} target")
    low, high = DEMO_START_PALM_P05_P95_M["right"]
    axes[0].axhspan(low, high, color="tab:blue", alpha=0.12,
                    label="demonstrated start palm p05-p95")
    axes[0].set_ylabel("right palm z\n(pelvis frame, m)")
    axes[1].set_ylabel("target right palm to\nnearest cube (m)")
    axes[1].set_xlabel("seconds after the measured onset")
    for axis in axes:
        axis.grid(alpha=0.25)
    axes[0].legend(fontsize=7, ncol=4)
    axes[0].set_title("Policy output (solid target, dotted realised), right arm")
    figure.tight_layout()
    figure.savefig(figures / "fig02_target_palm_and_distance.png", dpi=140)
    plt.close(figure)
    produced["fig02_target_palm_and_distance.png"] = {"series": True}

    # Fig 3 -- the paired effect against the pre-declared acceptance.
    if effects:
        figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        pairs = [row["pair"] for row in effects]
        z = [float(row["target_palm_z_right_first_5s_delta_m"]) for row in effects]
        d = [float(row["target_palm_cube_min_distance_right_first_5s_relative_delta"]) for row in effects]
        axes[0].bar(pairs, z, color="tab:green")
        axes[0].axhline(-ACCEPTANCE["target_palm_z_drop_m"], color="black", ls="--",
                        label="acceptance: >= 5 cm lower")
        axes[0].set_ylabel("B - A target right palm z (m)")
        axes[0].legend(fontsize=8)
        axes[1].bar(pairs, d, color="tab:green")
        axes[1].axhline(-ACCEPTANCE["target_palm_cube_min_distance_reduction"], color="black",
                        ls="--", label="acceptance: >= 30 % closer")
        axes[1].set_ylabel("B - A target to cube (relative)")
        axes[1].legend(fontsize=8)
        for axis in axes:
            axis.grid(alpha=0.25, axis="y")
        figure.suptitle("Paired A/B effect on the policy's own target, first 5 s (right arm)")
        figure.tight_layout()
        figure.savefig(figures / "fig03_paired_effect.png", dpi=140)
        plt.close(figure)
        produced["fig03_paired_effect.png"] = {"pairs": pairs}

    sheets = []
    for index in sorted({int(record["rollout"]) for record in records}):
        cells = [cell for cell in ("A", "B") if rollout_dirs(cell, out)]
        if len(cells) == 2:
            sheets.append(head_camera_sheet(out, cells, index))
    produced["head_camera_sheets"] = sheets
    write_json(figures / "figures.json", produced)
    return produced


# ----------------------------------------------------------------- manifest


def stage_manifest(args: argparse.Namespace) -> dict[str, Any]:
    out = repo_path(args.output_dir) if args.output_dir else DEFAULT_OUT
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "script": "scripts/groot-reset-pose-ab.py",
        "script_sha256": sha256(REPO_ROOT / "scripts" / "groot-reset-pose-ab.py"),
        "config": {
            "cells": {cell: {"driver": "scripts/blockstacking-rollout.py",
                             "reset_pose_file": cell in ("B", "R"),
                             "reset_pose_hold": cell == "B"}
                      for cell in CELLS},
            "windows_s": {name: list(bounds) for name, bounds in WINDOWS_S.items()},
            "acceptance": ACCEPTANCE,
            "start_window_rule": (
                f"median over the frames before the first {START_MOTION_SUSTAIN} consecutive samples "
                f"above {START_MOTION_RAD_S} rad/s, capped at {START_WINDOW_MAX_S}s"),
            "hold_rule": {
                "minimum_hold_s": 20.0, "stable_window_s": 2.0, "stable_rad": 0.02,
                "release_rad": 0.05, "release_sustain_s": 0.15,
            },
        },
        "inputs": {},
        "runs": {},
    }
    for name, path in (
        ("reset_pose_runtime", out / "tables" / "reset_pose_runtime.json"),
        ("reset_pose", out / "tables" / "reset_pose.json"),
        ("pose_candidates", out / "tables" / "pose_candidates.json"),
        ("start_state_population", out / "tables" / "start_state_population.json"),
        ("support_calibration", SUPPORT_DIR / "tables" / "support_calibration.npz"),
        ("channel_calibration", SUPPORT_DIR / "tables" / "channel_calibration.csv"),
        ("dataset_info", DATASET_ROOT / "meta" / "info.json"),
        ("dataset_episodes", DATASET_ROOT / "meta" / "episodes.jsonl"),
        ("profile", REPO_ROOT / "configs" / "profiles" / "isaac-g1-sonic-blockstacking-dex3.json"),
        ("campaign", CAMPAIGN / "campaign.json"),
    ):
        payload["inputs"][name] = {"path": rel(path), "sha256": sha256(path)}
    for cell in CELLS:
        session = session_dir(cell, out)
        meta = session / "session.json"
        if not meta.is_file():
            payload["runs"][cell] = {"present": False}
            continue
        session_payload = read_json(meta)
        payload["runs"][cell] = {
            "present": True,
            "session": rel(session),
            "session_json_sha256": sha256(meta),
            "command": session_payload.get("command"),
            "checkpoints": session_payload.get("checkpoints"),
            "checkpoint_step": session_payload.get("checkpoint_step"),
            "rollout_seconds": session_payload.get("rollout_seconds"),
            "settle_seconds": session_payload.get("settle_seconds"),
            "reset_pose": session_payload.get("reset_pose"),
            "rollouts": [directory.name for directory in rollout_dirs(cell, out)],
            "raw": {
                name: {"path": rel(session / "raw" / name), "sha256": sha256(session / "raw" / name)}
                for name in ("isaac.metrics.json", "isaac.tracking.parquet",
                             "isaac.tracking.jsonl", "isaac.samples.jsonl", "video.raw.mp4")
                if (session / "raw" / name).is_file()
            },
        }
    write_json(out / "manifest.json", payload)
    return {"cells": list(payload["runs"]), "manifest": str((out / "manifest.json").relative_to(REPO_ROOT))}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=("select-pose", "analyse", "figures", "manifest"))
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.stage == "select-pose":
        result = stage_select_pose(args)
        print(json.dumps(result, indent=2)[:4000])
    elif args.stage == "analyse":
        result = stage_analyse(args)
        print(json.dumps(result.get("decision_input", {}), indent=2)[:4000])
    elif args.stage == "figures":
        result = stage_figures(args)
        print(json.dumps(result, indent=2)[:2000])
    elif args.stage == "manifest":
        result = stage_manifest(args)
        print(json.dumps(result, indent=2)[:1000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
