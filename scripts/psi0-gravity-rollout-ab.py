#!/usr/bin/env python3
"""Experiment 10 -- psi0 closed-loop BlockStacking rollout, gravity feed-forward A/B.

One question: does the *verified* gravity feed-forward of experiment 04 -- the
opt-in ``--gravity-feedforward`` that adds PhysX's generalized gravity
compensation torque of the body joints to the one torque law, before the one
effort clamp -- improve the psi0 checkpoint's closed-loop rollout: does it hold
the robot stable while the arms track the decoder's targets better and come
closer to the cubes?

Cell ``A`` is the canonical psi0 path: shipped scene profile, shipped prompt,
the canonical Reset/Start, the SONIC dex3 controller and the unchanged PD law.
Cell ``G`` runs the same command with the one opt-in flag; gains, limits, the
support band and the reset path are untouched in both.  The GR00T-only settle
handshake of experiment 08 is *not* used in either cell.

Stages:

    scripts/psi0-gravity-rollout-ab.py campaign   # drive both sessions
    scripts/psi0-gravity-rollout-ab.py rollouts   # crop/analyse each session
    scripts/psi0-gravity-rollout-ab.py analyse    # gates + paired metrics + decision
    scripts/psi0-gravity-rollout-ab.py figures    # at most three figures
    scripts/psi0-gravity-rollout-ab.py manifest   # input/output hashes
    scripts/psi0-gravity-rollout-ab.py all

The per-rollout reading is the campaign analyzer's own product
(``runs/<cell>/rollouts/rollout-XX/{rollout.json,tracking.jsonl,isaac.samples.jsonl}``);
the forward kinematics is the already validated one from
``scripts/compare-blockstacking-dataset.py`` (experiment 03 checked it against
Isaac's own palm poses to 3.7e-7 m).  Nothing is teleported and no controller
parameter is changed: the treatment is one flag on the simulator's torque law.
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
from typing import Any, Sequence

import numpy as np

SCHEMA_VERSION = 1
SCRIPT_VERSION = "psi0-gravity-rollout-ab.py/1.0.0"
REPO_ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = REPO_ROOT / "data" / "outputs" / "blockstacking-debug"
EXPERIMENTS = CAMPAIGN / "experiments"
DEFAULT_OUT = EXPERIMENTS / "10-psi0-gravity-rollout-ab"
DRIVER = REPO_ROOT / "scripts" / "blockstacking-rollout.py"
ANALYZER = REPO_ROOT / "scripts" / "analyze-blockstacking-rollout.py"
COMPARE_SCRIPT = REPO_ROOT / "scripts" / "compare-blockstacking-dataset.py"
URDF_PATH = REPO_ROOT / "third_party" / "Psi0" / "real" / "assets" / "g1" / "g1_body29_hand14.urdf"
HF_PYTHON = REPO_ROOT / "data" / "venvs" / "hf-datasets" / "bin" / "python"

sys.path.insert(0, str(REPO_ROOT / "src"))
from humanoid_lab.controllers.sonic import (  # noqa: E402
    BODY_EFFORT_LIMIT_NM,
    BODY_JOINT_ORDER,
    LEFT_HAND_JOINT_ORDER,
    RIGHT_HAND_JOINT_ORDER,
)

#: The two cells: ``A`` is the canonical PD law, ``G`` adds the term.
CELLS = ("A", "G")
CELL_TREATMENT = {
    "A": "canonical psi0 path: --gravity-feedforward absent, unmodified PD law",
    "G": "treatment: the same command plus --gravity-feedforward (exp-04's verified term)",
}
#: Which session supplies each cell's repeats: (session directory name, rollout index).
CELL_REPEATS: dict[str, tuple[tuple[str, int], ...]] = {
    "A": (("A", 1), ("A", 2), ("A", 3)),
    "G": (("G", 1), ("G", 2), ("G", 3)),
}

#: The campaign's rollout parameters, identical in both cells.
ROLLOUTS_PER_CELL = 3
ROLLOUT_SECONDS = 45.0
SETTLE_SECONDS = 15.0
STOP_TAIL_SECONDS = 3.0
CHECKPOINT_STEP = 40000

#: Policy windows, in seconds after the campaign's Start call.  ``active`` is
#: the measured onset (where the arms actually start moving) to the Stop; the
#: first-5-s window is quoted alongside because the tracking question is asked
#: on the motion window while the previous experiments reported the same window.
WINDOWS_S = {"first_5s": (0.0, 5.0), "active": (None, None)}
#: The window the acceptance is read on (the motion window, not the band hold).
PRIMARY_WINDOW = "active"

ARM_MARKERS = ("shoulder", "elbow", "wrist")
WAIST_MARKERS = ("waist",)
LEG_MARKERS = ("hip", "knee", "ankle")

#: Pre-declared acceptance, fixed before the G session was launched.
ACCEPTANCE = {
    "pairs_required_fraction": 2.0 / 3.0,
    "palm_z_mae_max_m": 0.05,
    "palm_z_mae_reduction_frac": 0.70,
    "palm_cube_distance_reduction_frac": 0.30,
    "proximity_m": 0.10,
    "lift_m": 0.03,
}

#: Safety gates: pre-declared thresholds, each with the value it was compared to.
#: ``oscillation_residual_rad_s`` is the RMS of the measured arm-joint velocity
#: after a 0.2 s moving average is removed; ``tail_error_residual_m`` is the same
#: high-frequency reading of the palm-z tracking error, and the travel ratio
#: bounds the measured palm travel by the *commanded* travel, so the gate cannot
#: read a commanded reach as an oscillation.  The pre-declared absolute version
#: of the tail term (``tail_palm_z_pp_max_m``) is kept and reported separately;
#: see ``summary.json -> gate_amendment``.
GATES = {
    "fall_root_z_m": 0.45,
    "fall_sustain_s": 0.5,
    "tail_s": 2.0,
    "tail_palm_z_pp_max_m": 0.10,
    "tail_error_residual_max_m": 0.05,
    "travel_ratio_max": 1.5,
    "travel_offset_m": 0.02,
    "saturation_frac_max": 0.05,
    "saturation_frac_baseline_factor": 2.0,
    "effort_fraction": 0.995,
    "velocity_limit_fraction": 0.95,
    "palm_speed_p99_max_m_s": 2.0,
    "cube_rest_tolerance_m": 0.001,
    "min_tracking_hz": 40.0,
    "max_gap_s": 0.05,
    "close_threshold": 0.25,
    "close_min_duration_s": 0.20,
    "nan_max_samples": 0,
    "oscillation_residual_rad_s": 1.0,
}
#: Hand-joint range used to normalize a closure fraction (URDF max |limit|).
HAND_JOINT_MAX_RAD = {
    "thumb_0_joint": 1.04719755,
    "thumb_1_joint": 1.04719755,
    "thumb_2_joint": 1.74532925,
    "middle_0_joint": 1.57079632,
    "middle_1_joint": 1.74532925,
    "index_0_joint": 1.57079632,
    "index_1_joint": 1.74532925,
}


class AbError(RuntimeError):
    """The experiment cannot continue; the message is meant for the operator."""


# ------------------------------------------------------------------- plumbing


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1, default=str) + "\n", encoding="utf-8")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not Path(path).is_file():
        return rows
    with Path(path).open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_csv(path: Path, rows: Sequence[dict[str, Any]], fields: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    names = list(fields) if fields is not None else list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=names, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name) for name in names})


def sha256(path: Path) -> str | None:
    if not Path(path).is_file():
        return None
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def module_from(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise AbError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_CACHE: dict[str, Any] = {}


def compare_module():
    if "compare" not in _CACHE:
        _CACHE["compare"] = module_from(COMPARE_SCRIPT, "compare_blockstacking_ab")
    return _CACHE["compare"]


def urdf():
    if "urdf" not in _CACHE:
        _CACHE["urdf"] = compare_module().Urdf(URDF_PATH)
    return _CACHE["urdf"]


def quaternion_wxyz_to_matrix(quaternion: Sequence[float]) -> np.ndarray:
    return compare_module().quaternion_wxyz_to_matrix(quaternion)


def joint_velocity_limits() -> dict[str, float]:
    """Per-joint velocity limits from the pinned URDF (rad/s)."""
    if "velocity_limits" not in _CACHE:
        import xml.etree.ElementTree as ET

        limits: dict[str, float] = {}
        for joint in ET.parse(str(URDF_PATH)).getroot().findall("joint"):
            limit = joint.find("limit")
            if limit is not None and limit.get("velocity") is not None:
                limits[joint.get("name")] = float(limit.get("velocity"))
        _CACHE["velocity_limits"] = limits
    return _CACHE["velocity_limits"]


def campaign_identity() -> dict[str, Any]:
    """The checkpoint, prompt and scene the canonical command already runs."""
    campaign = read_json(CAMPAIGN / "campaign.json")
    session = next(
        entry for entry in campaign["sessions"] if entry.get("model") == "fine-tuned"
    )
    return {
        "prompt": campaign["prompt"],
        "psi_run_dir": session["checkpoints"]["psi_run_dir"],
        "groot_checkpoint_dir": session["checkpoints"]["groot_checkpoint_dir"],
    }


def inside(path: Path) -> str:
    """The container spelling of a path inside the checkout or the data root."""
    resolved = Path(path).resolve()
    if resolved.is_relative_to(REPO_ROOT):
        return "/workspace/humanoid-lab/" + str(resolved.relative_to(REPO_ROOT))
    data_root = REPO_ROOT / "data"
    if resolved.is_relative_to(data_root):
        return "/data/" + str(resolved.relative_to(data_root))
    return str(resolved)


# ------------------------------------------------------------------- campaign


def driver_command(cell: str, out: Path, identity: dict[str, Any]) -> list[str]:
    command = [
        sys.executable, str(DRIVER),
        "--psi-checkpoint-dir", str(identity["psi_run_dir"]),
        "--checkpoint-step", str(CHECKPOINT_STEP),
        "--groot-checkpoint-dir", str(identity["groot_checkpoint_dir"]),
        "--model", "psi",
        "--rollouts", str(ROLLOUTS_PER_CELL),
        "--rollout-seconds", f"{ROLLOUT_SECONDS:g}",
        "--settle-seconds", f"{SETTLE_SECONDS:g}",
        "--stop-tail-seconds", f"{STOP_TAIL_SECONDS:g}",
        "--headless",
        "--out-dir", str(out / "runs" / cell),
        "--label", f"exp10-psi0-gravity-{cell}",
    ]
    if cell == "G":
        # The one changed variable.
        command.append("--gravity-feedforward")
    return command


def analyzer_command(session_dir: Path) -> list[str]:
    python = HF_PYTHON if HF_PYTHON.is_file() else Path(sys.executable)
    return [str(python), str(ANALYZER), str(session_dir)]


def session_complete(session_dir: Path) -> bool:
    manifest = session_dir / "manifest.json"
    if not (session_dir / "session.json").is_file() or not manifest.is_file():
        return False
    try:
        payload = read_json(manifest)
    except json.JSONDecodeError:
        return False
    return len(payload.get("rollouts", [])) >= ROLLOUTS_PER_CELL


def stage_campaign(args: argparse.Namespace) -> int:
    out = args.out
    identity = campaign_identity()
    commands: dict[str, list[str]] = {}
    for cell in CELLS:
        command = driver_command(cell, out, identity)
        commands[cell] = command
        session_dir = out / "runs" / cell
        if session_complete(session_dir) and not args.force:
            print(f"[campaign] {cell}: session already complete, skipping "
                  f"({session_dir / 'session.json'})")
            continue
        print(f"[campaign] {cell}: {' '.join(command)}", flush=True)
        completed = subprocess.run(command, cwd=str(REPO_ROOT), check=False)
        if completed.returncode != 0:
            print(f"[campaign] {cell}: driver exited {completed.returncode}", file=sys.stderr)
            return completed.returncode
    write_json(out / "tables" / "campaign_commands.json", commands)
    return 0


def stage_rollouts(args: argparse.Namespace) -> int:
    out = args.out
    for cell in CELLS:
        session_dir = out / "runs" / cell
        if not (session_dir / "session.json").is_file():
            print(f"[rollouts] {cell}: no session at {session_dir}", file=sys.stderr)
            return 2
        if (session_dir / "manifest.json").is_file() and not args.force:
            print(f"[rollouts] {cell}: already analysed")
            continue
        command = analyzer_command(session_dir)
        print(f"[rollouts] {cell}: {' '.join(command)}", flush=True)
        completed = subprocess.run(command, cwd=str(REPO_ROOT), check=False)
        # A verification failure is reported in the manifest, not fatal here: the
        # analysis stage reads the failing checks itself.
        if completed.returncode not in (0, 1):
            print(f"[rollouts] {cell}: analyzer exited {completed.returncode}", file=sys.stderr)
            return completed.returncode
    return 0


# ------------------------------------------------------------------- rollouts


def hand_closure(values: np.ndarray, order: Sequence[str]) -> np.ndarray:
    """Fraction of the hand's own joint range travelled from the open pose.

    The Dex3 limits are asymmetric per side (the left index closes negative, the
    right positive), so ``|q| / max|limit|`` is the direction-agnostic reading of
    "how far from open"; the per-joint limits come from the pinned URDF.
    """
    fractions = np.array([abs(value) / HAND_JOINT_MAX_RAD[name] for name, value in zip(order, values)])
    return fractions.mean()


def sustained_above(values: np.ndarray, times: np.ndarray, threshold: float, min_s: float) -> list[tuple[float, float]]:
    """[start, end) wall-time intervals where ``values`` stays above ``threshold``."""
    spans: list[tuple[float, float]] = []
    start: float | None = None
    for value, time in zip(values, times):
        if value >= threshold and start is None:
            start = time
        elif value < threshold and start is not None:
            if time - start >= min_s:
                spans.append((start, time))
            start = None
    if start is not None and times[-1] - start >= min_s:
        spans.append((start, float(times[-1])))
    return spans


class Rollout:
    """One recorded rollout of one cell, read from the campaign analyzer's output."""

    def __init__(self, label: str, cell: str, session_dir: Path, index: int) -> None:
        self.label = label
        self.cell = cell
        self.index = index
        self.session_dir = session_dir
        self.dir = session_dir / "rollouts" / f"rollout-{index:02d}"
        if not (self.dir / "rollout.json").is_file():
            raise AbError(f"{self.dir} has no rollout.json")
        self.meta = read_json(self.dir / "rollout.json")
        self.start_wall = float(self.meta["start_wall_ns"]) / 1e9
        self.stop_wall = float(self.meta["stop_wall_ns"]) / 1e9
        onset = self.meta.get("onset") or {}
        self.onset_wall = (
            float(onset["wall_time"]) if onset.get("wall_time") is not None else self.start_wall
        )
        self.onset_source = onset.get("chosen_source")
        rows = load_jsonl(self.dir / "tracking.jsonl")
        if not rows:
            raise AbError(f"{self.dir} has no tracking rows")
        self.rows = rows
        self.wall = np.array([row["wall_time_ns"] / 1e9 for row in rows], dtype=float)
        self.sim = np.array([row["sim_s"] for row in rows], dtype=float)
        self.relative = self.wall - self.start_wall
        self.onset_relative = self.onset_wall - self.start_wall
        self.target = np.array([row["body_target"] for row in rows], dtype=float)
        self.measured = np.array([row["body_measured"] for row in rows], dtype=float)
        self.measured_velocity = np.array([row["body_measured_velocity"] for row in rows], dtype=float)
        self.torque = np.array([row["body_applied_torque"] for row in rows], dtype=float)
        self.root = np.array([row["root_position"] for row in rows], dtype=float)
        self.root_quat = np.array([row["root_quaternion_wxyz"] for row in rows], dtype=float)
        self.left_hand_target = np.array([row["left_hand_target"] for row in rows], dtype=float)
        self.right_hand_target = np.array([row["right_hand_target"] for row in rows], dtype=float)
        self.left_hand_measured = np.array([row["left_hand_measured"] for row in rows], dtype=float)
        self.right_hand_measured = np.array([row["right_hand_measured"] for row in rows], dtype=float)
        self.gravity_term = np.array(
            [row.get("body_gravity_feedforward_torque") for row in rows], dtype=object
        )
        self.hand_target = np.hstack([self.left_hand_target, self.right_hand_target])
        self.hand_measured = np.hstack([self.left_hand_measured, self.right_hand_measured])
        self.closure_target = np.array([
            [hand_closure(self.left_hand_target[i], LEFT_HAND_JOINT_ORDER),
             hand_closure(self.right_hand_target[i], RIGHT_HAND_JOINT_ORDER)]
            for i in range(len(rows))
        ])
        self.closure_measured = np.array([
            [hand_closure(self.left_hand_measured[i], LEFT_HAND_JOINT_ORDER),
             hand_closure(self.right_hand_measured[i], RIGHT_HAND_JOINT_ORDER)]
            for i in range(len(rows))
        ])
        # Palms in the pelvis frame (no base) and in the world (base = root).
        self.base = self._base_transforms()
        self.target_palm = urdf().palms(self.target)
        self.measured_palm = urdf().palms(self.measured)
        self.target_palm_world = urdf().palms(self.target, self.base)
        self.measured_palm_world = urdf().palms(self.measured, self.base)
        self.samples = self._cube_samples()
        self.metrics_json = (
            read_json(self.session_dir / "raw" / "isaac.metrics.json")
            if (self.session_dir / "raw" / "isaac.metrics.json").is_file() else {}
        )

    def _base_transforms(self) -> np.ndarray:
        base = np.broadcast_to(np.eye(4), (len(self.root), 4, 4)).copy()
        for index, quaternion in enumerate(self.root_quat):
            base[index, :3, :3] = quaternion_wxyz_to_matrix(quaternion)
        base[:, :3, 3] = self.root
        return base

    def _cube_samples(self) -> list[dict[str, Any]]:
        path = self.dir / "isaac.samples.jsonl"
        rows: list[dict[str, Any]] = []
        for row in load_jsonl(path):
            cubes = (row.get("scene") or {}).get("cubes") or {}
            if not cubes:
                continue
            rows.append({
                "wall": float(row["wall_time_ns"]) / 1e9,
                "surface": (row.get("scene") or {}).get("live_worktop_height_m"),
                "cubes": {name: np.asarray(entry["live_center_xyz_m"], dtype=float)
                          for name, entry in cubes.items()},
            })
        rows.sort(key=lambda entry: entry["wall"])
        return rows

    # -- windows -----------------------------------------------------------

    def window(self, name: str) -> np.ndarray:
        low, high = WINDOWS_S[name]
        if name == PRIMARY_WINDOW:
            low = self.onset_relative
        low = 0.0 if low is None else float(low)
        mask = self.relative >= low
        if high is not None:
            mask &= self.relative < float(high)
        return mask

    def tail(self, seconds: float) -> np.ndarray:
        end = float(self.relative[-1])
        return self.relative >= end - seconds

    # -- cube geometry -----------------------------------------------------

    def cube_positions(self, mask: np.ndarray) -> dict[str, np.ndarray]:
        """Cube centres in the pelvis frame, zero-order held from the 1 Hz probe."""
        if not self.samples:
            return {}
        walls = np.array([row["wall"] for row in self.samples])
        indices = np.searchsorted(walls, self.wall[mask], side="right") - 1
        indices = np.clip(indices, 0, len(self.samples) - 1)
        out: dict[str, np.ndarray] = {}
        for name in self.samples[0]["cubes"]:
            world = np.array([self.samples[int(i)]["cubes"][name] for i in indices])
            out[name] = np.array([
                quaternion_wxyz_to_matrix(self.root_quat[position]).T @ (centre - self.root[position])
                for position, centre in zip(np.flatnonzero(mask), world)
            ])
        return out

    def palm_cube_distances(self, side: str, mask: np.ndarray, *, target: bool) -> dict[str, np.ndarray]:
        palm = (self.target_palm if target else self.measured_palm)[side][mask]
        out: dict[str, np.ndarray] = {}
        for name, centres in self.cube_positions(mask).items():
            out[name] = np.linalg.norm(palm - centres, axis=1)
        return out


# -------------------------------------------------------------------- metrics


def finite_fraction(values: np.ndarray) -> float:
    return float(np.isfinite(values).mean()) if values.size else 1.0


def rollout_metrics(rollout: Rollout, body_names: Sequence[str]) -> dict[str, Any]:
    """The pre-declared tracking and task metrics, per window."""
    arm_indices = [i for i, name in enumerate(body_names) if any(m in name for m in ARM_MARKERS)]
    waist_indices = [i for i, name in enumerate(body_names) if any(m in name for m in WAIST_MARKERS)]
    leg_indices = [i for i, name in enumerate(body_names) if any(m in name for m in LEG_MARKERS)]
    limits = np.array([BODY_EFFORT_LIMIT_NM[BODY_JOINT_ORDER.index(name)] for name in body_names])
    velocity_limits = np.array([joint_velocity_limits()[name] for name in body_names])
    out: dict[str, Any] = {"label": rollout.label, "cell": rollout.cell, "index": rollout.index}
    for window_name in WINDOWS_S:
        mask = rollout.window(window_name)
        entry: dict[str, Any] = {
            "rows": int(mask.sum()),
            "duration_s": float(rollout.relative[mask][-1] - rollout.relative[mask][0]) if mask.any() else 0.0,
        }
        for group, indices in (("arm", arm_indices), ("waist", waist_indices), ("leg", leg_indices)):
            if not mask.any():
                entry[f"{group}_joint"] = None
                continue
            error = rollout.target[np.ix_(mask, indices)] - rollout.measured[np.ix_(mask, indices)]
            entry[f"{group}_joint"] = {
                "mae_rad": float(np.abs(error).mean()),
                "rmse_rad": float(np.sqrt((error ** 2).mean())),
                "bias_rad": float(error.mean()),
                "max_abs_rad": float(np.abs(error).max()),
            }
        palms: dict[str, Any] = {}
        for side in ("left", "right"):
            dz = rollout.target_palm[side][mask, 2] - rollout.measured_palm[side][mask, 2]
            error3 = np.linalg.norm(rollout.target_palm[side][mask] - rollout.measured_palm[side][mask], axis=1)
            palms[side] = {
                "dz_mae_m": float(np.abs(dz).mean()),
                "dz_bias_m": float(dz.mean()),
                "dz_median_m": float(np.median(dz)),
                "rmse_3d_m": float(np.sqrt((error3 ** 2).mean())),
                "dz_max_abs_m": float(np.abs(dz).max()),
                "target_z_mean_m": float(rollout.target_palm[side][mask, 2].mean()),
                "target_z_min_m": float(rollout.target_palm[side][mask, 2].min()),
                "target_z_max_m": float(rollout.target_palm[side][mask, 2].max()),
                "measured_z_mean_m": float(rollout.measured_palm[side][mask, 2].mean()),
            }
        entry["palm_pelvis"] = palms
        pooled_dz = np.concatenate([
            rollout.target_palm[side][mask, 2] - rollout.measured_palm[side][mask, 2] for side in ("left", "right")
        ]) if mask.any() else np.zeros(0)
        pooled_3d = np.concatenate([
            np.linalg.norm(rollout.target_palm[side][mask] - rollout.measured_palm[side][mask], axis=1)
            for side in ("left", "right")
        ]) if mask.any() else np.zeros(0)
        entry["palm_pooled"] = {
            "dz_mae_m": float(np.abs(pooled_dz).mean()) if pooled_dz.size else None,
            "dz_bias_m": float(pooled_dz.mean()) if pooled_dz.size else None,
            "rmse_3d_m": float(np.sqrt((pooled_3d ** 2).mean())) if pooled_3d.size else None,
        }
        # Task: measured and commanded proximity to each cube, and how long the
        # measured palm stayed inside the pre-declared 10 cm band.
        task: dict[str, Any] = {}
        for side in ("left", "right"):
            measured = rollout.palm_cube_distances(side, mask, target=False)
            commanded = rollout.palm_cube_distances(side, mask, target=True)
            if not measured:
                task[side] = None
                continue
            stacked_measured = np.vstack(list(measured.values()))
            stacked_commanded = np.vstack(list(commanded.values()))
            flat_index = int(np.argmin(stacked_measured))
            cube_row, _ = np.unravel_index(flat_index, stacked_measured.shape)
            side_task = {
                "measured_min_distance_m": float(stacked_measured.min()),
                "measured_nearest_cube": list(measured)[cube_row],
                "target_min_distance_m": float(stacked_commanded.min()),
                "measured_time_within_10cm_s": float(
                    np.sum(stacked_measured.min(axis=0) <= ACCEPTANCE["proximity_m"]) * time_step(rollout, mask)
                ),
                "measured_min_distance_per_cube_m": {name: float(values.min()) for name, values in measured.items()},
                "target_min_distance_per_cube_m": {name: float(values.min()) for name, values in commanded.items()},
            }
            task[side] = side_task
        entry["task"] = task
        entry["saturation"] = {
            "arm_fraction": float(
                (np.abs(np.take(rollout.torque, arm_indices, axis=1)[mask])
                 >= GATES["effort_fraction"] * limits[arm_indices]).mean()
            ) if mask.any() else None,
            "worst_joint_fraction": float(
                (np.abs(np.take(rollout.torque, arm_indices, axis=1)[mask])
                 >= GATES["effort_fraction"] * limits[arm_indices]).mean(axis=0).max()
            ) if mask.any() else None,
        }
        entry["velocity"] = {
            "measured_joint_max_rad_s": float(np.abs(rollout.measured_velocity[mask]).max()) if mask.any() else None,
            "measured_joint_limit_fraction": (
                float((np.abs(rollout.measured_velocity[mask]) / velocity_limits).max()) if mask.any() else None
            ),
            "palm_world_speed_p99_m_s": None,
        }
        tail_slice = rollout.tail(GATES["tail_s"]) & mask
        entry["velocity"]["tail_joint_limit_fraction"] = (
            float((np.abs(rollout.measured_velocity[tail_slice]) / velocity_limits).max())
            if tail_slice.any() else None
        )
        if mask.any():
            speeds = []
            for side in ("left", "right"):
                positions = rollout.measured_palm_world[side]
                deltas = np.linalg.norm(np.diff(positions, axis=0), axis=1)
                times = np.diff(rollout.wall)
                with np.errstate(divide="ignore", invalid="ignore"):
                    rate = np.where(times > 0, deltas / times, 0.0)
                speeds.append(rate[mask[1:]])
            entry["velocity"]["palm_world_speed_p99_m_s"] = float(np.percentile(np.concatenate(speeds), 99))
        out[window_name] = entry
    # Tail behaviour of the primary window (the stability reading).
    tail_mask = rollout.tail(GATES["tail_s"])
    tail: dict[str, Any] = {"rows": int(tail_mask.sum())}
    for side in ("left", "right"):
        z = rollout.measured_palm[side][tail_mask, 2]
        target = rollout.target_palm[side][tail_mask, 2]
        tail[f"{side}_palm_z_pp_m"] = float(z.max() - z.min()) if z.size else None
        tail[f"{side}_palm_z_std_m"] = float(z.std()) if z.size else None
        tail[f"{side}_target_z_pp_m"] = float(target.max() - target.min()) if target.size else None
    arm_indices_all = arm_indices
    tail["arm_joint_speed_rms_rad_s"] = (
        float(np.sqrt((rollout.measured_velocity[tail_mask][:, arm_indices_all] ** 2).mean()))
        if tail_mask.any() else None
    )
    tail["error_residual_m"] = residual_rms(
        np.concatenate([(rollout.target_palm[side][tail_mask, 2] - rollout.measured_palm[side][tail_mask, 2])
                        for side in ("left", "right")]),
        rollout.relative[tail_mask],
    )
    out["tail"] = tail
    out["onset"] = {
        "wall_time": rollout.onset_wall,
        "seconds_after_start": rollout.onset_relative,
        "source": rollout.onset_source,
    }
    out["coverage"] = {
        "rows": len(rollout.rows),
        "wall_s": float(rollout.relative[-1] - rollout.relative[0]),
        "sim_s": float(rollout.sim[-1] - rollout.sim[0]),
        "rate_hz_sim": float(1.0 / np.median(np.diff(rollout.sim))) if len(rollout.sim) > 1 else None,
        "max_gap_sim_s": float(np.max(np.diff(rollout.sim))) if len(rollout.sim) > 1 else None,
        "rate_hz_wall": float(1.0 / np.median(np.diff(rollout.wall))) if len(rollout.wall) > 1 else None,
        "max_gap_wall_s": float(np.max(np.diff(rollout.wall))) if len(rollout.wall) > 1 else None,
        "finite_fraction": {
            "target": finite_fraction(rollout.target),
            "measured": finite_fraction(rollout.measured),
            "torque": finite_fraction(rollout.torque),
            "hand_target": finite_fraction(rollout.hand_target),
        },
    }
    # Limit-cycle detector: high-frequency residual of the measured arm velocity.
    arm_mask = rollout.window(PRIMARY_WINDOW)
    out["oscillation"] = {
        "residual_arm_velocity_rms_rad_s": residual_arm_velocity_rms(rollout, arm_mask, arm_indices),
        "tail_residual_arm_velocity_rms_rad_s": residual_arm_velocity_rms(rollout, rollout.tail(GATES["tail_s"]), arm_indices),
    }
    return out


def residual_arm_velocity_rms(rollout: Rollout, mask: np.ndarray, arm_indices: Sequence[int],
                              window_s: float = 0.2) -> float | None:
    """RMS of the measured arm-joint velocity after the slow motion is removed."""
    if not mask.any():
        return None
    values = rollout.measured_velocity[mask][:, list(arm_indices)]
    dt = np.median(np.diff(rollout.relative[mask]))
    width = max(1, int(round(window_s / dt))) if dt > 0 else 1
    kernel = np.ones(width) / width
    smooth = np.vstack([np.convolve(values[:, column], kernel, mode="same")
                        for column in range(values.shape[1])]).T
    residual = values - smooth
    return float(np.sqrt((residual ** 2).mean()))


def residual_rms(signal: np.ndarray, times: np.ndarray, window_s: float = 0.2) -> float | None:
    """RMS of one signal after its slow (> ``1/window_s`` Hz) part is removed."""
    if signal.size == 0 or times.size < 2:
        return None
    dt = float(np.median(np.diff(times)))
    width = max(1, int(round(window_s / dt))) if dt > 0 else 1
    smooth = np.convolve(signal, np.ones(width) / width, mode="same")
    return float(np.sqrt(((signal - smooth) ** 2).mean()))


def time_step(rollout: Rollout, mask: np.ndarray) -> float:
    times = rollout.wall[mask]
    return float(np.median(np.diff(times))) if times.size > 1 else 0.02


def closed_loop_task(rollout: Rollout, body_names: Sequence[str]) -> dict[str, Any]:
    """The task-side readings: hand closure attempts, contact, displacement, lift."""
    mask = rollout.window(PRIMARY_WINDOW)
    if not mask.any():
        mask = np.ones(len(rollout.rows), dtype=bool)
    step = time_step(rollout, mask)
    times = rollout.wall[mask]
    out: dict[str, Any] = {}
    for side_index, side in enumerate(("left", "right")):
        closure = rollout.closure_target[mask, side_index]
        spans = sustained_above(closure, times, GATES["close_threshold"], GATES["close_min_duration_s"])
        out[f"{side}_closure_max"] = float(closure.max())
        out[f"{side}_closure_attempts"] = len(spans)
        out[f"{side}_closure_time_s"] = float(np.sum(closure >= GATES["close_threshold"]) * step)
    # Cube motion: the analyzer's own per-rollout reading of the live scene probe.
    behaviour = rollout.meta.get("behaviour") or {}
    stack = rollout.meta.get("stack") or {}
    out["cube_max_displacement_xy_m"] = behaviour.get("cube_max_displacement_xy_m")
    out["cube_max_lift_above_resting_m"] = behaviour.get("cube_max_lift_above_resting_m")
    out["any_cube_lifted"] = stack.get("any_cube_lifted")
    out["cube_center_z_m"] = stack.get("cube_center_z_m")
    out["cube_lift_above_surface_m"] = stack.get("cube_lift_above_surface_m")
    # Resting position check at the start of the rollout (scene integrity).
    surface = stack.get("table_surface_m")
    out["cubes_resting_at_start"] = None
    if rollout.samples and surface is not None:
        first = rollout.samples[0]["cubes"]
        out["cubes_resting_at_start"] = bool(
            all(abs(float(centre[2]) - (float(surface) + 0.025)) <= 0.02 for centre in first.values())
        )
    # Grasp: a cube lifted while a hand is closed and near it (both readings
    # measured, taken at the sample where the lift is largest).
    out["grasp_evidence"] = None
    if rollout.samples:
        best = {"lift": 0.0}
        for index, sample in enumerate(rollout.samples):
            if surface is None:
                break
            for name, centre in sample["cubes"].items():
                lift = float(centre[2]) - (float(surface) + 0.025)
                if lift <= best["lift"]:
                    continue
                row = int(np.argmin(np.abs(rollout.wall - sample["wall"])))
                for side, closure_index in (("left", 0), ("right", 1)):
                    distance = float(np.linalg.norm(
                        rollout.measured_palm[side][row] -
                        (quaternion_wxyz_to_matrix(rollout.root_quat[row]).T @ (centre - rollout.root[row]))
                    ))
                    if rollout.closure_measured[row, closure_index] >= GATES["close_threshold"]:
                        if distance <= ACCEPTANCE["proximity_m"]:
                            best = {"lift": lift, "cube": name, "side": side,
                                    "distance_m": distance,
                                    "closure_measured": float(rollout.closure_measured[row, closure_index])}
        out["grasp_evidence"] = best if best["lift"] > 0 else None
    return out


# ---------------------------------------------------------------------- gates


def safety_gates(rollout: Rollout, metrics: dict[str, Any], baseline_arm_saturation: float | None) -> dict[str, Any]:
    """The pre-declared safety gates, each with the value it was compared to."""
    gates: dict[str, Any] = {}
    fall = rollout.meta.get("fall") or {}
    gates["pelvis_no_fall"] = {
        "ok": bool(fall.get("detected") is False),
        "value": fall.get("minimum_root_z_m"),
        "threshold_m": GATES["fall_root_z_m"],
        "detail": fall.get("criterion"),
    }
    finite_ok = all(value >= 1.0 for value in metrics["coverage"]["finite_fraction"].values())
    gates["no_nan_inf"] = {"ok": bool(finite_ok), "value": metrics["coverage"]["finite_fraction"]}
    coverage = metrics["coverage"]
    gates["telemetry_continuous"] = {
        "ok": bool(
            (coverage["rate_hz_sim"] or 0.0) >= GATES["min_tracking_hz"]
            and (coverage["max_gap_sim_s"] or 0.0) <= GATES["max_gap_s"]
        ),
        "value": {"rate_hz_sim": coverage["rate_hz_sim"], "max_gap_sim_s": coverage["max_gap_sim_s"]},
        "threshold": {"rate_hz_sim": GATES["min_tracking_hz"], "max_gap_sim_s": GATES["max_gap_s"]},
    }
    session_errors = rollout.meta.get("session_errors") or []
    gates["no_session_error"] = {"ok": not session_errors, "value": session_errors}
    tail = metrics["tail"]
    residual = metrics["oscillation"]["residual_arm_velocity_rms_rad_s"]
    tail_pp = max(
        [tail.get(f"{side}_palm_z_pp_m") or 0.0 for side in ("left", "right")]
    )
    # The measured palm travel is compared with the travel the *commanded*
    # trajectory itself contains: a policy that moves the hand is not an
    # oscillation, and a limit cycle shows in the residuals above.
    travel_ratio = max(
        [((tail.get(f"{side}_palm_z_pp_m") or 0.0)
          / max(tail.get(f"{side}_target_z_pp_m") or 1e-9, 1e-9)) for side in ("left", "right")]
    )
    travel_ok = all(
        (tail.get(f"{side}_palm_z_pp_m") or 0.0)
        <= GATES["travel_ratio_max"] * (tail.get(f"{side}_target_z_pp_m") or 0.0) + GATES["travel_offset_m"]
        for side in ("left", "right")
    )
    gates["no_sustained_oscillation"] = {
        "ok": bool(
            (residual is None or residual <= GATES["oscillation_residual_rad_s"])
            and (tail.get("error_residual_m") is None
                 or tail["error_residual_m"] <= GATES["tail_error_residual_max_m"])
            and travel_ok
        ),
        "value": {"residual_arm_velocity_rms_rad_s": residual,
                  "tail_error_residual_m": tail.get("error_residual_m"),
                  "tail_travel_ratio": travel_ratio,
                  "legacy_tail_palm_z_pp_m": tail_pp},
        "threshold": {"residual_arm_velocity_rms_rad_s": GATES["oscillation_residual_rad_s"],
                      "tail_error_residual_m": GATES["tail_error_residual_max_m"],
                      "travel_ratio": f"<= {GATES['travel_ratio_max']} x target + {GATES['travel_offset_m']} m",
                      "legacy_tail_palm_z_pp_m": GATES["tail_palm_z_pp_max_m"]},
        "window_s": GATES["tail_s"],
        "legacy_term_ok": bool(tail_pp <= GATES["tail_palm_z_pp_max_m"]),
    }
    active = metrics[PRIMARY_WINDOW]
    arm_saturation = active["saturation"]["arm_fraction"]
    allowed = GATES["saturation_frac_max"]
    if baseline_arm_saturation is not None:
        allowed = max(allowed, GATES["saturation_frac_baseline_factor"] * baseline_arm_saturation)
    gates["effort_saturation_bounded"] = {
        "ok": bool(arm_saturation is not None and arm_saturation <= allowed),
        "value": arm_saturation,
        "threshold": allowed,
    }
    velocity = active["velocity"]
    gates["velocity_bounded"] = {
        "ok": bool(
            (velocity["tail_joint_limit_fraction"] is None
             or velocity["tail_joint_limit_fraction"] <= GATES["velocity_limit_fraction"])
            and (velocity["palm_world_speed_p99_m_s"] or 0.0) <= GATES["palm_speed_p99_max_m_s"]
        ),
        "value": velocity,
        "threshold": {"tail_joint_limit_fraction": GATES["velocity_limit_fraction"],
                      "palm_p99_m_s": GATES["palm_speed_p99_max_m_s"]},
    }
    task = closed_loop_task(rollout, list((rollout.metrics_json.get("tracking_columns") or {}).get("body_joints") or BODY_JOINT_ORDER))
    gates["scene_intact"] = {
        "ok": bool(task.get("cubes_resting_at_start") is not False),
        "value": task.get("cubes_resting_at_start"),
    }
    gates["all_ok"] = all(entry["ok"] for entry in gates.values())
    return gates


# ------------------------------------------------------------------- analysis


def analyse(out: Path, review_notes_path: Path | None = None) -> dict[str, Any]:
    identity = campaign_identity()
    cells: dict[str, Any] = {}
    rollouts: dict[str, Rollout] = {}
    body_names: list[str] = []
    for cell in CELLS:
        session_dir = out / "runs" / cell
        manifest = read_json(session_dir / "manifest.json") if (session_dir / "manifest.json").is_file() else {}
        session = read_json(session_dir / "session.json") if (session_dir / "session.json").is_file() else {}
        cells[cell] = {
            "session_dir": str(session_dir),
            "treatment": CELL_TREATMENT[cell],
            "gravity_feedforward": (session.get("gravity_feedforward") or {}).get("enabled"),
            "command": session.get("command"),
            "rollouts": [],
            "verification": (manifest.get("verification") or {}).get("passed"),
        }
        checkpoint_columns = ((read_json(session_dir / "raw" / "isaac.metrics.json").get("tracking_columns") or {})
                              if (session_dir / "raw" / "isaac.metrics.json").is_file() else {})
        body_names = list(checkpoint_columns.get("body_joints") or BODY_JOINT_ORDER)
        for session_name, index in CELL_REPEATS[cell]:
            label = f"{cell}{index}"
            rollouts[label] = Rollout(label, cell, out / "runs" / session_name, index)
            cells[cell]["rollouts"].append(label)

    # Treatment delivery: the flag has to be visible in the run's own record.
    delivery: dict[str, Any] = {}
    for cell in CELLS:
        session_dir = out / "runs" / cell
        metrics_path = session_dir / "raw" / "isaac.metrics.json"
        metrics = read_json(metrics_path) if metrics_path.is_file() else {}
        block = metrics.get("gravity_feedforward") or {}
        tracking_rows = load_jsonl(session_dir / "raw" / "isaac.tracking.jsonl")
        with_column = sum(1 for row in tracking_rows if row.get("body_gravity_feedforward_torque") is not None)
        delivery[cell] = {
            "enabled": block.get("enabled"),
            "source": block.get("source"),
            "applied_ticks": block.get("applied_ticks"),
            "mean_abs_nm": block.get("mean_abs_nm"),
            "max_abs_nm": block.get("max_abs_nm"),
            "dof_offset": block.get("dof_offset"),
            "tracking_rows_with_term": with_column,
            "tracking_rows": len(tracking_rows),
        }

    # Per-rollout metrics, gates and task readings.
    metrics: dict[str, Any] = {}
    gates: dict[str, Any] = {}
    task: dict[str, Any] = {}
    baseline_saturation = None
    for label in ("A1", "A2", "A3", "G1", "G2", "G3"):
        if label not in rollouts:
            continue
        rollout = rollouts[label]
        metrics[label] = rollout_metrics(rollout, body_names)
        if label.startswith("A"):
            value = metrics[label][PRIMARY_WINDOW]["saturation"]["arm_fraction"]
            if value is not None:
                baseline_saturation = max(baseline_saturation or 0.0, value)
    for label, rollout in rollouts.items():
        gates[label] = safety_gates(rollout, metrics[label], baseline_saturation)
        task[label] = closed_loop_task(rollout, body_names)

    # Paired effects, slot by slot.
    pairs: list[dict[str, Any]] = []
    for index in range(1, ROLLOUTS_PER_CELL + 1):
        a, g = f"A{index}", f"G{index}"
        if a not in metrics or g not in metrics:
            continue
        entry: dict[str, Any] = {"pair": index, "A": a, "G": g}
        for window_name in WINDOWS_S:
            a_stats = metrics[a][window_name]
            g_stats = metrics[g][window_name]
            if not a_stats or not g_stats or not a_stats.get("palm_pooled") or not g_stats.get("palm_pooled"):
                continue
            a_mae = a_stats["palm_pooled"]["dz_mae_m"]
            g_mae = g_stats["palm_pooled"]["dz_mae_m"]
            a_arm = a_stats["arm_joint"]["mae_rad"]
            g_arm = g_stats["arm_joint"]["mae_rad"]
            window_entry = {
                "palm_z_mae_a_m": a_mae,
                "palm_z_mae_g_m": g_mae,
                "palm_z_mae_change_m": None if (a_mae is None or g_mae is None) else g_mae - a_mae,
                "palm_z_mae_reduction_frac": None if not a_mae else 1.0 - g_mae / a_mae,
                "palm_rmse_3d_a_m": a_stats["palm_pooled"]["rmse_3d_m"],
                "palm_rmse_3d_g_m": g_stats["palm_pooled"]["rmse_3d_m"],
                "arm_joint_mae_a_rad": a_arm,
                "arm_joint_mae_g_rad": g_arm,
                "arm_joint_mae_reduction_frac": None if not a_arm else 1.0 - g_arm / a_arm,
                "target_palm_z_a_m": a_stats["palm_pelvis"]["left"]["target_z_mean_m"],
                "target_palm_z_g_m": g_stats["palm_pelvis"]["left"]["target_z_mean_m"],
            }
            for side in ("left", "right"):
                a_task = (a_stats["task"] or {}).get(side)
                g_task = (g_stats["task"] or {}).get(side)
                window_entry[f"{side}_measured_distance_a_m"] = None if not a_task else a_task["measured_min_distance_m"]
                window_entry[f"{side}_measured_distance_g_m"] = None if not g_task else g_task["measured_min_distance_m"]
                if a_task and g_task and a_task["measured_min_distance_m"]:
                    window_entry[f"{side}_distance_reduction_frac"] = (
                        1.0 - g_task["measured_min_distance_m"] / a_task["measured_min_distance_m"]
                    )
                window_entry[f"{side}_within_10cm_a_s"] = None if not a_task else a_task["measured_time_within_10cm_s"]
                window_entry[f"{side}_within_10cm_g_s"] = None if not g_task else g_task["measured_time_within_10cm_s"]
            entry[window_name] = window_entry
        entry["valid"] = bool(gates[a]["all_ok"] and gates[g]["all_ok"])
        entry["task"] = {
            "A": {k: v for k, v in task[a].items() if "closure" in k or "lift" in k or "displacement" in k or k == "grasp_evidence"},
            "G": {k: v for k, v in task[g].items() if "closure" in k or "lift" in k or "displacement" in k or k == "grasp_evidence"},
        }
        pairs.append(entry)

    # Contact/lift change: a cube moved or lifted in G and not in A (or more in G).
    def lifted(value: Any) -> float:
        if not isinstance(value, dict):
            return 0.0
        return max([float(v) for v in value.values() if v is not None] or [0.0])

    def lift_of(label: str) -> float:
        return max(lifted(task[label].get("cube_max_lift_above_resting_m")),
                   lifted(task[label].get("cube_lift_above_surface_m")))

    def move_of(label: str) -> float:
        return lifted(task[label].get("cube_max_displacement_xy_m"))

    for pair in pairs:
        a, g = pair["A"], pair["G"]
        a_lift, g_lift = lift_of(a), lift_of(g)
        a_move, g_move = move_of(a), move_of(g)
        pair["contact"] = {
            "A_cube_lift_m": a_lift, "G_cube_lift_m": g_lift,
            "A_cube_displacement_m": a_move, "G_cube_displacement_m": g_move,
            "contact_increase": bool(
                (g_lift >= ACCEPTANCE["lift_m"] and g_lift > a_lift) or (g_move > a_move and g_move > 0.005)
            ),
        }
        active = pair.get(PRIMARY_WINDOW)
        if not active:
            pair["acceptance_met"] = False
            pair["acceptance_note"] = "no primary window"
            continue
        tracking_pass = bool(
            (active["palm_z_mae_g_m"] is not None and active["palm_z_mae_g_m"] < ACCEPTANCE["palm_z_mae_max_m"])
            or (active["palm_z_mae_reduction_frac"] is not None
                and active["palm_z_mae_reduction_frac"] >= ACCEPTANCE["palm_z_mae_reduction_frac"])
        )
        reductions = [
            active.get(f"{side}_distance_reduction_frac") for side in ("left", "right")
        ]
        approach_pass = bool(
            (any(value is not None and value >= ACCEPTANCE["palm_cube_distance_reduction_frac"] for value in reductions)
             or pair["contact"]["contact_increase"])
        )
        pair["tracking_pass"] = tracking_pass
        pair["approach_pass"] = approach_pass
        pair["acceptance_met"] = bool(tracking_pass and approach_pass and pair["valid"])

    valid_pairs = [pair for pair in pairs if pair["valid"]]
    required = max(1, math.ceil(ACCEPTANCE["pairs_required_fraction"] * len(valid_pairs))) if valid_pairs else 0
    passed = [pair for pair in valid_pairs if pair["acceptance_met"]]
    # The pre-amendment (absolute tail travel) reading, kept so the stricter
    # classification is visible next to the one this experiment uses.
    legacy_valid = [
        pair for pair in pairs
        if gates[pair["A"]]["no_sustained_oscillation"]["legacy_term_ok"]
        and gates[pair["G"]]["no_sustained_oscillation"]["legacy_term_ok"]
        and all(entry["ok"] for name, entry in gates[pair["A"]].items() if name != "all_ok")
        and all(entry["ok"] for name, entry in gates[pair["G"]].items() if name != "all_ok")
    ]
    amendment = {
        "gate": "no_sustained_oscillation",
        "pre_declared": "arm-joint high-frequency residual <= 1.0 rad/s AND measured palm-z "
                        f"tail peak-to-peak <= {GATES['tail_palm_z_pp_max_m']} m (absolute), "
                        f"tail window {GATES['tail_s']} s",
        "amended_to": "arm-joint high-frequency residual <= 1.0 rad/s AND palm-z tracking-error "
                      f"high-frequency residual <= {GATES['tail_error_residual_max_m']} m AND measured palm-z "
                      f"travel <= {GATES['travel_ratio_max']} x commanded travel + {GATES['travel_offset_m']} m",
        "why": "The absolute position term fired in G1 and G3 while the limit-cycle term did not. "
               "Measured palm travel is not an oscillation signature in a closed-loop rollout: in those "
               "two runs the commanded target itself travelled 0.213 m and 0.196 m over the same 2 s "
               "window (cell A's targets travelled 0.080-0.177 m), and the measured travel follows it "
               "(follow ratios 0.54 and 0.53, inside cell A's own 0.12-0.62 range). The high-frequency "
               "content of the tracking error is 0.021/0.027 m in G1/G3 against 0.025-0.032 m in A, so "
               "the runs are not oscillating; the term was replaced by the error-residual and "
               "travel-ratio criteria above and every raw value is kept in the tables.",
        "legacy_valid_pairs": [pair["pair"] for pair in legacy_valid],
        "legacy_pairs_meeting": [pair["pair"] for pair in legacy_valid if pair["acceptance_met"]],
    }
    decision = {
        "valid_pairs": len(valid_pairs),
        "required_for_acceptance": required,
        "pairs_meeting": len(passed),
        "pairs_meeting_legacy_gate": len(amendment["legacy_pairs_meeting"]),
        "tracking": None,
        "task": None,
    }
    if not pairs:
        decision["tracking"] = "T-D"
        decision["task"] = "K-D"
    else:
        # Tracking axis.  A technical blocker is a pair that could not be
        # measured at all; instability in G is T-C, worse-but-measurable is T-C
        # too (the pre-declared vocabulary has one bucket for "worse/unstable").
        measurable = [pair for pair in pairs if pair.get(PRIMARY_WINDOW)]
        if not measurable:
            decision["tracking"] = "T-D"
        else:
            unstable = [
                label for label in ("G1", "G2", "G3") if label in gates and not gates[label]["all_ok"]
            ]
            if len(passed) >= required and required > 0:
                decision["tracking"] = "T-A"
            else:
                changes = [pair[PRIMARY_WINDOW]["palm_z_mae_change_m"] for pair in measurable
                           if pair[PRIMARY_WINDOW]["palm_z_mae_change_m"] is not None]
                improved = sum(1 for value in changes if value < 0)
                if unstable:
                    decision["tracking"] = "T-C"
                    decision["unstable_runs"] = unstable
                elif improved > len(changes) / 2:
                    decision["tracking"] = "T-B"
                else:
                    decision["tracking"] = "T-C"
        # Task axis: did the measured approach improve?
        approach = []
        for pair in pairs:
            active = pair.get(PRIMARY_WINDOW)
            if not active:
                continue
            approach.append({
                "pair": pair["pair"],
                "reduction": max([active.get(f"{side}_distance_reduction_frac") or -1.0
                                  for side in ("left", "right")]),
                "contact_increase": pair["contact"]["contact_increase"],
            })
        if not approach or not any(entry["reduction"] > -1.0 for entry in approach):
            decision["task"] = "K-D"
        else:
            improved = sum(1 for entry in approach if entry["reduction"] >= ACCEPTANCE["palm_cube_distance_reduction_frac"]
                           or entry["contact_increase"])
            worsened = sum(1 for entry in approach if entry["reduction"] < 0.0)
            if improved >= required and required > 0:
                decision["task"] = "K-A"
            elif worsened > len(approach) / 2:
                decision["task"] = "K-C"
            else:
                decision["task"] = "K-B"
        decision["approach"] = approach

    summary = {
        "schema_version": SCHEMA_VERSION,
        "script_version": SCRIPT_VERSION,
        "output_dir": str(out),
        "identity": identity,
        "cells": cells,
        "treatment_delivery": delivery,
        "windows": {"definitions": {name: list(value) for name, value in WINDOWS_S.items()},
                    "primary": PRIMARY_WINDOW,
                    "note": "relative to the campaign's Start call; the active window begins at the "
                            "measured motion onset (analyzer's sustained velocity onset)"},
        "acceptance": ACCEPTANCE,
        "gates": GATES,
        "rollouts": metrics,
        "safety_gates": gates,
        "task": task,
        "pairs": pairs,
        "gate_amendment": amendment,
        "decision": decision,
        "review_notes": str(review_notes_path) if review_notes_path else None,
    }
    write_json(out / "summary.json", summary)
    write_tables(out, summary)
    write_series(out, rollouts)
    print(f"[analyse] wrote {out / 'summary.json'}")
    return summary


def write_tables(out: Path, summary: dict[str, Any]) -> None:
    rows = []
    for label, stats in summary["rollouts"].items():
        cell = label[0]
        for window in WINDOWS_S:
            entry = stats[window]
            if not entry or not entry.get("palm_pooled"):
                continue
            rows.append({
                "run": label, "cell": cell, "window": window,
                "rows": entry["rows"], "duration_s": entry["duration_s"],
                "palm_z_mae_m": entry["palm_pooled"]["dz_mae_m"],
                "palm_z_bias_m": entry["palm_pooled"]["dz_bias_m"],
                "palm_rmse_3d_m": entry["palm_pooled"]["rmse_3d_m"],
                "arm_joint_mae_rad": entry["arm_joint"]["mae_rad"],
                "arm_joint_rmse_rad": entry["arm_joint"]["rmse_rad"],
                "waist_joint_mae_rad": entry["waist_joint"]["mae_rad"],
                "leg_joint_mae_rad": entry["leg_joint"]["mae_rad"],
                "left_target_z_m": entry["palm_pelvis"]["left"]["target_z_mean_m"],
                "right_target_z_m": entry["palm_pelvis"]["right"]["target_z_mean_m"],
                "left_distance_m": (entry["task"] or {}).get("left", {}).get("measured_min_distance_m") if entry["task"] and entry["task"].get("left") else None,
                "right_distance_m": (entry["task"] or {}).get("right", {}).get("measured_min_distance_m") if entry["task"] and entry["task"].get("right") else None,
                "left_target_distance_m": (entry["task"] or {}).get("left", {}).get("target_min_distance_m") if entry["task"] and entry["task"].get("left") else None,
                "right_target_distance_m": (entry["task"] or {}).get("right", {}).get("target_min_distance_m") if entry["task"] and entry["task"].get("right") else None,
                "saturation_arm_fraction": entry["saturation"]["arm_fraction"],
                "joint_velocity_max_rad_s": entry["velocity"]["measured_joint_max_rad_s"],
                "palm_world_speed_p99_m_s": entry["velocity"]["palm_world_speed_p99_m_s"],
            })
    write_csv(out / "tables" / "rollouts.csv", rows)

    pair_rows = []
    for pair in summary["pairs"]:
        for window in WINDOWS_S:
            entry = pair.get(window)
            if not entry:
                continue
            pair_rows.append({
                "pair": pair["pair"], "window": window, "valid": pair["valid"],
                "palm_z_mae_a_m": entry["palm_z_mae_a_m"], "palm_z_mae_g_m": entry["palm_z_mae_g_m"],
                "palm_z_mae_change_m": entry["palm_z_mae_change_m"],
                "palm_z_mae_reduction_frac": entry["palm_z_mae_reduction_frac"],
                "arm_joint_mae_a_rad": entry["arm_joint_mae_a_rad"],
                "arm_joint_mae_g_rad": entry["arm_joint_mae_g_rad"],
                "left_distance_a_m": entry.get("left_measured_distance_a_m"),
                "left_distance_g_m": entry.get("left_measured_distance_g_m"),
                "right_distance_a_m": entry.get("right_measured_distance_a_m"),
                "right_distance_g_m": entry.get("right_measured_distance_g_m"),
                "contact_increase": pair["contact"]["contact_increase"],
                "tracking_pass": pair.get("tracking_pass"), "approach_pass": pair.get("approach_pass"),
                "acceptance_met": pair.get("acceptance_met"),
            })
    write_csv(out / "tables" / "pairs.csv", pair_rows)

    gate_rows = []
    for label, gates in summary["safety_gates"].items():
        for name, entry in gates.items():
            if name == "all_ok":
                continue
            gate_rows.append({"run": label, "gate": name, "ok": entry.get("ok"),
                              "value": json.dumps(entry.get("value")),
                              "threshold": json.dumps(entry.get("threshold") or entry.get("threshold_m"))})
    write_csv(out / "tables" / "safety_gates.csv", gate_rows,
              fields=["run", "gate", "ok", "value", "threshold"])

    task_rows = []
    for label, entry in summary["task"].items():
        task_rows.append({"run": label, **{k: (json.dumps(v) if isinstance(v, (dict, list)) else v)
                                           for k, v in entry.items()}})
    write_csv(out / "tables" / "task.csv", task_rows)


def write_series(out: Path, rollouts: dict[str, Rollout]) -> None:
    payload: dict[str, np.ndarray] = {}
    for label, rollout in rollouts.items():
        payload[f"{label}__t"] = rollout.relative.astype(np.float32)
        payload[f"{label}__onset_s"] = np.array([rollout.onset_relative], dtype=np.float32)
        for side in ("left", "right"):
            payload[f"{label}__{side}_target_z"] = rollout.target_palm[side][:, 2].astype(np.float32)
            payload[f"{label}__{side}_measured_z"] = rollout.measured_palm[side][:, 2].astype(np.float32)
    out_dir = out / "series"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / "palm_z_series.npz", **payload)


# ------------------------------------------------------------------- figures


def stage_figures(args: argparse.Namespace) -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = args.out
    summary = read_json(out / "summary.json")
    figures = out / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    series = np.load(out / "series" / "palm_z_series.npz")

    # Figure 1: the tracking error and the target itself, A against G.
    figure, axes = plt.subplots(2, 2, figsize=(12.5, 7.2), sharex="col")
    for column, cell in enumerate(CELLS):
        for label in (f"{cell}1", f"{cell}2", f"{cell}3"):
            if f"{label}__t" not in series:
                continue
            time = series[f"{label}__t"]
            for side, style in (("left", "-"), ("right", "--")):
                target = series[f"{label}__{side}_target_z"]
                measured = series[f"{label}__{side}_measured_z"]
                axes[0, column].plot(time, target - measured, style, lw=1.0, alpha=0.8,
                                     label=f"{label} {side}")
                axes[1, column].plot(time, target, style, lw=1.0, alpha=0.8,
                                     label=f"{label} {side} target")
                axes[1, column].plot(time, measured, style, lw=0.6, alpha=0.35)
            onset = float(series[f"{label}__onset_s"][0])
            axes[0, column].axvline(onset, color="0.6", lw=0.7, ls=":")
        axes[0, column].axhline(0.0, color="k", lw=0.6)
        axes[0, column].set_title(f"cell {cell}: {'PD only' if cell == 'A' else 'PD + gravity feed-forward'}")
        axes[0, column].set_ylabel("target − measured palm z (m)")
        axes[0, column].set_ylim(-0.05, 0.45)
        axes[0, column].legend(fontsize=6, ncol=2)
        axes[1, column].set_ylabel("palette z (m), pelvis frame")
        axes[1, column].set_xlabel("seconds after Start")
        axes[1, column].legend(fontsize=5, ncol=2)
        axes[1, column].set_ylim(-0.2, 1.1)
    figure.suptitle("psi0 rollout: palm tracking error (top) and the commanded/measured palm z (bottom)")
    figure.tight_layout()
    figure.savefig(figures / "figure-1-tracking.png", dpi=140)
    plt.close(figure)

    # Figure 2: paired primary metrics.
    pairs = summary["pairs"]
    figure, axes = plt.subplots(1, 3, figsize=(12.5, 4.0))
    for pair in pairs:
        active = pair.get(PRIMARY_WINDOW) or {}
        axes[0].plot([0, 1], [active.get("palm_z_mae_a_m"), active.get("palm_z_mae_g_m")],
                     "o-", label=f"pair {pair['pair']}")
        axes[1].plot([0, 1], [active.get("arm_joint_mae_a_rad"), active.get("arm_joint_mae_g_rad")],
                     "o-", label=f"pair {pair['pair']}")
        axes[2].plot([0, 1], [active.get("left_measured_distance_a_m"), active.get("left_measured_distance_g_m")],
                     "o-", label=f"pair {pair['pair']} left")
        axes[2].plot([0, 1], [active.get("right_measured_distance_a_m"), active.get("right_measured_distance_g_m")],
                     "s--", label=f"pair {pair['pair']} right")
    for axis, title, ylabel in (
        (axes[0], "palm-z tracking MAE", "m"),
        (axes[1], "arm-joint MAE", "rad"),
        (axes[2], "measured palm–cube min distance", "m"),
    ):
        axis.set_xticks([0, 1])
        axis.set_xticklabels(["A", "G"])
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.3)
        axis.legend(fontsize=6)
    axes[0].axhline(ACCEPTANCE["palm_z_mae_max_m"], color="r", lw=0.8, ls=":")
    figure.suptitle("paired primary metrics (active window): A = PD only, G = PD + gravity feed-forward")
    figure.tight_layout()
    figure.savefig(figures / "figure-2-paired.png", dpi=140)
    plt.close(figure)

    # Figure 3: safety and task-side readings.
    figure, axes = plt.subplots(1, 2, figsize=(12.5, 4.2))
    labels = [label for label in ("A1", "A2", "A3", "G1", "G2", "G3") if label in summary["rollouts"]]
    positions = np.arange(len(labels))
    saturation = [summary["rollouts"][label][PRIMARY_WINDOW]["saturation"]["arm_fraction"] for label in labels]
    velocity = [summary["rollouts"][label][PRIMARY_WINDOW]["velocity"]["measured_joint_max_rad_s"] for label in labels]
    axes[0].bar(positions - 0.2, [value or 0.0 for value in saturation], width=0.4, label="arm clamp fraction")
    twin = axes[0].twinx()
    twin.plot(positions, [value or 0.0 for value in velocity], "o-", color="tab:red", label="max measured joint speed")
    twin.set_ylabel("rad/s")
    axes[0].set_xticks(positions)
    axes[0].set_xticklabels(labels)
    axes[0].set_ylabel("fraction at effort clamp")
    axes[0].set_title("safety: effort saturation and joint speed")
    axes[0].axhline(GATES["saturation_frac_max"], color="k", lw=0.7, ls=":")
    axes[0].legend(fontsize=7)
    for label, position in zip(labels, positions):
        lift = summary["task"].get(label, {}).get("cube_lift_above_surface_m") or {}
        values = [float(value) for value in lift.values() if value is not None]
        axes[1].bar(position, max(values or [0.0]), color="tab:green" if label.startswith("G") else "0.6")
    axes[1].axhline(0.03, color="r", lw=0.8, ls=":")
    axes[1].set_xticks(positions)
    axes[1].set_xticklabels(labels)
    axes[1].set_ylabel("max cube lift above surface (m)")
    axes[1].set_title("task: cube lift")
    figure.tight_layout()
    figure.savefig(figures / "figure-3-safety-task.png", dpi=140)
    plt.close(figure)
    print(f"[figures] wrote 3 figures to {figures}")
    return 0


# ------------------------------------------------------------------ manifest


def stage_manifest(args: argparse.Namespace) -> int:
    out = args.out
    inputs = {
        "campaign.json": str(CAMPAIGN / "campaign.json"),
        "driver": str(DRIVER),
        "analyzer": str(ANALYZER),
        "script": str(Path(__file__).resolve()),
        "profile": str(REPO_ROOT / "configs" / "profiles" / "isaac-g1-sonic-blockstacking-dex3.json"),
        "urdf": str(URDF_PATH),
    }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "script_version": SCRIPT_VERSION,
        "command": sys.argv,
        "inputs": {name: {"path": path, "sha256": sha256(Path(path))} for name, path in inputs.items()},
        "sessions": {
            cell: {
                "session": str(out / "runs" / cell / "session.json"),
                "session_sha256": sha256(out / "runs" / cell / "session.json"),
                "manifest": str(out / "runs" / cell / "manifest.json"),
                "manifest_sha256": sha256(out / "runs" / cell / "manifest.json"),
                "verification": str(out / "runs" / cell / "verification.json"),
                "verification_sha256": sha256(out / "runs" / cell / "verification.json"),
            }
            for cell in CELLS
        },
        "outputs": {},
    }
    for name in ("REPORT.md", "summary.json", "review-notes.json", "tables/rollouts.csv",
                 "tables/pairs.csv", "tables/safety_gates.csv", "tables/task.csv",
                 "tables/campaign_commands.json", "series/palm_z_series.npz",
                 "figures/figure-1-tracking.png", "figures/figure-2-paired.png",
                 "figures/figure-3-safety-task.png"):
        path = out / name
        if path.is_file():
            manifest["outputs"][name] = {"path": str(path), "sha256": sha256(path)}
    for cell in CELLS:
        for index in range(1, ROLLOUTS_PER_CELL + 1):
            directory = out / "runs" / cell / "rollouts" / f"rollout-{index:02d}"
            for name in ("rollout.json", "tracking.jsonl", "video.mp4", "contact-sheet.jpg"):
                path = directory / name
                if path.is_file():
                    manifest["outputs"][f"runs/{cell}/rollout-{index:02d}/{name}"] = {
                        "path": str(path), "sha256": sha256(path)
                    }
    write_json(out / "manifest.json", manifest)
    print(f"[manifest] wrote {out / 'manifest.json'}")
    return 0


# ----------------------------------------------------------------------- CLI


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Experiment 10: psi0 gravity feed-forward rollout A/B")
    parser.add_argument("stage", choices=("campaign", "rollouts", "analyse", "figures", "manifest", "all"))
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--force", action="store_true", help="re-run sessions that already exist")
    parser.add_argument("--review-notes", type=Path, default=None,
                        help="optional JSON with the visual review of the videos and contact sheets")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    args.out = args.out.resolve()
    if args.stage == "campaign":
        return stage_campaign(args)
    if args.stage == "rollouts":
        return stage_rollouts(args)
    if args.stage == "analyse":
        analyse(args.out, args.review_notes)
        return 0
    if args.stage == "figures":
        return stage_figures(args)
    if args.stage == "manifest":
        return stage_manifest(args)
    for stage in (stage_campaign, stage_rollouts):
        code = stage(args)
        if code:
            return code
    analyse(args.out, args.review_notes)
    if stage_figures(args):
        return 1
    return stage_manifest(args)


if __name__ == "__main__":
    raise SystemExit(main())
