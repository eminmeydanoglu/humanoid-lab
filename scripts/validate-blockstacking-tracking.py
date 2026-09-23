#!/usr/bin/env python3
"""Semantic and temporal validation of the recorded controller-tracking metric.

The dataset comparison report (`dataset-comparison/REPORT.md`) claims that the
simulator realises the SONIC controller's own joint target 0.21-0.28 m *below*
it.  This script re-derives that number from the recorded telemetry alone and
decides whether the two sides of the comparison are really the same quantity:
same joint semantics, same joint order, same frame, same instant.

Stages:

``analyse``  reads ``tracking.jsonl`` (the cropped rollout window) and
             ``raw/isaac.tracking.parquet`` (the same stream plus the pre-Start
             ramp and hold), the bridge telemetry and the URDF, and writes every
             metric plus its negative controls to JSON and CSV.  Needs pyarrow
             for the parquet (``data/venvs/hf-datasets``).
``figures``  renders the three figures from the JSON/CSV of ``analyse``.
``manifest`` writes the input hashes and the exact commands.

The forward kinematics is *not* re-implemented here: it is loaded from the
already validated ``scripts/compare-blockstacking-dataset.py``, whose FK was
checked against Isaac's own palm poses to 3.7e-7 m, and this script repeats
that check on the same data before it reports anything (``fk_gate``).

    data/venvs/hf-datasets/bin/python scripts/validate-blockstacking-tracking.py analyse
    python3 scripts/validate-blockstacking-tracking.py figures
    python3 scripts/validate-blockstacking-tracking.py manifest
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = REPO_ROOT / "data" / "outputs" / "blockstacking-debug"
OUTPUT = CAMPAIGN / "experiments" / "03-tracking-validation"
DEFAULT_URDF = REPO_ROOT / "third_party" / "Psi0" / "real" / "assets" / "g1" / "g1_body29_hand14.urdf"
COMPARE_SCRIPT = REPO_ROOT / "scripts" / "compare-blockstacking-dataset.py"

sys.path.insert(0, str(REPO_ROOT / "src"))
from humanoid_lab.controllers.sonic import (  # noqa: E402
    BODY_EFFORT_LIMIT_NM,
    BODY_JOINT_ORDER,
    DEFAULT_STANDING_POSE_RAD,
    SONIC_REFERENCE_JOINT_ORDER,
    deploy_gains,
)


def reorder(values: np.ndarray, source_names, target_names) -> np.ndarray:
    """Re-index (..., len(source_names)) values into ``target_names`` order.

    Same definition as ``humanoid_lab.datasets.sonic.joints.reorder``, inlined
    because importing that package pulls in the encoder's yaml dependency.
    """
    index = [list(source_names).index(name) for name in target_names]
    return np.asarray(values)[..., index]

SCHEMA_VERSION = 1
SCRIPT_VERSION = "validate-blockstacking-tracking.py/1.0.0"

TRACKING_DT_S = 0.02           # one tracking row = 4 physics ticks at 5 ms
ARM_JOINTS = tuple(
    name for name in BODY_JOINT_ORDER
    if any(token in name for token in ("shoulder", "elbow", "wrist"))
)
WAIST_JOINTS = ("waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint")
LAG_MIN_S, LAG_MAX_S = -2.0, 2.0
MOVE_RAD_S = 0.10              # arm-target speed norm above which a row counts as "moving"
STATIC_RAD_S = 0.01            # waist+arm target speed below which the target counts as "held"
MOVE_SENSITIVITY_RAD_S = (0.05, 0.20)
SMOOTH_WINDOW_S = 0.20         # diagnostic only: the target jitter the PD cannot chase
SATURATION_TOL_NM = 1e-4       # write quantisation of the applied torque

STANDING = np.array([DEFAULT_STANDING_POSE_RAD[name] for name in BODY_JOINT_ORDER], dtype=float)
ARM_INDEX = np.array([BODY_JOINT_ORDER.index(name) for name in ARM_JOINTS])
WAIST_INDEX = np.array([BODY_JOINT_ORDER.index(name) for name in WAIST_JOINTS])
LIMITS = np.array(BODY_EFFORT_LIMIT_NM, dtype=float)
LEFT_ARM_INDEX = np.array([BODY_JOINT_ORDER.index(n) for n in BODY_JOINT_ORDER
                           if n.startswith("left_")
                           and any(t in n for t in ("shoulder", "elbow", "wrist"))])
RIGHT_ARM_INDEX = np.array([BODY_JOINT_ORDER.index(n) for n in BODY_JOINT_ORDER
                            if n.startswith("right_")
                            and any(t in n for t in ("shoulder", "elbow", "wrist"))])


class ValidationError(RuntimeError):
    """The analysis cannot proceed; the message is for the operator."""


# ------------------------------------------------------------------- plumbing


def load_compare_module():
    specification = importlib.util.spec_from_file_location("compare_blockstacking_dataset", COMPARE_SCRIPT)
    if specification is None or specification.loader is None:
        raise ValidationError(f"cannot load {COMPARE_SCRIPT}")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with Path(path).open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1, default=str) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    names = fields or list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=names, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def describe(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return {"n": 0}
    return {"n": int(values.size), "mean": float(values.mean()), "median": float(np.median(values)),
            "p05": float(np.percentile(values, 5)), "p95": float(np.percentile(values, 95)),
            "min": float(values.min()), "max": float(values.max())}


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0:
        return float("nan")
    difference = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    return float(np.sqrt(np.mean(difference ** 2)))


def mae(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0:
        return float("nan")
    return float(np.mean(np.abs(np.asarray(a, dtype=float) - np.asarray(b, dtype=float))))


def rollout_metadata() -> list[dict]:
    campaign = json.loads((CAMPAIGN / "campaign.json").read_text())
    entries = []
    for session in campaign["sessions"]:
        for entry in session["rollouts"]:
            entries.append({
                "session": session["session_dir"], "model": session["model"], "tag": session["tag"],
                "index": int(entry["index"]), "rollout_dir": CAMPAIGN / entry["rollout_dir"],
                "raw_parquet": CAMPAIGN / session["session_dir"] / "raw" / "isaac.tracking.parquet",
                "onset_seconds_after_start": float(entry["onset_seconds_after_start"]),
                "onset_source": entry["onset_source"],
            })
    return entries


# --------------------------------------------------------------------- stream


class TrackingStream:
    """One rollout's controller command/state stream: cropped window + raw episode.

    The raw session parquet holds the same rows as the cropped
    ``tracking.jsonl`` *plus* the ramp to the standing pose that runs before the
    policy takes over.  The parquet carries no episode column, so episodes are
    separated by their physics-tick resets and each one is aligned to its
    cropped rollout by its first tick; the overlap is checked row by row.
    """

    KEYS = ("tick", "sim_s", "wall_time_ns", "support_active", "body_target", "body_velocity_target",
            "body_feedforward_torque", "body_kp", "body_kd", "body_applied_torque", "body_measured",
            "body_measured_velocity", "root_position", "root_quaternion_wxyz", "torso_quaternion_wxyz",
            "left_hand_target", "left_hand_measured", "right_hand_target", "right_hand_measured")

    def __init__(self, rollout_dir: Path, raw_parquet: Path):
        self.dir = Path(rollout_dir)
        self.crop_rows = read_jsonl(self.dir / "tracking.jsonl")
        rollout = json.loads((self.dir / "rollout.json").read_text())
        self.rollout = rollout
        self.start_wall_ns = int(rollout["start_wall_ns"])
        self.onset_wall_ns = self.start_wall_ns + int(round(rollout["onset"]["seconds_after_start"] * 1e9))
        self.surface = float(rollout["stack"]["table_surface_m"])
        self.cubes = {name: np.asarray(spec["center_xyz_m"], dtype=float)
                      for name, spec in rollout["stack"]["cube_pose_w"].items()}
        self.crop_tick = np.array([int(row["tick"]) for row in self.crop_rows], dtype=np.int64)
        self.crop_wall_ns = np.array([int(row["wall_time_ns"]) for row in self.crop_rows], dtype=np.int64)
        self.rows = self.crop_rows
        self.parquet_check = self._load_parquet(Path(raw_parquet))
        self.crop_mask = np.zeros(len(self.rows), dtype=bool)
        self.crop_mask[self.crop_slice] = True

    def _arrays_from(self, rows: list[dict]) -> None:
        self.row_count = len(rows)
        self.tick = np.array([int(row["tick"]) for row in rows], dtype=np.int64)
        self.sim_s = np.array([float(row["sim_s"]) for row in rows], dtype=float)
        self.wall_ns = np.array([int(row["wall_time_ns"]) for row in rows], dtype=np.int64)
        self.support = np.array([bool(row.get("support_active")) for row in rows], dtype=bool)
        for name, key, width in (("target", "body_target", 29), ("measured", "body_measured", 29),
                                 ("target_dq", "body_velocity_target", 29),
                                 ("measured_dq", "body_measured_velocity", 29),
                                 ("tau_ff", "body_feedforward_torque", 29),
                                 ("tau", "body_applied_torque", 29), ("kp", "body_kp", 29),
                                 ("kd", "body_kd", 29), ("root", "root_position", 3),
                                 ("root_quat", "root_quaternion_wxyz", 4),
                                 ("left_hand_target", "left_hand_target", 7),
                                 ("right_hand_target", "right_hand_target", 7),
                                 ("left_hand_measured", "left_hand_measured", 7),
                                 ("right_hand_measured", "right_hand_measured", 7)):
            values = [row.get(key) for row in rows]
            if any(value is None for value in values):
                setattr(self, name, np.full((len(rows), width), np.nan))
            else:
                setattr(self, name, np.asarray(values, dtype=float))

    def _load_parquet(self, raw_parquet: Path) -> dict:
        import pyarrow.parquet as pq

        columns = pq.ParquetFile(raw_parquet).read().to_pydict()
        blocks, start = [], 0
        for index in range(1, len(columns["tick"])):
            if columns["tick"][index] <= columns["tick"][index - 1]:
                blocks.append((start, index))
                start = index
        blocks.append((start, len(columns["tick"])))
        for block_start, block_stop in blocks:
            ticks = np.asarray(columns["tick"][block_start:block_stop], dtype=np.int64)
            stamps = np.asarray(columns["wall_time_ns"][block_start:block_stop], dtype=np.int64)
            offsets = np.where(ticks == self.crop_tick[0])[0]
            head = None
            for offset in offsets:
                if offset + len(self.crop_tick) > len(ticks):
                    continue
                if not np.array_equal(ticks[offset:offset + len(self.crop_tick)], self.crop_tick):
                    continue
                # Several episodes in one session share a tick range (each
                # starts from the same warm-up ticks); the wall clock is what
                # tells them apart.
                if not np.array_equal(stamps[offset:offset + len(self.crop_tick)], self.crop_wall_ns):
                    continue
                head = block_start + int(offset)
                break
            if head is None:
                continue
            rows = [{name: columns[name][index] for name in columns}
                    for index in range(block_start, block_stop)]
            self._arrays_from(rows)
            self.rows = rows
            self.crop_slice = slice(head - block_start, head - block_start + len(self.crop_rows))
            overlap = np.abs(np.asarray([row["body_target"] for row in self.rows[self.crop_slice]],
                                        dtype=float) - np.asarray([row["body_target"] for row in self.crop_rows]))
            return {
                "path": str(raw_parquet), "episode_rows": int(block_stop - block_start),
                "pre_start_rows": int(self.crop_slice.start),
                "crop_rows": int(len(self.crop_rows)),
                "crop_matches_jsonl_max_abs_rad": float(overlap.max()),
            }
        raise ValidationError(f"no raw episode matches the cropped rollout in {raw_parquet}")


# ------------------------------------------------------------------ kinematics


def base_matrix(root: np.ndarray, quaternion: np.ndarray, compare) -> np.ndarray:
    base = np.broadcast_to(np.eye(4), (len(root), 4, 4)).copy()
    for index, value in enumerate(quaternion):
        base[index, :3, :3] = compare.quaternion_wxyz_to_matrix(value)
    base[:, :3, 3] = root
    return base


GRAVITY_M_S2 = 9.81


def load_urdf(compare, path: Path):
    """The validated URDF reader plus the link masses and centres of mass."""

    class UrdfWithMass(compare.Urdf):
        def __init__(self, urdf_path: Path) -> None:
            super().__init__(urdf_path)
            import xml.etree.ElementTree as ET

            document = ET.parse(str(urdf_path)).getroot()
            self.masses: dict[str, float] = {}
            self.com_xyz: dict[str, np.ndarray] = {}
            self.link_names: list[str] = []
            for element in document.findall("link"):
                name = element.get("name")
                if not name:
                    continue
                self.link_names.append(name)
                inertial = element.find("inertial")
                if inertial is None:
                    continue
                self.masses[name] = float(inertial.find("mass").get("value"))
                origin = inertial.find("origin")
                self.com_xyz[name] = (
                    np.array([float(v) for v in origin.get("xyz").split()]) if origin is not None
                    else np.zeros(3))

    return UrdfWithMass(path)


def gravity_diagnostic(urdf, stream: TrackingStream, hold: np.ndarray, started: np.ndarray,
                       stride: int = 25) -> dict:
    """Is the droop the proportional term of a loop with no gravity feedforward?

    With ``tau_ff == 0`` the steady state of the applied PD is ``kp*e = gravity
    moment`` joint by joint.  The gravity moments here come from the pinned
    URDF's own link masses and centres of mass, so the comparison is between
    two independent sources: the recorded torque law and the model's own
    inertial data.
    """
    result: dict = {"stride": stride}
    for name, window in (("hold_pre_start", hold), ("post_start", started)):
        selected = np.where(window)[0][::stride]
        if selected.size < 3:
            result[name] = {"n_samples": int(selected.size)}
            continue
        torque = gravity_torques(urdf, stream.measured[selected])
        pd_term = (stream.kp * (stream.target - stream.measured))[selected]
        index = np.concatenate([WAIST_INDEX, ARM_INDEX])
        names = [BODY_JOINT_ORDER[i] for i in index]
        measured_torque = pd_term[:, index]
        modelled = torque[:, index]
        correlation = []
        for column in range(len(index)):
            if measured_torque[:, column].std() > 0 and modelled[:, column].std() > 0:
                correlation.append(float(np.corrcoef(measured_torque[:, column], modelled[:, column])[0, 1]))
        result[name] = {
            "n_samples": int(selected.size),
            "mean_abs_pd_torque_nm": float(np.abs(measured_torque).mean()),
            "mean_abs_gravity_torque_nm": float(np.abs(modelled).mean()),
            "ratio_pd_over_gravity": float(np.abs(measured_torque).mean() / np.abs(modelled).mean()),
            "per_joint_correlation_median": float(np.median(correlation)) if correlation else None,
            "per_joint": [{"joint": joint, "mean_abs_pd_torque_nm": float(np.abs(measured_torque[:, column]).mean()),
                           "mean_signed_pd_torque_nm": float(measured_torque[:, column].mean()),
                           "mean_signed_gravity_torque_nm": float(modelled[:, column].mean()),
                           "mean_abs_gravity_torque_nm": float(np.abs(modelled[:, column]).mean())}
                          for column, joint in enumerate(names)],
        }
    return result


def gravity_torques(urdf, q_body: np.ndarray) -> np.ndarray:
    """Gravity torque about every body joint, in the joint's own axis, at ``q_body``.

    Sums ``(r_link - r_joint) x (-m g z_hat) · axis`` over every URDF link distal
    to that joint, with the link masses and centres of mass taken from the
    pinned URDF.  Returned in ``BODY_JOINT_ORDER``; a joint that carries no
    distal mass (or is not on an arm chain) stays zero.
    """
    names = [joint["name"] for joint in urdf.joints]
    body_index = {name: index for index, name in enumerate(BODY_JOINT_ORDER)}
    chains = {link: urdf.chain(link) for link in urdf.link_names}
    out = np.zeros((len(q_body), len(BODY_JOINT_ORDER)))
    for link in urdf.link_names:
        mass = urdf.masses.get(link, 0.0)
        if mass <= 0.0:
            continue
        chain = chains[link]
        tip = urdf.fk_chain(chain, urdf.chain_joint_values(chain, q_body))
        centre = tip[:, :3, 3] + np.einsum("nij,j->ni", tip[:, :3, :3], urdf.com_xyz.get(link, np.zeros(3)))
        for index, joint in enumerate(chain):
            name = joint["name"]
            if name not in body_index:
                continue
            # Joint origin and axis in the pelvis frame.  The axis survives its
            # own rotation, so the child frame carries it into the pelvis frame.
            sub = chain[:index + 1]
            sub_tip = urdf.fk_chain(sub, urdf.chain_joint_values(sub, q_body))
            position = sub_tip[:, :3, 3]
            axis = np.einsum("nij,j->ni", sub_tip[:, :3, :3], np.asarray(joint["axis"], dtype=float))
            force = np.zeros_like(centre)
            force[:, 2] = -mass * GRAVITY_M_S2
            moment = np.cross(centre - position, force)
            out[:, body_index[name]] += np.einsum("ni,ni->n", moment, axis)
    _ = names
    return out


def joint_metrics(target: np.ndarray, measured: np.ndarray, index: np.ndarray) -> dict:
    difference = np.asarray(target)[:, index] - np.asarray(measured)[:, index]
    if difference.size == 0:
        return {"n_rows": 0}
    return {"n_rows": int(difference.shape[0]), "n_joints": int(len(index)),
            "mae_rad": float(np.abs(difference).mean()),
            "rmse_rad": float(np.sqrt(np.mean(difference ** 2))),
            "bias_rad": float(difference.mean()),
            "max_abs_rad": float(np.abs(difference).max())}


def palm_metrics(target_palm: dict, measured_palm: dict) -> dict:
    out = {}
    difference = {side: target_palm[side] - measured_palm[side] for side in ("left", "right")}
    for side, delta in difference.items():
        out[side] = {"n_rows": int(delta.shape[0]),
                     "dz_bias_m": float(delta[:, 2].mean()),
                     "dz_mae_m": float(np.abs(delta[:, 2]).mean()),
                     "dz_median_m": float(np.median(delta[:, 2])),
                     "dz_min_m": float(delta[:, 2].min()),
                     "dz_max_m": float(delta[:, 2].max()),
                     "rmse_3d_m": float(np.sqrt(np.mean(np.sum(delta ** 2, axis=1)))),
                     "horizontal_mae_m": float(np.mean(np.linalg.norm(delta[:, :2], axis=1)))}
    both = np.concatenate([difference["left"], difference["right"]], axis=0)
    out["pooled"] = {"n_rows": int(both.shape[0]),
                     "dz_bias_m": float(both[:, 2].mean()),
                     "dz_mae_m": float(np.abs(both[:, 2]).mean()),
                     "dz_median_m": float(np.median(both[:, 2])),
                     "rmse_3d_m": float(np.sqrt(np.mean(np.sum(both ** 2, axis=1))))}
    return out


# --------------------------------------------------------------- lag and gap


def lag_sweep(stream: TrackingStream, urdf, window: np.ndarray) -> list[dict]:
    target_pelvis = urdf.palms(stream.target)
    measured_pelvis = urdf.palms(stream.measured)
    count = len(stream.rows)
    rows = []
    for step in range(int(round(LAG_MIN_S / TRACKING_DT_S)), int(round(LAG_MAX_S / TRACKING_DT_S)) + 1):
        head = np.arange(count)
        tail = head + step
        shifted_window = np.zeros(count, dtype=bool)
        if step >= 0:
            shifted_window[:count - step] = window[step:]
        else:
            shifted_window[-step:] = window[:count + step]
        pair = window & shifted_window & (tail >= 0) & (tail < count)
        entry = {"lag_s": float(step * TRACKING_DT_S), "n_pairs": int(pair.sum())}
        if pair.sum() > 10:
            head_index, tail_index = head[pair], tail[pair]
            entry["arm_joint_rmse_rad"] = rmse(stream.target[head_index][:, ARM_INDEX],
                                               stream.measured[tail_index][:, ARM_INDEX])
            entry["arm_joint_mae_rad"] = mae(stream.target[head_index][:, ARM_INDEX],
                                             stream.measured[tail_index][:, ARM_INDEX])
            dz = np.concatenate([target_pelvis[side][head_index, 2] - measured_pelvis[side][tail_index, 2]
                                 for side in ("left", "right")])
            entry["palm_dz_mae_m"] = float(np.abs(dz).mean())
            entry["palm_dz_bias_m"] = float(dz.mean())
            # A pure time shift cannot remove a constant offset: the centred
            # RMSE is the part of the error a lag is able to explain at all.
            entry["palm_dz_rmse_centred_m"] = float(np.sqrt(np.mean((dz - dz.mean()) ** 2)))
            target_z = np.concatenate([target_pelvis[side][head_index, 2] for side in ("left", "right")])
            measured_z = np.concatenate([measured_pelvis[side][tail_index, 2] for side in ("left", "right")])
            if target_z.std() > 0 and measured_z.std() > 0:
                entry["palm_dz_correlation"] = float(np.corrcoef(target_z, measured_z)[0, 1])
            difference = np.concatenate([target_pelvis[side][head_index] - measured_pelvis[side][tail_index]
                                         for side in ("left", "right")], axis=0)
            entry["palm_rmse_3d_m"] = float(np.sqrt(np.mean(np.sum(difference ** 2, axis=1))))
        rows.append(entry)
    return rows


def per_joint_table(stream: TrackingStream, started: np.ndarray, hold: np.ndarray) -> list[dict]:
    gap = stream.target - stream.measured
    rows = []
    for index, name in enumerate(BODY_JOINT_ORDER):
        saturated = np.abs(stream.tau[:, index]) >= LIMITS[index] - SATURATION_TOL_NM
        rows.append({
            "joint": name,
            "mean_signed_gap_rad": float(gap[:, index].mean()),
            "mean_abs_gap_rad": float(np.abs(gap[:, index]).mean()),
            "max_abs_gap_rad": float(np.abs(gap[:, index]).max()),
            "mean_abs_gap_post_start_rad": float(np.abs(gap[started, index]).mean()) if started.any() else None,
            "mean_abs_gap_hold_pre_start_rad": float(np.abs(gap[hold, index]).mean()) if hold.any() else None,
            "saturation_fraction": float(saturated.mean()),
            "saturation_fraction_post_start": float(saturated[started].mean()) if started.any() else None,
            "mean_abs_torque_nm": float(np.abs(stream.tau[:, index]).mean()),
            "max_abs_torque_nm": float(np.abs(stream.tau[:, index]).max()),
            "kp": float(stream.kp[:, index].mean()),
            "kd": float(stream.kd[:, index].mean()),
            "effort_limit_nm": float(LIMITS[index]),
        })
    return rows


def control_table(stream: TrackingStream, urdf, compare, window: np.ndarray) -> list[dict]:
    target, measured = stream.target, stream.measured
    base = base_matrix(stream.root, stream.root_quat, compare)
    variants: list[tuple[str, np.ndarray, np.ndarray, str]] = [
        ("identity", target, measured, "the mapping this experiment validates"),
    ]
    swapped = target.copy()
    swapped[:, LEFT_ARM_INDEX] = target[:, RIGHT_ARM_INDEX]
    swapped[:, RIGHT_ARM_INDEX] = target[:, LEFT_ARM_INDEX]
    variants.append(("target_left_right_swapped", swapped, measured, "negative control: arm blocks exchanged"))
    swapped_measured = measured.copy()
    swapped_measured[:, LEFT_ARM_INDEX] = measured[:, RIGHT_ARM_INDEX]
    swapped_measured[:, RIGHT_ARM_INDEX] = measured[:, LEFT_ARM_INDEX]
    variants.append(("measured_left_right_swapped", target, swapped_measured, "negative control: measurement side"))
    variants.append(("target_in_reference_order",
                     reorder(target, BODY_JOINT_ORDER, SONIC_REFERENCE_JOINT_ORDER), measured,
                     "negative control: target read as SONIC reference order"))
    zeroed = target.copy()
    zeroed[:, ARM_INDEX] = 0.0
    variants.append(("target_arms_zeroed", zeroed, measured, "negative control: target treated as a residual"))
    waist_measured = target.copy()
    waist_measured[:, WAIST_INDEX] = measured[:, WAIST_INDEX]
    variants.append(("target_waist_replaced_by_measured", waist_measured, measured,
                     "ablation: waist contribution to the palm gap"))
    arms_only = measured.copy()
    arms_only[:, ARM_INDEX] = target[:, ARM_INDEX]
    arms_only[:, WAIST_INDEX] = target[:, WAIST_INDEX]
    variants.append(("measured_lower_body_with_target_arms", arms_only, measured,
                     "ablation: everything but the arms replaced by the measurement"))
    waist_only = measured.copy()
    waist_only[:, WAIST_INDEX] = target[:, WAIST_INDEX]
    variants.append(("measured_arms_with_target_waist", waist_only, measured,
                     "ablation: waist contribution isolated (arms and legs measured)"))

    rows = []
    for name, target_variant, measured_variant, note in variants:
        target_pelvis = urdf.palms(target_variant)
        measured_pelvis = urdf.palms(measured_variant)
        target_world = urdf.palms(target_variant, base)
        measured_world = urdf.palms(measured_variant, base)
        shared = palm_metrics({s: v[window] for s, v in target_pelvis.items()},
                              {s: v[window] for s, v in measured_pelvis.items()})
        world = palm_metrics({s: v[window] for s, v in target_world.items()},
                             {s: v[window] for s, v in measured_world.items()})
        mislabeled = palm_metrics({s: v[window] for s, v in target_pelvis.items()},
                                  {s: v[window] for s, v in measured_world.items()})
        rows.append({
            "variant": name, "note": note,
            "arm_joint_mae_rad": mae(target_variant[window][:, ARM_INDEX], measured_variant[window][:, ARM_INDEX]),
            "arm_joint_rmse_rad": rmse(target_variant[window][:, ARM_INDEX], measured_variant[window][:, ARM_INDEX]),
            "palm_dz_mae_m": shared["pooled"]["dz_mae_m"],
            "palm_dz_bias_m": shared["pooled"]["dz_bias_m"],
            "palm_dz_median_m": shared["pooled"]["dz_median_m"],
            "palm_rmse_3d_m": shared["pooled"]["rmse_3d_m"],
            "world_frame_palm_dz_median_m": world["pooled"]["dz_median_m"],
            "world_frame_palm_rmse_3d_m": world["pooled"]["rmse_3d_m"],
            "pelvis_target_read_as_world_rmse_3d_m": mislabeled["pooled"]["rmse_3d_m"],
        })
    return rows


# -------------------------------------------------------------- bridge clock


def bridge_clock(rollout_dir: Path, stream: TrackingStream) -> dict:
    """Align the policy stream with the tracking rows through the hand commands.

    The ``pose`` message carries the hand targets verbatim and the SONIC
    deployment forwards them unchanged, so an *exact* match between a tracking
    row's ``*_hand_target`` and a published ``applied_action`` proves both the
    shared time base and what the recorded command is.
    """
    events = read_jsonl(Path(rollout_dir) / "bridge-telemetry.jsonl")
    applied = [(int(event["wall_time_ns"]),
                np.asarray(event["fields"]["left_hand_joints"][0], dtype=float),
                np.asarray(event["fields"]["right_hand_joints"][0], dtype=float))
               for event in events if event.get("kind") == "applied_action"]
    observations = [event for event in events if event.get("kind") == "observation"]
    if not applied:
        return {"applied_actions": 0}
    wall = np.array([item[0] for item in applied], dtype=np.int64)
    hands = np.array([np.concatenate([item[1], item[2]]) for item in applied])
    lookup: dict[bytes, list[tuple[int, int]]] = {}
    for index in range(len(applied)):
        lookup.setdefault(hands[index].round(6).tobytes(), []).append((index, int(wall[index])))
    ages = []
    matched = 0
    for row in range(len(stream.rows)):
        key = np.concatenate([stream.left_hand_target[row], stream.right_hand_target[row]]).round(6).tobytes()
        candidates = lookup.get(key)
        if not candidates:
            continue
        matched += 1
        stamp = int(stream.wall_ns[row])
        ages.append((stamp - min(candidates, key=lambda item: abs(item[1] - stamp))[1]) / 1e9)
    ages_array = np.asarray(ages, dtype=float)
    state = np.array([event["state"] for event in observations], dtype=float) if observations else np.zeros((0, 43))
    state_wall = np.array([int(event["wall_time_ns"]) for event in observations], dtype=np.int64)
    state_error = None
    if observations:
        index = np.clip(np.searchsorted(stream.wall_ns, state_wall), 1, len(stream.wall_ns) - 1)
        nearest = np.where(np.abs(stream.wall_ns[index - 1] - state_wall) <
                           np.abs(stream.wall_ns[index] - state_wall), index - 1, index)
        difference = state[:, 15:29] - stream.measured[nearest][:, 15:29]
        state_error = {"mean_abs_rad": float(np.abs(difference).mean()),
                       "max_abs_rad": float(np.abs(difference).max()),
                       "n": int(difference.shape[0])}
    return {"applied_actions": len(applied), "observations": len(observations),
            "applied_rate_hz": float(1.0 / np.median(np.diff(wall) / 1e9)),
            "hand_target_verbatim_matches": matched,
            "hand_target_verbatim_fraction": float(matched / len(stream.rows)),
            "command_age_s": describe(ages_array),
            "observation_state_vs_measured_arm": state_error,
            "observation_arm_layout": "state[15:22] left arm, state[22:29] right arm (PSI0_LAYOUT)"}


# ------------------------------------------------------------------ analysis


def analyse_rollout(entry: dict, stream: TrackingStream, urdf, compare) -> dict:
    target, measured = stream.target, stream.measured
    base = base_matrix(stream.root, stream.root_quat, compare)
    target_pelvis, measured_pelvis = urdf.palms(target), urdf.palms(measured)
    target_world, measured_world = urdf.palms(target, base), urdf.palms(measured, base)

    wall = stream.wall_ns
    started = wall >= stream.onset_wall_ns
    standing_error = np.abs(target - STANDING[None, :]).max(axis=1)
    speed = np.zeros(len(stream.rows))
    speed[1:] = np.linalg.norm(np.diff(target[:, ARM_INDEX], axis=0), axis=1) / TRACKING_DT_S
    chain_speed = np.zeros(len(stream.rows))
    chain_speed[1:] = np.linalg.norm(np.diff(target[:, np.concatenate([WAIST_INDEX, ARM_INDEX])], axis=0),
                                     axis=1) / TRACKING_DT_S
    static_target = chain_speed <= STATIC_RAD_S
    fast_cut = float(np.median(speed[started])) if started.any() else MOVE_RAD_S
    masks = {
        "ramp_pre_start": (~started) & (~static_target),
        "hold_pre_start": (~started) & static_target,
        "post_start_all": started,
        # The policy target is never literally held, so "moving" and "quiet"
        # are the two halves of the post-Start window split at its own median
        # arm-target speed; the absolute thresholds are kept as a sensitivity.
        "post_start_fast_half": started & (speed > fast_cut),
        "post_start_slow_half": started & (speed <= fast_cut),
    }
    for threshold in MOVE_SENSITIVITY_RAD_S:
        masks[f"post_start_above_{threshold:.2f}rad_s"] = started & (speed > threshold)

    result: dict = {
        "session": entry["session"], "model": entry["model"], "rollout": entry["index"],
        "tag": f"{entry['session']}-r{entry['index']:02d}",
        "rollout_dir": str(entry["rollout_dir"]),
        "onset_seconds_after_start": entry["onset_seconds_after_start"],
        "onset_source": entry["onset_source"],
        "crop_rows": int(len(stream.crop_rows)),
        "parquet": stream.parquet_check,
        "clock": {
            "sim_start_s": float(stream.sim_s[0]), "sim_stop_s": float(stream.sim_s[-1]),
            "sim_span_s": float(stream.sim_s[-1] - stream.sim_s[0]),
            "wall_span_s": float((int(wall[-1]) - int(wall[0])) / 1e9),
            "sim_per_wall": float((stream.sim_s[-1] - stream.sim_s[0]) / ((wall[-1] - wall[0]) / 1e9)),
            "tracking_dt_s": TRACKING_DT_S,
            "root_z_range_m": [float(stream.root[:, 2].min()), float(stream.root[:, 2].max())],
            "support_rows": int(stream.support.sum()),
        },
        "segments": {}, "joint": {}, "palm_pelvis": {}, "palm_world": {},
        "palm_pelvis_crop": {}, "palm_world_crop": {},
        "per_joint": [], "torque": {}, "high_hand": {}, "parity_check": {}, "hand_channels": {},
    }
    for name, mask in masks.items():
        count = int(mask.sum())
        window = {"n_rows": count, "duration_s": count * TRACKING_DT_S,
                  "arm_target_speed_median_rad_s": float(np.median(speed[mask])) if count else None}
        if count >= 2:
            window["joint_arm"] = joint_metrics(target[mask], measured[mask], ARM_INDEX)
            window["joint_waist"] = joint_metrics(target[mask], measured[mask], WAIST_INDEX)
            window["palm_pelvis"] = palm_metrics({s: v[mask] for s, v in target_pelvis.items()},
                                                 {s: v[mask] for s, v in measured_pelvis.items()})
            window["palm_world"] = palm_metrics({s: v[mask] for s, v in target_world.items()},
                                                {s: v[mask] for s, v in measured_world.items()})
        result["segments"][name] = window

    result["joint"] = {
        "arm_post_start": joint_metrics(target[started], measured[started], ARM_INDEX),
        "waist_post_start": joint_metrics(target[started], measured[started], WAIST_INDEX),
        "arm_all_rows": joint_metrics(target, measured, ARM_INDEX),
        "arm_pre_start_hold": joint_metrics(target[masks["hold_pre_start"]], measured[masks["hold_pre_start"]],
                                            ARM_INDEX),
    }
    result["palm_pelvis"] = palm_metrics(target_pelvis, measured_pelvis)
    result["palm_world"] = palm_metrics(target_world, measured_world)
    result["palm_pelvis_crop"] = palm_metrics(
        {s: v[stream.crop_mask] for s, v in target_pelvis.items()},
        {s: v[stream.crop_mask] for s, v in measured_pelvis.items()})
    result["palm_world_crop"] = palm_metrics(
        {s: v[stream.crop_mask] for s, v in target_world.items()},
        {s: v[stream.crop_mask] for s, v in measured_world.items()})
    result["palm_pelvis_post_start"] = palm_metrics({s: v[started] for s, v in target_pelvis.items()},
                                                    {s: v[started] for s, v in measured_pelvis.items()})
    result["palm_world_post_start"] = palm_metrics({s: v[started] for s, v in target_world.items()},
                                                   {s: v[started] for s, v in measured_world.items()})

    result["per_joint"] = per_joint_table(stream, started, masks["hold_pre_start"])
    result["gravity_check"] = gravity_diagnostic(urdf, stream, masks["hold_pre_start"], started)

    saturated = np.abs(stream.tau) >= LIMITS[None, :] - SATURATION_TOL_NM
    saturated_arm = saturated[:, ARM_INDEX]
    result["torque"] = {
        "feedforward_abs_max_nm": float(np.abs(stream.tau_ff).max()),
        "applied_abs_max_nm": float(np.abs(stream.tau).max()),
        "arm_saturation_fraction_rows": float(saturated_arm.any(axis=1).mean()),
        "arm_saturation_fraction_joint_samples": float(saturated_arm.mean()),
        "arm_saturation_fraction_post_start": float(saturated_arm[started].mean()) if started.any() else None,
        "leg_saturation_fraction": float(saturated[:, :12].mean()),
        "waist_saturation_fraction": float(saturated[:, WAIST_INDEX].mean()),
        "arm_joints_at_limit": sorted({ARM_JOINTS[i] for i in np.unique(np.where(saturated_arm)[1])}),
        "arm_limits_nm": sorted({float(LIMITS[i]) for i in ARM_INDEX}),
        "max_abs_torque_arm_nm": float(np.abs(stream.tau[:, ARM_INDEX]).max()),
    }
    result["target_freshness"] = {
        "distinct_row_fraction": float(len(np.unique(stream.target.round(9), axis=0)) / len(stream.target)),
        "interpretation": "1.0 means the recorded target differs on every 20 ms row, i.e. it is a fresh "
                          "command and not a repeated hold of the last DDS message",
        "controller_command_age_ticks": None,
    }

    # Diagnostic, not a redefinition of the metric: how much of the gap is the
    # PD chasing jitter in the target rather than a pose it cannot hold.
    window = max(2, int(round(SMOOTH_WINDOW_S / TRACKING_DT_S)))
    kernel = np.ones(window) / window
    smoothed = np.stack([np.convolve(target[:, index], kernel, mode="same")
                         for index in range(target.shape[1])], axis=1)
    smoothed_pelvis = urdf.palms(smoothed)
    per_row_gap = np.mean([target_pelvis[s][started, 2] - measured_pelvis[s][started, 2]
                           for s in ("left", "right")], axis=0)
    smoothed_gap = np.concatenate([smoothed_pelvis[s][started, 2] - measured_pelvis[s][started, 2]
                                   for s in ("left", "right")])
    speed_started = speed[started]
    result["jitter_diagnostic"] = {
        "smooth_window_s": SMOOTH_WINDOW_S,
        "target_speed_median_rad_s": float(np.median(speed_started)) if speed_started.size else None,
        "target_speed_p95_rad_s": float(np.percentile(speed_started, 95)) if speed_started.size else None,
        "palm_dz_mae_raw_m": float(np.abs(per_row_gap).mean()) if per_row_gap.size else None,
        "palm_dz_mae_smoothed_target_m": float(np.abs(smoothed_gap).mean()),
        "palm_dz_bias_smoothed_target_m": float(smoothed_gap.mean()),
        "gap_vs_target_speed_correlation": float(np.corrcoef(speed_started, np.abs(per_row_gap))[0, 1])
        if per_row_gap.size and per_row_gap.std() > 0 else None,
        "note": "the smoothed variant replaces the target with its own 0.20 s moving average; it is reported "
                "to show how much of the gap is fast reference motion, not as the validated metric",
    }

    torque = stream.tau_ff + stream.kp * (target - measured) + stream.kd * (stream.target_dq - stream.measured_dq)
    residual = stream.tau - np.clip(torque, -LIMITS[None, :], LIMITS[None, :])
    result["parity_check"] = {
        "max_abs_residual_nm": float(np.abs(residual).max()),
        "mean_abs_residual_nm": float(np.abs(residual).mean()),
        "n_joint_samples": int(residual.size),
        "formula": "tau_applied = clamp(tau_ff + kp*(body_target - body_measured) "
                   "+ kd*(dq_target - dq_measured), +-effort_limit)",
    }

    for side, hand_target, hand_measured in (("left", stream.left_hand_target, stream.left_hand_measured),
                                             ("right", stream.right_hand_target, stream.right_hand_measured)):
        difference = hand_target - hand_measured
        result["hand_channels"][side] = {"mae_rad": float(np.abs(difference).mean()),
                                         "bias_rad": float(difference.mean()),
                                         "median_abs_gap_rad": float(np.median(np.abs(difference)))}

    for side in ("left", "right"):
        gap = target_pelvis[side][:, 2] - measured_pelvis[side][:, 2]
        cubes = {}
        for name, centre in stream.cubes.items():
            target_distance = np.linalg.norm(target_world[side][started] - centre[None, :], axis=1)
            measured_distance = np.linalg.norm(measured_world[side][started] - centre[None, :], axis=1)
            cubes[name] = {
                "target_min_distance_m": float(target_distance.min()) if target_distance.size else None,
                "measured_min_distance_m": float(measured_distance.min()) if measured_distance.size else None,
            }
        result["high_hand"][side] = {
            "target_minus_measured_dz_median_all_m": float(np.median(gap)),
            "target_minus_measured_dz_median_post_start_m": float(np.median(gap[started])),
            "target_minus_measured_dz_median_world_crop_m": float(np.median(
                target_world[side][stream.crop_mask, 2] - measured_world[side][stream.crop_mask, 2])),
            "measured_z_median_post_start_m": float(np.median(measured_pelvis[side][started, 2])),
            "target_z_median_post_start_m": float(np.median(target_pelvis[side][started, 2])),
            "cubes": cubes,
        }
    return result


def fk_gate(stream: TrackingStream, entry: dict, urdf, compare) -> dict:
    rows = []
    samples = read_jsonl(entry["rollout_dir"] / "isaac.samples.jsonl")
    for sample in samples:
        pose = sample.get("palm_pose_w")
        if not pose:
            continue
        matches = np.where(stream.tick == int(sample["physics_tick"]))[0]
        if matches.size == 0:
            continue
        index = int(matches[0])
        base = base_matrix(stream.root[index:index + 1], stream.root_quat[index:index + 1], compare)
        measured = stream.measured[index:index + 1]
        swapped = measured.copy()
        swapped[0, LEFT_ARM_INDEX] = measured[0, RIGHT_ARM_INDEX]
        swapped[0, RIGHT_ARM_INDEX] = measured[0, LEFT_ARM_INDEX]
        zeroed = measured.copy()
        zeroed[0, ARM_INDEX] = 0.0
        record = {"tick": int(sample["physics_tick"]), "identity": [], "swapped": [], "zeroed": []}
        for side in ("left", "right"):
            simulator = np.asarray(pose[f"{side}_hand_palm_link"][:3], dtype=float)
            record["identity"].append(float(np.linalg.norm(urdf.palms(measured, base)[side][0] - simulator)))
            record["swapped"].append(float(np.linalg.norm(urdf.palms(swapped, base)[side][0] - simulator)))
            record["zeroed"].append(float(np.linalg.norm(urdf.palms(zeroed, base)[side][0] - simulator)))
        rows.append(record)
    if not rows:
        return {"samples": len(samples), "matched": 0}
    return {
        "samples": len(samples), "matched": len(rows),
        "identity_max_error_m": float(max(max(row["identity"]) for row in rows)),
        "identity_mean_error_m": float(np.mean([np.mean(row["identity"]) for row in rows])),
        "swapped_arms_mean_error_m": float(np.mean([np.mean(row["swapped"]) for row in rows])),
        "zeroed_arms_mean_error_m": float(np.mean([np.mean(row["zeroed"]) for row in rows])),
    }


# ------------------------------------------------------------------- figures


def render_figures(outdir: Path, summary: dict) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures = outdir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    written = []

    series = np.load(outdir / "series" / "palm_series.npz")
    tags = sorted({key.rsplit("__", 1)[0] for key in series.files})
    figure, axes = plt.subplots(2, 2, figsize=(13.5, 7.5), sharex=False)
    for axis, tag in zip(axes.ravel(), tags):
        meta = json.loads(str(series[f"{tag}__meta"].tolist()))
        time = series[f"{tag}__t"]
        for side, colour in (("left", "#1f77b4"), ("right", "#d62728")):
            axis.plot(time, series[f"{tag}__measured_{side}"], color=colour, linewidth=1.0,
                      label=f"realised {side[0].upper()}")
            axis.plot(time, series[f"{tag}__target_{side}"], color=colour, linewidth=1.0, linestyle="--",
                      alpha=0.9, label=f"controller target {side[0].upper()}")
        axis.axvline(meta["start_s"], color="black", linewidth=0.9, linestyle=":",
                     label="policy Start" if tag == tags[0] else None)
        axis.axhline(meta["table_z"], color="green", linewidth=0.9, linestyle="-.",
                     label="worktop (pelvis frame)" if tag == tags[0] else None)
        axis.set_title(f"{meta['session']} / rollout-0{meta['rollout']} ({meta['model']})", fontsize=10)
        axis.set_xlabel("sim time (s)")
        axis.grid(alpha=0.25)
    for row in (0, 1):
        axes[row][0].set_ylabel("palm z, pelvis frame (m)")
    axes[0][0].legend(fontsize=7, ncol=2)
    figure.suptitle("Controller target vs realised palm height; the pre-Start segment is the raw session "
                    "ramp/hold, the rest is the rollout", fontsize=10)
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    path = figures / "target_vs_measured_palm_z.png"
    figure.savefig(path, dpi=110)
    plt.close(figure)
    written.append(str(path))

    sweeps = summary["lag_sweep"]
    figure, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))
    colours = {"fine-tuned": "#1f77b4", "groot": "#d62728"}
    for model, colour in colours.items():
        for field, linestyle, label in (("arm_joint_rmse_rad", "-", "arm joint RMSE"),
                                        ("palm_dz_mae_m", "--", "palm dz MAE")):
            curves = sorted((row["lag_s"], row[field]) for row in sweeps
                            if row.get("model") == model and row.get(field) is not None)
            if not curves:
                continue
            lag = np.array([item[0] for item in curves])
            values = np.array([item[1] for item in curves])
            axes[0].plot(lag, (values - values.min()) / (values.max() - values.min() + 1e-12),
                         color=colour, linestyle=linestyle, label=f"{model} · {label}")
            axes[1].plot(lag, values, color=colour, linestyle=linestyle, label=f"{model} · {label}")
            best = lag[int(np.argmin(values))]
            axes[1].plot([best], [values.min()], marker="v", color=colour, markersize=7)
            axes[1].annotate(f"{best:+.2f}s", (best, values.min()), fontsize=7, color=colour)
    for axis in axes:
        axis.axvline(0.0, color="black", linewidth=0.9)
        axis.set_xlabel("lag applied to the measured side (s)")
        axis.grid(alpha=0.3)
        axis.legend(fontsize=8)
    axes[0].set_title("lag sweep, min-max normalised (0 would mean a pure time shift)")
    axes[1].set_title("lag sweep, absolute error (▼ = best lag)")
    figure.tight_layout()
    path = figures / "lag_sweep.png"
    figure.savefig(path, dpi=110)
    plt.close(figure)
    written.append(str(path))

    joints = summary["joint_gap_series"]
    names = joints["joints"]
    tags = sorted(joints["series"])
    figure, axes = plt.subplots(1, 2, figsize=(13.5, 5.4))
    width = 0.2
    for position, tag in enumerate(tags):
        offsets = np.arange(len(names)) + (position - (len(tags) - 1) / 2) * width
        axes[0].bar(offsets, joints["series"][tag]["mean_signed_gap_rad"], width=width, label=tag)
        axes[1].bar(offsets, np.array(joints["series"][tag]["saturation_fraction"]) * 100.0,
                    width=width, label=tag)
    axes[0].axhline(0.0, color="black", linewidth=0.8)
    axes[0].set_ylabel("mean(target − measured)  (rad)")
    axes[0].set_title("signed controller-target gap per joint, whole recording")
    axes[1].set_ylabel("applied torque at the ±limit (%)")
    axes[1].set_title("applied-torque saturation per joint (observational)")
    for axis in axes:
        axis.set_xticks(np.arange(len(names)))
        axis.set_xticklabels(names, rotation=90, fontsize=7)
        axis.grid(alpha=0.3, axis="y")
        axis.legend(fontsize=7)
    figure.tight_layout()
    path = figures / "joint_gap_and_saturation.png"
    figure.savefig(path, dpi=110)
    plt.close(figure)
    written.append(str(path))
    return written


# ---------------------------------------------------------------------- main


def stage_analyse(args: argparse.Namespace) -> int:
    compare = load_compare_module()
    urdf = load_urdf(compare, args.urdf)
    (OUTPUT / "tables").mkdir(parents=True, exist_ok=True)
    (OUTPUT / "series").mkdir(parents=True, exist_ok=True)

    results, lag_rows, control_rows, joint_rows, segment_rows, clock_rows = [], [], [], [], [], []
    series: dict[str, np.ndarray] = {}
    gap_series: dict[str, dict] = {}

    for entry in rollout_metadata():
        tag = f"{entry['session']}-r{entry['index']:02d}"
        print(f"[analyse] {tag}", flush=True)
        stream = TrackingStream(entry["rollout_dir"], entry["raw_parquet"])
        started = stream.wall_ns >= stream.onset_wall_ns
        result = analyse_rollout(entry, stream, urdf, compare)
        result["fk_gate"] = fk_gate(stream, entry, urdf, compare)
        result["bridge_clock"] = bridge_clock(entry["rollout_dir"], stream)
        metrics_path = CAMPAIGN / entry["session"] / "raw" / "isaac.metrics.json"
        if metrics_path.is_file():
            metrics = json.loads(metrics_path.read_text())
            result["target_freshness"]["controller_command_age_ticks"] = (
                metrics.get("controller", {}) or {}).get("command_age_ticks")
        results.append(result)

        sweep = lag_sweep(stream, urdf, started)
        for row in sweep:
            row.update({"session": entry["session"], "model": entry["model"],
                        "rollout": entry["index"], "tag": tag})
        lag_rows.extend(sweep)
        usable = [row for row in sweep if "arm_joint_rmse_rad" in row]
        zero = min(usable, key=lambda row: abs(row["lag_s"]))
        joint_best = min(usable, key=lambda row: row["arm_joint_rmse_rad"])
        palm_best = min(usable, key=lambda row: row["palm_dz_mae_m"])
        result["lag"] = {
            "zero_lag_arm_joint_rmse_rad": zero["arm_joint_rmse_rad"],
            "zero_lag_palm_dz_mae_m": zero["palm_dz_mae_m"],
            "best_arm_joint": {"lag_s": joint_best["lag_s"], "rmse_rad": joint_best["arm_joint_rmse_rad"]},
            "best_palm_dz": {"lag_s": palm_best["lag_s"], "mae_m": palm_best["palm_dz_mae_m"]},
            "arm_rmse_gain_at_best_lag_percent": float(
                100.0 * (zero["arm_joint_rmse_rad"] - joint_best["arm_joint_rmse_rad"]) /
                zero["arm_joint_rmse_rad"]),
            "palm_mae_gain_at_best_lag_percent": float(
                100.0 * (zero["palm_dz_mae_m"] - palm_best["palm_dz_mae_m"]) / zero["palm_dz_mae_m"]),
        }

        for row in control_table(stream, urdf, compare, started):
            row.update({"session": entry["session"], "model": entry["model"],
                        "rollout": entry["index"], "tag": tag})
            control_rows.append(row)
        for row in result["per_joint"]:
            row.update({"session": entry["session"], "model": entry["model"],
                        "rollout": entry["index"], "tag": tag})
            joint_rows.append(row)
        for name, window in result["segments"].items():
            segment_rows.append({
                "session": entry["session"], "model": entry["model"], "rollout": entry["index"],
                "segment": name, "n_rows": window["n_rows"], "duration_s": window["duration_s"],
                "arm_joint_mae_rad": window.get("joint_arm", {}).get("mae_rad"),
                "arm_joint_rmse_rad": window.get("joint_arm", {}).get("rmse_rad"),
                "arm_joint_bias_rad": window.get("joint_arm", {}).get("bias_rad"),
                "waist_joint_mae_rad": window.get("joint_waist", {}).get("mae_rad"),
                "palm_dz_mae_m": window.get("palm_pelvis", {}).get("pooled", {}).get("dz_mae_m"),
                "palm_dz_bias_m": window.get("palm_pelvis", {}).get("pooled", {}).get("dz_bias_m"),
                "palm_dz_median_m": window.get("palm_pelvis", {}).get("pooled", {}).get("dz_median_m"),
                "palm_rmse_3d_m": window.get("palm_pelvis", {}).get("pooled", {}).get("rmse_3d_m"),
                "world_palm_dz_median_m": window.get("palm_world", {}).get("pooled", {}).get("dz_median_m"),
                "world_palm_rmse_3d_m": window.get("palm_world", {}).get("pooled", {}).get("rmse_3d_m"),
                "arm_target_speed_median_rad_s": window.get("arm_target_speed_median_rad_s"),
            })
        clock = dict(result["bridge_clock"])
        clock.update({"session": entry["session"], "model": entry["model"],
                      "rollout": entry["index"], "tag": tag})
        clock_rows.append(clock)

        gap_series[tag] = {
            "mean_signed_gap_rad": [row["mean_signed_gap_rad"] for row in result["per_joint"]],
            "saturation_fraction": [row["saturation_fraction"] for row in result["per_joint"]],
        }
        target_pelvis, measured_pelvis = urdf.palms(stream.target), urdf.palms(stream.measured)
        series[f"{tag}__t"] = stream.sim_s.astype(np.float32)
        for side in ("left", "right"):
            series[f"{tag}__target_{side}"] = target_pelvis[side][:, 2].astype(np.float32)
            series[f"{tag}__measured_{side}"] = measured_pelvis[side][:, 2].astype(np.float32)
        series[f"{tag}__meta"] = np.array(json.dumps({
            "session": entry["session"], "model": entry["model"], "rollout": entry["index"],
            "start_s": float(stream.sim_s[int(np.argmax(stream.wall_ns >= stream.start_wall_ns))]),
            "table_z": float(stream.surface - stream.root[-1, 2]),
        }))
        del stream

    np.savez_compressed(OUTPUT / "series" / "palm_series.npz", **series)
    write_csv(OUTPUT / "tables" / "joint_gap.csv", joint_rows)
    write_csv(OUTPUT / "tables" / "lag_sweep.csv", lag_rows)
    write_csv(OUTPUT / "tables" / "controls.csv", control_rows)
    write_csv(OUTPUT / "tables" / "segments.csv", segment_rows)
    write_csv(OUTPUT / "tables" / "bridge_clock.csv",
              [{key: value for key, value in row.items() if not isinstance(value, dict)} |
               {"command_age_median_s": (row.get("command_age_s") or {}).get("median"),
                "command_age_p95_s": (row.get("command_age_s") or {}).get("p95"),
                "state_vs_measured_arm_mean_abs_rad":
                    (row.get("observation_state_vs_measured_arm") or {}).get("mean_abs_rad")}
               for row in clock_rows])

    models: dict[str, dict] = {}
    for row in results:
        bucket = models.setdefault(row["model"], {"rollouts": []})
        bucket["rollouts"].append(row["tag"])
    for model, bucket in models.items():
        selected = [row for row in results if row["model"] == model]
        bucket["arm_joint_mae_rad_mean_post_start"] = float(
            np.mean([row["joint"]["arm_post_start"]["mae_rad"] for row in selected]))
        bucket["arm_joint_rmse_rad_mean_post_start"] = float(
            np.mean([row["joint"]["arm_post_start"]["rmse_rad"] for row in selected]))
        bucket["palm_pelvis_dz_mae_m_mean_post_start"] = float(
            np.mean([row["palm_pelvis_post_start"]["pooled"]["dz_mae_m"] for row in selected]))
        bucket["palm_pelvis_dz_bias_m_mean_post_start"] = float(
            np.mean([row["palm_pelvis_post_start"]["pooled"]["dz_bias_m"] for row in selected]))
        bucket["palm_pelvis_rmse_3d_m_mean_post_start"] = float(
            np.mean([row["palm_pelvis_post_start"]["pooled"]["rmse_3d_m"] for row in selected]))
        bucket["palm_world_dz_median_m_mean_crop"] = float(
            np.mean([row["palm_world_crop"]["pooled"]["dz_median_m"] for row in selected]))
        bucket["arm_saturation_fraction_joint_samples"] = float(
            np.mean([row["torque"]["arm_saturation_fraction_joint_samples"] for row in selected]))

    summary = {"schema_version": SCHEMA_VERSION, "script_version": SCRIPT_VERSION,
               "output_dir": str(OUTPUT), "models": models, "rollouts": results,
               "lag_sweep": lag_rows, "controls": control_rows, "segments": segment_rows,
               "joint_gap_series": {"joints": list(BODY_JOINT_ORDER), "series": gap_series},
               "decision_inputs": decision_inputs(results, control_rows)}
    write_json(OUTPUT / "summary.json", summary)
    print(f"[analyse] wrote {OUTPUT / 'summary.json'} ({len(results)} rollouts)")
    return 0


def decision_inputs(results: list[dict], control_rows: list[dict]) -> dict:
    """The numbers the A/B/C/D verdict is read off, derived, never asserted."""
    pelvis = [row["palm_pelvis_post_start"]["pooled"]["dz_bias_m"] for row in results]
    world = [row["palm_world_crop"]["pooled"]["dz_median_m"] for row in results]
    frame_difference = [abs(row["palm_world_crop"]["pooled"]["dz_median_m"] -
                            row["palm_pelvis_crop"]["pooled"]["dz_median_m"]) for row in results]
    identity = {(row["tag"], row["variant"]): row for row in control_rows}
    controls = {}
    for tag, variant in sorted({(row["tag"], row["variant"]) for row in control_rows}):
        row = identity[(tag, variant)]
        controls.setdefault(variant, {}).setdefault(tag, {
            "arm_joint_mae_rad": row["arm_joint_mae_rad"],
            "palm_dz_bias_m": row["palm_dz_bias_m"],
            "palm_rmse_3d_m": row["palm_rmse_3d_m"],
        })
    return {
        "palm_dz_bias_pelvis_post_start_m": [float(value) for value in pelvis],
        "palm_dz_bias_pelvis_post_start_range_m": [float(min(pelvis)), float(max(pelvis))],
        "palm_dz_world_crop_median_range_m": [float(min(world)), float(max(world))],
        "frame_choice_effect_max_m": float(max(frame_difference)),
        "best_lag_gain_palm_mae_percent_max": float(max(row["lag"]["palm_mae_gain_at_best_lag_percent"]
                                                         for row in results)),
        "best_lag_gain_arm_rmse_percent_max": float(max(row["lag"]["arm_rmse_gain_at_best_lag_percent"]
                                                        for row in results)),
        "parity_max_residual_nm": float(max(row["parity_check"]["max_abs_residual_nm"] for row in results)),
        "feedforward_torque_abs_max_nm": float(max(row["torque"]["feedforward_abs_max_nm"] for row in results)),
        "fk_gate_identity_max_error_m": float(max(row["fk_gate"]["identity_max_error_m"] for row in results)),
        "fk_gate_swapped_arms_mean_error_m": float(np.mean(
            [row["fk_gate"]["swapped_arms_mean_error_m"] for row in results])),
        "arm_target_speed_median_rad_s": [float(row["jitter_diagnostic"]["target_speed_median_rad_s"])
                                          for row in results],
        "palm_dz_mae_raw_vs_smoothed_target_m": [
            [row["jitter_diagnostic"]["palm_dz_mae_raw_m"],
             row["jitter_diagnostic"]["palm_dz_mae_smoothed_target_m"]] for row in results],
        "controls": controls,
        "pelvis_target_read_as_world_rmse_3d_m": {
            row["tag"]: row["pelvis_target_read_as_world_rmse_3d_m"] for row in control_rows
            if row["variant"] == "identity"},
    }


def stage_figures(args: argparse.Namespace) -> int:
    summary = json.loads((OUTPUT / "summary.json").read_text())
    print("[figures] wrote " + ", ".join(render_figures(OUTPUT, summary)))
    return 0


def build_manifest() -> dict:
    inputs = []
    for entry in rollout_metadata():
        for name in ("tracking.jsonl", "bridge-telemetry.jsonl", "isaac.samples.jsonl", "rollout.json"):
            path = Path(entry["rollout_dir"]) / name
            inputs.append({"path": str(path.relative_to(REPO_ROOT)), "sha256": sha256(path),
                           "bytes": path.stat().st_size})
        raw = entry["raw_parquet"]
        inputs.append({"path": str(raw.relative_to(REPO_ROOT)), "sha256": sha256(raw),
                       "bytes": raw.stat().st_size})
    for path in (COMPARE_SCRIPT, DEFAULT_URDF):
        inputs.append({"path": str(path.relative_to(REPO_ROOT)), "sha256": sha256(path),
                       "bytes": path.stat().st_size})
    return {
        "schema_version": SCHEMA_VERSION,
        "script": {"path": str(Path(__file__).resolve().relative_to(REPO_ROOT)),
                   "sha256": sha256(Path(__file__).resolve()), "version": SCRIPT_VERSION},
        "commands": [
            "data/venvs/hf-datasets/bin/python scripts/validate-blockstacking-tracking.py analyse",
            "python3 scripts/validate-blockstacking-tracking.py figures",
            "python3 scripts/validate-blockstacking-tracking.py manifest",
        ],
        "inputs": inputs,
        "constants": {
            "tracking_dt_s": TRACKING_DT_S, "lag_range_s": [LAG_MIN_S, LAG_MAX_S],
            "moving_threshold_rad_s": MOVE_RAD_S, "move_sensitivity_rad_s": list(MOVE_SENSITIVITY_RAD_S),
            "saturation_tolerance_nm": SATURATION_TOL_NM,
            "arm_joints": list(ARM_JOINTS), "waist_joints": list(WAIST_JOINTS),
        },
        "pinned_controller_values": {
            "source": "src/humanoid_lab/controllers/sonic.py (copied from the pinned SONIC deployment)",
            "arm_kp_nm_per_rad": float(deploy_gains()[0][BODY_JOINT_ORDER.index("left_elbow_joint")]),
            "arm_kd": float(deploy_gains()[1][BODY_JOINT_ORDER.index("left_elbow_joint")]),
            "arm_effort_limit_nm": float(LIMITS[BODY_JOINT_ORDER.index("left_elbow_joint")]),
            "wrist_pitch_yaw_effort_limit_nm": float(LIMITS[BODY_JOINT_ORDER.index("left_wrist_pitch_joint")]),
            "note": "read only; no controller parameter was changed by this experiment",
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", nargs="?", default="analyse", choices=("analyse", "figures", "manifest"))
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    args = parser.parse_args(argv)
    if args.stage == "analyse":
        return stage_analyse(args)
    if args.stage == "figures":
        return stage_figures(args)
    write_json(OUTPUT / "manifest.json", build_manifest())
    print(f"[manifest] wrote {OUTPUT / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
