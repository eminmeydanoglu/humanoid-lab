#!/usr/bin/env python3
"""Compare the BlockStacking rollouts against the training demonstrations.

The question this answers is whether the "hands stay high" behaviour visible in
``data/outputs/blockstacking-debug`` is (a) a property of the demonstrations the
policies were trained on, (b) a property of the train/eval transform, or (c) a
property of the policies.

Neither dataset copy stores a hand pose: ``observation.state`` carries joint
positions only.  Palm poses are produced by forward kinematics on the robot
description, with the ``pelvis`` link as the root, so *base-relative* output
needs no world assumption about the dataset.  The FK is not trusted on faith:
``fk-check`` reproduces the simulator's own measured palm world poses from the
recorded joint values plus the recorded root pose, and fails loudly if that
agreement is lost.

Frames (all computed, never translated by hand):

* ``pelvis`` -- palm position in the pelvis frame.  The one frame both sides
  share without a scene assumption: the dataset has no world pose and no scene
  metadata, and the rollouts' world height is inflated by the SONIC support band,
  which holds the robot ~0.19 m above standing until the policy takes over.
* ``torso``  -- palm position in ``torso_link``.  Identical to ``pelvis`` for the
  dataset, whose waist joints are the synthetic zero constant.
* ``table``  -- palm height above the worktop.  Exact for the rollouts
  (``scene.table.surface_height_m``, re-measured live every run).  The dataset's
  worktop is unknown by construction; it is estimated from the demonstrated
  reaches (``TABLE_PROXY_*``) and every table-relative number is reported with
  its +-0.05 m sensitivity, biased so that the dataset's heights are
  *under*-stated rather than overstated.

    scripts/compare-blockstacking-dataset.py all
    scripts/compare-blockstacking-dataset.py fk-check --urdf third_party/.../g1_body29_hand14.urdf
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Sequence

import numpy as np

SCHEMA_VERSION = 2
SCRIPT_VERSION = "compare-blockstacking-dataset.py/1.0.0"

REPO_ROOT = Path(__file__).resolve().parents[1]

PSI0_TRAIN = Path("psi0-unitree-dex3-sonic-v1/train")
GROOT_TRAIN = Path("groot/unitree-dex3-sonic-v1/train")
RAW_FIRST_TUR_HAM = Path("first_tur_ham/unitree-g1-dex3")
DEFAULT_URDF = REPO_ROOT / "third_party" / "Psi0" / "real" / "assets" / "g1" / "g1_body29_hand14.urdf"

TASK_INDEX_COLLECTIONS = (
    "G1_Dex3_BlockStacking_Dataset", "G1_Dex3_CameraPackaging_Dataset",
    "G1_Dex3_GraspSquare_Dataset", "G1_Dex3_ObjectPlacement_Dataset",
    "G1_Dex3_PickApple_Dataset", "G1_Dex3_PickBottle_Dataset",
    "G1_Dex3_PickCharger_Dataset", "G1_Dex3_PickDoll_Dataset",
    "G1_Dex3_PickGum_Dataset", "G1_Dex3_PickSnack_Dataset",
    "G1_Dex3_PickTissue_Dataset", "G1_Dex3_Pouring_Dataset",
    "G1_Dex3_ToastedBread_Dataset",
)
BLOCKSTACKING_TASK_INDEX = 0
BLOCKSTACKING_PROMPT = "Stack the three cubic blocks on the black tape in the order red, yellow, blue."

BODY_JOINT_ORDER = (
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)
ARM_JOINTS_LEFT = BODY_JOINT_ORDER[15:22]
ARM_JOINTS_RIGHT = BODY_JOINT_ORDER[22:29]

#: psi0 copy state layout (``configs/datasets/psi0/unitree_dex3_sonic_v1.yaml``
#: ``state.joint_layout``, channel-exact against the raw source; see ``recon``).
PSI0_LAYOUT = {"left_arm": slice(15, 22), "right_arm": slice(22, 29),
               "left_hand": slice(29, 36), "right_hand": slice(36, 43)}
#: groot copy state layout (``meta/modality.json``); the same joints, a different
#: block order and a different within-hand order -- see ``layout`` in recon.
GROOT_LAYOUT = {"left_arm": slice(15, 22), "left_hand": slice(22, 29),
                "right_arm": slice(29, 36), "right_hand": slice(36, 43)}
GROOT_LEFT_HAND_ORDER = ("index_0_joint", "index_1_joint", "middle_0_joint", "middle_1_joint",
                         "thumb_0_joint", "thumb_1_joint", "thumb_2_joint")

PALM_LINKS = {"left": "left_hand_palm_link", "right": "right_hand_palm_link"}

TABLE_PROXY_CUBE_M = 0.05   # block edge; the palm cannot be lower than the block it grips
TABLE_PROXY_BAND_M = 0.05   # reported sensitivity of every table-relative number
SMOOTH_WINDOW_S = 0.20
ONSET_DELTA_M = 0.03
ONSET_SUSTAIN_S = 0.20
START_WINDOW_S = 0.50
HEIGHT_BANDS_M = (0.10, 0.15, 0.20)
ROLLOUT_TAGS = ("psi0-rollout-1", "groot-rollout-1")


class ComparisonError(RuntimeError):
    """The comparison cannot run; the message is meant for the operator."""


# ------------------------------------------------------------------ locations


def data_root(explicit: Path | None) -> Path:
    if explicit is not None:
        return Path(explicit)
    for candidate in (Path("/data/datasets"), REPO_ROOT / "data" / "datasets"):
        if candidate.is_dir():
            return candidate
    raise ComparisonError("cannot locate the dataset root (pass --data-root)")


def output_dir(explicit: Path | None) -> Path:
    return Path(explicit) if explicit is not None else (
        REPO_ROOT / "data" / "outputs" / "blockstacking-debug" / "dataset-comparison")


def campaign_dir() -> Path:
    return REPO_ROOT / "data" / "outputs" / "blockstacking-debug"


# ----------------------------------------------------------------------- URDF


def rpy_to_matrix(rpy: Sequence[float]) -> np.ndarray:
    roll, pitch, yaw = (float(v) for v in rpy)
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return (np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1.0]])
            @ np.array([[cp, 0, sp], [0, 1.0, 0], [-sp, 0, cp]])
            @ np.array([[1.0, 0, 0], [0, cr, -sr], [0, sr, cr]]))


def axis_angle(axis: Sequence[float], angle: np.ndarray) -> np.ndarray:
    """Rodrigues rotation vectorised over ``angle`` -> (..., 3, 3)."""
    a = np.asarray(axis, dtype=float)
    norm = float(np.linalg.norm(a))
    q = np.asarray(angle, dtype=float)
    if norm == 0.0:
        return np.broadcast_to(np.eye(3), q.shape + (3, 3)).copy()
    x, y, z = a / norm
    c, s = np.cos(q), np.sin(q)
    C = 1.0 - c
    out = np.empty(q.shape + (3, 3), dtype=float)
    out[..., 0, 0] = c + x * x * C
    out[..., 0, 1] = x * y * C - z * s
    out[..., 0, 2] = x * z * C + y * s
    out[..., 1, 0] = y * x * C + z * s
    out[..., 1, 1] = c + y * y * C
    out[..., 1, 2] = y * z * C - x * s
    out[..., 2, 0] = z * x * C - y * s
    out[..., 2, 1] = z * y * C + x * s
    out[..., 2, 2] = c + z * z * C
    return out


class Urdf:
    """Minimal URDF reader with a vectorised chain forward kinematics."""

    def __init__(self, path: Path) -> None:
        root = ET.parse(str(path)).getroot()
        self.path = Path(path)
        self.joints: list[dict[str, Any]] = []
        for element in root.findall("joint"):
            origin = element.find("origin")
            xyz = np.array([float(v) for v in (origin.get("xyz") or "0 0 0").split()]) if origin is not None else np.zeros(3)
            rpy = np.array([float(v) for v in (origin.get("rpy") or "0 0 0").split()]) if origin is not None else np.zeros(3)
            axis_element = element.find("axis")
            axis = (np.array([float(v) for v in axis_element.get("xyz").split()])
                    if axis_element is not None else np.array([0.0, 0.0, 1.0]))
            transform = np.eye(4)
            transform[:3, :3] = rpy_to_matrix(rpy)
            transform[:3, 3] = xyz
            self.joints.append({"name": element.get("name"), "type": element.get("type"),
                                "parent": element.find("parent").get("link"),
                                "child": element.find("child").get("link"),
                                "origin": transform, "axis": axis})
        links = {j["child"] for j in self.joints} | {j["parent"] for j in self.joints}
        self.root_link = next(link for link in links if all(j["child"] != link for j in self.joints))
        self._chains: dict[str, list[dict[str, Any]]] = {}

    def chain(self, target_link: str) -> list[dict[str, Any]]:
        if target_link in self._chains:
            return self._chains[target_link]
        chain: list[dict[str, Any]] = []
        link = target_link
        while link != self.root_link:
            joint = next(j for j in self.joints if j["child"] == link)
            chain.append(joint)
            link = joint["parent"]
        chain.reverse()
        self._chains[target_link] = chain
        return chain

    def fk_chain(self, chain: Sequence[dict[str, Any]], q: np.ndarray,
                 base: np.ndarray | None = None) -> np.ndarray:
        """(N, 4, 4) tip transform; ``q`` is (N, len(chain)) in chain order."""
        q = np.atleast_2d(np.asarray(q, dtype=float))
        n = q.shape[0]
        current = np.broadcast_to(np.eye(4), (n, 4, 4)).copy()
        if base is not None:
            current[:] = np.asarray(base, dtype=float)
        for index, joint in enumerate(chain):
            step = np.broadcast_to(joint["origin"], (n, 4, 4)).copy()
            if joint["type"] != "fixed":
                rotation = np.broadcast_to(np.eye(4), (n, 4, 4)).copy()
                rotation[:, :3, :3] = axis_angle(joint["axis"], q[:, index])
                step = step @ rotation
            current = current @ step
        return current

    def chain_joint_values(self, chain: Sequence[dict[str, Any]], q_body: np.ndarray) -> np.ndarray:
        """Pick the chain's joints out of canonical body-order joint vectors."""
        values = np.zeros((q_body.shape[0], len(chain)))
        for index, joint in enumerate(chain):
            if joint["name"] in BODY_JOINT_ORDER:
                values[:, index] = q_body[:, BODY_JOINT_ORDER.index(joint["name"])]
        return values

    def palms(self, q_body: np.ndarray, base: np.ndarray | None = None) -> dict[str, np.ndarray]:
        out = {}
        for side, link in PALM_LINKS.items():
            chain = self.chain(link)
            tip = self.fk_chain(chain, self.chain_joint_values(chain, q_body), base)
            out[side] = tip[:, :3, 3]
        return out


def quaternion_wxyz_to_matrix(quaternion: Sequence[float]) -> np.ndarray:
    w, x, y, z = (float(v) for v in quaternion)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


# ------------------------------------------------------------------- plumbing


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not Path(path).is_file():
        return rows
    with Path(path).open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def read_parquet(path: Path, columns: Sequence[str]) -> dict[str, Any]:
    import pyarrow.parquet as pq

    return pq.read_table(str(path), columns=list(columns)).to_pydict()


def episode_path(root: Path, episode: int) -> Path:
    return Path(root) / "data" / f"chunk-{episode // 1000:03d}" / f"episode_{episode:06d}.parquet"


def write_json(path: Path, payload: Any) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: Sequence[dict[str, Any]], columns: Sequence[str] | None = None) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        Path(path).write_text("", encoding="utf-8")
        return
    names = list(columns) if columns else list(rows[0])
    lines = [",".join(names)]
    for row in rows:
        cells = []
        for name in names:
            value = row.get(name)
            if isinstance(value, float):
                cells.append("" if math.isnan(value) else f"{value:.9g}")
            else:
                cells.append("" if value is None else str(value))
        lines.append(",".join(cells))
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def sha256(path: Path) -> str | None:
    if not Path(path).is_file():
        return None
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def percentiles(values: np.ndarray, fractions: Sequence[float]) -> list[float]:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return [float("nan")] * len(fractions)
    return [float(np.percentile(values, fraction * 100.0)) for fraction in fractions]


def describe(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"count": 0}
    q = percentiles(values, (0.0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0))
    return {"count": int(values.size), "mean": float(values.mean()), "std": float(values.std()),
            "min": q[0], "p01": q[1], "p05": q[2], "p25": q[3], "median": q[4],
            "p75": q[5], "p95": q[6], "p99": q[7], "max": q[8]}


def empirical_percentile(sample: np.ndarray, value: float) -> float | None:
    sample = np.asarray(sample, dtype=float)
    sample = sample[np.isfinite(sample)]
    if sample.size == 0 or not math.isfinite(value):
        return None
    return float((sample < value).mean() * 100.0 + 0.5 * (sample == value).mean() * 100.0)


def robust_z(value: float, values: np.ndarray) -> float | None:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0 or not math.isfinite(value):
        return None
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    return None if mad == 0.0 else 0.6745 * (value - median) / mad


def smooth(values: np.ndarray, dt: float, window_s: float = SMOOTH_WINDOW_S) -> np.ndarray:
    width = max(1, int(round(window_s / dt)))
    values = np.asarray(values, dtype=float)
    if width <= 1 or values.size == 0:
        return values
    pad_left = width // 2
    pad_right = width - 1 - pad_left
    padded = np.concatenate([np.full(pad_left, values[0]), values, np.full(pad_right, values[-1])])
    return np.convolve(padded, np.ones(width) / width, mode="valid")[: values.size]


# -------------------------------------------------------------- phase metrics


def phase_metrics(t: np.ndarray, z: np.ndarray) -> dict[str, Any]:
    """Shared, automatic phase description of one palm-height series."""
    out: dict[str, Any] = {}
    t = np.asarray(t, dtype=float)
    z = np.asarray(z, dtype=float)
    if t.size < 5:
        return out
    dt = float(np.median(np.diff(t)))
    if not math.isfinite(dt) or dt <= 0:
        return out
    zs = smooth(z, dt)
    start_samples = max(1, int(round(START_WINDOW_S / dt)))
    start = float(np.median(zs[:start_samples]))
    out.update({"z_start_m": start, "z_min_m": float(zs.min()), "z_max_m": float(zs.max()),
                "z_range_m": float(zs.max() - zs.min()), "duration_s": float(t[-1] - t[0]),
                "t_min_s": float(t[int(np.argmin(zs))])})

    departure = np.abs(zs - start) > ONSET_DELTA_M
    sustain = max(1, int(round(ONSET_SUSTAIN_S / dt)))
    onset_index, run = None, 0
    for index, flag in enumerate(departure):
        run = run + 1 if flag else 0
        if run >= sustain:
            onset_index = index - sustain + 1
            break
    out["onset_s"] = float(t[onset_index]) if onset_index is not None else float("nan")
    out["start_hold_s"] = (out["onset_s"] - float(t[0])) if onset_index is not None else float("nan")

    velocity = np.gradient(zs, dt)
    out["v_down_min_m_s"] = float(velocity.min())
    out["v_down_p10_m_s"] = float(np.percentile(velocity, 10.0))
    out["v_up_p90_m_s"] = float(np.percentile(velocity, 90.0))
    out["down_amplitude_m"] = start - out["z_min_m"]
    out["up_amplitude_m"] = out["z_max_m"] - out["z_min_m"]

    min_index = int(np.argmin(zs))
    if min_index > start_samples:
        peak = float(zs[: min_index + 1].max())
        out["approach_amplitude_m"] = peak - out["z_min_m"]
        if onset_index is not None and min_index > onset_index:
            seconds = float(t[min_index] - t[onset_index])
            out["approach_seconds"] = seconds
            if seconds > 0:
                out["approach_rate_m_s"] = out["approach_amplitude_m"] / seconds
    return out


def band_fractions(values: np.ndarray, table_z: float, bands: Sequence[float] = HEIGHT_BANDS_M) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return {}
    return {f"frac_above_{int(round(b * 100)):02d}cm": float((values > table_z + b).mean()) for b in bands}


# ---------------------------------------------------------- recon / layout


def verify_layouts(root: Path) -> dict[str, Any]:
    """Channel-exact identification of each copy's state blocks against the source."""
    raw_path = (Path(root) / RAW_FIRST_TUR_HAM / TASK_INDEX_COLLECTIONS[BLOCKSTACKING_TASK_INDEX]
                / "data" / "chunk-000" / "file-000.parquet")
    if not raw_path.is_file():
        return {"available": False, "reason": f"raw source not found at {raw_path}"}
    columns = read_parquet(raw_path, ["episode_index", "observation.state", "action"])
    mask = np.asarray(columns["episode_index"]) == 0
    raw_state = np.asarray(columns["observation.state"], dtype=float)[mask]
    raw_action = np.asarray(columns["action"], dtype=float)[mask]
    raw_blocks = {"left_arm": raw_state[:, 0:7], "right_arm": raw_state[:, 7:14],
                  "left_hand": raw_state[:, 14:21], "right_hand": raw_state[:, 21:28]}
    raw_hand_names = ["Lthumb0", "Lthumb1", "Lthumb2", "Lmid0", "Lmid1", "Lidx0", "Lidx1",
                      "Rthumb0", "Rthumb1", "Rthumb2", "Ridx0", "Ridx1", "Rmid0", "Rmid1"]

    psi0 = read_parquet(episode_path(Path(root) / PSI0_TRAIN, 0), ["observation.state", "action"])
    groot = read_parquet(episode_path(Path(root) / GROOT_TRAIN, 0),
                         ["observation.state", "teleop.left_hand_joints", "teleop.right_hand_joints"])
    result: dict[str, Any] = {"available": True, "reference": str(raw_path),
                              "episode": 0, "raw_hand_names": raw_hand_names}
    for label, state, layout, fps in (
        ("psi0", np.asarray(psi0["observation.state"], dtype=float), PSI0_LAYOUT, 30.0),
        ("groot", np.asarray(groot["observation.state"], dtype=float), GROOT_LAYOUT, 50.0),
    ):
        index = np.clip((np.arange(state.shape[0]) * (30.0 / fps)).round().astype(int), 0, len(raw_state) - 1)
        blocks = {}
        for block in ("left_arm", "right_arm", "left_hand", "right_hand"):
            mine = state[:, layout[block]]
            scored = sorted(((float(np.abs(mine - candidate[index]).mean()), name)
                             for name, candidate in raw_blocks.items()))
            blocks[block] = {"slice": [layout[block].start, layout[block].stop],
                             "matches_raw_block": scored[0][1],
                             "mean_abs_diff": scored[0][0],
                             "second_best": {"block": scored[1][1], "mean_abs_diff": scored[1][0]}}
        result[label] = blocks

    # Within-hand order of each copy, one column at a time, then a
    # reconstruction check: putting the copy's block back into the raw channel
    # order must reproduce the raw hand block.  This is what turns "the names
    # disagree" into "the values are a permutation of the same measurements".
    index = np.clip((np.arange(len(groot["observation.state"])) * (30.0 / 50.0)).round().astype(int),
                    0, len(raw_state) - 1)
    for label, state, layout in (("psi0", np.asarray(psi0["observation.state"], dtype=float), PSI0_LAYOUT),
                                 ("groot", np.asarray(groot["observation.state"], dtype=float), GROOT_LAYOUT)):
        row_index = None if label == "psi0" else index
        for block in ("left_hand", "right_hand"):
            mine = state[:, layout[block]]
            reference = raw_blocks[block]
            if row_index is not None:
                reference = reference[row_index]
            order = []
            permutation = []
            for column in range(mine.shape[1]):
                scored = sorted(((float(np.abs(mine[:, column] - reference[:, other]).mean()), other)
                                 for other in range(reference.shape[1])))
                order.append({"column": column,
                              "best_raw_channel": raw_hand_names[(0 if block == "left_hand" else 7) + scored[0][1]],
                              "mean_abs_diff": scored[0][0],
                              "runner_up_diff": scored[1][0]})
                permutation.append(scored[0][1])
            reconstructed = mine[:, np.argsort(permutation)]
            result.setdefault("hand_channel_order", {})[f"{label}_{block}"] = {
                "per_column": order,
                "is_permutation_of_raw": sorted(permutation) == list(range(7)),
                "reconstruction_mean_abs_diff": float(np.abs(reconstructed - reference).mean()),
            }

    psi_action = np.asarray(psi0["action"], dtype=float)
    diff = np.abs(psi_action - raw_action[:, 14:28])
    result["psi0_action_vs_raw_hand_command"] = {
        "mean_abs_diff": float(diff.mean()), "max_abs_diff": float(diff.max()),
        "note": "psi0 action is the frozen SONIC 78D[64:78] hand slice; agreement is order-level",
    }
    return result


def compare_copies(root: Path, episodes: Sequence[int]) -> dict[str, Any]:
    """Equivalence of the two copies, compared on the shared time axis.

    Both copies carry a ``timestamp`` column in seconds from the clip start, so
    the 50 Hz copy is compared after linear resampling onto its own timestamps:
    a naive index ratio would fold a half-sample of interpolation error into a
    claim about the data.
    """
    rows = []
    for episode in episodes:
        psi0 = read_parquet(episode_path(Path(root) / PSI0_TRAIN, episode),
                            ["observation.state", "action.body_token_v1_1", "timestamp"])
        groot = read_parquet(episode_path(Path(root) / GROOT_TRAIN, episode),
                             ["observation.state", "action.motion_token", "timestamp"])
        ps = np.asarray(psi0["observation.state"], dtype=float)
        gs = np.asarray(groot["observation.state"], dtype=float)
        pt = np.asarray(psi0["timestamp"], dtype=float)
        gt = np.asarray(groot["timestamp"], dtype=float)
        diffs = {}
        for block_psi, block_groot in (("left_arm", "left_arm"), ("right_arm", "right_arm")):
            for column in range(7):
                reference = ps[:, PSI0_LAYOUT[block_psi].start + column]
                values = gs[:, GROOT_LAYOUT[block_groot].start + column]
                diffs.setdefault(block_psi, []).append(
                    float(np.abs(np.interp(gt, pt, reference) - values).mean()))
        token_diff = []
        psi_token = np.asarray(psi0["action.body_token_v1_1"], dtype=float)
        groot_token = np.asarray(groot["action.motion_token"], dtype=float)
        for column in range(psi_token.shape[1]):
            token_diff.append(float(np.abs(np.interp(gt, pt, psi_token[:, column]) - groot_token[:, column]).mean()))
        rows.append({"episode": episode, "psi0_frames": int(len(ps)), "groot_frames": int(len(gs)),
                     "rate_ratio": float(len(gs) / len(ps)),
                     "psi0_fps": float(1.0 / np.median(np.diff(pt))),
                     "groot_fps": float(1.0 / np.median(np.diff(gt))),
                     "left_arm_mean_abs_diff": float(np.mean(diffs["left_arm"])),
                     "right_arm_mean_abs_diff": float(np.mean(diffs["right_arm"])),
                     "body_token_mean_abs_diff": float(np.mean(token_diff)),
                     "body_token_max_abs_diff": float(np.max(np.abs(np.interp(gt, pt, psi_token[:, 0]) - groot_token[:, 0])))})
    summary = {"episodes": rows, "sampled": len(rows),
               "method": "psi0 signal linearly interpolated onto groot timestamps"}
    for key in ("left_arm_mean_abs_diff", "right_arm_mean_abs_diff",
                "body_token_mean_abs_diff", "rate_ratio"):
        if rows:
            summary[key] = float(np.mean([row[key] for row in rows]))
    summary["token_quantization"] = {
        "psi0_body_token_values_on_1_16_grid": None, "groot_body_token_values_on_1_16_grid": None}
    if rows:
        psi0 = read_parquet(episode_path(Path(root) / PSI0_TRAIN, rows[0]["episode"]),
                            ["action.body_token_v1_1"])
        groot = read_parquet(episode_path(Path(root) / GROOT_TRAIN, rows[0]["episode"]),
                             ["action.motion_token"])
        for label, values in (("psi0", np.asarray(psi0["action.body_token_v1_1"], dtype=float)),
                              ("groot", np.asarray(groot["action.motion_token"], dtype=float))):
            on_grid = np.abs(values * 16.0 - np.round(values * 16.0)) < 1e-6
            summary["token_quantization"][f"{label}_body_token_values_on_1_16_grid"] = float(on_grid.mean())
            summary["token_quantization"][f"{label}_unique_values"] = int(np.unique(values).size)
    return summary


def stage_recon(args: argparse.Namespace) -> dict[str, Any]:
    root = data_root(args.data_root)
    out = output_dir(args.output_dir)
    result: dict[str, Any] = {"schema_version": SCHEMA_VERSION, "script_version": SCRIPT_VERSION}
    for name, relative in (("psi0", PSI0_TRAIN), ("groot", GROOT_TRAIN)):
        path = Path(root) / relative
        info = json.loads((path / "meta" / "info.json").read_text())
        tasks = [json.loads(line) for line in (path / "meta" / "tasks.jsonl").read_text().splitlines() if line.strip()]
        episodes = [json.loads(line) for line in (path / "meta" / "episodes.jsonl").read_text().splitlines() if line.strip()]
        per_task = {}
        for task in tasks:
            label = task["task"]
            members = [e for e in episodes if label in e["tasks"]]
            per_task[str(task["task_index"])] = {
                "task": label, "collection": TASK_INDEX_COLLECTIONS[task["task_index"]],
                "episodes": len(members), "frames": int(sum(e["length"] for e in members)),
                "episode_indices": [e["episode_index"] for e in members],
            }
        result[name] = {"path": str(path), "fps": info["fps"], "robot_type": info["robot_type"],
                        "codebase_version": info["codebase_version"],
                        "total_episodes": info["total_episodes"], "total_frames": info["total_frames"],
                        "total_tasks": info["total_tasks"],
                        "features": {key: {"dtype": spec["dtype"], "shape": spec["shape"]}
                                     for key, spec in info["features"].items()},
                        "state_names": (info["features"].get("observation.state") or {}).get("names"),
                        "action_names": (info["features"].get("action") or {}).get("names"),
                        "per_task": per_task}
    result["blockstacking_train"] = result["psi0"]["per_task"][str(BLOCKSTACKING_TASK_INDEX)]
    result["prompt"] = BLOCKSTACKING_PROMPT
    result["layout"] = verify_layouts(root)
    result["copy_equivalence"] = compare_copies(root, args.recon_episodes)
    write_json(out / "tables" / "dataset_recon.json", result)
    write_csv(out / "tables" / "corpus_tasks.csv",
              [{"source": name, "task_index": int(index), "collection": entry["collection"],
                "task": entry["task"], "episodes": entry["episodes"], "frames": entry["frames"],
                "fps": result[name]["fps"]}
               for name in ("psi0", "groot")
               for index, entry in sorted(result[name]["per_task"].items(), key=lambda item: int(item[0]))])
    return result


# --------------------------------------------------------------- dataset stage


def stage_dataset(args: argparse.Namespace) -> dict[str, Any]:
    """Palm and joint statistics of the demonstrations, for every task."""
    root = data_root(args.data_root)
    out = output_dir(args.output_dir)
    urdf = Urdf(args.urdf)
    psi_root = Path(root) / PSI0_TRAIN
    episodes_meta = [json.loads(line) for line in (psi_root / "meta" / "episodes.jsonl").read_text().splitlines() if line.strip()]
    by_task: dict[int, list[dict[str, Any]]] = {}
    for entry in episodes_meta:
        task_index = TASK_INDEX_COLLECTIONS.index(entry["source_collection"])
        by_task.setdefault(task_index, []).append(entry)

    rng = np.random.default_rng(args.seed)
    per_task: dict[str, Any] = {}
    blockstacking_series: dict[str, Any] = {}
    per_episode_rows: list[dict[str, Any]] = []
    for task_index in sorted(by_task):
        entries = sorted(by_task[task_index], key=lambda e: e["episode_index"])
        collection = TASK_INDEX_COLLECTIONS[task_index]
        sample = entries
        if args.sample_per_task and len(entries) > args.sample_per_task:
            picks = rng.choice(len(entries), size=args.sample_per_task, replace=False)
            sample = [entries[int(i)] for i in sorted(picks)]
        raw_task = (Path(root) / RAW_FIRST_TUR_HAM / collection)
        frame_pool = {"left": [], "right": []}
        episode_stats: list[dict[str, Any]] = []
        joint_columns: dict[str, list[np.ndarray]] = {}
        for entry in sample:
            episode = entry["episode_index"]
            # The raw 30 Hz source is the only place the task's own measured
            # state lives for a strided sample; psi0 carries the same values but
            # with the lower body replaced by the standing constant.
            columns = read_parquet(episode_path(psi_root, episode),
                                   ["observation.state", "action", "timestamp", "anchor_valid"])
            state = np.asarray(columns["observation.state"], dtype=float)
            timestamp = np.asarray(columns["timestamp"], dtype=float)
            body = np.zeros((state.shape[0], 29))
            body[:, 0:15] = state[:, 0:15]
            body[:, 15:22] = state[:, PSI0_LAYOUT["left_arm"]]
            body[:, 22:29] = state[:, PSI0_LAYOUT["right_arm"]]
            palms = urdf.palms(body)
            # The raw episodes are the reference for which frames are training
            # samples: psi0 marks the first ``frames_valid`` frames as usable.
            valid = int(entry.get("frames_valid", state.shape[0]))
            for side in ("left", "right"):
                z = palms[side][:valid, 2]
                frame_pool[side].append(z)
            hand_joints = state[:valid, 29:43]
            joint_columns.setdefault("left_hand", []).append(hand_joints[:, 0:7])
            joint_columns.setdefault("right_hand", []).append(hand_joints[:, 7:14])
            joint_columns.setdefault("left_arm", []).append(state[:valid, PSI0_LAYOUT["left_arm"]])
            joint_columns.setdefault("right_arm", []).append(state[:valid, PSI0_LAYOUT["right_arm"]])
            row: dict[str, Any] = {"task_index": task_index, "collection": collection,
                                   "episode_index": episode, "source_episode_index": entry["source_episode_index"],
                                   "frames": int(state.shape[0]), "frames_valid": valid,
                                   "duration_s": float(timestamp[-1] - timestamp[0])}
            for side in ("left", "right"):
                z = palms[side][:valid, 2]
                metrics = phase_metrics(timestamp[:valid], z)
                row[f"{side}_z_start_m"] = metrics.get("z_start_m")
                row[f"{side}_z_min_m"] = metrics.get("z_min_m")
                row[f"{side}_z_max_m"] = metrics.get("z_max_m")
                row[f"{side}_z_median_m"] = float(np.median(z))
                row[f"{side}_approach_amplitude_m"] = metrics.get("approach_amplitude_m")
                row[f"{side}_approach_rate_m_s"] = metrics.get("approach_rate_m_s")
                row[f"{side}_start_hold_s"] = metrics.get("start_hold_s")
                row[f"{side}_duration_s"] = metrics.get("duration_s")
            episode_stats.append(row)
            per_episode_rows.append(row)
            if task_index == BLOCKSTACKING_TASK_INDEX and args.keep_series:
                blockstacking_series[f"t_{episode}"] = timestamp[:valid].astype(np.float32)
                for side in ("left", "right"):
                    blockstacking_series[f"z_{side}_{episode}"] = palms[side][:valid, 2].astype(np.float32)
                    blockstacking_series[f"x_{side}_{episode}"] = palms[side][:valid, 0].astype(np.float32)
                    blockstacking_series[f"y_{side}_{episode}"] = palms[side][:valid, 1].astype(np.float32)
                blockstacking_series[f"body_{episode}"] = body[:valid].astype(np.float32)
        pool = {side: np.concatenate(frame_pool[side]) if frame_pool[side] else np.zeros(0) for side in frame_pool}
        entry_stats = [{"episode": row["episode_index"], "z_min_m": row[f"{side}_z_min_m"],
                        "z_max_m": row[f"{side}_z_max_m"], "z_median_m": row[f"{side}_z_median_m"]}
                       for row in episode_stats for side in ("left", "right")]
        episode_min = np.array([e["z_min_m"] for e in entry_stats])
        episode_median = np.array([e["z_median_m"] for e in entry_stats])
        per_task[str(task_index)] = {
            "collection": collection,
            "task": result_task_label(task_index),
            "episodes_in_train": len(entries), "episodes_sampled": len(sample),
            "sampled_episode_indices": [int(e["episode_index"]) for e in sample],
            "per_frame": {side: describe(pool[side]) for side in ("left", "right")},
            "per_episode_min": describe(episode_min),
            "per_episode_median": describe(episode_median),
            "per_episode_mean_min": float(np.mean(episode_min)),
            "joints": {name: {"median": [float(v) for v in np.median(np.concatenate(arrays), axis=0)],
                              "p05": [float(v) for v in np.percentile(np.concatenate(arrays), 5, axis=0)],
                              "p95": [float(v) for v in np.percentile(np.concatenate(arrays), 95, axis=0)]}
                       for name, arrays in joint_columns.items()},
        }
    write_json(out / "tables" / "dataset_task_palm_stats.json",
               {"schema_version": SCHEMA_VERSION, "per_task": per_task,
                "sampled_per_task": args.sample_per_task, "seed": args.seed})
    write_csv(out / "tables" / "dataset_episode_palm_stats.csv", per_episode_rows)
    if args.keep_series and blockstacking_series:
        np.savez_compressed(out / "tables" / "blockstacking_palm_series.npz", **blockstacking_series)
    per_task_csv = []
    for index, entry in sorted(per_task.items(), key=lambda item: int(item[0])):
        for side in ("left", "right"):
            per_task_csv.append({
                "task_index": int(index), "collection": entry["collection"], "hand": side,
                "episodes_in_train": entry["episodes_in_train"], "episodes_sampled": entry["episodes_sampled"],
                "frames_sampled": entry["per_frame"][side]["count"],
                "frame_median_m": entry["per_frame"][side]["median"],
                "frame_p95_m": entry["per_frame"][side]["p95"],
                "frame_p05_m": entry["per_frame"][side]["p05"],
                "episode_min_median_m": entry["per_episode_min"]["median"],
                "episode_median_median_m": entry["per_episode_median"]["median"],
            })
    write_csv(out / "tables" / "dataset_task_palm_summary.csv", per_task_csv)
    return {"per_task": per_task, "episodes": len(per_episode_rows)}


def result_task_label(task_index: int) -> str:
    return TASK_INDEX_COLLECTIONS[task_index]


# --------------------------------------------------------------- rollout stage


def tracking_series(rollout: Path) -> dict[str, Any]:
    rows = read_jsonl(rollout / "tracking.jsonl")
    return {
        "rows": rows,
        "sim_s": np.array([r["sim_s"] for r in rows], dtype=float),
        "wall_ns": np.array([r["wall_time_ns"] for r in rows], dtype=float),
        "root": np.array([r["root_position"] for r in rows], dtype=float),
        "root_quat": np.array([r["root_quaternion_wxyz"] for r in rows], dtype=float),
        "measured": np.array([r["body_measured"] for r in rows], dtype=float),
        "target": np.array([r["body_target"] for r in rows], dtype=float),
    }


def stage_rollout(args: argparse.Namespace) -> dict[str, Any]:
    out = output_dir(args.output_dir)
    urdf = Urdf(args.urdf)
    campaign = json.loads((campaign_dir() / "campaign.json").read_text())
    result: dict[str, Any] = {"schema_version": SCHEMA_VERSION, "sessions": []}
    series_store: dict[str, Any] = {}
    rows_out: list[dict[str, Any]] = []
    for session in campaign["sessions"]:
        tag = session["session_dir"]
        manifest = json.loads((campaign_dir() / tag / "manifest.json").read_text())
        surface = None
        entry: dict[str, Any] = {"session": tag, "model": session["model"], "rollouts": []}
        for rollout_meta in session["rollouts"]:
            rollout_dir = campaign_dir() / session["session_dir"] / "rollouts" / f"rollout-{rollout_meta['index']:02d}"
            rollout_json = json.loads((rollout_dir / "rollout.json").read_text())
            tracking = tracking_series(rollout_dir)
            if not tracking["rows"]:
                continue
            surface = rollout_json["stack"]["table_surface_m"]
            base = np.broadcast_to(np.eye(4), (len(tracking["rows"]), 4, 4)).copy()
            for index, quaternion in enumerate(tracking["root_quat"]):
                base[index, :3, :3] = quaternion_wxyz_to_matrix(quaternion)
            base[:, :3, 3] = tracking["root"]
            measured_world = urdf.palms(tracking["measured"], base)
            target_world = urdf.palms(tracking["target"], base)
            measured_base = urdf.palms(tracking["measured"])
            target_base = urdf.palms(tracking["target"])
            torso_chain = urdf.chain("torso_link")
            torso = urdf.fk_chain(torso_chain, urdf.chain_joint_values(torso_chain, tracking["measured"]), base)
            measured_torso = {side: np.einsum("nij,nj->ni", torso[:, :3, :3].transpose(0, 2, 1),
                                              measured_world[side] - torso[:, :3, 3])
                              for side in PALM_LINKS}
            onset_s = tracking["sim_s"][0] + rollout_json["onset"]["seconds_after_start"] if rollout_json.get("onset") else None
            onset_wall_ns = rollout_json["start_wall_ns"] + int(rollout_meta["onset_seconds_after_start"] * 1e9)
            active = tracking["wall_ns"] >= onset_wall_ns
            entry_row: dict[str, Any] = {
                "session": tag, "model": session["model"], "rollout": rollout_meta["index"],
                "onset_seconds_after_start": rollout_meta["onset_seconds_after_start"],
                "onset_source": rollout_meta["onset_source"],
                "duration_s": rollout_json["duration_wall_s"],
                "sim_seconds_recorded": rollout_json["sim_seconds_recorded"],
                "table_surface_m": surface,
                "applied_actions": rollout_json["applied_actions"],
                "observations": rollout_json["observations"],
                "cubes_lifted": rollout_json["stack"]["any_cube_lifted"],
                "max_cube_lift_m": max(rollout_json["stack"]["cube_lift_above_surface_m"].values()),
                "tracking_rows": len(tracking["rows"]),
                "active_rows": int(active.sum()),
            }
            for side in PALM_LINKS:
                world = measured_world[side][:, 2]
                base_z = measured_base[side][:, 2]
                torso_z = measured_torso[side][:, 2]
                table_z = world - surface
                entry_row[f"{side}_pelvis_min_m"] = float(base_z.min())
                entry_row[f"{side}_pelvis_median_m"] = float(np.median(base_z))
                entry_row[f"{side}_pelvis_p95_m"] = float(np.percentile(base_z, 95))
                entry_row[f"{side}_pelvis_start_m"] = float(np.median(base_z[: max(1, int(2.0 / 0.02))]))
                entry_row[f"{side}_table_min_m"] = float(table_z.min())
                entry_row[f"{side}_table_median_m"] = float(np.median(table_z))
                entry_row[f"{side}_table_p95_m"] = float(np.percentile(table_z, 95))
                entry_row[f"{side}_table_max_m"] = float(table_z.max())
                entry_row[f"{side}_torso_median_m"] = float(np.median(torso_z))
                entry_row.update({f"{side}_{key}": value for key, value in band_fractions(table_z, 0.0).items()})
                if rollout_json["behaviour"]["palm_height_above_table_m"].get(PALM_LINKS[side]):
                    stage1 = rollout_json["behaviour"]["palm_height_above_table_m"][PALM_LINKS[side]]
                    entry_row[f"{side}_stage1_median_above_table_m"] = stage1["median_above_table_m"]
                    entry_row[f"{side}_stage1_fraction_above_0_10_m"] = stage1["fraction_above_0_10_m"]
                # Active-window only (the policy's own behaviour, not the hold).
                for key, value in band_fractions(table_z[active], 0.0).items():
                    entry_row[f"{side}_active_{key}"] = value
                entry_row[f"{side}_active_table_median_m"] = float(np.median(table_z[active])) if active.any() else float("nan")
                entry_row[f"{side}_active_pelvis_median_m"] = float(np.median(base_z[active])) if active.any() else float("nan")
                entry_row[f"{side}_active_pelvis_min_m"] = float(base_z[active].min()) if active.any() else float("nan")
                # Commanded vs realized: does the policy ask to go down?
                entry_row[f"{side}_cmd_pelvis_min_m"] = float(target_base[side][:, 2].min())
                entry_row[f"{side}_cmd_pelvis_median_m"] = float(np.median(target_base[side][:, 2]))
                entry_row[f"{side}_cmd_active_pelvis_median_m"] = (
                    float(np.median(target_base[side][active, 2])) if active.any() else float("nan"))
                entry_row[f"{side}_cmd_active_table_median_m"] = (
                    float(np.median(target_world[side][active, 2] - surface)) if active.any() else float("nan"))
                entry_row[f"{side}_cmd_minus_measured_pelvis_median_m"] = float(
                    np.median(target_base[side][:, 2] - base_z))
                metrics = phase_metrics(tracking["sim_s"][active], base_z[active]) if active.sum() > 5 else {}
                for key in ("z_min_m", "z_max_m", "approach_amplitude_m", "approach_rate_m_s",
                            "start_hold_s", "v_down_min_m_s"):
                    entry_row[f"{side}_active_{key}"] = metrics.get(key)
            rows_out.append(entry_row)
            entry["rollouts"].append(entry_row)
            if args.keep_series:
                series_store[f"t_{tag}_{rollout_meta['index']}"] = tracking["sim_s"].astype(np.float32)
                series_store[f"active_{tag}_{rollout_meta['index']}"] = active
                for side in PALM_LINKS:
                    series_store[f"z_{side}_{tag}_{rollout_meta['index']}"] = measured_base[side][:, 2].astype(np.float32)
                    series_store[f"ztab_{side}_{tag}_{rollout_meta['index']}"] = (
                        measured_world[side][:, 2] - surface).astype(np.float32)
                    series_store[f"cmd_{side}_{tag}_{rollout_meta['index']}"] = target_base[side][:, 2].astype(np.float32)
                    series_store[f"world_{side}_{tag}_{rollout_meta['index']}"] = measured_world[side].astype(np.float32)
                series_store[f"root_z_{tag}_{rollout_meta['index']}"] = tracking["root"][:, 2].astype(np.float32)
                series_store[f"measured_{tag}_{rollout_meta['index']}"] = tracking["measured"].astype(np.float32)
                series_store[f"target_{tag}_{rollout_meta['index']}"] = tracking["target"].astype(np.float32)
        entry["table_surface_m"] = surface
        result["sessions"].append(entry)
    write_json(out / "tables" / "rollout_palm_stats.json", result)
    write_csv(out / "tables" / "rollout_palm_stats.csv", rows_out)
    if args.keep_series and series_store:
        np.savez_compressed(out / "tables" / "rollout_palm_series.npz", **series_store)
    return result


# ------------------------------------------------------------------- comparison


def stage_compare(args: argparse.Namespace) -> dict[str, Any]:
    out = output_dir(args.output_dir)
    dataset = json.loads((out / "tables" / "dataset_task_palm_stats.json").read_text())
    blockstacking = dataset["per_task"][str(BLOCKSTACKING_TASK_INDEX)]
    rollout = json.loads((out / "tables" / "rollout_palm_stats.json").read_text())
    series_path = out / "tables" / "blockstacking_palm_series.npz"
    result: dict[str, Any] = {"schema_version": SCHEMA_VERSION}

    # The dataset worktop is not recorded anywhere.  A demonstration can only
    # reach its palm minimum by reaching to a block, and a block is a 5 cm cube
    # resting on the worktop, so the corpus median of the per-episode minima is
    # an upper bound on "worktop + cube".  Subtracting the block edge gives a
    # worktop proxy that is at or below the true surface, i.e. every dataset
    # height above the table reported here is a *lower* bound.
    table_proxy = float(blockstacking["per_episode_min"]["median"]) - TABLE_PROXY_CUBE_M
    result["table_proxy"] = {
        "pelvis_frame_z_m": table_proxy,
        "method": "median over BlockStacking episodes of the per-episode palm-z minimum, minus the 5 cm block edge",
        "per_episode_min_median_m": blockstacking["per_episode_min"]["median"],
        "per_episode_min_p05_m": blockstacking["per_episode_min"]["p05"],
        "per_episode_min_p95_m": blockstacking["per_episode_min"]["p95"],
        "band_m": TABLE_PROXY_BAND_M,
        "direction": "proxy <= true worktop, so dataset heights above it are lower bounds",
    }

    # Pooled dataset distributions: frame-pooled (what a training batch looks
    # like) and episode-pooled (each demonstration weighted equally).
    pool: dict[str, np.ndarray] = {}
    if series_path.is_file():
        with np.load(series_path) as payload:
            for side in ("left", "right"):
                keys = [key for key in payload.files if key.startswith(f"z_{side}_")]
                pool[side] = np.concatenate([payload[key] for key in keys])
    result["dataset_frame_pool"] = {side: describe(values) for side, values in pool.items()}
    result["dataset_frame_pool_above_table"] = {
        side: {"table_minus_proxy_m": None, **_describe_above(values, table_proxy)}
        for side, values in pool.items()}

    per_task = {}
    for index, entry in dataset["per_task"].items():
        for side in ("left", "right"):
            per_task.setdefault(side, []).append({
                "task_index": int(index), "collection": entry["collection"],
                "episodes_in_train": entry["episodes_in_train"],
                "episodes_sampled": entry["episodes_sampled"],
                "frames_sampled": entry["per_frame"][side]["count"],
                "frame_median_m": entry["per_frame"][side]["median"],
                "frame_p05_m": entry["per_frame"][side]["p05"],
                "frame_p95_m": entry["per_frame"][side]["p95"],
                "episode_min_median_m": entry["per_episode_min"]["median"],
                "episode_median_median_m": entry["per_episode_median"]["median"],
                "above_table": _describe_above_scalar(entry, side, table_proxy),
            })
    result["per_task"] = per_task

    # Where do the rollouts sit inside the BlockStacking distribution?
    comparisons = []
    for session in rollout["sessions"]:
        for row in session["rollouts"]:
            for side in PALM_LINKS:
                side_key = "left" if side == "left" else "right"
                observations = {
                    "pelvis_median": row.get(f"{side}_pelvis_median_m"),
                    "pelvis_min": row.get(f"{side}_pelvis_min_m"),
                    "active_pelvis_median": row.get(f"{side}_active_pelvis_median_m"),
                    "table_median": row.get(f"{side}_table_median_m"),
                    "table_min": row.get(f"{side}_table_min_m"),
                }
                samples = {
                    "pelvis_median": pool.get(side_key, np.zeros(0)),
                    "pelvis_min": np.zeros(0),
                    "active_pelvis_median": pool.get(side_key, np.zeros(0)),
                    "table_median": pool.get(side_key, np.zeros(0)) - table_proxy,
                    "table_min": np.zeros(0),
                }
                entry = {"session": row["session"], "model": row["model"], "rollout": row["rollout"],
                         "hand": side, "values": {}, "percentile_in_dataset": {}, "robust_z": {}}
                for name, value in observations.items():
                    sample = samples[name]
                    entry["values"][name] = value
                    entry["percentile_in_dataset"][name] = empirical_percentile(sample, value) if sample.size else None
                    entry["robust_z"][name] = robust_z(value, sample) if sample.size else None
                entry["table_fraction_comparison"] = {
                    "rollout_above_10cm": row.get(f"{side}_frac_above_10cm"),
                    "rollout_above_15cm": row.get(f"{side}_frac_above_15cm"),
                    "rollout_above_20cm": row.get(f"{side}_frac_above_20cm"),
                }
                comparisons.append(entry)
    result["rollout_vs_dataset"] = comparisons

    # Dataset-side fractions above the table (lower bounds, by construction).
    dataset_fractions = {}
    for side, values in pool.items():
        dataset_fractions[side] = {**band_fractions(values, table_proxy),
                                   "median_above_table_m": float(np.median(values - table_proxy)),
                                   "min_above_table_m": float(values.min() - table_proxy),
                                   "p95_above_table_m": float(np.percentile(values, 95) - table_proxy)}
    result["dataset_above_table"] = dataset_fractions
    write_json(out / "tables" / "comparison.json", result)
    write_csv(out / "tables" / "comparison_rollout_percentiles.csv",
              [{"session": e["session"], "model": e["model"], "rollout": e["rollout"], "hand": e["hand"],
                **{f"{key}": e["values"][key] for key in e["values"]},
                **{f"pct_{key}": e["percentile_in_dataset"][key] for key in e["percentile_in_dataset"]},
                **{f"z_{key}": e["robust_z"][key] for key in e["robust_z"]},
                } for e in comparisons])
    write_csv(out / "tables" / "comparison_per_task.csv",
              [{**row} for rows in per_task.values() for row in rows])
    return result


def _describe_above(values: np.ndarray, table_proxy: float) -> dict[str, Any]:
    return {"median_above_table_m": float(np.median(values) - table_proxy),
            "p95_above_table_m": float(np.percentile(values, 95) - table_proxy),
            "min_above_table_m": float(values.min() - table_proxy),
            **band_fractions(values, table_proxy)}


def _describe_above_scalar(entry: dict[str, Any], side: str, table_proxy: float) -> dict[str, Any]:
    return {"episode_min_median_above_table_m": entry["per_episode_min"]["median"] - table_proxy,
            "frame_median_above_table_m": entry["per_frame"][side]["median"] - table_proxy}


# ------------------------------------------------------------------ diagnosis


def stage_diagnose(args: argparse.Namespace) -> dict[str, Any]:
    """Where the rollouts sit in the demonstration distribution, and why.

    Two questions are answered separately, because they have different causes:

    * *command vs execution* -- the SONIC controller's decoded joint target
      (what the policy asked for, after the token decoder) against the joint
      value the simulator realized.  ``body_target`` and ``body_measured`` are
      both in the articulation's own order, so the difference is a controller
      property, not a policy property.
    * *grasp geometry* -- the distance from each palm to each cube over time and
      the palm height above the cube top at the closest approach.  A policy can
      hold its hands high and still be misaligned, so height alone is not the
      failure description.
    """
    out = output_dir(args.output_dir)
    urdf = Urdf(args.urdf)
    campaign = json.loads((campaign_dir() / "campaign.json").read_text())
    result: dict[str, Any] = {"schema_version": SCHEMA_VERSION, "sessions": [], "joint_reference": {}}

    # Demonstration arm-joint envelope, for the "is the command in-distribution" test.
    dataset = json.loads((out / "tables" / "dataset_task_palm_stats.json").read_text())["per_task"]
    blockstacking = dataset[str(BLOCKSTACKING_TASK_INDEX)]
    result["joint_reference"]["blockstacking_arm_joints"] = blockstacking["joints"]

    for session in campaign["sessions"]:
        tag = session["session_dir"]
        metrics = json.loads((campaign_dir() / tag / "raw" / "isaac.metrics.json").read_text())
        body_names = metrics["tracking_columns"]["body_joints"]
        session_entry: dict[str, Any] = {"session": tag, "model": session["model"], "rollouts": []}
        for rollout_meta in session["rollouts"]:
            rollout_dir = campaign_dir() / tag / "rollouts" / f"rollout-{rollout_meta['index']:02d}"
            rollout_json = json.loads((rollout_dir / "rollout.json").read_text())
            tracking = tracking_series(rollout_dir)
            if not tracking["rows"]:
                continue
            surface = rollout_json["stack"]["table_surface_m"]
            cube_size = float(np.mean([
                value for cube in campaign_cubes(tag)
                for value in cube[1]])) if campaign_cubes(tag) else 0.05
            cubes = {name: np.asarray(spec["center_xyz_m"], dtype=float)
                     for name, spec in rollout_json["stack"]["cube_pose_w"].items()}
            base = np.broadcast_to(np.eye(4), (len(tracking["rows"]), 4, 4)).copy()
            for index, quaternion in enumerate(tracking["root_quat"]):
                base[index, :3, :3] = quaternion_wxyz_to_matrix(quaternion)
            base[:, :3, 3] = tracking["root"]
            measured_world = urdf.palms(tracking["measured"], base)
            target_world = urdf.palms(tracking["target"], base)
            active = tracking["wall_ns"] >= rollout_json["start_wall_ns"] + int(
                rollout_meta["onset_seconds_after_start"] * 1e9)

            entry: dict[str, Any] = {"rollout": rollout_meta["index"],
                                     "table_surface_m": surface, "cube_size_m": cube_size,
                                     "onset_seconds_after_start": rollout_meta["onset_seconds_after_start"],
                                     "joint_tracking": {}, "grasp_geometry": {}}
            for name in body_names:
                index = body_names.index(name)
                command = tracking["target"][active, index]
                realized = tracking["measured"][active, index]
                if not command.size:
                    continue
                entry["joint_tracking"][name] = {
                    "command_mean_rad": float(command.mean()),
                    "measured_mean_rad": float(realized.mean()),
                    "mean_signed_gap_rad": float((command - realized).mean()),
                    "mean_abs_gap_rad": float(np.abs(command - realized).mean()),
                    "max_abs_gap_rad": float(np.abs(command - realized).max()),
                }
            arm_names = tuple(n for n in body_names if "shoulder" in n or "elbow" in n or "wrist" in n)
            entry["joint_tracking_summary"] = {
                "arm_mean_abs_gap_rad": float(np.mean([entry["joint_tracking"][n]["mean_abs_gap_rad"] for n in arm_names])),
                "arm_max_abs_gap_rad": float(np.max([entry["joint_tracking"][n]["max_abs_gap_rad"] for n in arm_names])),
                "body_mean_abs_gap_rad": float(np.mean([v["mean_abs_gap_rad"] for v in entry["joint_tracking"].values()])),
                "arm_kp": float(np.mean([row["body_kp"][body_names.index(n)] for row in tracking["rows"] for n in arm_names[:1]])),
                "arm_kd": float(np.mean([row["body_kd"][body_names.index(n)] for row in tracking["rows"] for n in arm_names[:1]])),
                "arm_torque_absmax_nm": float(max(
                    abs(value) for row in tracking["rows"] for index, value in enumerate(row["body_applied_torque"])
                    if body_names[index] in arm_names)),
            }
            for side, link in PALM_LINKS.items():
                world = measured_world[side]
                commanded = target_world[side]
                entry["grasp_geometry"][side] = {
                    "measured_minus_commanded_z_median_m": float(np.median(commanded[:, 2] - world[:, 2])),
                    "cubes": {},
                }
                for cube_name, centre in cubes.items():
                    distance = np.linalg.norm(world - centre[None, :], axis=1)
                    horizontal = np.linalg.norm((world - centre[None, :])[:, :2], axis=1)
                    closest = int(np.argmin(distance))
                    entry["grasp_geometry"][side]["cubes"][cube_name] = {
                        "min_distance_m": float(distance.min()),
                        "min_horizontal_distance_m": float(horizontal.min()),
                        "z_above_cube_centre_at_closest_m": float(world[closest, 2] - centre[2]),
                        "z_above_table_at_closest_m": float(world[closest, 2] - surface),
                        "seconds_within_10cm": float((distance < 0.10).sum() * 0.02),
                        "seconds_within_10cm_horizontal": float((horizontal < 0.10).sum() * 0.02),
                        "active_min_distance_m": float(distance[active].min()) if active.any() else None,
                    }
            # Realized palm heights, for the in-distribution test against the demos.
            for side in PALM_LINKS:
                values = urdf.palms(tracking["measured"])[side][:, 2]
                entry.setdefault("realized_pelvis_z", {})[side] = describe(values[active] if active.any() else values)
            session_entry["rollouts"].append(entry)
        result["sessions"].append(session_entry)
    write_json(out / "tables" / "diagnosis.json", result)
    return result


def campaign_cubes(tag: str) -> list[tuple[str, tuple[float, float, float]]]:
    profile = json.loads((REPO_ROOT / "configs" / "profiles" / "isaac-g1-sonic-blockstacking-dex3.json").read_text())
    return [(cube["color"], tuple(cube["size_m"])) for cube in profile["scene"]["cubes"]]


# --------------------------------------------------------------------- figures


def stage_figures(args: argparse.Namespace) -> dict[str, Any]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = output_dir(args.output_dir)
    figures = out / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    produced = []
    series_path = out / "tables" / "blockstacking_palm_series.npz"
    rollout_path = out / "tables" / "rollout_palm_series.npz"
    comparison = json.loads((out / "tables" / "comparison.json").read_text())

    if series_path.is_file() and rollout_path.is_file():
        with np.load(series_path) as dataset, np.load(rollout_path) as rollout:
            # 1. BlockStacking palm height, pelvis frame: all demos + rollouts.
            figure, axes = plt.subplots(1, 2, figsize=(15, 5.2), sharey=True)
            for side, axis in zip(("left", "right"), axes):
                keys = sorted(key for key in dataset.files if key.startswith(f"z_{side}_"))
                for key in keys:
                    z = dataset[key]
                    t = dataset[key.replace(f"z_{side}_", "t_")]
                    axis.plot(t, z, color="#1f77b4", alpha=0.06, linewidth=0.7)
                for tag in ROLLOUT_TAGS:
                    rkeys = [key for key in rollout.files if key.startswith(f"z_{side}_{tag}_")]
                    for key in rkeys:
                        z = rollout[key]
                        t = rollout[key.replace(f"z_{side}_", "t_")]
                        index = key.rsplit("_", 1)[-1]
                        t0 = t[0]
                        axis.plot(t - t0, z, color="#d62728" if "groot" in tag else "#2ca02c",
                                  alpha=0.9, linewidth=1.6)
                        active = rollout[f"active_{tag}_{index}"]
                        axis.plot(t[active] - t0, z[active], color="black", alpha=0.9, linewidth=0.8)
                axis.axhline(comparison["table_proxy"]["pelvis_frame_z_m"], color="black",
                             linestyle="--", linewidth=1.0)
                axis.set_title(f"{side} palm, pelvis frame\nblue: 286 demos   red: GR00T   green: psi0")
                axis.set_xlabel("time from clip start (s)")
                axis.grid(alpha=0.25)
            axes[0].set_ylabel("palm z in pelvis frame (m)")
            figure.tight_layout()
            path = figures / "palm_height_pelvis_frame.png"
            figure.savefig(path, dpi=130)
            plt.close(figure)
            produced.append(path.name)

            # 2. Distribution comparison.
            figure, axes = plt.subplots(1, 2, figsize=(14, 4.6))
            for side, axis in zip(("left", "right"), axes):
                keys = sorted(key for key in dataset.files if key.startswith(f"z_{side}_"))
                demo = np.concatenate([dataset[key] for key in keys])
                axis.hist(demo, bins=120, density=True, color="#1f77b4", alpha=0.55,
                          label=f"BlockStacking demos (n={demo.size})")
                for tag, colour in zip(ROLLOUT_TAGS, ("#d62728", "#2ca02c")):
                    rkeys = [key for key in rollout.files if key.startswith(f"z_{side}_{tag}_")]
                    values = np.concatenate([rollout[key] for key in rkeys])
                    axis.axvline(np.median(values), color=colour, linewidth=2.0,
                                 label=f"{tag} median {np.median(values):.3f}")
                    axis.hist(values, bins=80, density=True, color=colour, alpha=0.35)
                axis.set_title(f"{side} palm z, pelvis frame")
                axis.set_xlabel("z (m)")
                axis.grid(alpha=0.25)
                axis.legend(fontsize=8)
            figure.tight_layout()
            path = figures / "palm_height_histograms.png"
            figure.savefig(path, dpi=130)
            plt.close(figure)
            produced.append(path.name)

            # 3. Commanded vs measured, per rollout.
            figure, axes = plt.subplots(2, 2, figsize=(15, 7), sharex="col")
            for column, tag in enumerate(ROLLOUT_TAGS):
                keys = [key for key in rollout.files if key.startswith(f"t_{tag}_")]
                for row, side in enumerate(("left", "right")):
                    axis = axes[row][column]
                    for key in sorted(keys):
                        index = key.rsplit("_", 1)[-1]
                        t = rollout[key]
                        axis.plot(t - t[0], rollout[f"cmd_{side}_{tag}_{index}"], color="#ff7f0e",
                                  linewidth=1.1, label="commanded (SONIC target)" if index == "01" and row == 0 else None)
                        axis.plot(t - t[0], rollout[f"z_{side}_{tag}_{index}"], color="#1f77b4",
                                  linewidth=1.1, label="measured" if index == "01" and row == 0 else None)
                        axis.plot(t - t[0], rollout[f"ztab_{side}_{tag}_{index}"], color="#7f7f7f",
                                  linewidth=0.8, alpha=0.8, label="measured above table" if index == "01" and row == 0 else None)
                    axis.set_title(f"{tag} {side} palm")
                    axis.grid(alpha=0.25)
                    if row == 0 and column == 0:
                        axis.legend(fontsize=8)
            figure.tight_layout()
            path = figures / "rollout_commanded_vs_measured.png"
            figure.savefig(path, dpi=130)
            plt.close(figure)
            produced.append(path.name)

    # 4. Corpus-wide task comparison.
    per_task_path = out / "tables" / "dataset_task_palm_stats.json"
    if per_task_path.is_file():
        payload = json.loads(per_task_path.read_text())["per_task"]
        order = sorted(payload.items(), key=lambda item: int(item[0]))
        labels = [f"{index}: {entry['collection'].replace('G1_Dex3_', '').replace('_Dataset', '')}"
                  for index, entry in order]
        figure, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
        for side, axis in zip(("left", "right"), axes):
            medians = [entry["per_frame"][side]["median"] for _, entry in order]
            low = [entry["per_frame"][side]["p05"] for _, entry in order]
            high = [entry["per_frame"][side]["p95"] for _, entry in order]
            positions = np.arange(len(order))
            axis.errorbar(medians, positions, xerr=[np.array(medians) - np.array(low), np.array(high) - np.array(medians)],
                          fmt="o", color="#1f77b4", ecolor="#1f77b4", capsize=3)
            axis.set_yticks(positions)
            axis.set_yticklabels(labels, fontsize=8)
            axis.set_title(f"{side} palm z (median, p05-p95) per task")
            axis.set_xlabel("palm z in pelvis frame (m)")
            axis.grid(alpha=0.25)
        figure.tight_layout()
        path = figures / "corpus_task_palm_heights.png"
        figure.savefig(path, dpi=130)
        plt.close(figure)
        produced.append(path.name)

    result = {"figures": produced}
    write_json(out / "tables" / "figures.json", result)
    return result


def stage_contact_sheets(args: argparse.Namespace) -> dict[str, Any]:
    """Frame strips from the demonstration videos, chosen by measured height."""
    out = output_dir(args.output_dir)
    videos_dir = out / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)
    root = data_root(args.data_root)
    produced: list[dict[str, Any]] = []
    series_path = out / "tables" / "blockstacking_palm_series.npz"
    if not series_path.is_file():
        return {"sheets": []}
    with np.load(series_path) as dataset:
        episodes = sorted({int(key.split("_")[-1]) for key in dataset.files if key.startswith("z_left_")})
        summaries = []
        for episode in episodes:
            z = dataset[f"z_left_{episode}"]
            summaries.append((float(np.median(z)), float(z.min()), float(z.max()) - float(z.min()), episode))
        summaries.sort()
        picks = {
            "high_median": summaries[-1],
            "median_median": summaries[len(summaries) // 2],
            "low_median": summaries[0],
            "widest_range": max(summaries, key=lambda item: item[2]),
        }
        choices = {}
        for label, (median, minimum, span, episode) in picks.items():
            t = dataset[f"t_{episode}"]
            fractions = np.linspace(0.02, 0.98, 6)
            times = [float(t[int(fraction * (len(t) - 1))]) for fraction in fractions]
            video = (Path(root) / PSI0_TRAIN / "videos" / f"chunk-{episode // 1000:03d}"
                     / "observation.images.egocentric" / f"episode_{episode:06d}.mp4")
            if not video.is_file():
                continue
            target = videos_dir / f"dataset_ep{episode:06d}_{label}.jpg"
            select = "+".join(f"eq(n\\,{int(round(time * 30))})" for time in times)
            command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(video),
                       "-vf", f"select='{select}',scale=320:-1,tile=3x2", "-frames:v", "1", str(target)]
            completed = subprocess.run(command, capture_output=True, text=True, check=False)
            choices[label] = {"episode": episode, "median_m": median, "min_m": minimum,
                              "range_m": span, "frame_times_s": times, "sheet": target.name,
                              "ffmpeg_rc": completed.returncode, "video": str(video),
                              "stderr": completed.stderr.strip()[:200]}
            produced.append(choices[label])
        write_json(out / "tables" / "dataset_contact_sheets.json", {"sheets": produced, "picks": choices})

    # The policies' own input: the frames the bridge handed to each model.
    headcam: list[dict[str, Any]] = []
    for session in json.loads((campaign_dir() / "campaign.json").read_text())["sessions"]:
        tag = session["session_dir"]
        for rollout_meta in session["rollouts"]:
            rollout_dir = campaign_dir() / tag / "rollouts" / f"rollout-{rollout_meta['index']:02d}"
            frames = []
            for row in read_jsonl(rollout_dir / "bridge-telemetry.jsonl"):
                if row.get("kind") == "observation" and row.get("camera_jpeg"):
                    path = Path(str(row["camera_jpeg"]).replace("/outputs/", str(campaign_dir().parent) + "/"))
                    if path.is_file():
                        frames.append(path)
            if len(frames) < 6:
                continue
            picks = [frames[int(round(index * (len(frames) - 1) / 5.0))] for index in range(6)]
            target = videos_dir / f"headcam_{tag}_r{rollout_meta['index']}.jpg"
            scratch = videos_dir / "scratch"
            if scratch.exists():
                shutil.rmtree(scratch)
            scratch.mkdir(parents=True, exist_ok=True)
            for index, path in enumerate(picks):
                shutil.copy(path, scratch / f"{index:02d}.jpg")
            completed = subprocess.run(
                ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(scratch / "%02d.jpg"),
                 "-vf", "scale=320:-1,tile=3x2", "-frames:v", "1", str(target)],
                capture_output=True, text=True, check=False)
            shutil.rmtree(scratch, ignore_errors=True)
            headcam.append({"session": tag, "rollout": rollout_meta["index"], "sheet": target.name,
                            "frames_available": len(frames),
                            "picked": [frame.name for frame in picks],
                            "ffmpeg_rc": completed.returncode})
    write_json(out / "tables" / "rollout_head_camera_sheets.json", {"sheets": headcam})
    return {"sheets": produced, "head_camera_sheets": headcam}


# ---------------------------------------------------------------------- visual


def stage_visual(args: argparse.Namespace) -> dict[str, Any]:
    """Colour/blob measurements that turn "the video looks different" into numbers.

    Only two questions are asked of the pixels, both cheap and robust:

    * *camera scale* -- how many pixels wide is the red block in the demonstrations
      versus in the simulator's head camera (the only block whose colour class has no
      large object behind it in the simulated scene).  Apparent size scales as 1/depth,
      so this compares the two cameras' working distance without any calibration.
    * *object identity* -- the hue of the third block in the demonstrations, which the
      instruction calls "blue".
    """
    result: dict[str, Any] = {"schema_version": SCHEMA_VERSION}
    try:
        import cv2
    except ImportError:
        result["available"] = False
        result["reason"] = "cv2 is not installed in this interpreter"
        write_json(output_dir(args.output_dir) / "tables" / "camera_scale.json", result)
        return result
    root = data_root(args.data_root)
    result["available"] = True
    result["blocks"] = {}

    def colour_blobs(image: np.ndarray) -> dict[str, Any]:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hue, saturation, value = cv2.split(hsv)
        classes = {
            "red": (((hue < 10) | (hue > 170)) & (saturation > 120) & (value > 70)),
            "yellow": ((hue >= 15) & (hue < 35) & (saturation > 110) & (value > 90)),
            "green": ((hue >= 45) & (hue < 95) & (saturation > 80) & (value > 50)),
        }
        out = {}
        for name, mask in classes.items():
            mask = mask.astype(np.uint8)
            count, _labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
            areas = [(int(stats[index][4]), tuple(np.round(centroids[index]).astype(int)))
                     for index in range(1, count) if stats[index][4] >= 60]
            if areas:
                areas.sort(reverse=True)
                out[name] = {"area_px": areas[0][0], "centroid_xy": areas[0][1]}
        return out

    def video_frame(path: Path, seconds: float) -> np.ndarray | None:
        completed = subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-ss", f"{seconds:.3f}", "-i", str(path),
             "-frames:v", "1", "-f", "image2pipe", "-pix_fmt", "bgr24", "-vcodec", "rawvideo", "-"],
            capture_output=True, check=False)
        if len(completed.stdout) < 480 * 640 * 3:
            return None
        return np.frombuffer(completed.stdout[: 480 * 640 * 3], dtype=np.uint8).reshape(480, 640, 3)

    dataset_areas: dict[str, list[int]] = {}
    demo_hues: list[dict[str, Any]] = []
    for episode in args.visual_episodes:
        video = (Path(root) / PSI0_TRAIN / "videos" / f"chunk-{episode // 1000:03d}"
                 / "observation.images.egocentric" / f"episode_{episode:06d}.mp4")
        if not video.is_file():
            continue
        for fraction in (0.05, 0.3, 0.6):
            image = video_frame(video, fraction * args.visual_video_seconds)
            if image is None:
                continue
            for name, payload in colour_blobs(image).items():
                dataset_areas.setdefault(name, []).append(payload["area_px"])
            blobs = colour_blobs(image)
            if episode == args.visual_episodes[0] and fraction == 0.05:
                for name, payload in blobs.items():
                    y, x = payload["centroid_xy"][1], payload["centroid_xy"][0]
                    patch = image[max(0, y - 3): y + 4, max(0, x - 3): x + 4].reshape(-1, 3).mean(axis=0)
                    demo_hues.append({"class": name, "area_px": payload["area_px"],
                                      "rgb": [int(round(float(v))) for v in patch[::-1]]})
    sim_areas: dict[str, list[int]] = {}
    for tag in ROLLOUT_TAGS:
        for path in sorted((campaign_dir() / tag / "raw" / "telemetry" / "head_camera").glob("*.jpg"))[::8]:
            image = cv2.imread(str(path))
            if image is None:
                continue
            for name, payload in colour_blobs(image).items():
                sim_areas.setdefault(name, []).append(payload["area_px"])
    for label, areas in (("dataset_egocentric", dataset_areas), ("sim_head_camera", sim_areas)):
        result["blocks"][label] = {
            name: {"frames_with_blob": len(values), "median_area_px": float(np.median(values)),
                   "median_apparent_width_px": float(np.sqrt(np.median(values)))}
            for name, values in sorted(areas.items()) if values}
    result["demonstration_third_block_colour"] = demo_hues
    result["note"] = ("yellow/blue classes also match the storage crates under the simulated table; "
                      "only the red class is used for the cross-domain scale comparison")
    write_json(output_dir(args.output_dir) / "tables" / "camera_scale.json", result)
    return result


# --------------------------------------------------------------------- manifest


def stage_manifest(args: argparse.Namespace) -> dict[str, Any]:
    out = output_dir(args.output_dir)
    root = data_root(args.data_root)
    campaign = json.loads((campaign_dir() / "campaign.json").read_text())
    inputs: dict[str, Any] = {
        "campaign.json": {"path": str(campaign_dir() / "campaign.json"),
                          "sha256": sha256(campaign_dir() / "campaign.json")},
        "urdf": {"path": str(args.urdf), "sha256": sha256(args.urdf)},
    }
    for tag in ROLLOUT_TAGS:
        for name in ("manifest.json", "verification.json"):
            path = campaign_dir() / tag / name
            inputs[f"{tag}/{name}"] = {"path": str(path), "sha256": sha256(path)}
        for rollout in sorted((campaign_dir() / tag / "rollouts").iterdir()):
            if rollout.is_dir():
                for name in ("rollout.json", "tracking.jsonl", "isaac.samples.jsonl"):
                    path = rollout / name
                    if path.is_file():
                        inputs[f"{tag}/{rollout.name}/{name}"] = {"path": str(path), "sha256": sha256(path)}
    for relative in (PSI0_TRAIN, GROOT_TRAIN):
        for name in ("meta/info.json", "meta/tasks.jsonl", "meta/episodes.jsonl", "meta/modality.json"):
            path = Path(root) / relative / name
            if path.is_file():
                inputs[f"{relative}/{name}"] = {"path": str(path), "sha256": sha256(path)}
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "script": SCRIPT_VERSION,
        "script_path": str(Path(__file__).resolve()),
        "script_sha256": sha256(Path(__file__).resolve()),
        "commands": args.commands,
        "constants": {
            "table_proxy_cube_m": TABLE_PROXY_CUBE_M,
            "table_proxy_band_m": TABLE_PROXY_BAND_M,
            "smooth_window_s": SMOOTH_WINDOW_S,
            "onset_delta_m": ONSET_DELTA_M,
            "onset_sustain_s": ONSET_SUSTAIN_S,
            "start_window_s": START_WINDOW_S,
            "height_bands_m": list(HEIGHT_BANDS_M),
            "psi0_state_layout": {k: [v.start, v.stop] for k, v in PSI0_LAYOUT.items()},
            "groot_state_layout": {k: [v.start, v.stop] for k, v in GROOT_LAYOUT.items()},
            "seed": args.seed,
            "sample_per_task": args.sample_per_task,
        },
        "campaign": {"task": campaign["task"], "prompt": campaign["prompt"],
                     "sessions": [s["session_dir"] for s in campaign["sessions"]]},
        "inputs": inputs,
        "outputs": sorted(str(path.relative_to(out)) for path in out.rglob("*") if path.is_file()),
    }
    write_json(out / "manifest.json", manifest)
    return manifest


# ------------------------------------------------------------------------- main


VALID_STAGES = ("recon", "fk-check", "dataset", "rollout", "compare", "diagnose",
                "visual", "figures", "sheets", "manifest", "all")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stages", nargs="*", default=["all"],
                        help=f"one or more of {', '.join(VALID_STAGES)}")
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    parser.add_argument("--seed", type=int, default=20260921)
    parser.add_argument("--sample-per-task", type=int, default=0,
                        help="episodes per task for the corpus comparison (0 = all)")
    parser.add_argument("--recon-episodes", type=int, nargs="*", default=[0, 1, 2, 7, 100, 400, 1200, 2500])
    parser.add_argument("--no-series", dest="keep_series", action="store_false", default=True)
    parser.add_argument("--visual-episodes", type=int, nargs="*",
                        default=[0, 2, 5, 12, 21, 33, 44, 53, 67, 88, 101, 133])
    parser.add_argument("--visual-video-seconds", type=float, default=30.0,
                        help="nominal clip length used to place the sampled video frames")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    stages = []
    for stage in args.stages:
        if stage not in VALID_STAGES:
            print(f"[compare-blockstacking-dataset] unknown stage {stage!r}", file=sys.stderr)
            return 2
        stages.extend(VALID_STAGES[:-1] if stage == "all" else [stage])
    args.commands = [" ".join([sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]])]
    out = output_dir(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for stage in stages:
        print(f"[compare-blockstacking-dataset] stage {stage}", flush=True)
        if stage == "recon":
            payload = stage_recon(args)
            print(f"    blockstacking train episodes: {payload['blockstacking_train']['episodes']} "
                  f"frames: {payload['blockstacking_train']['frames']}", flush=True)
        elif stage == "fk-check":
            payload = fk_check(args)
            print(f"    worst identity-mapping palm error: {payload.get('worst_identity_error_m')} m", flush=True)
        elif stage == "dataset":
            payload = stage_dataset(args)
        elif stage == "rollout":
            payload = stage_rollout(args)
        elif stage == "compare":
            payload = stage_compare(args)
        elif stage == "diagnose":
            payload = stage_diagnose(args)
        elif stage == "figures":
            payload = stage_figures(args)
        elif stage == "visual":
            payload = stage_visual(args)
            print(f"    red block apparent width: "
                  f"{(payload.get('blocks', {}).get('dataset_egocentric', {}).get('red') or {}).get('median_apparent_width_px')} px (demos) vs "
                  f"{(payload.get('blocks', {}).get('sim_head_camera', {}).get('red') or {}).get('median_apparent_width_px')} px (sim)", flush=True)
        elif stage == "sheets":
            payload = stage_contact_sheets(args)
        elif stage == "manifest":
            stage_manifest(args)
    return 0


# --------------------------------------------------------------------- fk check


def fk_check(args: argparse.Namespace) -> dict[str, Any]:
    """Reproduce the simulator's measured palm poses from the recorded joints."""
    urdf = Urdf(args.urdf)
    chains = {side: urdf.chain(link) for side, link in PALM_LINKS.items()}
    result: dict[str, Any] = {"schema_version": SCHEMA_VERSION, "script_version": SCRIPT_VERSION,
                              "urdf": str(args.urdf), "root_link": urdf.root_link,
                              "chains": {side: [j["name"] for j in chain] for side, chain in chains.items()},
                              "sessions": []}
    for tag in ROLLOUT_TAGS:
        for rollout in sorted((campaign_dir() / tag / "rollouts").iterdir()):
            if not rollout.is_dir():
                continue
            rows = read_jsonl(rollout / "tracking.jsonl")
            samples = read_jsonl(rollout / "isaac.samples.jsonl")
            if not rows or not samples:
                continue
            tracking = {row["tick"]: row for row in rows}
            errors = {side: [] for side in PALM_LINKS}
            controls = {"swapped_arms": {side: [] for side in PALM_LINKS},
                        "zero_joints": {side: [] for side in PALM_LINKS}}
            for sample in samples:
                row = tracking.get(sample.get("physics_tick"))
                if row is None:
                    continue
                base = np.eye(4)
                base[:3, :3] = quaternion_wxyz_to_matrix(row["root_quaternion_wxyz"])
                base[:3, 3] = np.asarray(row["root_position"], dtype=float)
                measured = np.asarray(row["body_measured"], dtype=float)
                mapping = {name: measured[index] for index, name in enumerate(BODY_JOINT_ORDER)}
                for side, chain in chains.items():
                    link = PALM_LINKS[side]
                    reference = np.asarray(sample["palm_pose_w"][link][:3], dtype=float)
                    for label, transform in (
                        ("identity", lambda name: mapping.get(name, 0.0)),
                        ("swapped_arms", lambda name: mapping.get(
                            (("right" + name[4:]) if name.startswith("left") else
                             ("left" + name[5:]) if name.startswith("right") else name), 0.0)),
                        ("zero_joints", lambda name: 0.0),
                    ):
                        values = np.array([[transform(joint["name"]) for joint in chain]])
                        tip = urdf.fk_chain(chain, values, base)[0]
                        error = float(np.linalg.norm(tip[:3, 3] - reference))
                        (errors if label == "identity" else controls[label])[side].append(error)
            result["sessions"].append({
                "session": tag, "rollout": rollout.name, "samples": len(samples),
                "matched": len(errors["left"]),
                "identity_mapping_max_error_m": max(errors["left"] + errors["right"]),
                "identity_mapping_mean_error_m": float(np.mean(errors["left"] + errors["right"])),
                "swapped_arms_mean_error_m": float(np.mean(controls["swapped_arms"]["left"] + controls["swapped_arms"]["right"])),
                "zero_joints_mean_error_m": float(np.mean(controls["zero_joints"]["left"] + controls["zero_joints"]["right"])),
            })
    if result["sessions"]:
        result["worst_identity_error_m"] = max(s["identity_mapping_max_error_m"] for s in result["sessions"])
        result["verdict"] = ("FK reproduces the simulator's measured palm poses"
                             if result["worst_identity_error_m"] < 1e-3 else "MISMATCH: do not trust the FK")
    write_json(output_dir(args.output_dir) / "tables" / "fk_validation.json", result)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
