#!/usr/bin/env python3
"""Is the rollouts' 43D state inside the BlockStacking demonstration support?

One question only: do the psi0 and GR00T rollout states (the observation the
policy was fed, and the measured state the simulator realised) lie inside the
support of the BlockStacking demonstration state distribution, and where and
when does the first and the strongest departure appear?

Everything downstream of that answer -- model diagnosis, decoder replay,
controller tuning -- is deliberately out of scope.

Method
------

* **Reference.**  The Task-0 dataset copy (``psi0-unitree-dex3-sonic-v1/train``),
  task index 0 (``G1_Dex3_BlockStacking_Dataset``), every frame the converter
  marked valid (``frames_valid``).  Demonstrations are strongly time-correlated,
  so each episode carries equal total weight: per-channel statistics are
  weighted quantiles with frame weight ``1 / frames_valid``, and the support
  cloud is a fixed-seed balanced sample of ``FRAMES_PER_EPISODE`` frames per
  episode.  Both constructions are recorded.
* **Scaling.**  Robust per channel: ``scale = 1.4826 * MAD`` about the weighted
  median.  A channel whose MAD is exactly zero carries no support information at
  all (the BlockStacking lower body and waist are a synthetic constant in every
  episode), so it is reported on an absolute scale instead of being divided by an
  invented one.
* **Three complementary support metrics**, all per channel group so that
  dimension cannot drive the answer:
  1. Euclidean k-nearest-neighbour distance to the demo cloud (k=1, k=5);
  2. Mahalanobis distance under the demo group covariance -- far more
     discriminative than k-NN when the cloud is dense, and its quadratic form
     decomposes exactly into per-joint contributions, which is what names the
     responsible joints;
  3. a per-channel demonstration-quantile envelope.
* **Thresholds and severity come only from leave-one-episode-out (LOEO)
  demonstrations**: every demo episode is scored against a model built from the
  other 285, and the p95/p99 of that distribution become the thresholds.  A
  rollout is "in support" when its distance is inside the demo's own LOEO p99.
  Nothing is calibrated on the rollouts.
* **Negative controls, including a metric-power control.**  The rollouts are
  re-scored (a) with the upper body mirrored left/right, (b) against equal-sized
  demo clouds from two other tasks, and (c) -- the control that actually tests
  whether the metric has teeth -- *demonstration frames from another task* are
  scored against the BlockStacking cloud.  If (c) is not clearly worse, an
  "in support" verdict is not evidence of anything.

Layout is verified, never assumed: the canonical 43D channel names are read from
the dataset's own ``meta/info.json`` and must equal the hard-coded list, the
simulator's body/hand joint order is read from ``isaac.metrics.json``, and the
psi0 observation recorded in the bridge telemetry is checked channel by channel
against the tracked measured state.

    data/venvs/hf-datasets/bin/python scripts/analyze-blockstacking-state-support.py support
    python3 scripts/analyze-blockstacking-state-support.py figures
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np

SCHEMA_VERSION = 1
SCRIPT_VERSION = "analyze-blockstacking-state-support.py/1.0.0"
REPO_ROOT = Path(__file__).resolve().parents[1]

CAMPAIGN_DIR = REPO_ROOT / "data" / "outputs" / "blockstacking-debug"
DEFAULT_OUT = CAMPAIGN_DIR / "experiments" / "02-state-support"
DEFAULT_DATASET = REPO_ROOT / "data" / "datasets" / "psi0-unitree-dex3-sonic-v1" / "train"

COLLECTION = "G1_Dex3_BlockStacking_Dataset"
PROMPT = "Stack the three cubic blocks on the black tape in the order red, yellow, blue."
SEED = 20260921
FRAMES_PER_EPISODE = 100
NEIGHBOURS = (1, 5)
THRESHOLD_QUANTILES = (0.95, 0.99)
ENVELOPE_QUANTILES = (0.01, 0.99)
RIDGE_RELATIVE = 1e-6
#: Absolute deviation bands for the zero-variance groups, where no relative
#: scale exists: "the block of joints is within this many radians of the single
#: demonstrated value".
ABSOLUTE_BANDS_RAD = (0.01, 0.05, 0.10)
#: The pre-policy settle series is the last this many frames of the raw-tracking
#: block that immediately precedes the rollout crop (5 s at the 50 Hz log rate).
SETTLE_TAIL_FRAMES = 250
#: Wrong-task reference clouds.  ``PickGum`` is the corpus's lowest-hand task,
#: ``CameraPackaging`` the closest tabletop task to BlockStacking.
WRONG_TASKS = ("G1_Dex3_PickGum_Dataset", "G1_Dex3_CameraPackaging_Dataset")

#: Canonical 43D order: legs (12) | waist (3) | left arm (7) | right arm (7) |
#: left hand (7) | right hand (7).  Identical to the psi0 copy's own declared
#: names, to the simulator's articulation order and to the psi0 bridge's
#: ``[body_q(29) | left_hand_q(7) | right_hand_q(7)]``; the first two are
#: re-asserted at runtime below, the third in the reported checks.
CHANNELS: tuple[str, ...] = (
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
    "left_hand_thumb_0_joint", "left_hand_thumb_1_joint", "left_hand_thumb_2_joint",
    "left_hand_middle_0_joint", "left_hand_middle_1_joint",
    "left_hand_index_0_joint", "left_hand_index_1_joint",
    "right_hand_thumb_0_joint", "right_hand_thumb_1_joint", "right_hand_thumb_2_joint",
    "right_hand_index_0_joint", "right_hand_index_1_joint",
    "right_hand_middle_0_joint", "right_hand_middle_1_joint",
)
STATE_DIM = len(CHANNELS)

GROUPS: dict[str, slice] = {
    "lower_body": slice(0, 12),
    "waist": slice(12, 15),
    "left_arm": slice(15, 22),
    "right_arm": slice(22, 29),
    "arms": slice(15, 29),
    "left_hand": slice(29, 36),
    "right_hand": slice(36, 43),
    "hands": slice(29, 43),
    "upper_body": slice(15, 43),
    "all_43d_informative": slice(0, 43),
    "all_43d": slice(0, 43),
}
#: ``all_43d`` and ``all_43d_informative`` both cover channels 0..42.  The 15
#: zero-variance channels are dropped from every k-NN and Mahalanobis metric
#: (they have no scale to be measured in, and a zero-variance channel cannot be
#: in or out of a support), so the two groups coincide numerically and are
#: reported only to make that exclusion visible.  Those 15 channels are measured
#: against their single demonstrated constant instead, on an absolute scale.
#: ``FLOOR_GROUPS`` names the groups whose *scaled space* uses the floor scale.
FLOOR_GROUPS = ("all_43d",)
#: Groups the "which group is worst / first out" ranking runs over.  ``all_43d``
#: has no relative scale, ``all_43d_informative`` is the same 28 channels as
#: ``upper_body``, and the two constant groups have no threshold at all.
RANKING_GROUPS = ("left_arm", "right_arm", "arms", "left_hand", "right_hand", "hands", "upper_body")

#: Left/right mirror of the upper body, used as the joint-permuted negative
#: control: ``permuted[:, i] = state[:, MIRROR[i]]``.
MIRROR_PERMUTATION = np.array(
    list(range(6, 12)) + list(range(0, 6)) + list(range(12, 15))
    + list(range(22, 29)) + list(range(15, 22))
    + list(range(36, 43)) + list(range(29, 36)), dtype=int)


class SupportError(RuntimeError):
    """The measurement cannot run; the message is meant for the operator."""


# ------------------------------------------------------------------ plumbing


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


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


def write_json(path: Path, payload: Any) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n", encoding="utf-8")


def union_columns(rows: Sequence[dict[str, Any]]) -> list[str]:
    """Column names in first-seen order, so partial rows do not lose their fields."""
    names: dict[str, None] = {}
    for row in rows:
        for name in row:
            names.setdefault(name, None)
    return list(names)


def write_csv(path: Path, rows: Sequence[dict[str, Any]], columns: Sequence[str] | None = None) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        Path(path).write_text("", encoding="utf-8")
        return
    names = list(columns) if columns else union_columns(rows)
    lines = [",".join(names)]
    for row in rows:
        cells = []
        for name in names:
            value = row.get(name)
            if isinstance(value, bool):
                cells.append("true" if value else "false")
            elif isinstance(value, float):
                cells.append("" if not math.isfinite(value) else f"{value:.9g}")
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


def stat_block(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"count": 0}
    q = np.percentile(values, [0, 5, 25, 50, 75, 95, 99, 100])
    return {"count": int(values.size), "mean": float(values.mean()), "std": float(values.std()),
            "min": float(q[0]), "p05": float(q[1]), "p25": float(q[2]), "median": float(q[3]),
            "p75": float(q[4]), "p95": float(q[5]), "p99": float(q[6]), "max": float(q[7])}


def empirical_percentile(sample: np.ndarray, value: float) -> float | None:
    sample = np.asarray(sample, dtype=float)
    sample = sample[np.isfinite(sample)]
    if sample.size == 0 or not math.isfinite(value):
        return None
    return float((sample < value).mean() * 100.0 + 0.5 * (sample == value).mean() * 100.0)


def weighted_channel_quantiles(values: np.ndarray, weights: np.ndarray,
                               fractions: Sequence[float]) -> np.ndarray:
    """``(len(fractions), channels)`` weighted quantiles, midpoint convention."""
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    fractions = np.asarray(fractions, dtype=float)
    total = float(weights.sum())
    out = np.empty((fractions.size, values.shape[1]))
    for channel in range(values.shape[1]):
        order = np.argsort(values[:, channel])
        ordered = values[order, channel]
        cumulative = np.cumsum(weights[order]) - 0.5 * weights[order]
        out[:, channel] = np.interp(fractions * total, cumulative, ordered)
    return out


def channel_group(channel: int) -> str:
    for name, block in GROUPS.items():
        if name == "all_43d":
            continue
        if block.start <= channel < block.stop:
            return name
    return "all_43d"


# ------------------------------------------------------------------ dataset


def blockstacking_episodes(dataset_root: Path) -> list[dict[str, Any]]:
    selected = [e for e in read_jsonl(dataset_root / "meta" / "episodes.jsonl")
                if e["source_collection"] == COLLECTION]
    selected.sort(key=lambda e: e["episode_index"])
    if not selected:
        raise SupportError(f"no {COLLECTION} episodes under {dataset_root}")
    return selected


def episode_path(dataset_root: Path, episode: int) -> Path:
    return dataset_root / "data" / f"chunk-{episode // 1000:03d}" / f"episode_{episode:06d}.parquet"


def verify_dataset_layout(dataset_root: Path) -> dict[str, Any]:
    info = read_json(dataset_root / "meta" / "info.json")
    names = (info["features"].get("observation.state") or {}).get("names")
    if names is None:
        raise SupportError("dataset meta/info.json declares no observation.state names")
    flat = [str(n) for n in names]
    if flat != list(CHANNELS):
        mismatch = [(i, a, b) for i, (a, b) in enumerate(zip(flat, CHANNELS)) if a != b]
        raise SupportError(f"dataset channel names differ from the canonical order: {mismatch[:5]}")
    return {"channel_names_match_canonical": True, "fps": info["fps"],
            "total_episodes": info["total_episodes"], "codebase_version": info["codebase_version"]}


def verify_simulator_layout(tag: str) -> dict[str, Any]:
    """The rollout tracking columns must be the canonical order, block for block."""
    metrics = read_json(CAMPAIGN_DIR / tag / "raw" / "isaac.metrics.json")
    columns = metrics["tracking_columns"]
    body = [str(n) for n in columns["body_joints"]]
    left = [str(n) for n in columns["left_hand_joints"]]
    right = [str(n) for n in columns["right_hand_joints"]]
    if body != list(CHANNELS[:29]):
        raise SupportError(f"{tag}: simulator body joint order differs from canonical")
    if left != list(CHANNELS[29:36]):
        raise SupportError(f"{tag}: simulator left-hand order differs from canonical")
    if right != list(CHANNELS[36:43]):
        raise SupportError(f"{tag}: simulator right-hand order differs from canonical")
    return {"body_matches_canonical": True, "left_hand_matches_canonical": True,
            "right_hand_matches_canonical": True}


def load_all_valid_states(dataset_root: Path, episodes: Sequence[dict[str, Any]]
                          ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Every valid frame, with its episode index and an episode-balanced weight."""
    states: list[np.ndarray] = []
    weights: list[np.ndarray] = []
    owners: list[int] = []
    for entry in episodes:
        episode = int(entry["episode_index"])
        columns = read_parquet(episode_path(dataset_root, episode), ["observation.state"])
        state = np.asarray(columns["observation.state"], dtype=float)
        valid = min(int(entry.get("frames_valid", state.shape[0])), state.shape[0])
        if valid <= 0:
            continue
        states.append(state[:valid])
        weights.append(np.full(valid, 1.0 / valid))
        owners.extend([episode] * valid)
    if not states:
        raise SupportError("no valid demonstration frames")
    return (np.concatenate(states, axis=0), np.concatenate(weights, axis=0),
            np.asarray(owners, dtype=int))


def balanced_cloud(states: np.ndarray, owners: np.ndarray, episodes: Sequence[int],
                   per_episode: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Fixed-seed balanced sample: an equal number of frames per episode."""
    rng = np.random.default_rng(seed)
    picks: list[np.ndarray] = []
    labels: list[int] = []
    for episode in episodes:
        rows = np.flatnonzero(owners == episode)
        if rows.size == 0:
            continue
        take = min(per_episode, rows.size)
        picks.append(rows[np.sort(rng.choice(rows.size, size=take, replace=False))])
        labels.extend([episode] * take)
    if not picks:
        raise SupportError("balanced cloud sampled no frames")
    return states[np.concatenate(picks)], np.asarray(labels, dtype=int)


def loeo_envelope_rate(cloud: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    """Demo pass rate of the envelope when each episode is scored out of sample."""
    inside_p: list[np.ndarray] = []
    inside_hull: list[np.ndarray] = []
    for episode in np.unique(labels):
        rows = np.flatnonzero(labels == episode)
        other = np.flatnonzero(labels != episode)
        low = np.percentile(cloud[other], ENVELOPE_QUANTILES[0] * 100.0, axis=0)
        high = np.percentile(cloud[other], ENVELOPE_QUANTILES[1] * 100.0, axis=0)
        held = cloud[rows]
        inside_p.append((held >= low[None, :]) & (held <= high[None, :]))
        inside_hull.append((held >= cloud[other].min(axis=0)[None, :])
                           & (held <= cloud[other].max(axis=0)[None, :]))
    inside_p = np.concatenate(inside_p)
    inside_hull = np.concatenate(inside_hull)
    return {
        "channel_inside_p1_p99_frac": float(inside_p.mean()),
        "frame_all_channels_inside_p1_p99_frac": float(inside_p.all(axis=1).mean()),
        "channel_inside_hull_frac": float(inside_hull.mean()),
        "frame_all_channels_inside_hull_frac": float(inside_hull.all(axis=1).mean()),
        "frames": int(inside_p.shape[0]),
        "note": ("bounds rebuilt from the 285 episodes a held-out episode is not in; the bounds "
                 "applied to the rollouts come from the full cloud and are therefore wider"),
    }


# ------------------------------------------------------- kNN / Mahalanobis


def _squared_distances(query: np.ndarray, cloud: np.ndarray) -> np.ndarray:
    return np.maximum((query ** 2).sum(1)[:, None] + (cloud ** 2).sum(1)[None, :]
                      - 2.0 * query @ cloud.T, 0.0)


def knn_distances(query: np.ndarray, cloud: np.ndarray, index: np.ndarray,
                  neighbours: Sequence[int] = NEIGHBOURS, block: int = 512) -> dict[int, np.ndarray]:
    query = np.asarray(query, dtype=float)[:, index]
    cloud = np.asarray(cloud, dtype=float)[:, index]
    k_max = max(neighbours)
    out = {k: np.empty(query.shape[0]) for k in neighbours}
    for start in range(0, query.shape[0], block):
        stop = min(start + block, query.shape[0])
        part = np.partition(_squared_distances(query[start:stop], cloud), k_max - 1, axis=1)[:, :k_max]
        for k in neighbours:
            out[k][start:stop] = np.sqrt(part[:, :k].mean(axis=1))
    return out


def loeo_knn(cloud: np.ndarray, labels: np.ndarray, index: np.ndarray,
             neighbours: Sequence[int] = NEIGHBOURS, block: int = 256) -> tuple[np.ndarray, ...]:
    """k-NN distances of every demo frame to a cloud built without its own episode."""
    cloud = np.asarray(cloud, dtype=float)[:, index]
    k_max = max(neighbours)
    out = {k: np.empty(cloud.shape[0]) for k in neighbours}
    order = np.argsort(labels, kind="stable")
    sorted_labels = labels[order]
    blocks = {}
    for episode in np.unique(labels):
        rows = np.flatnonzero(sorted_labels == episode)
        blocks[int(episode)] = (int(rows[0]), int(rows[-1]) + 1)
    for start in range(0, order.size, block):
        rows = order[start:min(start + block, order.size)]
        distances = _squared_distances(cloud[rows], cloud)
        for episode in np.unique(labels[rows]):
            begin, end = blocks[int(episode)]
            distances[labels[rows] == episode, begin:end] = np.inf
        part = np.partition(distances, k_max - 1, axis=1)[:, :k_max]
        for k in neighbours:
            out[k][rows] = np.sqrt(part[:, :k].mean(axis=1))
    return tuple(out[k] for k in neighbours)


def _regularised_covariance(samples: np.ndarray) -> np.ndarray:
    count = samples.shape[0]
    mean = samples.mean(axis=0)
    covariance = samples.T @ samples / count - np.outer(mean, mean)
    scale = float(np.trace(covariance)) / covariance.shape[0]
    return covariance + (RIDGE_RELATIVE * scale) * np.eye(covariance.shape[0])


class MahalanobisModel:
    """Mahalanobis distance with an exact per-channel contribution split."""

    def __init__(self, samples: np.ndarray) -> None:
        self.mean = samples.mean(axis=0)
        covariance = _regularised_covariance(samples)
        self.precision = np.linalg.inv(covariance)
        self.cholesky = np.linalg.cholesky(covariance)

    def distance(self, query: np.ndarray) -> np.ndarray:
        centred = np.asarray(query, dtype=float) - self.mean[None, :]
        solved = np.linalg.solve(self.cholesky, centred.T)
        return np.sqrt((solved ** 2).sum(axis=0))

    def contributions(self, query: np.ndarray) -> np.ndarray:
        """(N, d) additive split of the squared distance; rows sum to d^2."""
        centred = np.asarray(query, dtype=float) - self.mean[None, :]
        weighted = centred @ self.precision
        return centred * weighted


def loeo_mahalanobis(cloud: np.ndarray, labels: np.ndarray, index: np.ndarray) -> np.ndarray:
    """Mahalanobis distance of each demo frame to a model without its own episode."""
    samples = np.asarray(cloud, dtype=float)[:, index]
    total = samples.shape[0]
    total_sum = samples.sum(axis=0)
    total_gram = samples.T @ samples
    out = np.empty(total)
    for episode in np.unique(labels):
        rows = np.flatnonzero(labels == episode)
        held = samples[rows]
        remaining = total - rows.size
        sums = total_sum - held.sum(axis=0)
        gram = total_gram - held.T @ held
        mean = sums / remaining
        covariance = gram / remaining - np.outer(mean, mean)
        scale = float(np.trace(covariance)) / covariance.shape[0]
        covariance = covariance + (RIDGE_RELATIVE * scale) * np.eye(covariance.shape[0])
        cholesky = np.linalg.cholesky(covariance)
        solved = np.linalg.solve(cholesky, (held - mean).T)
        out[rows] = np.sqrt((solved ** 2).sum(axis=0))
    return out


# ------------------------------------------------------------------ rollouts


def tracked_state(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """``(t_wall_ns, (N, 43))`` measured state in canonical order."""
    rows = read_jsonl(path)
    if not rows:
        raise SupportError(f"no tracking rows under {path}")
    times = np.array([r["wall_time_ns"] for r in rows], dtype=float)
    state = np.stack([np.concatenate([np.asarray(r["body_measured"], dtype=float),
                                      np.asarray(r["left_hand_measured"], dtype=float),
                                      np.asarray(r["right_hand_measured"], dtype=float)])
                      for r in rows])
    if state.shape[1] != STATE_DIM:
        raise SupportError(f"{path}: measured state has {state.shape[1]} channels")
    return times, state


def observed_state(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """``(t_wall_ns, (N, 43))`` recorded policy observation; empty when absent."""
    times: list[float] = []
    states: list[np.ndarray] = []
    for row in read_jsonl(path):
        if row.get("kind") != "observation":
            continue
        state = np.asarray(row["state"], dtype=float)
        if state.shape[0] != STATE_DIM:
            raise SupportError(f"{path}: observation state has {state.shape[0]} channels")
        times.append(float(row["wall_time_ns"]))
        states.append(state)
    if not times:
        return np.zeros(0), np.zeros((0, STATE_DIM))
    return np.array(times), np.stack(states)


def settle_block(tag: str, crop_first_ns: float) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Last ``SETTLE_TAIL_FRAMES`` frames of the raw-tracking block before the crop.

    The campaign crop begins essentially at the policy start, so the only real
    pre-policy settle window lives in the session-level log.
    """
    raw = read_jsonl(CAMPAIGN_DIR / tag / "raw" / "isaac.tracking.jsonl")
    if not raw:
        return np.zeros(0), np.zeros((0, STATE_DIM)), {"available": False, "reason": "no raw tracking"}
    wall = np.array([r["wall_time_ns"] for r in raw], dtype=float)
    sim = np.array([r["sim_s"] for r in raw], dtype=float)
    stop = int(np.searchsorted(wall, crop_first_ns))
    if stop <= 0:
        return np.zeros(0), np.zeros((0, STATE_DIM)), {"available": False,
                                                       "reason": "crop starts at the log start"}
    begin = stop - 1
    while begin > 0 and wall[begin - 1] < wall[begin] and 0.0 <= sim[begin] - sim[begin - 1] < 0.1:
        begin -= 1
    first = max(begin, stop - SETTLE_TAIL_FRAMES)
    rows = raw[first:stop]
    state = np.stack([np.concatenate([np.asarray(r["body_measured"], dtype=float),
                                      np.asarray(r["left_hand_measured"], dtype=float),
                                      np.asarray(r["right_hand_measured"], dtype=float)])
                      for r in rows])
    support = np.array([bool(r.get("support_active", False)) for r in rows])
    return wall[first:stop], state, {
        "available": True, "frames_used": int(len(rows)),
        "block_frames_available": int(stop - begin), "tail_frames_cap": SETTLE_TAIL_FRAMES,
        "sim_s_start": float(sim[first]), "sim_s_stop": float(sim[stop - 1]),
        "support_active_frac": float(support.mean()),
    }


def measure_series(times: np.ndarray, state: np.ndarray, onset_ns: int,
                   calibration: dict[str, Any]) -> dict[str, Any]:
    relative = (times - onset_ns) / 1e9
    scaled = (state - calibration["median"][None, :]) / calibration["scale"][None, :]
    per_group: dict[str, Any] = {}
    frame_support: dict[str, np.ndarray] = {}
    for name, block in GROUPS.items():
        index = np.arange(*block.indices(STATE_DIM))
        informative = np.array([i for i in index if not calibration["degenerate"][i]])
        entry: dict[str, Any] = {
            "channels": [CHANNELS[i] for i in index],
            "informative_channels": [CHANNELS[i] for i in informative],
            "degenerate_channels": [CHANNELS[i] for i in index if calibration["degenerate"][i]],
        }
        if informative.size == 0:
            deviation = np.abs(state[:, index] - calibration["median"][index][None, :]).max(axis=1)
            entry["absolute_deviation_rad"] = stat_block(deviation)
            entry["demonstrated_constant_rad"] = {CHANNELS[i]: float(calibration["median"][i])
                                                  for i in index}
            for band in ABSOLUTE_BANDS_RAD:
                entry[f"frame_within_{int(band * 1000):03d}mrad_frac"] = float((deviation <= band).mean())
            frame_support[f"{name}_max_abs_dev"] = deviation
            per_group[name] = entry
            continue

        group_cloud = calibration["scaled_cloud"][:, informative]
        thresholds = calibration["thresholds"][name]
        if name in calibration["mahalanobis_groups"]:
            model = MahalanobisModel(group_cloud)
            mahalanobis = model.distance(scaled[:, informative])
            entry["mahalanobis"] = stat_block(mahalanobis)
            for quantile in THRESHOLD_QUANTILES:
                key = f"q{int(quantile * 100)}"
                inside = mahalanobis <= thresholds[key]["mahalanobis"]
                entry[f"in_support_mahalanobis_{key}_frac"] = float(inside.mean())
                entry[f"first_exceed_mahalanobis_{key}_s_rel"] = (
                    float(relative[int(np.argmax(~inside))]) if not inside.all() else None)
                frame_support[f"{name}_inside_mahalanobis_{key}"] = inside
            entry["mahalanobis_over_p99_median"] = float(
                np.median(mahalanobis) / thresholds["q99"]["mahalanobis"])
            entry["mahalanobis_loeo_percentile_of_median"] = empirical_percentile(
                calibration["loeo"][name]["mahalanobis"], float(np.median(mahalanobis)))
            frame_support[f"{name}_mahalanobis"] = mahalanobis
            frame_support[f"{name}_mahalanobis_contrib"] = model.contributions(scaled[:, informative])

        distances = knn_distances(scaled, calibration["scaled_cloud"], informative)
        low, high = calibration["envelope_p_low"], calibration["envelope_p_high"]
        inside_p = ((state[:, informative] >= low[informative][None, :])
                    & (state[:, informative] <= high[informative][None, :]))
        inside_hull = ((state[:, informative] >= calibration["envelope_min"][informative][None, :])
                       & (state[:, informative] <= calibration["envelope_max"][informative][None, :]))
        entry["channel_inside_p1_p99_frac"] = float(inside_p.mean())
        entry["frame_all_channels_inside_p1_p99_frac"] = float(inside_p.all(axis=1).mean())
        entry["channel_inside_hull_frac"] = float(inside_hull.mean())
        entry["frame_all_channels_inside_hull_frac"] = float(inside_hull.all(axis=1).mean())
        for k in NEIGHBOURS:
            threshold = thresholds["q99"][f"k{k}"]
            inside = distances[k] <= threshold
            entry[f"knn_k{k}"] = stat_block(distances[k])
            entry[f"knn_k{k}_over_p99_median"] = float(np.median(distances[k]) / threshold)
            for quantile in THRESHOLD_QUANTILES:
                key = f"q{int(quantile * 100)}"
                sub = distances[k] <= thresholds[key][f"k{k}"]
                entry[f"in_support_knn_k{k}_{key}_frac"] = float(sub.mean())
                entry[f"first_exceed_knn_k{k}_{key}_s_rel"] = (
                    float(relative[int(np.argmax(~sub))]) if not sub.all() else None)
            frame_support[f"{name}_d{k}"] = distances[k]
            frame_support[f"{name}_inside_k{k}_q99"] = inside
        if name.startswith("all_43d"):
            entry["note"] = (
                f"the {len(entry['degenerate_channels'])} zero-variance channels are excluded from "
                "every scaled metric (no scale exists for them), so this group is numerically "
                "identical to the same metric over the 28 informative channels; those channels are "
                "measured against their single demonstrated constant instead")
        per_group[name] = entry
    return {"per_group": per_group, "relative_seconds": relative, "frame_support": frame_support}


def window_view(series: dict[str, Any], mask: np.ndarray) -> dict[str, Any]:
    out: dict[str, Any] = {"frames": int(mask.sum())}
    if not mask.any():
        return out
    relative = series["relative_seconds"][mask]
    out["seconds_rel"] = [float(relative[0]), float(relative[-1])]
    for name in GROUPS:
        block: dict[str, Any] = {}
        if f"{name}_mahalanobis" in series["frame_support"]:
            values = series["frame_support"][f"{name}_mahalanobis"][mask]
            inside = series["frame_support"][f"{name}_inside_mahalanobis_q99"][mask]
            block.update({
                "mahalanobis_median": float(np.median(values)),
                "mahalanobis_p95": float(np.percentile(values, 95)),
                "mahalanobis_max": float(values.max()),
                "mahalanobis_in_support_q99_frac": float(inside.mean()),
            })
            if (~inside).any():
                block["mahalanobis_first_exceed_s_rel"] = float(relative[int(np.argmax(~inside))])
        if f"{name}_d1" in series["frame_support"]:
            values = series["frame_support"][f"{name}_d1"][mask]
            inside = series["frame_support"][f"{name}_inside_k1_q99"][mask]
            block.update({
                "knn_k1_median": float(np.median(values)),
                "knn_k1_p95": float(np.percentile(values, 95)),
                "knn_k1_in_support_q99_frac": float(inside.mean()),
            })
            if (~inside).any():
                block["knn_k1_first_exceed_s_rel"] = float(relative[int(np.argmax(~inside))])
        if f"{name}_max_abs_dev" in series["frame_support"]:
            values = series["frame_support"][f"{name}_max_abs_dev"][mask]
            block.update({"max_abs_dev_median": float(np.median(values)),
                          "max_abs_dev_max": float(values.max())})
            for band in ABSOLUTE_BANDS_RAD:
                block[f"within_{int(band * 1000):03d}mrad_frac"] = float((values <= band).mean())
        if block:
            out[name] = block
    return out


# --------------------------------------------------------------------- stages


def stage_support(args: argparse.Namespace) -> dict[str, Any]:
    dataset_root = Path(args.dataset_root)
    out = Path(args.output_dir)
    tables = out / "tables"
    tables.mkdir(parents=True, exist_ok=True)

    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION, "script_version": SCRIPT_VERSION,
        "question": ("are the psi0 / GR00T rollout observation and measured states inside the "
                     "BlockStacking demonstration state support?"),
        "layout_verification": {
            "dataset": verify_dataset_layout(dataset_root),
            "simulator": {tag: verify_simulator_layout(tag)
                          for tag in ("psi0-rollout-1", "groot-rollout-1")}},
    }

    episodes = blockstacking_episodes(dataset_root)
    states, weights, owners = load_all_valid_states(dataset_root, episodes)
    result["demo_pool"] = {
        "dataset": str(dataset_root), "collection": COLLECTION, "episodes": len(episodes),
        "valid_frames": int(states.shape[0]),
        "episode_frames_min": int(min(e["frames_valid"] for e in episodes)),
        "episode_frames_max": int(max(e["frames_valid"] for e in episodes)),
        "weighting": "per-frame weight 1/frames_valid so every episode carries total weight 1",
    }
    quantiles = weighted_channel_quantiles(
        states, weights, [0.5, ENVELOPE_QUANTILES[0], ENVELOPE_QUANTILES[1]])
    median = quantiles[0]
    mad = weighted_channel_quantiles(np.abs(states - median[None, :]), weights, [0.5])[0]
    degenerate = mad <= 0.0
    informative_scales = 1.4826 * mad[~degenerate]
    floor_scale = float(np.median(informative_scales)) if informative_scales.size else 1.0
    scale = np.where(degenerate, floor_scale, 1.4826 * mad)
    result["channel_calibration"] = {
        "degenerate_channels": [CHANNELS[i] for i in np.flatnonzero(degenerate)],
        "degenerate_channel_count": int(degenerate.sum()),
        "floor_scale_rad": floor_scale,
        "floor_scale_definition": ("median robust scale of the non-degenerate channels; only the "
                                   "all_43d group uses it, so the zero-variance channels have a "
                                   "documented scale instead of an invented one"),
        "smallest_informative_scale_rad": float(informative_scales.min()),
        "largest_informative_scale_rad": float(informative_scales.max()),
        "weighted_median_rad": {CHANNELS[i]: float(median[i]) for i in range(STATE_DIM)},
        "weighted_mad_rad": {CHANNELS[i]: float(mad[i]) for i in range(STATE_DIM)},
    }

    cloud, labels = balanced_cloud(states, owners, [int(e["episode_index"]) for e in episodes],
                                   args.frames_per_episode, args.seed)
    cloud_median, cloud_mad, cloud_scale = scaled_space(cloud, degenerate, floor_scale)
    scaled_cloud = (cloud - cloud_median[None, :]) / cloud_scale[None, :]
    result["demo_cloud"] = {
        "frames": int(cloud.shape[0]), "per_episode": args.frames_per_episode,
        "episodes": int(np.unique(labels).size), "seed": args.seed,
        "method": "fixed-seed balanced sample, equal frame count per episode",
        "max_abs_cloud_median_minus_weighted_median": float(np.abs(cloud_median - median).max()),
        "max_abs_cloud_scale_minus_weighted_scale": float(np.abs(cloud_scale - scale).max()),
    }

    low, high = np.percentile(cloud, [q * 100.0 for q in ENVELOPE_QUANTILES], axis=0)
    # A group that contains a zero-variance channel has a singular covariance, so
    # the Mahalanobis metric is defined only where every channel varies.  This
    # excludes all_43d (whose 15 constant channels are reported on an absolute
    # scale instead) and the two degenerate-only groups.
    mahalanobis_groups = tuple(name for name, block in GROUPS.items()
                               if not any(degenerate[i] for i in range(block.start, block.stop)))
    calibration: dict[str, Any] = {
        "median": median, "scale": scale, "degenerate": degenerate, "scaled_cloud": scaled_cloud,
        "envelope_p_low": low, "envelope_p_high": high,
        "envelope_min": cloud.min(axis=0), "envelope_max": cloud.max(axis=0),
        "thresholds": {}, "loeo": {}, "mahalanobis_groups": mahalanobis_groups,
    }
    for name, block in GROUPS.items():
        index = np.arange(*block.indices(STATE_DIM))
        informative = np.array([i for i in index if not degenerate[i]])
        record: dict[str, Any] = {}
        if informative.size == 0:
            calibration["thresholds"][name] = {}
            calibration["loeo"][name] = {}
            result.setdefault("loeo_calibration", {})[name] = {
                "degenerate_only": True,
                "channels": [CHANNELS[i] for i in index],
                "reason": "no demonstrated variance, so no relative scale and no distance threshold exists",
            }
            continue
        d1, d5 = loeo_knn(scaled_cloud, labels, informative)
        loeo_maha = (loeo_mahalanobis(scaled_cloud, labels, informative)
                     if name in mahalanobis_groups else None)
        calibration["loeo"][name] = {"k1": d1, "k5": d5}
        if loeo_maha is not None:
            calibration["loeo"][name]["mahalanobis"] = loeo_maha
        thresholds = {f"q{int(q * 100)}": {"k1": float(np.percentile(d1, q * 100.0)),
                                          "k5": float(np.percentile(d5, q * 100.0))}
                      for q in THRESHOLD_QUANTILES}
        if loeo_maha is not None:
            for key, value in thresholds.items():
                value["mahalanobis"] = float(np.percentile(loeo_maha, float(key[1:])))
        calibration["thresholds"][name] = thresholds
        record = {
            "thresholds": thresholds,
            "demo_in_support_knn_k1_q99_frac": float((d1 <= thresholds["q99"]["k1"]).mean()),
            "loeo_knn_k1": stat_block(d1), "loeo_knn_k5": stat_block(d5),
        }
        if loeo_maha is not None:
            record["demo_in_support_mahalanobis_q99_frac"] = float(
                (loeo_maha <= thresholds["q99"]["mahalanobis"]).mean())
            record["loeo_mahalanobis"] = stat_block(loeo_maha)
        result.setdefault("loeo_calibration", {})[name] = record
    envelope_loeo = loeo_envelope_rate(cloud, labels)
    result.setdefault("loeo_calibration", {})["envelope"] = envelope_loeo
    write_json(tables / "loeo_calibration.json",
               {"groups": result["loeo_calibration"], "envelope": envelope_loeo,
                "constants": {"seed": args.seed, "frames_per_episode": args.frames_per_episode,
                              "threshold_quantiles": list(THRESHOLD_QUANTILES),
                              "envelope_quantiles": list(ENVELOPE_QUANTILES),
                              "ridge_relative": RIDGE_RELATIVE}})

    # ---------------------------------------------------------------- queries
    campaign = read_json(CAMPAIGN_DIR / "campaign.json")
    frame_rows: list[dict[str, Any]] = []
    group_rows: list[dict[str, Any]] = []
    channel_rows: list[dict[str, Any]] = []
    controls: dict[str, Any] = {}
    wrong_task_cache: dict[str, Any] = {}
    result["series"] = {}
    for session in campaign["sessions"]:
        tag = session["session_dir"]
        for rollout_meta in session["rollouts"]:
            index = int(rollout_meta["index"])
            rollout_dir = CAMPAIGN_DIR / tag / "rollouts" / f"rollout-{index:02d}"
            rollout_json = read_json(rollout_dir / "rollout.json")
            onset_ns = rollout_json["start_wall_ns"] + int(rollout_meta["onset_seconds_after_start"] * 1e9)
            measured_t, measured = tracked_state(rollout_dir / "tracking.jsonl")
            obs_t, obs = observed_state(rollout_dir / "bridge-telemetry.jsonl")
            settle_t, settle_x, settle_info = settle_block(tag, measured_t[0])
            sources: list[tuple[str, np.ndarray, np.ndarray]] = []
            if obs.size:
                sources.append(("observation", obs_t, obs))
            sources.append(("measured", measured_t, measured))
            if settle_x.size:
                sources.append(("settle_raw", settle_t, settle_x))

            record: dict[str, Any] = {
                "onset_seconds_after_start": rollout_meta["onset_seconds_after_start"],
                "onset_source": rollout_meta["onset_source"], "onset_wall_ns": int(onset_ns),
                "settle_block": settle_info, "rows": {}}
            if obs.size:
                nearest = np.clip(np.searchsorted(measured_t, obs_t), 0, len(measured_t) - 1)
                interpolated = np.stack(
                    [np.interp(obs_t, measured_t, measured[:, c]) for c in range(STATE_DIM)], axis=1)
                bias = np.abs((obs - interpolated).mean(axis=0))
                record["observation_vs_measured"] = {
                    "frames": int(obs.shape[0]),
                    "nearest_sample_mean_abs_rad": float(np.abs(obs - measured[nearest]).mean()),
                    "nearest_sample_max_abs_rad": float(np.abs(obs - measured[nearest]).max()),
                    "interpolated_mean_abs_rad": float(np.abs(obs - interpolated).mean()),
                    "per_channel_mean_bias_max_abs_rad": float(bias.max()),
                    "worst_bias_channel": CHANNELS[int(np.argmax(bias))],
                    "per_channel_correlation_min": float(min(
                        np.corrcoef(obs[:, c], interpolated[:, c])[0, 1] for c in range(STATE_DIM))),
                }
            for kind, times, state in sources:
                series = measure_series(times, state, onset_ns, calibration)
                scaled = (state - calibration["median"][None, :]) / calibration["scale"][None, :]
                relative = series["relative_seconds"]
                active = relative >= 0.0
                settle = ~active
                label = f"{tag}/rollout-{index:02d}/{kind}"
                windows = {"all": np.ones(times.size, dtype=bool), "settle": settle, "active": active}
                entry: dict[str, Any] = {
                    "kind": kind, "frames": int(times.size), "settle_frames": int(settle.sum()),
                    "active_frames": int(active.sum()),
                    "settle_window_s": [float(relative[settle].min()), float(relative[settle].max())]
                    if settle.any() else None,
                    "active_window_s": [float(relative[active].min()), float(relative[active].max())]
                    if active.any() else None,
                    "per_group": series["per_group"],
                    **{name: window_view(series, mask) for name, mask in windows.items()},
                }
                ranking_mask = active if active.any() else np.ones(times.size, dtype=bool)
                ranking = []
                for name in RANKING_GROUPS:
                    if f"{name}_mahalanobis" not in series["frame_support"]:
                        continue
                    values = series["frame_support"][f"{name}_mahalanobis"][ranking_mask]
                    inside = series["frame_support"][f"{name}_inside_mahalanobis_q99"][ranking_mask]
                    knn_values = series["frame_support"][f"{name}_d1"][ranking_mask]
                    knn_inside = series["frame_support"][f"{name}_inside_k1_q99"][ranking_mask]
                    ranking.append({
                        "group": name,
                        "median_mahalanobis": float(np.median(values)),
                        "mahalanobis_over_p99": float(np.median(values)
                                                      / calibration["thresholds"][name]["q99"]["mahalanobis"]),
                        "mahalanobis_loeo_percentile_of_median": empirical_percentile(
                            calibration["loeo"][name]["mahalanobis"], float(np.median(values))),
                        "mahalanobis_in_support_q99_frac": float(inside.mean()),
                        "first_mahalanobis_exceed_s_rel": (
                            float(series["relative_seconds"][ranking_mask][int(np.argmax(~inside))])
                            if not inside.all() else None),
                        "median_knn_k1": float(np.median(knn_values)),
                        "knn_k1_loeo_percentile_of_median": empirical_percentile(
                            calibration["loeo"][name]["k1"], float(np.median(knn_values))),
                        "knn_k1_in_support_q99_frac": float(knn_inside.mean()),
                        "first_knn_k1_exceed_s_rel": (
                            float(series["relative_seconds"][ranking_mask][int(np.argmax(~knn_inside))])
                            if not knn_inside.all() else None),
                    })
                entry["group_ranking"] = sorted(
                    ranking, key=lambda r: -(r["mahalanobis_over_p99"] or 0.0))
                entry["strongest_group"] = entry["group_ranking"][0]["group"] if ranking else None
                for metric, key in (("mahalanobis", "first_mahalanobis_exceed_s_rel"),
                                    ("knn_k1", "first_knn_k1_exceed_s_rel")):
                    outside = [r for r in ranking if r[key] is not None]
                    entry[f"first_group_out_of_support_{metric}"] = (
                        min(outside, key=lambda r: r[key])["group"] if outside else None)
                    entry[f"first_group_out_of_support_{metric}_s_rel"] = (
                        min(r[key] for r in outside) if outside else None)
                entry["start_state"] = {
                    "seconds_rel": float(relative[0]),
                    "mahalanobis_in_support": {name: bool(series["frame_support"][f"{name}_inside_mahalanobis_q99"][0])
                                               for name in GROUPS
                                               if f"{name}_inside_mahalanobis_q99" in series["frame_support"]},
                    "knn_in_support": {name: bool(series["frame_support"][f"{name}_inside_k1_q99"][0])
                                       for name in GROUPS
                                       if f"{name}_inside_k1_q99" in series["frame_support"]},
                    "lower_body_max_abs_dev_rad": float(
                        series["frame_support"]["lower_body_max_abs_dev"][0]),
                }
                record["rows"][kind] = entry

                # Per-channel position and exact Mahalanobis contribution.  The
                # contribution index is the channel's offset inside the group the
                # quadratic form was built on (``upper_body`` for the informative
                # part of all_43d, which shares the same channel range).
                channel_lookup: dict[int, dict[str, Any]] = {}
                if active.any():
                    for channel in range(STATE_DIM):
                        row: dict[str, Any] = {
                            "series": label, "kind": kind, "channel_index": channel,
                            "channel": CHANNELS[channel], "group": channel_group(channel),
                            "degenerate": bool(degenerate[channel]),
                            "median_abs_scaled_dev": float(np.median(np.abs(scaled[active, channel]))),
                            "p95_abs_scaled_dev": float(np.percentile(np.abs(scaled[active, channel]), 95)),
                            "active_median_rad": float(np.median(state[active, channel])),
                            "demo_median_rad": float(median[channel]),
                            "inside_demo_p1_p99_frac": float(
                                ((state[active, channel] >= low[channel])
                                 & (state[active, channel] <= high[channel])).mean()),
                        }
                        source = "upper_body" if channel >= 15 else channel_group(channel)
                        contributions = series["frame_support"].get(f"{source}_mahalanobis_contrib")
                        if contributions is not None:
                            offset = channel - GROUPS[source].start
                            squared = float((series["frame_support"][f"{source}_mahalanobis"][active] ** 2).sum())
                            row["mean_mahalanobis_contribution"] = float(contributions[active, offset].mean())
                            row["mahalanobis_contribution_share"] = float(
                                contributions[active, offset].sum() / max(squared, 1e-12))
                        channel_lookup[channel] = row
                        channel_rows.append(row)
                for row in range(times.size):
                    frame_rows.append({
                        "series": label, "kind": kind, "frame": row,
                        "seconds_rel": float(relative[row]),
                        "window": "active" if relative[row] >= 0 else "settle",
                        "mahalanobis_upper_body": float(series["frame_support"]["upper_body_mahalanobis"][row]),
                        "d1_upper_body": float(series["frame_support"]["upper_body_d1"][row]),
                        "inside_upper_body_mahalanobis": bool(
                            series["frame_support"]["upper_body_inside_mahalanobis_q99"][row]),
                        "inside_upper_body_knn": bool(
                            series["frame_support"]["upper_body_inside_k1_q99"][row]),
                        "mahalanobis_left_arm": float(series["frame_support"]["left_arm_mahalanobis"][row]),
                        "mahalanobis_right_arm": float(series["frame_support"]["right_arm_mahalanobis"][row]),
                        "mahalanobis_left_hand": float(series["frame_support"]["left_hand_mahalanobis"][row]),
                        "mahalanobis_right_hand": float(series["frame_support"]["right_hand_mahalanobis"][row]),
                        "d1_all_43d": float(series["frame_support"]["all_43d_d1"][row]),
                        "inside_all_43d_knn": bool(series["frame_support"]["all_43d_inside_k1_q99"][row]),
                        "inside_left_arm_mahalanobis": bool(
                            series["frame_support"]["left_arm_inside_mahalanobis_q99"][row]),
                        "inside_right_arm_mahalanobis": bool(
                            series["frame_support"]["right_arm_inside_mahalanobis_q99"][row]),
                        "max_abs_dev_lower_body": float(
                            series["frame_support"]["lower_body_max_abs_dev"][row]),
                    })
                for name in GROUPS:
                    for window_name, mask in windows.items():
                        view = window_view(series, mask).get(name, {})
                        group_rows.append({
                            "series": label, "session": tag, "rollout": f"rollout-{index:02d}",
                            "kind": kind, "group": name, "window": window_name,
                            "frames": view.get("frames", 0),
                            "n_channels": len(series["per_group"][name]["channels"]),
                            "n_degenerate_channels": len(series["per_group"][name]["degenerate_channels"]),
                            "threshold_q95_knn_k1": calibration["thresholds"][name].get("q95", {}).get("k1"),
                            "threshold_q99_knn_k1": calibration["thresholds"][name].get("q99", {}).get("k1"),
                            "threshold_q95_mahalanobis": calibration["thresholds"][name].get("q95", {}).get("mahalanobis"),
                            "threshold_q99_mahalanobis": calibration["thresholds"][name].get("q99", {}).get("mahalanobis"),
                            **{k: v for k, v in view.items() if k not in ("frames", "seconds_rel")},
                        })

            # ---------------------------------------------------- controls
            active_mask = (measured_t - onset_ns) / 1e9 >= 0.0
            forward = np.arange(15, STATE_DIM)
            measured_scaled = (measured - calibration["median"][None, :]) / calibration["scale"][None, :]
            model = MahalanobisModel(scaled_cloud[:, forward])
            true_knn = knn_distances(measured_scaled, scaled_cloud, forward)
            true_maha = model.distance(measured_scaled[:, forward])
            entry_controls: dict[str, Any] = {}
            for control_name, permutation in (("joint_permuted_left_right_mirror", MIRROR_PERMUTATION),):
                permuted = measured_scaled[:, permutation]
                permuted_knn = knn_distances(permuted, scaled_cloud, forward)
                permuted_maha = model.distance(permuted[:, forward])
                entry_controls[control_name] = {
                    "channels_remapped": [f"{CHANNELS[i]}<-{CHANNELS[permutation[i]]}"
                                          for i in range(STATE_DIM) if permutation[i] != i],
                    "true_knn_k1_median": float(np.median(true_knn[1][active_mask])),
                    "permuted_knn_k1_median": float(np.median(permuted_knn[1][active_mask])),
                    "knn_ratio_permuted_over_true": float(
                        np.median(permuted_knn[1][active_mask])
                        / max(float(np.median(true_knn[1][active_mask])), 1e-12)),
                    "true_mahalanobis_median": float(np.median(true_maha[active_mask])),
                    "permuted_mahalanobis_median": float(np.median(permuted_maha[active_mask])),
                    "mahalanobis_ratio_permuted_over_true": float(
                        np.median(permuted_maha[active_mask])
                        / max(float(np.median(true_maha[active_mask])), 1e-12)),
                    "permuted_mahalanobis_percentile_in_demo_loeo": empirical_percentile(
                        calibration["loeo"]["upper_body"]["mahalanobis"],
                        float(np.median(permuted_maha[active_mask]))),
                    "permuted_in_support_frac_q99": float(
                        (permuted_maha[active_mask]
                         <= calibration["thresholds"]["upper_body"]["q99"]["mahalanobis"]).mean()),
                }
            for collection in WRONG_TASKS:
                wrong = wrong_task_reference(dataset_root, collection, cloud.shape[0], args.seed,
                                             degenerate, floor_scale, wrong_task_cache)
                # Each model is applied in its own robust-scaled units, so the
                # ratio between the two median distances is dimensionless.
                wrong_query = (measured - wrong["median"][None, :]) / wrong["scale"][None, :]
                wrong_knn = knn_distances(wrong_query, wrong["scaled_cloud"], forward)
                wrong_maha = wrong["model"].distance(wrong_query[:, forward])
                entry_controls[f"wrong_task_cloud::{collection}"] = {
                    "cloud_frames": int(wrong["cloud"].shape[0]),
                    "wrong_task_knn_k1_median": float(np.median(wrong_knn[1][active_mask])),
                    "knn_ratio_wrong_over_true": float(
                        np.median(wrong_knn[1][active_mask])
                        / max(float(np.median(true_knn[1][active_mask])), 1e-12)),
                    "wrong_task_mahalanobis_median": float(np.median(wrong_maha[active_mask])),
                    "mahalanobis_ratio_wrong_over_true": float(
                        np.median(wrong_maha[active_mask])
                        / max(float(np.median(true_maha[active_mask])), 1e-12)),
                    "wrong_task_own_loeo_q99_mahalanobis": wrong["loeo_q99"],
                    "in_support_frac_vs_wrong_task_own_threshold": float(
                        (wrong_maha[active_mask] <= wrong["loeo_q99"]).mean()),
                    "in_support_frac_vs_true_task_threshold": float(
                        (wrong_maha[active_mask]
                         <= calibration["thresholds"]["upper_body"]["q99"]["mahalanobis"]).mean()),
                }
            controls[f"{tag}/rollout-{index:02d}"] = entry_controls
            result["series"].setdefault(tag, {})[f"rollout-{index:02d}"] = record

    # Metric-power control: another task's demonstrations must be clearly outside.
    power: dict[str, Any] = {}
    model = MahalanobisModel(scaled_cloud[:, np.arange(15, STATE_DIM)])
    for collection in WRONG_TASKS:
        wrong = wrong_task_reference(dataset_root, collection, cloud.shape[0], args.seed,
                                     degenerate, floor_scale, wrong_task_cache)
        raw = wrong["cloud"]
        scaled = (raw - median[None, :]) / scale[None, :]
        maha = model.distance(scaled[:, np.arange(15, STATE_DIM)])
        knn = knn_distances(scaled, scaled_cloud, np.arange(15, STATE_DIM))
        power[collection] = {
            "demo_frames_scored": int(raw.shape[0]),
            "mahalanobis_median": float(np.median(maha)),
            "mahalanobis_p95": float(np.percentile(maha, 95)),
            "mahalanobis_percentile_in_blockstacking_loeo": empirical_percentile(
                calibration["loeo"]["upper_body"]["mahalanobis"], float(np.median(maha))),
            "in_support_frac_q99": float(
                (maha <= calibration["thresholds"]["upper_body"]["q99"]["mahalanobis"]).mean()),
            "knn_k1_median": float(np.median(knn[1])),
            "knn_k1_in_support_q99_frac": float(
                (knn[1] <= calibration["thresholds"]["upper_body"]["q99"]["k1"]).mean()),
        }
    result["metric_power_control"] = power
    result["controls"] = controls

    write_json(tables / "controls.json", {"rollouts": controls, "metric_power": power})
    write_json(tables / "support_by_group.json", {"series": result["series"]})
    write_csv(tables / "support_by_group.csv", group_rows)
    write_csv(tables / "channel_contributions.csv", channel_rows)
    write_csv(tables / "frame_support.csv", frame_rows)
    write_csv(tables / "channel_calibration.csv", [{
        "channel_index": i, "channel": CHANNELS[i], "group": channel_group(i),
        "weighted_median_rad": float(median[i]), "weighted_mad_rad": float(mad[i]),
        "robust_scale_rad": float(scale[i]), "degenerate": bool(degenerate[i]),
        "cloud_min_rad": float(calibration["envelope_min"][i]),
        "cloud_p01_rad": float(low[i]), "cloud_p99_rad": float(high[i]),
        "cloud_max_rad": float(calibration["envelope_max"][i]),
    } for i in range(STATE_DIM)])
    np.savez_compressed(
        out / "tables" / "support_calibration.npz",
        channel_median=median, channel_scale=scale, envelope_p_low=low, envelope_p_high=high,
        cloud_min=calibration["envelope_min"], cloud_max=calibration["envelope_max"],
        **{f"loeo_{name}_mahalanobis": calibration["loeo"][name]["mahalanobis"]
           for name in mahalanobis_groups},
        **{f"loeo_{name}_knn_k1": calibration["loeo"][name]["k1"]
           for name in GROUPS if calibration["loeo"].get(name)})
    write_json(out / "summary.json", result)
    write_json(out / "manifest.json", build_manifest(args, dataset_root, out))
    return result


def scaled_space(cloud: np.ndarray, degenerate: np.ndarray, floor_scale: float
                 ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    median = np.median(cloud, axis=0)
    mad = np.median(np.abs(cloud - median[None, :]), axis=0)
    scale = np.where((mad <= 0.0) | degenerate, floor_scale, 1.4826 * mad)
    return median, mad, scale


def wrong_task_reference(dataset_root: Path, collection: str, target_frames: int, seed: int,
                         degenerate: np.ndarray, floor_scale: float,
                         cache: dict[str, Any] | None = None) -> dict[str, Any]:
    """An equal-sized reference cloud from another task, with its own robust scaling."""
    if cache is not None and collection in cache:
        return cache[collection]
    entries = [e for e in read_jsonl(dataset_root / "meta" / "episodes.jsonl")
               if e["source_collection"] == collection]
    if not entries:
        raise SupportError(f"no episodes for control collection {collection}")
    entries.sort(key=lambda e: e["episode_index"])
    per_episode = max(1, int(round(target_frames / len(entries))))
    states, _, owners = load_all_valid_states(dataset_root, entries)
    cloud, labels = balanced_cloud(states, owners, [int(e["episode_index"]) for e in entries],
                                   per_episode, seed)
    median, _, scale = scaled_space(cloud, degenerate, floor_scale)
    scaled = (cloud - median[None, :]) / scale[None, :]
    forward = np.arange(15, STATE_DIM)
    loeo = loeo_mahalanobis(scaled, labels, forward)
    reference = {"collection": collection, "cloud": cloud, "scaled_cloud": scaled,
                 "labels": labels, "median": median, "scale": scale,
                 "model": MahalanobisModel(scaled[:, forward]),
                 "loeo_q99": float(np.percentile(loeo, 99.0))}
    if cache is not None:
        cache[collection] = reference
    return reference


def build_manifest(args: argparse.Namespace, dataset_root: Path, out: Path) -> dict[str, Any]:
    inputs: dict[str, Any] = {}
    for path in (dataset_root / "meta" / "info.json", dataset_root / "meta" / "episodes.jsonl",
                 dataset_root / "meta" / "modality.json", CAMPAIGN_DIR / "campaign.json"):
        if path.is_file():
            inputs[str(path)] = {"sha256": sha256(path), "bytes": path.stat().st_size}
    for tag in ("psi0-rollout-1", "groot-rollout-1"):
        for name in ("raw/isaac.metrics.json", "raw/isaac.tracking.jsonl"):
            path = CAMPAIGN_DIR / tag / name
            inputs[str(path)] = {"sha256": sha256(path), "bytes": path.stat().st_size}
        for index in (1, 2):
            for name in ("tracking.jsonl", "bridge-telemetry.jsonl", "rollout.json"):
                path = CAMPAIGN_DIR / tag / "rollouts" / f"rollout-{index:02d}" / name
                if path.is_file():
                    inputs[str(path)] = {"sha256": sha256(path), "bytes": path.stat().st_size}
    return {
        "schema_version": SCHEMA_VERSION, "script_version": SCRIPT_VERSION,
        "script": str(Path(__file__).resolve()), "script_sha256": sha256(Path(__file__).resolve()),
        "command": (f"data/venvs/hf-datasets/bin/python scripts/{Path(__file__).name} support "
                    f"--dataset-root {dataset_root} --output-dir {out}"),
        "figures_command": (f"python3 scripts/{Path(__file__).name} figures "
                            f"--dataset-root {dataset_root} --output-dir {out}"),
        "constants": {"seed": args.seed, "frames_per_episode": args.frames_per_episode,
                      "neighbours": list(NEIGHBOURS),
                      "threshold_quantiles": list(THRESHOLD_QUANTILES),
                      "envelope_quantiles": list(ENVELOPE_QUANTILES),
                      "absolute_bands_rad": list(ABSOLUTE_BANDS_RAD),
                      "settle_tail_frames": SETTLE_TAIL_FRAMES,
                      "ridge_relative": RIDGE_RELATIVE,
                      "collection": COLLECTION, "prompt": PROMPT,
                      "wrong_task_controls": list(WRONG_TASKS),
                      "frames_valid_semantics": ("the first `frames_valid` frames of each episode "
                                                 "are the training-valid ones")},
        "inputs": inputs,
        "channel_order": list(CHANNELS),
        "groups": {name: [block.start, block.stop] for name, block in GROUPS.items()},
        "metric_definitions": {
            "knn_k1": "Euclidean distance to the nearest demo cloud frame, robust-scaled space",
            "mahalanobis": ("distance under the demo group covariance (ridge 1e-6 relative); its "
                            "quadratic form splits exactly into per-joint contributions"),
            "envelope": "per-channel demo p1/p99 and min/max bounds",
            "in_support": "distance <= the demonstration's own leave-one-episode-out p99",
        },
    }


# ------------------------------------------------------------------- figures


def stage_figures(args: argparse.Namespace) -> dict[str, Any]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(args.output_dir)
    figures = out / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    summary = read_json(out / "summary.json")
    calibration = read_json(out / "tables" / "loeo_calibration.json")["groups"]
    colours = {"psi0-rollout-1/rollout-01": "#1f77b4", "psi0-rollout-1/rollout-02": "#17becf",
               "groot-rollout-1/rollout-01": "#d62728", "groot-rollout-1/rollout-02": "#ff7f0e"}

    def csv_rows(name: str) -> list[dict[str, str]]:
        lines = (out / "tables" / name).read_text().splitlines()
        header = lines[0].split(",")
        return [dict(zip(header, line.split(","))) for line in lines[1:]]

    frame_rows = csv_rows("frame_support.csv")
    channel_rows = csv_rows("channel_contributions.csv")
    calibration_rows = csv_rows("channel_calibration.csv")
    channel_order = {row["channel"]: i for i, row in enumerate(calibration_rows)}

    # series available per rollout, and which one is the policy input
    available: dict[tuple[str, str], list[str]] = {}
    for row in frame_rows:
        tag, rollout, kind = row["series"].split("/")
        available.setdefault((tag, rollout), [])
        if kind not in available[(tag, rollout)]:
            available[(tag, rollout)].append(kind)
    primary_kind = {key: ("observation" if "observation" in kinds else "measured")
                    for key, kinds in available.items()}
    points: dict[str, list[tuple[float, float]]] = {}
    for row in frame_rows:
        points.setdefault(row["series"], []).append(
            (float(row["seconds_rel"]), float(row["mahalanobis_upper_body"])))
    produced: list[str] = []

    # 1. upper-body Mahalanobis distance over time, per model
    figure, axes = plt.subplots(1, 2, figsize=(14, 5.0), sharey=True)
    for axis, tag in zip(axes, sorted({key[0] for key in available})):
        for (series_tag, rollout), kinds in sorted(available.items()):
            if series_tag != tag:
                continue
            colour = colours.get(f"{series_tag}/{rollout}", "#666666")
            primary = primary_kind[(series_tag, rollout)]
            if "measured" in kinds and primary != "measured":
                series = sorted(points[f"{tag}/{rollout}/measured"])
                axis.plot([p[0] for p in series], [p[1] for p in series], color=colour,
                          linewidth=0.7, alpha=0.30)
            series = sorted(points[f"{tag}/{rollout}/{primary}"])
            axis.plot([p[0] for p in series], [p[1] for p in series], color=colour,
                      linewidth=1.1, label=f"{rollout} — policy input ({primary})")
            if "settle_raw" in kinds:
                series = sorted(points[f"{tag}/{rollout}/settle_raw"])
                axis.plot([p[0] for p in series], [p[1] for p in series], color=colour,
                          linewidth=1.8, alpha=0.85, label=f"{rollout} — pre-policy settle")
        for quantile, style in (("q95", ":"), ("q99", "--")):
            axis.axhline(calibration["upper_body"]["thresholds"][quantile]["mahalanobis"],
                         color="#444444", linestyle=style, linewidth=0.9,
                         label=f"demo LOEO {quantile}")
        axis.axvline(0.0, color="#2ca02c", linewidth=1.2)
        axis.set_yscale("log")
        axis.set_xlabel("seconds relative to policy start")
        axis.set_title(tag)
        axis.grid(alpha=0.25)
        axis.legend(fontsize=7, loc="lower left")
    axes[0].set_ylabel("Mahalanobis distance to the demo cloud\nupper body 15:43 (28 channels)")
    figure.suptitle("Distance from the demonstration state cloud: flat segments at the left are the "
                    "pre-policy SONIC hold", fontsize=11)
    figure.tight_layout()
    figure.savefig(figures / "fig01_distance_over_time.png", dpi=140)
    plt.close(figure)
    produced.append("fig01_distance_over_time.png")

    # 2. in-support fraction per channel group, both metrics
    groups = ["left_arm", "right_arm", "arms", "left_hand", "right_hand", "hands",
              "upper_body", "all_43d"]
    keys = sorted({f"{tag}/{rollout}/{kind}" for (tag, rollout), kinds in available.items()
                   for kind in [primary_kind[(tag, rollout)]]})
    figure, axes = plt.subplots(1, 2, figsize=(14, 5.0), sharey=True)
    width = 0.8 / max(len(keys), 1)
    for axis, (metric, label) in zip(axes, (("mahalanobis", "Mahalanobis"), ("knn_k1", "k-NN k=1"))):
        for position, key in enumerate(keys):
            tag, rollout, kind = key.split("/")
            active = summary["series"][tag][rollout]["rows"][kind]["active"]
            values = [active.get(group, {}).get(f"{metric}_in_support_q99_frac", float("nan"))
                      for group in groups]
            axis.bar(np.arange(len(groups)) + position * width, values, width,
                     color=colours[f"{tag}/{rollout}"], label=key)
        axis.set_xticks(np.arange(len(groups)) + 0.4 - width / 2)
        axis.set_xticklabels(groups, rotation=20, ha="right")
        axis.set_title(f"{label}: frame fraction within the demo LOEO p99")
        axis.grid(alpha=0.25, axis="y")
    axes[0].set_ylabel("in-support fraction, active window")
    axes[0].set_ylim(0, 1.05)
    axes[1].legend(fontsize=7, loc="lower left")
    figure.suptitle("In-support fraction by channel group — lower body and waist have no "
                    "demonstrated scale, so they are absent", fontsize=11)
    figure.tight_layout()
    figure.savefig(figures / "fig02_group_in_support.png", dpi=140)
    plt.close(figure)
    produced.append("fig02_group_in_support.png")

    # 3. arm and hand channels against the demonstrated envelope
    subset = [row for row in calibration_rows if int(row["channel_index"]) >= 15]
    positions = np.arange(len(subset))
    figure, axis = plt.subplots(figsize=(13, 5.2))
    axis.vlines(positions, [float(r["cloud_p01_rad"]) for r in subset],
                [float(r["cloud_p99_rad"]) for r in subset],
                color="#dddddd", linewidth=9, label="demonstrated p1–p99 envelope")
    axis.scatter(positions, [float(r["weighted_median_rad"]) for r in subset],
                 s=20, color="#222222", zorder=3, label="demonstrated weighted median")
    for key in keys:
        tag, rollout, kind = key.split("/")
        values = np.full(len(subset), np.nan)
        for row in channel_rows:
            if row["series"] == key and row["kind"] == kind:
                position = channel_order[row["channel"]] - 15
                if 0 <= position < len(subset):
                    values[position] = float(row["active_median_rad"])
        axis.plot(positions, values, linewidth=1.2, marker="o", markersize=3,
                  color=colours[f"{tag}/{rollout}"], label=key)
    axis.set_xticks(positions)
    axis.set_xticklabels([r["channel"].replace("_joint", "").replace("_", " ") for r in subset],
                         rotation=90, fontsize=6.5)
    axis.set_ylabel("median joint value over active frames [rad]")
    axis.set_title("Arm and hand chords against the demonstrated per-joint envelope "
                   "(channels 15–42, all layout-verified)")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=7.5, ncol=3)
    figure.tight_layout()
    figure.savefig(figures / "fig03_channel_deviation.png", dpi=140)
    plt.close(figure)
    produced.append("fig03_channel_deviation.png")

    write_json(figures / "figures.json", {
        "figures": produced, "primary_series": keys,
        "note": ("fig01–03 use the robust-scaled space for distances and raw radians for joint "
                 "envelopes; the 15 zero-variance channels are excluded from fig03")})
    return {"figures": produced}


# ---------------------------------------------------------------------- main


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("stage", choices=("support", "figures", "all"))
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--frames-per-episode", type=int, default=FRAMES_PER_EPISODE)
    args = parser.parse_args(argv)
    if args.stage in ("support", "all"):
        result = stage_support(args)
        print(json.dumps({
            "demo_valid_frames": result["demo_pool"]["valid_frames"],
            "demo_cloud_frames": result["demo_cloud"]["frames"],
            "degenerate_channels": result["channel_calibration"]["degenerate_channel_count"],
            "series": {tag: {rollout: sorted(record["rows"]) for rollout, record in rollouts.items()}
                       for tag, rollouts in result["series"].items()},
        }, indent=1))
    if args.stage in ("figures", "all"):
        print(json.dumps(stage_figures(args), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
