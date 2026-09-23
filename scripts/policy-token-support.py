#!/usr/bin/env python3
"""Are the rollouts' 64D SONIC tokens inside the BlockStacking demonstration token support?

One question only: the 64D motion tokens the two checkpoints drove the BlockStacking
rollouts with -- after the inference path has done whatever it does to them -- are they
inside the support of the token distribution the Task-0 demonstration copy contains, and
does the token departure track the high-palm segments of the rollout?

Scope, deliberately: no rollout, no model inference, no decoder replay, no controller or
scene change.  Read the recorded dataset, the recorded telemetry and the recorded
tracking, measure, and stop.

Method
------

* **What the telemetry fields are**, established from the pinned source *and* checked on
  the recordings (``telemetry_semantics``): for the PSI0 session the recorder writes
  ``target_action`` = the raw 80D model reply before any adaptation, and
  ``applied_action`` = the Protocol v4 payload the router forwarded to SONIC, whose
  ``token_state`` went through ``ActionAdapter.pack`` -> ``fsq_quantize``.  The two are
  joined on ``frame_index`` and the equality ``applied == fsq(target[:64])`` is asserted
  frame by frame, so "post-quantization" is a measured fact, not a reading of a name.
  The GR00T session is driven by NVIDIA's own VLA client, which publishes straight onto
  the router's input port: the bridge quantizes nothing, the router forwards the payload
  verbatim, and the recorder stores it verbatim.  GR00T therefore has **no raw target** in
  this telemetry; its published token is what it is, and every GR00T number below is
  computed on that published token.

* **Reference.**  The Task-0 dataset copy (``psi0-unitree-dex3-sonic-v1/train``),
  ``G1_Dex3_BlockStacking_Dataset``, every frame the converter marked valid, channel
  ``action.body_token_v1_1``.  Demonstrations are strongly time-correlated, so episodes
  carry equal weight: the support cloud is a fixed-seed balanced sample of
  ``FRAMES_PER_EPISODE`` frames per episode.

* **Distance in token space.**  The token is a symbol stream, not a joint vector: the
  recorded demonstrations sit exactly on the WBC FSQ grid (``[-0.625, 0.625]``, step
  ``0.0625``, 21 levels), so every metric is reported both in value units and in grid
  levels -- value L1, grid-level L1 (sum of |level difference|), Hamming (number of
  channels whose level differs) -- plus a robust-scaled Euclidean k-NN for continuity
  with the state-support experiment.  Nothing here treats "not exactly on the grid" as an
  error: the PSI0 raw output is expected to be continuous and is snapped by the client by
  contract, so the raw stream is judged against the rounding it undergoes, not against a
  grid membership rule.

* **Thresholds come only from the demonstration**, leave-one-episode-out: every cloud frame
  is scored against a cloud built without its own episode, and the p95/p99 of that
  distribution are the thresholds; the envelope pass rate is measured the same way.  No
  threshold is calibrated on a rollout.

* **Negative controls.**  The same metric is applied to (a) demonstration frames of the
  other 12 tasks scored against the Task-0 cloud -- the control that tests whether the
  metric has teeth at all -- (b) the rollout tokens against equal-sized clouds of each
  other task, (c) the rollout tokens with their 64 channels permuted.  If (a) and (c) are
  not clearly worse than the true pairing, an "in support" verdict is reported as
  non-discriminative instead of as evidence.

* **High palms.**  The rollout tokens are joined to the recorded tracking rows by wall
  clock, and the palm height of every row is computed by the already validated FK
  (imported from ``scripts/compare-blockstacking-dataset.py``, checked against Isaac to
  3e-7 m by experiment 03).  Support distance is then reported per palm-height segment and
  with its correlation to palm height.  Correlation only: nothing here claims a cause.

    data/venvs/groot-n17/bin/python scripts/policy-token-support.py support
    python3 scripts/policy-token-support.py figures
    python3 scripts/policy-token-support.py manifest
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np

SCHEMA_VERSION = 1
SCRIPT_VERSION = "policy-token-support.py/1.0.0"
REPO_ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN_DIR = REPO_ROOT / "data" / "outputs" / "blockstacking-debug"
DEFAULT_OUT = CAMPAIGN_DIR / "experiments" / "05-policy-token-support"
DEFAULT_DATASET = REPO_ROOT / "data" / "datasets" / "psi0-unitree-dex3-sonic-v1" / "train"
DEFAULT_VAL_DATASET = REPO_ROOT / "data" / "datasets" / "psi0-unitree-dex3-sonic-v1" / "val"
COMPARE_SCRIPT = REPO_ROOT / "scripts" / "compare-blockstacking-dataset.py"
DEFAULT_URDF = REPO_ROOT / "third_party" / "Psi0" / "real" / "assets" / "g1" / "g1_body29_hand14.urdf"

TOKEN_DIM = 64
FSQ_MIN, FSQ_MAX, FSQ_STEP = -0.625, 0.625, 0.0625
GRID_LEVELS = int(round((FSQ_MAX - FSQ_MIN) / FSQ_STEP)) + 1

COLLECTION = "G1_Dex3_BlockStacking_Dataset"
TASK_INDEX = 0
SEED = 20260921
#: Frames per episode in the reference cloud.  The same cloud is used for the thresholds
#: and for the rollout queries, so the comparison is density-matched by construction; the
#: density sensitivity is reported separately (``DENSITY_FRAMES_PER_EPISODE``).
FRAMES_PER_EPISODE = 40
#: Cloud used only for the density-sensitivity check (a sparser reference is a looser one).
DENSITY_FRAMES_PER_EPISODE = 20
#: Frames per episode scored in the leave-one-episode-out calibration (the reference side
#: stays the whole cloud).
CALIBRATION_FRAMES_PER_EPISODE = 20
#: Wrong-task clouds are built to the same frame budget as the reference cloud, so a
#: cloud-size difference cannot explain a distance difference between tasks.
WRONG_TASK_BUDGET = 1200
WRONG_TASK_MAX_EPISODES = 200
#: The channel-permutation and wrong-task controls run on every this many-th token; the
#: primary metrics always use the whole stream.
CONTROL_STRIDE = 2
THRESHOLD_QUANTILES = (0.95, 0.99)
ENVELOPE_QUANTILES = (0.01, 0.99)
RIDGE_RELATIVE = 1e-6
#: A published token further than this from its tracking row is not joined to a palm.
MAX_JOIN_OFFSET_S = 0.05
#: A permuted stream that is not at least this much further than the true stream would
#: mean the metric cannot tell the two apart at all.
CONTROL_RATIO_FLOOR = 1.05

CHANNELS = tuple(f"token_{index:02d}" for index in range(TOKEN_DIM))


class SupportError(RuntimeError):
    """The measurement cannot run; the message is meant for the operator."""


# ------------------------------------------------------------------ token space


def fsq_quantize(values: np.ndarray) -> np.ndarray:
    """The client's own snap onto the WBC FSQ grid (``psi0_bridge/actions.py``)."""
    array = np.asarray(values, dtype=np.float32)
    quantized = np.round(np.clip(array, FSQ_MIN, FSQ_MAX) / FSQ_STEP) * FSQ_STEP
    return np.clip(quantized, FSQ_MIN, FSQ_MAX).astype(np.float32)


def grid_level(values: np.ndarray) -> np.ndarray:
    """Nearest FSQ level index in ``0..GRID_LEVELS-1`` (0 == -0.625)."""
    array = np.asarray(values, dtype=float)
    level = np.round((np.clip(array, FSQ_MIN, FSQ_MAX) - FSQ_MIN) / FSQ_STEP)
    return level.astype(np.int16)


def grid_residual(values: np.ndarray) -> np.ndarray:
    """Distance from each value to its nearest point of the FSQ lattice, in value units.

    Measured against the unclipped lattice (``round(x/step)*step``): a value outside
    ``[-0.625, 0.625]`` is still off the lattice by the same rule, and how often values
    leave the range is a separate question, reported separately as
    ``raw_outside_fsq_range_frac``.
    """
    array = np.asarray(values, dtype=float)
    return np.abs(array - np.round(array / FSQ_STEP) * FSQ_STEP)


def grid_span(residual: np.ndarray, tolerance: float = 1e-6) -> float:
    """Fraction of channel values that sit exactly on the grid."""
    residual = np.asarray(residual, dtype=float)
    return float((residual <= tolerance).mean())


# -------------------------------------------------------------------- plumbing


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


def assert_jsonable(payload: Any, path: str = "summary") -> None:
    """Fail with the offending path instead of a bare serialization error.

    numpy scalars are fine (``write_json`` converts them); a whole array reaching the
    summary is a mistake in the analysis, not a formatting problem, so it is named.
    """
    if isinstance(payload, dict):
        for key, value in payload.items():
            assert_jsonable(value, f"{path}.{key}")
        return
    if isinstance(payload, (list, tuple)):
        for position, value in enumerate(payload):
            assert_jsonable(value, f"{path}[{position}]")
        return
    if payload is None or isinstance(payload, (str, bool, int, float, np.generic)):
        return
    raise SupportError(f"{path}: {type(payload).__name__} is not serializable")


def write_json(path: Path, payload: Any) -> None:
    assert_jsonable(payload)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, indent=1, sort_keys=True, default=float) + "\n",
                          encoding="utf-8")


def write_csv(path: Path, rows: Sequence[dict[str, Any]], columns: Sequence[str] | None = None) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        Path(path).write_text("", encoding="utf-8")
        return
    names = list(columns) if columns else list(rows[0].keys())
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


def describe(values: np.ndarray) -> dict[str, Any]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"n": 0}
    q = np.percentile(values, [0, 5, 25, 50, 75, 95, 99, 100])
    return {"n": int(values.size), "mean": float(values.mean()), "median": float(q[3]),
            "p05": float(q[1]), "p25": float(q[2]), "p75": float(q[4]), "p95": float(q[5]),
            "p99": float(q[6]), "min": float(q[0]), "max": float(q[7])}


def empirical_percentile(sample: np.ndarray, value: float) -> float | None:
    sample = np.asarray(sample, dtype=float)
    sample = sample[np.isfinite(sample)]
    if sample.size == 0 or not math.isfinite(value):
        return None
    return float((sample < value).mean() * 100.0 + 0.5 * (sample == value).mean() * 100.0)


def spearman(a: np.ndarray, b: np.ndarray) -> float | None:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    keep = np.isfinite(a) & np.isfinite(b)
    if keep.sum() < 3 or a[keep].std() == 0 or b[keep].std() == 0:
        return None
    order_a = np.argsort(np.argsort(a[keep]))
    order_b = np.argsort(np.argsort(b[keep]))
    return float(np.corrcoef(order_a, order_b)[0, 1])


# ----------------------------------------------------------------- dataclasses


def blockstacking_episodes(dataset_root: Path) -> list[dict[str, Any]]:
    selected = [entry for entry in read_jsonl(dataset_root / "meta" / "episodes.jsonl")
                if entry["source_collection"] == COLLECTION]
    selected.sort(key=lambda entry: entry["episode_index"])
    if not selected:
        raise SupportError(f"no {COLLECTION} episodes under {dataset_root}")
    return selected


def episode_path(dataset_root: Path, episode: int) -> Path:
    return dataset_root / "data" / f"chunk-{episode // 1000:03d}" / f"episode_{episode:06d}.parquet"


def episode_file_map(dataset_root: Path) -> dict[str, list[int]]:
    """Collection name -> its episodes, read from the copy's own episode list."""
    out: dict[str, list[int]] = {}
    for entry in read_jsonl(dataset_root / "meta" / "episodes.jsonl"):
        out.setdefault(str(entry["source_collection"]), []).append(int(entry["episode_index"]))
    for value in out.values():
        value.sort()
    return out


def task_index_of(dataset_root: Path, episodes: Sequence[int]) -> int:
    columns = read_parquet(episode_path(dataset_root, int(episodes[0])), ["task_index"])
    return int(columns["task_index"][0])


def load_episode_tokens(dataset_root: Path, episode: int, valid: int | None = None) -> np.ndarray:
    columns = read_parquet(episode_path(dataset_root, episode), ["action.body_token_v1_1"])
    tokens = np.asarray(columns["action.body_token_v1_1"], dtype=np.float32)
    if tokens.ndim != 2 or tokens.shape[1] != TOKEN_DIM:
        raise SupportError(f"episode {episode}: token column has shape {tokens.shape}")
    if valid is not None:
        tokens = tokens[:min(int(valid), tokens.shape[0])]
    return tokens


def load_demo_pool(dataset_root: Path, episodes: Sequence[dict[str, Any]]
                   ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Every valid demonstration frame: ``(tokens, owners, weights)``."""
    parts: list[np.ndarray] = []
    owners: list[int] = []
    weights: list[np.ndarray] = []
    for entry in episodes:
        episode = int(entry["episode_index"])
        tokens = load_episode_tokens(dataset_root, episode, int(entry.get("frames_valid", 0)))
        if tokens.shape[0] == 0:
            continue
        parts.append(tokens)
        owners.extend([episode] * tokens.shape[0])
        weights.append(np.full(tokens.shape[0], 1.0 / tokens.shape[0]))
    if not parts:
        raise SupportError("no valid demonstration frames")
    return (np.concatenate(parts, axis=0).astype(np.float32),
            np.asarray(owners, dtype=int), np.concatenate(weights))


def balanced_cloud(tokens: np.ndarray, owners: np.ndarray, episode_list: Sequence[int],
                   per_episode: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    picks: list[np.ndarray] = []
    labels: list[int] = []
    for episode in episode_list:
        rows = np.flatnonzero(owners == episode)
        if rows.size == 0:
            continue
        take = min(int(per_episode), rows.size)
        picks.append(rows[np.sort(rng.choice(rows.size, size=take, replace=False))])
        labels.extend([int(episode)] * take)
    if not picks:
        raise SupportError("balanced cloud sampled no frames")
    return tokens[np.concatenate(picks)], np.asarray(labels, dtype=int)


def sampled_task_cloud(dataset_root: Path, episodes: Sequence[int], budget: int,
                       seed: int, max_episodes: int) -> np.ndarray:
    """Balanced cloud from another task's episodes, without retaining the task."""
    rng = np.random.default_rng(seed)
    chosen = list(episodes)
    if len(chosen) > max_episodes:
        picked = np.sort(rng.choice(len(chosen), size=max_episodes, replace=False))
        chosen = [chosen[int(index)] for index in picked]
    per_episode = max(1, budget // max(1, len(chosen)))
    parts: list[np.ndarray] = []
    for episode in chosen:
        tokens = load_episode_tokens(dataset_root, int(episode))
        take = min(per_episode, tokens.shape[0])
        if take == 0:
            continue
        rows = np.sort(rng.choice(tokens.shape[0], size=take, replace=False))
        parts.append(tokens[rows])
    return np.concatenate(parts, axis=0).astype(np.float32)


# ----------------------------------------------------------------- distances


def _token_distance(query: np.ndarray, reference: np.ndarray, metric: str,
                    block: int = 128, reference_block: int = 2048) -> np.ndarray:
    """Per-query minimum distance to ``reference`` under one of the token metrics.

    Blocked over both sides so the pairwise tensor never has to exist: the token stream is
    64 wide, and a full query-by-cloud tensor at these sizes would be gigabytes.
    """
    query = np.asarray(query, dtype=np.float32)
    reference = np.asarray(reference, dtype=np.float32)
    if query.shape[0] == 0 or reference.shape[0] == 0:
        raise SupportError("distance needs at least one query and one reference row")
    out = np.empty(query.shape[0], dtype=float)
    grid_query = grid_level(query) if metric in ("grid_l1", "hamming") else None
    grid_reference = grid_level(reference) if metric in ("grid_l1", "hamming") else None
    for start in range(0, query.shape[0], block):
        stop = min(start + block, query.shape[0])
        best = np.full(stop - start, np.inf, dtype=np.float32)
        for lower in range(0, reference.shape[0], reference_block):
            upper = min(lower + reference_block, reference.shape[0])
            if metric in ("value_l1", "scaled_l2"):
                difference = query[start:stop, None, :] - reference[None, lower:upper, :]
                values = (np.abs(difference).sum(axis=2) if metric == "value_l1"
                          else np.sqrt((difference ** 2).sum(axis=2)))
            else:
                levels = (grid_query[start:stop, None, :].astype(np.int16)
                          - grid_reference[None, lower:upper, :].astype(np.int16))
                values = (np.abs(levels).sum(axis=2) if metric == "grid_l1"
                          else (levels != 0).sum(axis=2)).astype(np.float32)
            best = np.minimum(best, values.min(axis=1))
            del values
        out[start:stop] = best.astype(float)
    return out


def loeo_cloud_distance(cloud: np.ndarray, labels: np.ndarray, metric: str,
                        frames_per_episode: int | None = None, seed: int = 0) -> np.ndarray:
    """Leave-one-episode-out distance of every scored cloud frame to the rest of the cloud.

    The query side may be thinned (a fixed-seed subsample per episode) while the reference
    side stays the whole cloud: an episode's own frames are removed from the reference in
    every case, so no frame ever sees itself or its own episode.
    """
    rng = np.random.default_rng(seed)
    scores = np.empty(cloud.shape[0], dtype=float)
    scored = np.zeros(cloud.shape[0], dtype=bool)
    for episode in np.unique(labels):
        rows = np.flatnonzero(labels == episode)
        other = np.flatnonzero(labels != episode)
        if other.size == 0:
            continue
        take = rows
        if frames_per_episode is not None and rows.size > frames_per_episode:
            take = np.sort(rng.choice(rows, size=frames_per_episode, replace=False))
        scores[take] = _token_distance(cloud[take], cloud[other], metric)
        scored[take] = True
    return scores[scored]


# ----------------------------------------------------------------- calibration


def weighted_quantiles(values: np.ndarray, weights: np.ndarray, fractions: Sequence[float]) -> np.ndarray:
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


def robust_scales(pool: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    quantiles = weighted_quantiles(pool, weights, [0.5, ENVELOPE_QUANTILES[0], ENVELOPE_QUANTILES[1]])
    median = quantiles[0]
    mad = weighted_quantiles(np.abs(pool - median[None, :]), weights, [0.5])[0]
    degenerate = mad <= 0.0
    informative = 1.4826 * mad[~degenerate]
    floor = float(np.median(informative)) if informative.size else 1.0
    scale = np.where(degenerate, floor, 1.4826 * mad)
    return median, scale, degenerate


def envelope_loeo(cloud: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
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
        "note": ("bounds rebuilt from the episodes a held-out episode is not in; the bounds applied "
                 "to the rollouts come from the whole demonstration pool and are therefore wider"),
    }


def delta_loeo(cloud: np.ndarray, labels: np.ndarray, metric: str) -> dict[str, Any]:
    """Step-size calibration: each episode's own steps against the other episodes' steps."""
    steps: list[np.ndarray] = []
    step_labels: list[int] = []
    for episode in np.unique(labels):
        rows = np.flatnonzero(labels == episode)
        tokens = cloud[rows]
        if tokens.shape[0] < 2:
            continue
        if metric == "value_l1":
            step = np.abs(np.diff(tokens.astype(np.float64), axis=0)).sum(axis=1)
        elif metric == "grid_l1":
            step = np.abs(np.diff(grid_level(tokens).astype(np.int32), axis=0)).sum(axis=1)
        elif metric == "hamming":
            step = (np.diff(grid_level(tokens).astype(np.int32), axis=0) != 0).sum(axis=1)
        else:
            raise SupportError(f"unknown delta metric {metric!r}")
        steps.append(step)
        step_labels.extend([int(episode)] * step.size)
    all_steps = np.concatenate(steps)
    step_labels_array = np.asarray(step_labels, dtype=int)
    thresholds: dict[str, float] = {}
    exceed: dict[str, float] = {}
    for quantile in THRESHOLD_QUANTILES:
        key = f"q{int(quantile * 100)}"
        per_episode = [float(np.percentile(all_steps[step_labels_array != episode], quantile * 100.0))
                       for episode in np.unique(step_labels_array)]
        thresholds[key] = float(np.median(per_episode))
        # Fraction of a held-out episode's steps above the threshold built without it.
        above = np.concatenate([
            all_steps[step_labels_array == episode] >
            np.percentile(all_steps[step_labels_array != episode], quantile * 100.0)
            for episode in np.unique(step_labels_array)])
        exceed[key] = float(above.mean())
    return {"steps": int(all_steps.size), "stat": describe(all_steps), "thresholds": thresholds,
            "loeo_exceed_frac": exceed}


# ------------------------------------------------------------------- rollout

PSI0 = "psi0-rollout-1"
GROOT = "groot-rollout-1"
MODEL_LABEL = {PSI0: "psi0", GROOT: "groot"}


def session_rollouts() -> list[dict[str, Any]]:
    campaign = read_json(CAMPAIGN_DIR / "campaign.json")
    out: list[dict[str, Any]] = []
    for session in campaign["sessions"]:
        tag = session["session_dir"]
        if tag not in (PSI0, GROOT):
            continue
        for entry in session["rollouts"]:
            out.append({"tag": tag, "model": MODEL_LABEL[tag], "index": int(entry["index"]),
                        "dir": CAMPAIGN_DIR / tag / "rollouts" / f"rollout-{int(entry['index']):02d}"})
    return out


def telemetry_tokens(path: Path) -> dict[str, Any]:
    """Every action event of one rollout, split into raw targets and published tokens."""
    raw_rows: list[dict[str, Any]] = []
    applied_rows: list[dict[str, Any]] = []
    for row in read_jsonl(path):
        kind = row.get("kind")
        if kind == "target_action":
            raw_rows.append(row)
        elif kind == "applied_action":
            applied_rows.append(row)
    applied = np.asarray([row["fields"]["token_state"][0] for row in applied_rows], dtype=np.float32)
    applied_time = np.asarray([int(row["wall_time_ns"]) for row in applied_rows], dtype=np.int64)
    applied_index = np.asarray([int(row["fields"]["frame_index"][0]) for row in applied_rows], dtype=np.int64)
    raw = np.asarray([row["action"][:TOKEN_DIM] for row in raw_rows], dtype=np.float32) \
        if raw_rows else np.zeros((0, TOKEN_DIM), dtype=np.float32)
    raw_index = np.asarray([int(row.get("frame_index", -1)) for row in raw_rows], dtype=np.int64)
    raw_published = np.asarray([bool(row.get("published")) for row in raw_rows], dtype=bool)
    return {"raw": raw, "raw_index": raw_index, "raw_published": raw_published,
            "applied": applied, "applied_index": applied_index, "applied_time": applied_time}


def join_frames(raw: dict[str, Any]) -> dict[str, Any]:
    """The raw target stream against the published stream, joined on the frame index."""
    published = np.flatnonzero(raw["raw_published"])
    if published.size == 0:
        # No raw target stream at all (the GR00T path).  Every key is still present, so a
        # caller never has to guess whether a missing number means zero or "not recorded".
        return {"raw_targets": int(raw["raw"].shape[0]), "published_targets": 0,
                "published_without_applied": 0,
                "applied_without_published_target": int(raw["applied"].shape[0]),
                "raw_applied_index_match_frac": None,
                "applied_equals_fsq_raw_frac": None, "max_abs_applied_minus_fsq_raw": None}
    published_index = raw["raw_index"][published]
    lookup = {int(index): position for position, index in enumerate(raw["applied_index"])}
    matched = np.array([lookup.get(int(index), -1) for index in published_index], dtype=np.int64)
    found = matched >= 0
    exact: float | None = None
    max_difference: float | None = None
    if found.any():
        quantized = fsq_quantize(raw["raw"][published][found])
        difference = np.abs(raw["applied"][matched[found]] - quantized)
        max_difference = float(difference.max()) if difference.size else 0.0
        exact = float((difference == 0.0).all(axis=1).mean()) if difference.size else 0.0
    unmatched_applied = int(raw["applied"].shape[0] - len(set(published_index.tolist()) & set(lookup)))
    return {
        "raw_targets": int(raw["raw"].shape[0]),
        "published_targets": int(published.size),
        "published_without_applied": int((~found).sum()),
        "applied_without_published_target": unmatched_applied,
        "raw_applied_index_match_frac": float(found.mean()),
        "applied_equals_fsq_raw_frac": exact,
        "max_abs_applied_minus_fsq_raw": max_difference,
    }


def quantization_report(raw: np.ndarray) -> dict[str, Any]:
    """What the client's snap does to the raw model token.

    Three quantities, kept apart because they answer different questions: the distance
    from a raw value to the *nearest grid value* (how far off the lattice the model
    actually is), the displacement the snap applies (raw -> published), and the distance
    to the rounding boundary (a value sitting exactly on a tie is the only case where the
    round-half rule could matter).
    """
    if raw.shape[0] == 0:
        return {"available": False}
    quantized = fsq_quantize(raw)
    displacement = np.abs(quantized.astype(np.float64) - raw.astype(np.float64))
    nearest_distance = grid_residual(raw)
    # Margin to the rounding boundary, in level units: |x/step - floor(x/step) - 0.5|.
    index = raw.astype(np.float64) / FSQ_STEP
    boundary = np.abs(index - np.floor(index) - 0.5)
    clipped = np.abs(raw) > FSQ_MAX + 1e-9
    return {
        "available": True,
        "raw_tokens": int(raw.shape[0]),
        "raw_on_grid_frac": grid_span(nearest_distance),
        "raw_outside_fsq_range_frac": float(clipped.mean()),
        "raw_abs_max": float(np.abs(raw).max()),
        "nearest_grid_distance_levels": describe(nearest_distance.ravel() / FSQ_STEP),
        "nearest_grid_distance_value": describe(nearest_distance.ravel()),
        "rounding_boundary_margin_levels": describe(boundary.ravel()),
        "at_boundary_within_1e-6_levels_frac": float((boundary <= 1e-6).mean()),
        "exact_half_level_frac": float((np.abs(boundary - 0.5) <= 1e-9).mean()),
        "per_channel_displacement_value": describe(displacement.ravel()),
        "per_channel_displacement_levels": describe(displacement.ravel() / FSQ_STEP),
        "per_token_l1_displacement": describe(displacement.sum(axis=1)),
        "per_token_max_displacement": describe(displacement.max(axis=1)),
        "per_token_channels_moved": describe((displacement > 0).sum(axis=1)),
        "per_token_channels_moved_frac": float((displacement > 0).any(axis=1).mean()),
        "uniform_rounding_expectation": {
            "mean_abs_displacement_value": FSQ_STEP / 4.0,
            "mean_l1_displacement_per_token": FSQ_STEP / 4.0 * TOKEN_DIM,
            "note": ("a raw value uniformly distributed inside its cell snaps by E|U-0.5|*step = "
                     "step/4 = 0.015625; the measured displacement is compared against that"),
        },
    }


# -------------------------------------------------------------------- palms


def load_compare_module():
    specification = importlib.util.spec_from_file_location("compare_blockstacking_dataset", COMPARE_SCRIPT)
    if specification is None or specification.loader is None:
        raise SupportError(f"cannot load {COMPARE_SCRIPT}")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def tracking_series(path: Path) -> dict[str, np.ndarray]:
    rows = read_jsonl(path)
    if not rows:
        raise SupportError(f"no tracking rows under {path}")
    return {
        "wall_ns": np.asarray([int(row["wall_time_ns"]) for row in rows], dtype=np.int64),
        "sim_s": np.asarray([float(row["sim_s"]) for row in rows], dtype=float),
        "target": np.asarray([row["body_target"] for row in rows], dtype=float),
        "measured": np.asarray([row["body_measured"] for row in rows], dtype=float),
        "support": np.asarray([bool(row.get("support_active")) for row in rows], dtype=bool),
    }


def palm_z_series(series: dict[str, np.ndarray], urdf) -> dict[str, np.ndarray]:
    """Palm height per tracking row, pelvis frame, for the target and the measurement."""
    target = urdf.palms(series["target"])
    measured = urdf.palms(series["measured"])
    return {"target_left": target["left"][:, 2], "target_right": target["right"][:, 2],
            "measured_left": measured["left"][:, 2], "measured_right": measured["right"][:, 2]}


def join_to_rows(applied_time: np.ndarray, row_wall: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Nearest tracking row per published token, plus the offset it was joined at."""
    position = np.clip(np.searchsorted(row_wall, applied_time), 0, len(row_wall) - 1)
    lower = np.clip(position - 1, 0, len(row_wall) - 1)
    choose = np.where(np.abs(row_wall[lower] - applied_time) <= np.abs(row_wall[position] - applied_time),
                      lower, position)
    return choose, (applied_time - row_wall[choose]) / 1e9


# --------------------------------------------------------------------- stages


def measure_stream(tokens: np.ndarray, calibration: dict[str, Any]) -> dict[str, np.ndarray]:
    """Every support metric of one token stream."""
    cloud = calibration["cloud"]
    out = {metric: _token_distance(tokens, cloud, metric)
           for metric in ("value_l1", "grid_l1", "hamming", "scaled_l2")}
    informative = ~calibration["degenerate"]
    scaled_tokens = ((tokens.astype(np.float64) - calibration["median"][None, :])
                     / calibration["scale"][None, :])[:, informative]
    scaled_cloud = calibration["scaled_cloud"]
    out["scaled_l2"] = _token_distance(scaled_tokens, scaled_cloud, "scaled_l2")
    low, high = calibration["envelope_low"], calibration["envelope_high"]
    inside_channel = (tokens >= low[None, :]) & (tokens <= high[None, :])
    out["envelope_channels_outside"] = (~inside_channel).sum(axis=1)
    out["envelope_frame_outside"] = (~inside_channel).any(axis=1)
    out["envelope_channel_outside_frac"] = 1.0 - inside_channel.mean(axis=1)
    hull_low, hull_high = calibration["hull_low"], calibration["hull_high"]
    inside_hull = (tokens >= hull_low[None, :]) & (tokens <= hull_high[None, :])
    out["hull_frame_outside"] = (~inside_hull).any(axis=1)
    out["hull_channel_outside_frac"] = 1.0 - inside_hull.mean(axis=1)
    levels = grid_level(tokens)
    unseen = np.zeros(levels.shape, dtype=bool)
    for channel in range(TOKEN_DIM):
        unseen[:, channel] = ~calibration["levels_used"][channel][levels[:, channel]]
    out["channels_unseen_level"] = unseen.sum(axis=1)
    out["any_unseen_level"] = unseen.any(axis=1)
    out["level_outside_levels"] = unseen.sum(axis=1) / TOKEN_DIM
    return out


def step_metrics(tokens: np.ndarray) -> dict[str, Any]:
    if tokens.shape[0] < 2:
        return {"steps": 0}
    difference = np.diff(tokens.astype(np.float64), axis=0)
    levels = grid_level(tokens).astype(np.int32)
    level_difference = np.diff(levels, axis=0)
    return {
        "steps": int(tokens.shape[0] - 1),
        "value_l1": describe(np.abs(difference).sum(axis=1)),
        "value_l1_per_step_series": np.abs(difference).sum(axis=1),
        "grid_l1": describe(np.abs(level_difference).sum(axis=1)),
        "hamming": describe((level_difference != 0).sum(axis=1)),
        "hamming_series": (level_difference != 0).sum(axis=1),
        "changed_any_frac": float((level_difference != 0).any(axis=1).mean()),
        "changed_channels_mean": float((level_difference != 0).sum(axis=1).mean()),
        "repeated_token_frac": float((level_difference == 0).all(axis=1).mean()),
    }


def occupancy(tokens: np.ndarray, calibration: dict[str, Any]) -> dict[str, Any]:
    levels = grid_level(tokens).astype(np.int32)
    unique = np.unique(levels, axis=0).shape[0]
    per_channel = [int(np.unique(levels[:, channel]).size) for channel in range(TOKEN_DIM)]
    unseen = np.zeros(levels.shape, dtype=bool)
    for channel in range(TOKEN_DIM):
        unseen[:, channel] = ~calibration["levels_used"][channel][levels[:, channel]]
    return {
        "frames": int(tokens.shape[0]),
        "unique_level_vectors": int(unique),
        "unique_per_thousand_frames": float(unique / max(1, tokens.shape[0]) * 1000.0),
        "channels_at_single_level": int(sum(1 for value in per_channel if value == 1)),
        "channel_level_cardinality_median": float(np.median(per_channel)),
        "channel_level_cardinality_max": int(max(per_channel)),
        "channel_samples_outside_demonstrated_levels": float(unseen.mean()),
        "frames_with_any_undemonstrated_level": float(unseen.any(axis=1).mean()),
    }


def segment_report(support: dict[str, np.ndarray], target_height: np.ndarray,
                   measured_height: np.ndarray, relative: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    """Support metrics over one window; the palm columns are the joined values."""
    if mask.sum() == 0:
        return {"frames": 0}
    block = {
        "frames": int(mask.sum()),
        "seconds_rel": [float(relative[mask].min()), float(relative[mask].max())],
        "palm_target_z_median_m": float(np.median(target_height[mask])),
        "palm_measured_z_median_m": float(np.median(measured_height[mask])),
        "value_l1_median": float(np.median(support["value_l1"][mask])),
        "grid_l1_median": float(np.median(support["grid_l1"][mask])),
        "hamming_median": float(np.median(support["hamming"][mask])),
        "scaled_l2_median": float(np.median(support["scaled_l2"][mask])),
        "value_l1_p95": float(np.percentile(support["value_l1"][mask], 95)),
        "envelope_channel_outside_frac": float(support["envelope_channel_outside_frac"][mask].mean()),
        "envelope_frame_outside_frac": float(support["envelope_frame_outside"][mask].mean()),
        "hull_frame_outside_frac": float(support["hull_frame_outside"][mask].mean()),
        "undemonstrated_level_frac": float(support["level_outside_levels"][mask].mean()),
        "frames_with_undemonstrated_level_frac": float(support["any_unseen_level"][mask].mean()),
    }
    return block


def heldout_baseline(val_root: Path, cloud: np.ndarray, calibration: dict[str, Any],
                     frames_per_episode: int, seed: int) -> dict[str, Any]:
    """The val split's own BlockStacking episodes scored against the train cloud.

    These episodes are in neither the cloud nor its calibration, so this is exactly the
    operation applied to a rollout -- the difference is only that the source is a recorded
    demonstration of the same task.  Every threshold question ("is this token where a
    demonstration's token would be?") is therefore answered against a like-for-like
    reference rather than against a hand-picked number.
    """
    episodes = blockstacking_episodes(val_root)
    if not episodes:
        raise SupportError(f"the val copy {val_root} has no {COLLECTION} episodes")
    parts: list[np.ndarray] = []
    rng = np.random.default_rng(seed)
    for entry in episodes:
        tokens = load_episode_tokens(val_root, int(entry["episode_index"]),
                                     int(entry.get("frames_valid", 0)))
        if tokens.shape[0] == 0:
            continue
        take = min(frames_per_episode, tokens.shape[0])
        parts.append(tokens[np.sort(rng.choice(tokens.shape[0], size=take, replace=False))])
    held = np.concatenate(parts, axis=0).astype(np.float32)
    support = measure_stream(held, calibration)
    steps = step_metrics(held)
    envelope_channels_outside = support["envelope_channels_outside"].astype(float) / TOKEN_DIM
    summary_block = {
        "dataset": str(val_root), "episodes": len(episodes), "frames": int(held.shape[0]),
        "frames_per_episode": frames_per_episode, "seed": seed,
        "value_l1": describe(support["value_l1"]),
        "grid_l1": describe(support["grid_l1"]),
        "hamming": describe(support["hamming"]),
        "scaled_l2": describe(support["scaled_l2"]),
        "envelope_frame_outside_frac": float(support["envelope_frame_outside"].mean()),
        "hull_frame_outside_frac": float(support["hull_frame_outside"].mean()),
        "envelope_channel_outside_frac": float(envelope_channels_outside.mean()),
        "undemonstrated_level_frac": float(support["level_outside_levels"].mean()),
        "steps": {key: value for key, value in steps.items() if not key.endswith("_series")},
        "note": ("recorded demonstrations of the same task that were never used to build the "
                 "support cloud; the numbers below are the like-for-like reference for a rollout"),
    }
    series = {
        "value_l1": support["value_l1"].astype(np.float32),
        "grid_l1": support["grid_l1"].astype(np.float32),
        "hamming": support["hamming"].astype(np.float32),
        "scaled_l2": support["scaled_l2"].astype(np.float32),
        "envelope_channels_outside": support["envelope_channels_outside"].astype(np.float32),
        "envelope_frame_outside": support["envelope_frame_outside"],
        "hull_frame_outside": support["hull_frame_outside"],
        "level_outside_levels": support["level_outside_levels"].astype(np.float32),
        "token": held,
    }
    return {"summary": summary_block, "series": series}


def decide(model: str, result: dict[str, Any], calibration: dict[str, Any]) -> dict[str, Any]:
    """A/B/C/D for one model, from named numbers only.

    The reference is the held-out val episodes of the same task, scored against the same
    cloud with the same metrics.  Three metric families are judged separately, because they
    answer different questions and the evidence does not have to agree:

    * **distance** -- the nearest-neighbour distance of a token to the demonstration cloud
      (``value_l1`` and ``grid_l1``): inside the held-out p95 is "in", beyond the held-out
      p99 is "out", between is "borderline";
    * **envelope** -- the share of tokens with at least one channel outside the demonstration
      p1-p99 band: "in" up to twice the held-out rate, "out" beyond ten times it;
    * **dynamics** -- the median step between consecutive tokens against the held-out step
      distribution: "in" inside its p25-p75, otherwise "out" in the direction it moved;

    A: all three in.  B: all three out.  C: anything mixed, or the controls do not separate
    (a verdict of "in support" is then reported as C, because it is not evidence).
    D: no token telemetry for that model.
    """
    if not result.get("tokens"):
        return {"decision": "D", "reason": "no token telemetry for this model"}
    heldout = calibration["heldout"]
    windows = result["windows"]
    thresholds = calibration["thresholds"]

    distance_family: dict[str, Any] = {}
    for metric in ("value_l1", "grid_l1"):
        median = windows[f"{metric}_median"]
        family = ("in" if median <= heldout[metric]["p95"]
                  else "out" if median > heldout[metric]["p99"] else "borderline")
        distance_family[metric] = {
            "rollout_median": float(median),
            "heldout_median": float(heldout[metric]["median"]),
            "heldout_p95": float(heldout[metric]["p95"]),
            "heldout_p99": float(heldout[metric]["p99"]),
            "ratio_to_heldout_median": float(median / heldout[metric]["median"]),
            "ratio_to_heldout_p99": float(median / heldout[metric]["p99"]),
            "ratio_to_loeo_p99": float(median / thresholds["q99"][metric]),
            "verdict": family,
        }
    envelope_heldout_rate = float(heldout["envelope_frame_outside_frac"])
    envelope = {
        "rollout_frame_outside_frac": float(windows["envelope_frame_outside_frac"]),
        "heldout_frame_outside_frac": envelope_heldout_rate,
        "ratio": float(windows["envelope_frame_outside_frac"] / max(1e-9, envelope_heldout_rate)),
        "rollout_channel_outside_frac": float(windows["envelope_channel_outside_frac"]),
        "heldout_channel_outside_frac": float(heldout["envelope_channel_outside_frac"]),
        "rollout_hull_outside_frac": float(windows["hull_frame_outside_frac"]),
        "heldout_hull_outside_frac": float(heldout["hull_frame_outside_frac"]),
        "loeo_frame_outside_frac": float(1.0 - calibration["envelope_loeo"][
            "frame_all_channels_inside_p1_p99_frac"]),
    }
    # The min-max hull is the strictest envelope and the held-out demonstrations never
    # leave it, so the ratio is not defined there; the two raw rates are reported instead.
    envelope["hull_ratio"] = (None if envelope["heldout_hull_outside_frac"] == 0.0
                              else float(envelope["rollout_hull_outside_frac"]
                                         / envelope["heldout_hull_outside_frac"]))
    envelope["verdict"] = ("in" if envelope["ratio"] <= 2.0
                           else "out" if envelope["ratio"] > 10.0 else "borderline")
    steps_heldout = heldout["steps"]["value_l1"]
    dynamics = {
        "rollout_step_value_l1_median": float(windows["step_value_l1_median"]),
        "heldout_step_value_l1_median": float(steps_heldout["median"]),
        "heldout_step_value_l1_p25": float(steps_heldout["p25"]),
        "heldout_step_value_l1_p75": float(steps_heldout["p75"]),
        "rollout_step_grid_l1_median": float(windows["step_grid_l1_median"]),
        "heldout_step_grid_l1_median": float(heldout["steps"]["grid_l1"]["median"]),
        "rollout_repeated_token_frac": float(result.get("repeated_token_frac", float("nan"))),
        "heldout_repeated_token_frac": float(heldout["steps"]["repeated_token_frac"]),
    }
    # Tested against the held-out interquartile range, not its p5-p95: the demonstration
    # step distribution is wide, so a p5-p95 band would call almost any step size "inside"
    # and would not answer the question being asked.
    if dynamics["heldout_step_value_l1_p25"] <= dynamics["rollout_step_value_l1_median"] \
            <= dynamics["heldout_step_value_l1_p75"]:
        dynamics["verdict"] = "in"
        dynamics["direction"] = "within"
    elif dynamics["rollout_step_value_l1_median"] < dynamics["heldout_step_value_l1_p25"]:
        dynamics["verdict"] = "out"
        dynamics["direction"] = "slower"
    else:
        dynamics["verdict"] = "out"
        dynamics["direction"] = "faster"
    families = {"distance": distance_family, "envelope": envelope, "dynamics": dynamics}

    control_permuted = result["controls"].get("permuted_median_ratio")
    control_wrong_task = result["controls"].get("wrong_task_median_ratio")
    discriminative = (
        (control_permuted is None or control_permuted >= CONTROL_RATIO_FLOOR)
        and (control_wrong_task is None or control_wrong_task >= CONTROL_RATIO_FLOOR))
    verdicts = [distance_family["value_l1"]["verdict"], distance_family["grid_l1"]["verdict"],
                envelope["verdict"], dynamics["verdict"]]
    if all(item == "in" for item in verdicts):
        decision = "A"
    elif all(item == "out" for item in verdicts):
        decision = "B"
    else:
        decision = "C"
    if not discriminative and decision == "A":
        decision = "C"
    return {
        "decision": decision,
        "families": families,
        "metric_power": result.get("metric_power"),
        "controls_discriminative": bool(discriminative),
        "rule": ("A: all three families inside the held-out reference; B: all three outside; "
                 "C: mixed, or a non-discriminative control set"),
        "note": ("the held-out reference is the val split's own BlockStacking episodes -- "
                 "recorded demonstrations of the same task that the cloud never saw; 'D' would "
                 "mean no token telemetry exists for the model at all"),
    }


def stage_support(args: argparse.Namespace) -> int:
    dataset_root = Path(args.dataset_root)
    out = Path(args.output_dir)
    tables = out / "tables"
    series_dir = out / "series"
    tables.mkdir(parents=True, exist_ok=True)
    series_dir.mkdir(parents=True, exist_ok=True)
    heldout_series: dict[str, np.ndarray] = {}

    summary: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "script_version": SCRIPT_VERSION,
        "question": ("are the 64D SONIC tokens the BlockStacking rollouts were driven with inside "
                     "the Task-0 demonstration token support, and does the departure track high palms?"),
        "seed": args.seed,
    }

    # ---------------------------------------------------------- 1. reference
    episodes = blockstacking_episodes(dataset_root)
    pool, owners, weights = load_demo_pool(dataset_root, episodes)
    median, scale, degenerate = robust_scales(pool, weights)
    envelope = weighted_quantiles(pool, weights, ENVELOPE_QUANTILES)
    cloud, labels = balanced_cloud(pool, owners, [int(e["episode_index"]) for e in episodes],
                                   args.frames_per_episode, args.seed)
    levels_used = [np.zeros(GRID_LEVELS, dtype=bool) for _ in range(TOKEN_DIM)]
    pool_levels = grid_level(pool)
    for channel in range(TOKEN_DIM):
        levels_used[channel][np.unique(pool_levels[:, channel].astype(np.int32))] = True
    summary["demo_reference"] = {
        "dataset": str(dataset_root), "collection": COLLECTION, "task_index": TASK_INDEX,
        "episodes": len(episodes),
        "valid_frames": int(pool.shape[0]),
        "episode_frames_min": int(min(e["frames_valid"] for e in episodes)),
        "episode_frames_max": int(max(e["frames_valid"] for e in episodes)),
        "cloud_frames": int(cloud.shape[0]), "cloud_per_episode": args.frames_per_episode,
        "cloud_seed": args.seed,
        "weighting": "per-frame weight 1/frames_valid so every episode carries total weight 1",
        "grid": {"min": FSQ_MIN, "max": FSQ_MAX, "step": FSQ_STEP, "levels": GRID_LEVELS},
        "pool_on_grid_frac": grid_span(grid_residual(pool)),
        "pool_unique_level_vectors": occupancy(pool, {"levels_used": levels_used}),
        "pool_value_min": float(pool.min()), "pool_value_max": float(pool.max()),
        "degenerate_channels": [CHANNELS[i] for i in np.flatnonzero(degenerate)],
        "degenerate_channel_count": int(degenerate.sum()),
        "weighted_median_value": {CHANNELS[i]: float(median[i]) for i in range(TOKEN_DIM)},
        "weighted_robust_scale": {CHANNELS[i]: float(scale[i]) for i in range(TOKEN_DIM)},
        "channels_unused_levels": {CHANNELS[i]: int((~levels_used[i]).sum()) for i in range(TOKEN_DIM)},
    }
    write_json(tables / "demo_token_stats.json", summary["demo_reference"])

    # ------------------------------------------------------- 2. calibration
    loeo_values: dict[str, np.ndarray] = {}
    for metric in ("value_l1", "grid_l1", "hamming", "scaled_l2"):
        loeo_values[metric] = loeo_cloud_distance(
            cloud, labels, metric, frames_per_episode=args.calibration_frames_per_episode,
            seed=args.seed)
    thresholds: dict[str, dict[str, float]] = {}
    for quantile in THRESHOLD_QUANTILES:
        key = f"q{int(quantile * 100)}"
        thresholds[key] = {metric: float(np.percentile(values, quantile * 100.0))
                           for metric, values in loeo_values.items()}

    # Density sensitivity: a sparser cloud is a looser reference.  The same leave-one-
    # episode-out construction is repeated on it so the verdict can be checked against two
    # reference densities instead of one.
    sparse_cloud, sparse_labels = balanced_cloud(
        pool, owners, [int(e["episode_index"]) for e in episodes],
        args.density_frames_per_episode, args.seed + 7)
    sparse_loeo = {metric: loeo_cloud_distance(sparse_cloud, sparse_labels, metric,
                                               frames_per_episode=args.density_frames_per_episode,
                                               seed=args.seed + 7)
                   for metric in ("value_l1", "grid_l1")}
    sparse_thresholds = {f"q{int(quantile * 100)}": {
        metric: float(np.percentile(values, quantile * 100.0))
        for metric, values in sparse_loeo.items()} for quantile in THRESHOLD_QUANTILES}
    envelope_calibration = envelope_loeo(cloud, labels)
    delta_calibration = {metric: delta_loeo(cloud, labels, metric)
                         for metric in ("value_l1", "grid_l1", "hamming")}
    calibration = {
        "median": median, "scale": scale, "degenerate": degenerate,
        "envelope_low": envelope[0], "envelope_high": envelope[1],
        "hull_low": pool.min(axis=0), "hull_high": pool.max(axis=0),
        "cloud": cloud, "labels": labels, "levels_used": levels_used,
        "scaled_cloud": ((cloud.astype(np.float64) - median[None, :]) / scale[None, :])[:, ~degenerate],
        "thresholds": thresholds,
        "sparse_cloud": sparse_cloud, "sparse_thresholds": sparse_thresholds,
        "sparse_loeo": sparse_loeo,
        "envelope_loeo": envelope_calibration,
        "delta_loeo": delta_calibration,
        "loeo_values": loeo_values,
    }
    # Held-out baseline: the val split's own BlockStacking episodes never entered the
    # support cloud, so scoring them against it is the same operation as scoring a rollout
    # and needs no leave-one-out construction.  This is the baseline the verdict uses.
    heldout = heldout_baseline(Path(args.val_dataset_root), cloud, calibration,
                               args.frames_per_episode, args.seed)
    calibration["heldout"] = heldout["summary"]
    summary["heldout_baseline"] = heldout["summary"]
    write_json(tables / "heldout_baseline.json", heldout["summary"])
    for key, values in heldout["series"].items():
        heldout_series[f"heldout__{key}"] = values

    summary["loeo_calibration"] = {
        "thresholds": thresholds,
        "knn_loeo": {metric: describe(values) for metric, values in loeo_values.items()},
        "envelope": envelope_calibration,
        "steps": {metric: {"steps": block["steps"], "stat": block["stat"],
                           "thresholds": block["thresholds"], "loeo_exceed_frac": block["loeo_exceed_frac"]}
                  for metric, block in delta_calibration.items()},
        "density_sensitivity": {
            "sparse_cloud_frames": int(sparse_cloud.shape[0]),
            "sparse_cloud_per_episode": args.density_frames_per_episode,
            "sparse_thresholds": sparse_thresholds,
            "loeo": {metric: describe(values) for metric, values in sparse_loeo.items()},
            "note": ("a sparser reference cloud is a looser support: the rollout distance is "
                     "therefore reported against both densities and the verdict must survive both"),
        },
        "note": "every threshold is built from the cloud without the episode being scored",
    }
    np.savez_compressed(
        tables / "calibration.npz",
        thresholds=np.array([thresholds[key][metric] for key in sorted(thresholds)
                             for metric in sorted(thresholds[key])]),
        **{f"loeo_{metric}": values for metric, values in loeo_values.items()},
        envelope_low=envelope[0], envelope_high=envelope[1],
        hull_low=calibration["hull_low"], hull_high=calibration["hull_high"])
    write_json(tables / "loeo_calibration.json",
               {"thresholds": thresholds, "envelope": envelope_calibration,
                "knn_loeo": {metric: describe(values) for metric, values in loeo_values.items()},
                "steps": summary["loeo_calibration"]["steps"],
                "constants": {"seed": args.seed, "frames_per_episode": args.frames_per_episode,
                              "threshold_quantiles": list(THRESHOLD_QUANTILES),
                              "envelope_quantiles": list(ENVELOPE_QUANTILES)}})

    # ---------------------------------------------- 3. other-task reference
    compare = load_compare_module()
    urdf = compare.Urdf(Path(args.urdf))
    collections = episode_file_map(dataset_root)
    wrong_tasks: dict[str, Any] = {}
    for collection, episode_list in sorted(collections.items()):
        if collection == COLLECTION:
            continue
        task_index = task_index_of(dataset_root, episode_list)
        cloud_other = sampled_task_cloud(dataset_root, episode_list, args.wrong_task_budget,
                                         args.seed + task_index, args.wrong_task_max_episodes)
        wrong_tasks[collection] = {"task_index": task_index, "episodes": len(episode_list),
                                   "cloud_frames": int(cloud_other.shape[0]), "cloud": cloud_other}
    summary["wrong_task_reference"] = {
        name: {"task_index": block["task_index"], "episodes": block["episodes"],
               "cloud_frames": block["cloud_frames"]}
        for name, block in wrong_tasks.items()}
    summary["wrong_task_reference"]["note"] = (
        "equal frame budget per task and a balanced per-episode sample, so a cloud-size "
        "difference cannot produce the distance difference")

    # metric-power control: other-task demonstration frames scored against the Task-0 cloud.
    # The reference is the full cloud (28.5k frames -- the same size the leave-one-episode-out
    # calibration uses), so the two distance distributions are comparable.
    metric_power: dict[str, Any] = {}
    for name, block in wrong_tasks.items():
        metric_power[name] = {
            "median_value_l1": float(np.median(_token_distance(block["cloud"], cloud, "value_l1"))),
            "median_grid_l1": float(np.median(_token_distance(block["cloud"], cloud, "grid_l1"))),
            "median_hamming": float(np.median(_token_distance(block["cloud"], cloud, "hamming"))),
        }
    summary["metric_power_control"] = {
        "rule": ("each other task's demonstration tokens are scored against the Task-0 cloud and "
                 "compared with the same metric's leave-one-episode-out value for Task-0 frames; the "
                 "metric has teeth only if the other tasks are clearly further"),
        "reference_cloud_frames": int(cloud.shape[0]),
        "task0_loeo": {metric: describe(values) for metric, values in loeo_values.items()},
        "other_task": metric_power,
        "median_ratio_vs_task0_loeo_median": {
            name: {"value_l1": block["median_value_l1"] / float(np.median(loeo_values["value_l1"])),
                   "grid_l1": block["median_grid_l1"] / float(np.median(loeo_values["grid_l1"])),
                   "hamming": block["median_hamming"] / float(np.median(loeo_values["hamming"]))}
            for name, block in metric_power.items()},
    }

    # ---------------------------------------------------------- 4. rollouts
    series: dict[str, np.ndarray] = {}
    window_rows: list[dict[str, Any]] = []
    control_rows: list[dict[str, Any]] = []
    palm_rows: list[dict[str, Any]] = []
    quant_rows: list[dict[str, Any]] = []
    semantics_rows: list[dict[str, Any]] = []
    for entry in session_rollouts():
        tag, model, index = entry["tag"], entry["model"], entry["index"]
        rollout_dir = entry["dir"]
        label = f"{tag}-r{index:02d}"
        print(f"[support] {label}", flush=True)
        telemetry = telemetry_tokens(rollout_dir / "bridge-telemetry.jsonl")
        join = join_frames(telemetry)
        applied_residual = grid_residual(telemetry["applied"])
        semantics_rows.append({
            "tag": tag, "model": model, "rollout": index,
            "raw_target_events": join["raw_targets"], "applied_events": int(telemetry["applied"].shape[0]),
            "published_targets": join["published_targets"],
            "published_without_applied": join["published_without_applied"],
            "applied_without_published_target": join["applied_without_published_target"],
            "raw_applied_index_match_frac": join["raw_applied_index_match_frac"],
            "applied_equals_fsq_raw_frac": join["applied_equals_fsq_raw_frac"],
            "max_abs_applied_minus_fsq_raw": join["max_abs_applied_minus_fsq_raw"],
            "raw_on_grid_frac": grid_span(grid_residual(telemetry["raw"])) if join["raw_targets"] else None,
            "applied_on_grid_frac": grid_span(applied_residual),
            "applied_outside_fsq_range_frac": float((np.abs(telemetry["applied"]) > FSQ_MAX + 1e-9).mean()),
            "applied_values_per_frame": float(TOKEN_DIM),
            "target_action_field": "raw model reply, pre-adaptation" if join["raw_targets"] else None,
            "applied_action_field": ("Protocol v4 payload after fsq_quantize"
                                     if join["applied_equals_fsq_raw_frac"] == 1.0
                                     else "Protocol v4 payload, forwarded verbatim (no bridge quantization)"),
        })
        if join["raw_targets"]:
            report = quantization_report(telemetry["raw"])
            report.update({"tag": tag, "model": model, "rollout": index})
            quant_rows.append({key: value for key, value in report.items()
                               if not isinstance(value, dict)})
            summary.setdefault("quantization", {})[label] = report

        tokens = telemetry["applied"]
        times = telemetry["applied_time"]
        support = measure_stream(tokens, calibration)
        steps = step_metrics(tokens)
        occ = occupancy(tokens, calibration)

        tracking = tracking_series(rollout_dir / "tracking.jsonl")
        palm = palm_z_series(tracking, urdf)
        rows_position, offset = join_to_rows(times, tracking["wall_ns"])
        rollout_json = read_json(rollout_dir / "rollout.json")
        onset_ns = int(rollout_json["start_wall_ns"]) + int(
            rollout_json["onset"]["seconds_after_start"] * 1e9)
        relative_onset = (times - onset_ns) / 1e9
        height = np.maximum(palm["target_left"], palm["target_right"])[rows_position]
        measured_height = np.maximum(palm["measured_left"], palm["measured_right"])[rows_position]
        joined = np.abs(offset) <= MAX_JOIN_OFFSET_S
        usable = np.isfinite(height)

        windows: dict[str, np.ndarray] = {
            "all": np.ones(tokens.shape[0], dtype=bool),
            "active": relative_onset >= 0.0,
            "pre_onset": relative_onset < 0.0,
        }
        if usable.sum() > 10:
            height_cut = float(np.median(height[usable]))
            decile = float(np.percentile(height[usable], 90))
            windows["low_palm_half"] = usable & joined & (height <= height_cut)
            windows["high_palm_half"] = usable & joined & (height > height_cut)
            windows["top_palm_decile"] = usable & joined & (height >= decile)

        record: dict[str, Any] = {
            "tag": tag, "model": model, "rollout": index, "tokens": int(tokens.shape[0]),
            "on_grid_frac": grid_span(applied_residual), "occupancy": occ,
            "steps": {key: value for key, value in steps.items() if not key.endswith("_series")},
            "steps_series": {key: value for key, value in steps.items() if key.endswith("_series")},
            "join": {"tracking_rows": int(tracking["wall_ns"].size),
                     "offset_s": describe(offset), "joined_frac": float(joined.mean()),
                     "join_tolerance_s": MAX_JOIN_OFFSET_S},
            "windows": {}, "correlations": {}, "controls": {}}
        for name, mask in windows.items():
            record["windows"][name] = segment_report(support, height, measured_height,
                                                     relative_onset, mask)
        for height_name, values in (("target_palm_z", height), ("measured_palm_z", measured_height)):
            keep = usable & joined
            for metric in ("value_l1", "grid_l1", "hamming", "scaled_l2"):
                if keep.sum() > 10 and values[keep].std() > 0:
                    record["correlations"][f"{height_name}_vs_{metric}"] = {
                        "pearson": float(np.corrcoef(values[keep], support[metric][keep])[0, 1]),
                        "spearman": spearman(values[keep], support[metric][keep]),
                        "frames": int(keep.sum()),
                    }
        for name in ("low_palm_half", "high_palm_half", "top_palm_decile"):
            mask = windows.get(name)
            if mask is None or mask.sum() < 5:
                continue
            palm_rows.append({
                "tag": tag, "model": model, "rollout": index, "segment": name,
                "frames": int(mask.sum()),
                "palm_target_z_median_m": float(np.median(height[mask])),
                "palm_measured_z_median_m": float(np.median(measured_height[mask])),
                "value_l1_median": float(np.median(support["value_l1"][mask])),
                "grid_l1_median": float(np.median(support["grid_l1"][mask])),
                "hamming_median": float(np.median(support["hamming"][mask])),
                "envelope_frame_outside_frac": float(support["envelope_frame_outside"][mask].mean()),
                "undemonstrated_level_frac": float(support["level_outside_levels"][mask].mean()),
            })
        low_mask = windows.get("low_palm_half")
        high_mask = windows.get("high_palm_half")
        if low_mask is not None and low_mask.sum() > 5 and high_mask is not None and high_mask.sum() > 5:
            record["high_vs_low_palm"] = {
                "definition": "median split of max(target left, right) palm z over the joined tokens",
                "split_m": float(np.median(height[usable])),
                "frames_low": int(low_mask.sum()), "frames_high": int(high_mask.sum()),
                "value_l1_median_low": float(np.median(support["value_l1"][low_mask])),
                "value_l1_median_high": float(np.median(support["value_l1"][high_mask])),
                "grid_l1_median_low": float(np.median(support["grid_l1"][low_mask])),
                "grid_l1_median_high": float(np.median(support["grid_l1"][high_mask])),
                "hamming_median_low": float(np.median(support["hamming"][low_mask])),
                "hamming_median_high": float(np.median(support["hamming"][high_mask])),
                "envelope_outside_low": float(support["envelope_frame_outside"][low_mask].mean()),
                "envelope_outside_high": float(support["envelope_frame_outside"][high_mask].mean()),
            }

        # controls ---------------------------------------------------------
        # The primary metrics above use every token; the controls are computed on a strided
        # subsample because they only compare distributions, and the stride is reported.
        stride = max(1, args.control_stride)
        tokens_control = tokens[::stride]
        rng = np.random.default_rng(args.seed + index)
        permutation = rng.permutation(TOKEN_DIM)
        permuted = _token_distance(tokens_control[:, permutation], cloud, "value_l1")
        reverse_permuted = _token_distance(tokens_control[:, ::-1], cloud, "value_l1")
        paired_permuted = _token_distance(
            tokens_control.reshape(tokens_control.shape[0], 2, 32)[:, ::-1, :].reshape(-1, 64),
            cloud, "value_l1")
        rolled = _token_distance(np.roll(tokens_control, 137, axis=0), cloud, "value_l1")
        module_permuted = _token_distance(np.roll(tokens_control, 1, axis=1), cloud, "value_l1")
        wrong_task_distances = {name: _token_distance(tokens_control, block["cloud"], "value_l1")
                                for name, block in wrong_tasks.items()}
        # Density sensitivity on the whole stream, against the sparser cloud.
        sparse_distance = _token_distance(tokens[::max(1, stride)], calibration["sparse_cloud"],
                                          "value_l1")
        true_median = float(np.median(support["value_l1"]))
        best_wrong = min(wrong_task_distances.items(), key=lambda item: float(np.median(item[1])))
        record["controls"] = {
            "control_stride": stride,
            "true_value_l1_median": true_median,
            "channel_permuted_median": float(np.median(permuted)),
            "channel_permuted_ratio": float(np.median(permuted) / true_median),
            "channel_reversed_median": float(np.median(reverse_permuted)),
            "channel_reversed_ratio": float(np.median(reverse_permuted) / true_median),
            "channel_half_swapped_ratio": float(np.median(paired_permuted) / true_median),
            "channel_rolled_ratio": float(np.median(module_permuted) / true_median),
            "time_shifted_137_ratio": float(np.median(rolled) / true_median),
            "permuted_median_ratio": float(np.median(permuted) / true_median),
            "closest_wrong_task": best_wrong[0],
            "closest_wrong_task_median": float(np.median(best_wrong[1])),
            "closest_wrong_task_ratio": float(np.median(best_wrong[1]) / true_median),
            "wrong_task_median_ratio": float(np.median(best_wrong[1]) / true_median),
            "sparse_cloud_median": float(np.median(sparse_distance)),
            "sparse_cloud_ratio_to_sparse_p99": float(
                np.median(sparse_distance) / calibration["sparse_thresholds"]["q99"]["value_l1"]),
            "sparse_cloud_ratio_to_dense_p99": float(
                np.median(sparse_distance) / calibration["thresholds"]["q99"]["value_l1"]),
            "per_wrong_task": {name: {"median": float(np.median(values)),
                                      "ratio": float(np.median(values) / true_median)}
                               for name, values in wrong_task_distances.items()},
            "note": "each other task's cloud has the same frame budget as the reference cloud",
        }
        control_rows.append({
            "tag": tag, "model": model, "rollout": index,
            **{key: value for key, value in record["controls"].items() if not isinstance(value, dict)},
        })

        for key in ("value_l1", "grid_l1", "hamming", "scaled_l2", "envelope_channels_outside",
                    "envelope_channel_outside_frac", "envelope_frame_outside", "hull_frame_outside",
                    "channels_unseen_level", "level_outside_levels", "any_unseen_level"):
            series[f"{label}__{key}"] = support[key].astype(np.float32)
        series[f"{label}__time_rel"] = relative_onset.astype(np.float32)
        series[f"{label}__token"] = tokens
        series[f"{label}__palm_target_z"] = height.astype(np.float32)
        series[f"{label}__palm_measured_z"] = measured_height.astype(np.float32)
        series[f"{label}__joined"] = joined
        for key, values in record["steps_series"].items():
            series[f"{label}__step_{key.replace('_series', '')}"] = np.asarray(values, dtype=np.float32)
        record.pop("steps_series")
        series[f"{label}__meta"] = np.array(json.dumps(
            {"tag": tag, "model": model, "rollout": index,
             "onset_seconds_after_start": float(rollout_json["onset"]["seconds_after_start"])}))
        for name in windows:
            window_rows.append({"tag": tag, "model": model, "rollout": index, "window": name,
                                **record["windows"][name]})
        summary.setdefault("rollouts", {})[label] = record

    summary["telemetry_semantics"] = {
        "rows": semantics_rows,
        "psi0": ("target_action = raw model reply before adaptation, recorded with its own frame "
                 "index and a published flag; applied_action = the Protocol v4 payload the router "
                 "forwarded, whose token_state is the FSQ-snapped token"),
        "groot": ("the VLA client publishes onto the router's input port and the router forwards the "
                  "payload verbatim, so the telemetry has no raw target for this model; the published "
                  "token is unquantized (no quantization step exists on that path)"),
        "source": ["src/humanoid_lab/psi0_bridge/actions.py", "src/humanoid_lab/psi0_bridge/session.py",
                   "src/humanoid_lab/psi0_bridge/action_router.py", "src/humanoid_lab/psi0_bridge/telemetry.py",
                   "src/humanoid_lab/psi0_bridge/groot_backend.py"],
    }
    write_json(tables / "telemetry_semantics.json", summary["telemetry_semantics"])
    if quant_rows:
        write_csv(tables / "quantization.csv", quant_rows)

    # ------------------------------------------------------- 5. summary rows
    summary["support_by_window"] = window_rows
    write_csv(tables / "support_by_window.csv", window_rows)
    write_csv(tables / "support_controls.csv", control_rows)
    write_csv(tables / "palm_correlation.csv", palm_rows)
    np.savez_compressed(series_dir / "token_support.npz", **series, **heldout_series)

    # ---------------------------------------------------------------- 6. A-D
    decisions: dict[str, Any] = {}
    for model in ("psi0", "groot"):
        records = {label: record for label, record in summary["rollouts"].items()
                   if record["model"] == model}
        if not records:
            decisions[model] = {"decision": "D", "reason": "no rollout telemetry for this model"}
            continue
        # Model-level numbers are the median over rollouts of the window medians.
        medians = {key: float(np.median([record["windows"]["active"][key] for record in records.values()]))
                   for key in ("value_l1_median", "grid_l1_median", "hamming_median",
                               "scaled_l2_median", "envelope_frame_outside_frac",
                               "envelope_channel_outside_frac", "hull_frame_outside_frac",
                               "undemonstrated_level_frac")}
        medians["step_value_l1_median"] = float(np.median(
            [record["steps"]["value_l1"]["median"] for record in records.values()]))
        medians["step_grid_l1_median"] = float(np.median(
            [record["steps"]["grid_l1"]["median"] for record in records.values()]))
        controls = {key: float(np.median([record["controls"][key] for record in records.values()]))
                    for key in ("channel_permuted_ratio", "channel_reversed_ratio", "channel_rolled_ratio",
                                "time_shifted_137_ratio", "closest_wrong_task_ratio")}
        corpus = {"tokens": int(sum(record["tokens"] for record in records.values())),
                  "rollouts": len(records), "windows": medians, "controls": controls}
        corpus["controls"]["note"] = ("medians over rollouts of the per-rollout control ratios; the "
                                      "wrong-task ratio uses each rollout's closest other task")
        corpus["high_vs_low_palm"] = [record.get("high_vs_low_palm") for record in records.values()]
        corpus["correlations"] = {record["tag"]: record["correlations"] for record in records.values()}
        corpus["steps"] = {record["tag"]: record["steps"] for record in records.values()}
        corpus["repeated_token_frac"] = float(np.median(
            [record["steps"]["repeated_token_frac"] for record in records.values()]))
        corpus["occupancy"] = {record["tag"]: record["occupancy"] for record in records.values()}
        corpus["metric_power"] = summary["metric_power_control"]["median_ratio_vs_task0_loeo_median"]
        corpus["decisions"] = decide(model, corpus, calibration)
        decisions[model] = corpus
    summary["decisions"] = decisions
    summary["grid_note"] = (
        "value_l1/grid_l1/hamming are reported on the same 0.0625 grid step for both models; for "
        "GR00T the level index is the nearest grid level, and its own off-grid residual is reported "
        "separately, because no quantization step exists on that path")
    write_json(out / "summary.json", summary)
    print(json.dumps({model: block.get("decisions", block) for model, block in decisions.items()},
                     indent=1, default=float)[:4000], flush=True)
    return 0


# -------------------------------------------------------------------- figures


def stage_figures(args: argparse.Namespace) -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(args.output_dir)
    summary = read_json(out / "summary.json")
    series = np.load(out / "series" / "token_support.npz")
    calibration = read_json(out / "tables" / "loeo_calibration.json")
    figures = out / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    written: list[str] = []

    labels = sorted({key.rsplit("__", 1)[0] for key in series.files if not key.startswith("heldout__")})
    heldout = series["heldout__value_l1"]

    # 1. support distance over time -----------------------------------------
    figure, axes = plt.subplots(len(labels) + 1, 1, figsize=(12.0, 2.0 * (len(labels) + 1)),
                                sharex=False)
    axes = np.atleast_1d(axes)
    heldout_summary = summary["heldout_baseline"]
    for axis, label in zip(axes, labels):
        time = series[f"{label}__time_rel"]
        distance = series[f"{label}__value_l1"]
        meta = json.loads(str(series[f"{label}__meta"].tolist()))
        axis.plot(time, distance, linewidth=0.8, color="#1f77b4" if meta["model"] == "psi0" else "#d62728")
        for value, style, note in ((heldout_summary["value_l1"]["p99"], "--", "held-out p99"),
                                   (heldout_summary["value_l1"]["median"], ":", "held-out median")):
            axis.axhline(value, color="black", linewidth=0.9 if style == "--" else 0.7,
                         linestyle=style, label=note if label == labels[0] else None)
        axis.set_ylabel("value L1")
        axis.set_title(f"{meta['tag']} r{meta['rollout']:02d} ({meta['model']}), "
                       f"{len(time) / max(1e-9, time[-1] - time[0]):.0f} tokens/s", fontsize=9)
        axis.grid(alpha=0.25)
    axes[-1].hist(heldout, bins=40, color="#7f7f7f", alpha=0.85)
    axes[-1].set_title("held-out val BlockStacking episodes scored against the same cloud "
                       "(the like-for-like reference)", fontsize=9)
    axes[-1].set_ylabel("tokens")
    axes[-1].grid(alpha=0.25)
    axes[-1].set_xlabel("held-out episodes: nearest-neighbour value L1")
    for axis in axes[:-1]:
        axis.set_xlabel("seconds relative to the rollout's own onset")
    axes[0].legend(fontsize=7)
    figure.suptitle("Published-token distance to the Task-0 demonstration cloud (nearest neighbour)",
                    fontsize=11)
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    path = figures / "fig01_support_distance_over_time.png"
    figure.savefig(path, dpi=110)
    plt.close(figure)
    written.append(str(path))

    # 2. per-channel envelope ----------------------------------------------
    demo = summary["demo_reference"]
    bounds = np.load(out / "tables" / "calibration.npz")
    figure, axes = plt.subplots(1, 2, figsize=(13.0, 4.6))
    channels = np.arange(64)
    low, high = bounds["envelope_low"], bounds["envelope_high"]
    axes[0].fill_between(channels, low, high, color="#7f7f7f", alpha=0.3,
                         label="demonstration p1-p99 (train pool)")
    axes[0].plot(channels, [demo["weighted_median_value"][f"token_{i:02d}"] for i in channels],
                 color="black", linewidth=1.0, label="demonstration median")
    heldout_tokens = series["heldout__token"]
    axes[1].plot(channels, np.sort(((heldout_tokens < low[None, :])
                                    | (heldout_tokens > high[None, :])).mean(axis=0)) * 100.0,
                 color="#7f7f7f", linewidth=1.8, linestyle="--",
                 label="held-out val episodes (reference)")
    for label in labels:
        meta = json.loads(str(series[f"{label}__meta"].tolist()))
        tokens = series[f"{label}__token"]
        colour = "#1f77b4" if meta["model"] == "psi0" else "#d62728"
        outside = ((tokens < low[None, :]) | (tokens > high[None, :])).mean(axis=0)
        axes[1].plot(channels, np.sort(outside) * 100.0, linewidth=1.0, color=colour, alpha=0.85,
                     label=f"{meta['tag']} r{meta['rollout']:02d}")
        axes[0].plot(channels, np.median(tokens, axis=0), linewidth=1.0, alpha=0.75, color=colour,
                     label=f"{meta['tag']} r{meta['rollout']:02d}")
    axes[0].set_title("per-channel envelope: demonstration vs published tokens")
    axes[0].set_ylabel("token value")
    axes[0].set_xlabel("token channel")
    axes[1].set_title("per-channel out-of-envelope rate, channels sorted")
    axes[1].set_ylabel("% of tokens outside")
    axes[1].set_xlabel("channel rank")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend(fontsize=6, ncol=2)
    figure.tight_layout()
    path = figures / "fig02_channel_envelope.png"
    figure.savefig(path, dpi=110)
    plt.close(figure)
    written.append(str(path))

    # 3. palm height vs support distance ------------------------------------
    figure, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))
    for axis, model in zip(axes, ("psi0", "groot")):
        colour = "#1f77b4" if model == "psi0" else "#d62728"
        for label in labels:
            meta = json.loads(str(series[f"{label}__meta"].tolist()))
            if meta["model"] != model:
                continue
            height = series[f"{label}__palm_target_z"]
            distance = series[f"{label}__value_l1"]
            joined = series[f"{label}__joined"]
            keep = joined & np.isfinite(height)
            axis.scatter(height[keep], distance[keep], s=2, alpha=0.25, color=colour)
            edges = np.linspace(np.nanmin(height[keep]), np.nanmax(height[keep]), 13)
            index = np.digitize(height[keep], edges)
            centres, medians = [], []
            for bin_index in np.unique(index):
                rows = index == bin_index
                centres.append(float(np.mean(height[keep][rows])))
                medians.append(float(np.median(distance[keep][rows])))
            axis.plot(centres, medians, color="black", linewidth=1.6, marker="o", markersize=3,
                      label="binned median" if meta["rollout"] == 1 else None)
        axis.set_title(f"{model}: token support distance vs commanded palm height")
        axis.set_xlabel("max(target palm z, left/right), pelvis frame (m)")
        axis.set_ylabel("nearest-neighbour value L1")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    figure.tight_layout()
    path = figures / "fig03_palm_vs_support.png"
    figure.savefig(path, dpi=110)
    plt.close(figure)
    written.append(str(path))
    write_json(out / "figures" / "figures.json", {"written": written})
    for item in written:
        print(f"[figures] {item}")
    return 0


# ------------------------------------------------------------------- manifest


def stage_manifest(args: argparse.Namespace) -> int:
    out = Path(args.output_dir)
    dataset_root = Path(args.dataset_root)
    inputs: list[dict[str, Any]] = []
    for path in (CAMPAIGN_DIR / "campaign.json",):
        inputs.append({"path": str(path), "sha256": sha256(path)})
    for entry in session_rollouts():
        for name in ("bridge-telemetry.jsonl", "tracking.jsonl", "rollout.json"):
            path = entry["dir"] / name
            inputs.append({"path": str(path), "sha256": sha256(path),
                           "bytes": path.stat().st_size if path.is_file() else None})
    meta_dir = dataset_root / "meta"
    for name in ("episodes.jsonl", "info.json", "tasks.jsonl", "modality.json"):
        path = meta_dir / name
        inputs.append({"path": str(path), "sha256": sha256(path)})
    script = Path(__file__).resolve()
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "script": {"path": str(script.relative_to(REPO_ROOT)), "sha256": sha256(script),
                   "version": SCRIPT_VERSION},
        "commands": [
            "data/venvs/groot-n17/bin/python scripts/policy-token-support.py support",
            "python3 scripts/policy-token-support.py figures",
            "python3 scripts/policy-token-support.py manifest",
        ],
        "inputs": inputs,
        "dataset_root": str(dataset_root),
        "campaign_dir": str(CAMPAIGN_DIR),
        "urdf": {"path": str(Path(args.urdf)), "sha256": sha256(Path(args.urdf))},
        "constants": {"seed": args.seed, "frames_per_episode": args.frames_per_episode,
                      "density_frames_per_episode": args.density_frames_per_episode,
                      "calibration_frames_per_episode": args.calibration_frames_per_episode,
                      "control_stride": args.control_stride,
                      "wrong_task_budget": args.wrong_task_budget,
                      "wrong_task_max_episodes": args.wrong_task_max_episodes,
                      "threshold_quantiles": list(THRESHOLD_QUANTILES),
                      "envelope_quantiles": list(ENVELOPE_QUANTILES),
                      "fsq": {"min": FSQ_MIN, "max": FSQ_MAX, "step": FSQ_STEP, "levels": GRID_LEVELS},
                      "join_tolerance_s": MAX_JOIN_OFFSET_S},
        "outputs": sorted(str(path.relative_to(REPO_ROOT)) for path in out.rglob("*") if path.is_file()),
        "environment": {"python": sys.version.split()[0], "numpy": np.__version__},
    }
    write_json(out / "manifest.json", manifest)
    print(f"[manifest] {out / 'manifest.json'}")
    return 0


# ----------------------------------------------------------------------- main


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=("support", "figures", "manifest", "all"))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--dataset-root", default=str(DEFAULT_DATASET))
    parser.add_argument("--val-dataset-root", default=str(DEFAULT_VAL_DATASET))
    parser.add_argument("--urdf", default=str(DEFAULT_URDF))
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--frames-per-episode", type=int, default=FRAMES_PER_EPISODE)
    parser.add_argument("--density-frames-per-episode", type=int, default=DENSITY_FRAMES_PER_EPISODE)
    parser.add_argument("--calibration-frames-per-episode", type=int,
                        default=CALIBRATION_FRAMES_PER_EPISODE)
    parser.add_argument("--control-stride", type=int, default=CONTROL_STRIDE)
    parser.add_argument("--wrong-task-budget", type=int, default=WRONG_TASK_BUDGET)
    parser.add_argument("--wrong-task-max-episodes", type=int, default=WRONG_TASK_MAX_EPISODES)
    args = parser.parse_args(argv)
    stages = ("support", "figures", "manifest") if args.stage == "all" else (args.stage,)
    for stage in stages:
        if stage == "support":
            stage_support(args)
        elif stage == "figures":
            stage_figures(args)
        else:
            stage_manifest(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
