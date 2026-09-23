"""Per-episode conversion into the Psi0 30 Hz training contract (Gate 1).

One converted episode is the raw 30 Hz episode on its own timeline carrying:

* ``observation.state``   43D = synthetic standing legs/waist + measured arms/hands,
* ``action.body_token_v1_1`` 64D SONIC v1.1 body latent,
* ``action``                 14D Dex3 hand target,

with the 50 Hz SONIC corpus sampled by nearest timestamp and the strict
chunk-validity mask carried along. Nothing here decides a layout: the canonical
order and the raw 78D slices come from :mod:`.contract`.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .contract import (
    BODY_TOKEN_DIM,
    CANONICAL_HAND_NAMES,
    CANONICAL_STATE_NAMES,
    HAND_DIM,
    ACTION_MODEL_DIM,
    PHYSICAL_HAND_SAMPLE_BOUND_RAD,
    SOURCE_STATE_DIM,
    ConversionConfig,
    source_state_permutation,
    standing_lower_body,
)


@dataclass(frozen=True)
class HandRepair:
    """Which physically impossible measured samples were replaced, and how.

    ``channels`` maps a canonical joint name to how many of that episode's
    frames were repaired; ``invalid_source_samples`` is their total.  The raw
    values are never lost: they stay in the frozen source parquet and this
    record is carried into the conversion manifest next to the episode, so a
    repaired frame is always attributable.
    """

    policy: str
    threshold_rad: float
    channels: dict[str, int]

    @property
    def invalid_source_samples(self) -> int:
        return int(sum(self.channels.values()))

    @property
    def repaired(self) -> bool:
        return bool(self.channels)

    def as_dict(self) -> dict[str, Any]:
        return {
            "policy": self.policy,
            "threshold_rad": self.threshold_rad,
            "channels": dict(sorted(self.channels.items())),
            "invalid_source_samples": self.invalid_source_samples,
        }


#: Fail closed rather than impute: a measured sample this far outside the model
#: is non-physical, and a corpus that suddenly contains many of them is a
#: collector regression, not something a converter should quietly paper over.
#: The observed rate over the frozen corpus is 314 samples in 2,587,515 frames.
MAX_REPAIRABLE_SAMPLES_PER_EPISODE = 256


def repair_nonphysical_samples(
    state: np.ndarray,
    names: tuple[str, ...],
    *,
    bound_rad: float = PHYSICAL_HAND_SAMPLE_BOUND_RAD,
) -> tuple[np.ndarray, HandRepair]:
    """Replace physically impossible samples by interpolation over valid ones.

    A single-revolution joint cannot hold a position beyond the model's largest
    absolute limit, so a value outside ``±bound_rad`` is a recording artefact
    (the frozen corpus contains isolated spikes up to 3363 rad).  Such samples
    are replaced per channel by linear interpolation between the nearest valid
    samples, which keeps the episode on its own timeline and preserves the
    temporal structure the action chunk depends on.

    Leading/trailing invalid runs have no valid sample on one side; they are
    filled with the nearest valid value.  A channel that is invalid everywhere
    is refused, because there is nothing left to interpolate from.

    ``names`` labels each channel so the repair record is readable and stable
    against a future reordering of the source layout.
    """
    measured = np.asarray(state, dtype=np.float64)
    if measured.ndim != 2:
        raise ValueError(f"measured state must be [T, D], got {measured.shape}")
    if len(names) != measured.shape[1]:
        raise ValueError(
            f"{len(names)} channel names for {measured.shape[1]} measured columns"
        )
    invalid = np.abs(measured) > bound_rad
    channels: dict[str, int] = {}
    if not invalid.any():
        return measured, HandRepair("none", float(bound_rad), channels)

    total = int(invalid.sum())
    if total > MAX_REPAIRABLE_SAMPLES_PER_EPISODE:
        raise ValueError(
            f"{total} measured samples exceed the physical bound {bound_rad} rad in one "
            f"episode (limit {MAX_REPAIRABLE_SAMPLES_PER_EPISODE}); refusing to impute a "
            "corpus-scale defect"
        )

    repaired = measured.copy()
    positions = np.arange(measured.shape[0], dtype=np.float64)
    for channel in range(measured.shape[1]):
        bad = invalid[:, channel]
        if not bad.any():
            continue
        good = ~bad
        if not good.any():
            raise ValueError(
                f"measured channel {names[channel]!r} is outside {bound_rad} rad on every "
                "frame; there is no valid sample to interpolate from"
            )
        # np.interp clamps outside the sampled range, which fills a leading or
        # trailing invalid run with the nearest valid value.
        repaired[bad, channel] = np.interp(
            positions[bad], positions[good], measured[good, channel]
        )
        channels[names[channel]] = int(bad.sum())

    policy = "linear_interpolation_over_valid_source_samples"
    return repaired, HandRepair(policy, float(bound_rad), channels)


@dataclass(frozen=True)
class RawEpisode:
    """One raw 30 Hz episode: measured state, timeline and camera segment."""

    collection: str
    episode_index: int
    state: np.ndarray          # [T, 28] measured arm/hand joint positions
    timestamp: np.ndarray      # [T] seconds, episode-relative
    state_names: tuple[str, ...]
    data_file: str
    video_file: Path
    video_start_s: float
    video_stop_s: float
    #: Absolute path of the frozen source parquet these rows were read from, so
    #: an auditor can re-read the untouched values without re-deriving layout.
    data_path: Path | None = None
    #: Row range ``[from, to)`` of this episode inside ``data_path``; one source
    #: parquet holds every episode of its collection.
    data_slice: tuple[int, int] | None = None
    #: Which physically impossible samples were replaced before this episode was
    #: used.  Empty for the 99.9% of episodes that need no repair; the raw
    #: values remain in the frozen source parquet, referenced by ``data_file``.
    repair: HandRepair = field(default_factory=lambda: HandRepair("none", PHYSICAL_HAND_SAMPLE_BOUND_RAD, {}))

    @property
    def frames(self) -> int:
        return int(self.state.shape[0])


@dataclass(frozen=True)
class ConvertedEpisode:
    """One episode in the Psi0 training contract."""

    collection: str
    episode_index: int
    state: np.ndarray               # [T, 43]
    body_token: np.ndarray          # [T, 64]
    hand_action: np.ndarray         # [T, 14]
    action_mask: np.ndarray         # [T, 80] per-frame, per-dimension target validity
    timestamp: np.ndarray           # [T] uniform on the dataset timeline
    training_valid_mask: np.ndarray  # [T] target frame usable
    anchor_valid: np.ndarray        # [T] every one of the next 30 targets is usable
    source_timestamp: np.ndarray    # [T] raw episode timeline used for the mapping
    source_index: np.ndarray        # [T] selected 50 Hz corpus index per frame
    raw: RawEpisode

    @property
    def frames(self) -> int:
        return int(self.state.shape[0])


def nearest_indices(source_timestamps: np.ndarray, target_timestamps: np.ndarray) -> np.ndarray:
    """Nearest source index for every target timestamp; ties pick the lower index.

    Linear interpolation in the latent action space is not validated, so the
    50->30 Hz reduction selects an existing 50 Hz sample rather than inventing
    one.
    """
    source = np.asarray(source_timestamps, dtype=np.float64)
    target = np.asarray(target_timestamps, dtype=np.float64)
    if source.ndim != 1 or source.size == 0:
        raise ValueError("the source timeline must be a non-empty 1D array")
    if target.ndim != 1:
        raise ValueError("the target timeline must be a 1D array")
    if source.size > 1 and not bool(np.all(np.diff(source) > 0)):
        raise ValueError("the source timeline must be strictly increasing")
    if not bool(np.isfinite(source).all()) or not bool(np.isfinite(target).all()):
        raise ValueError("the timelines must be finite")
    position = np.clip(np.searchsorted(source, target, side="left"), 1, source.size - 1)
    left, right = source[position - 1], source[position]
    index = np.where(np.abs(target - left) <= np.abs(right - target), position - 1, position)
    return index.astype(np.int64)


def mapping_error(source_timestamps: np.ndarray, target_timestamps: np.ndarray, index: np.ndarray) -> np.ndarray:
    """Distance in seconds between every target timestamp and the sample it selected."""
    source = np.asarray(source_timestamps, dtype=np.float64)
    target = np.asarray(target_timestamps, dtype=np.float64)
    return np.abs(source[np.asarray(index, dtype=np.int64)] - target)


def assert_mapping(
    error: np.ndarray,
    index: np.ndarray,
    anchor: np.ndarray,
    chunk: int,
    config: ConversionConfig,
    label: str,
) -> None:
    """Fail closed on a mapping the contract does not allow.

    Two regions are distinguished because the frozen 50 Hz corpus is sometimes a
    frame or two shorter than the ideal resampling of the raw 30 Hz timeline:

    * every frame a valid anchor depends on -- the anchor itself and the 30
      action targets that follow it -- must be within
      ``max_timestamp_error_s``. The two timelines share a ``1/150 s`` grid, so
      the achievable in-corpus bound is ``1/150`` and this is what is checked;
    * only beyond the last valid anchor may the error grow to
      ``tail_timestamp_error_s``, where the corpus has simply ended and the tail
      carries no supervision.

    The selection must also be monotone: a non-decreasing mapping is what keeps
    the 30 targets a real forward window instead of a replayed or reordered one.
    """
    if index.size > 1 and bool((np.diff(index) < 0).any()):
        raise ValueError(f"{label}: the 50->30 Hz selection is not monotone")
    worst = float(error.max()) if error.size else 0.0
    if worst > config.tail_timestamp_error_s:
        raise ValueError(
            f"{label}: 50->30 Hz nearest-timestamp error {worst:.5f}s exceeds the declared "
            f"tail bound {config.tail_timestamp_error_s}s"
        )
    anchors = np.flatnonzero(anchor)
    if not anchors.size:
        return
    last = int(anchors[-1])
    supervised = float(error[: last + chunk].max())
    if supervised > config.max_timestamp_error_s:
        raise ValueError(
            f"{label}: a supervised action target is {supervised:.5f}s from its nearest 50 Hz "
            f"sample, above the declared {config.max_timestamp_error_s}s (only the tail beyond "
            f"the last valid anchor may be coarser)"
        )


def anchor_validity(valid: np.ndarray, chunk: int) -> np.ndarray:
    """``anchor_valid[i]`` iff all ``chunk`` targets starting at ``i`` are usable.

    An episode tail whose future window was clamped upstream is a valid replay
    frame but not valid supervision, and one bad target disqualifies the whole
    anchor rather than leaking into the loss as a short chunk.
    """
    mask = np.asarray(valid, dtype=bool)
    if chunk <= 0:
        raise ValueError("the action chunk size must be positive")
    frames = mask.shape[0]
    anchors = np.zeros(frames, dtype=bool)
    if frames >= chunk:
        windows = np.lib.stride_tricks.sliding_window_view(mask, chunk)
        anchors[: frames - chunk + 1] = windows.all(axis=1)
    return anchors


def _assert_uniform(timestamp: np.ndarray, fps: int, tolerance: float = 5e-4) -> None:
    expected = np.arange(len(timestamp), dtype=np.float64) / fps
    if float(np.abs(np.asarray(timestamp, dtype=np.float64) - expected).max()) > tolerance:
        raise ValueError(f"the source timeline is not uniform at {fps} Hz")


def build_state(state28: np.ndarray, permutation: tuple[int, ...]) -> np.ndarray:
    """43D state = constant standing legs/waist + measured arms and hands."""
    measured = np.asarray(state28, dtype=np.float64)
    if measured.ndim != 2 or measured.shape[1] != SOURCE_STATE_DIM:
        raise ValueError(f"raw observation.state must be [T, {SOURCE_STATE_DIM}], got {measured.shape}")
    lower = np.tile(standing_lower_body(), (measured.shape[0], 1))
    return np.concatenate([lower, measured[:, list(permutation)]], axis=1).astype(np.float32)


def convert_episode(
    raw: RawEpisode,
    sonic_action: np.ndarray,
    sonic_timestamp: np.ndarray,
    sonic_valid: np.ndarray,
    config: ConversionConfig,
) -> ConvertedEpisode:
    """Map one raw episode and its 50 Hz SONIC action onto the 30 Hz contract."""
    action = np.asarray(sonic_action, dtype=np.float32)
    valid = np.asarray(sonic_valid, dtype=bool)
    source_timestamp = np.asarray(sonic_timestamp, dtype=np.float64)
    if action.ndim != 2 or action.shape[1] != BODY_TOKEN_DIM + HAND_DIM:
        raise ValueError(f"the frozen SONIC action must be [T, 78], got {action.shape}")
    if source_timestamp.shape != (action.shape[0],) or valid.shape != (action.shape[0],):
        raise ValueError("the SONIC action, timestamp and validity mask must share one frame axis")
    chunk = config.action_chunk_size

    target_timestamp = np.asarray(raw.timestamp, dtype=np.float64)
    if target_timestamp.shape != (raw.frames,):
        raise ValueError("the raw episode timeline does not match its frame count")
    _assert_uniform(target_timestamp, config.source_fps)

    index = nearest_indices(source_timestamp, target_timestamp)
    error = mapping_error(source_timestamp, target_timestamp, index)
    selected = action[index]
    body_token = np.ascontiguousarray(selected[:, 0:BODY_TOKEN_DIM])
    hand_action = np.ascontiguousarray(selected[:, BODY_TOKEN_DIM : BODY_TOKEN_DIM + HAND_DIM])
    training_valid = np.ascontiguousarray(valid[index])
    anchor = anchor_validity(training_valid, chunk)
    assert_mapping(error, index, anchor, chunk, config, f"{raw.collection}/episode {raw.episode_index}")

    return ConvertedEpisode(
        collection=raw.collection,
        episode_index=raw.episode_index,
        state=build_state(raw.state, source_state_permutation(raw.state_names)),
        body_token=body_token,
        hand_action=hand_action,
        action_mask=action_mask(training_valid, config),
        timestamp=(np.arange(raw.frames, dtype=np.float64) / config.dataset_fps),
        training_valid_mask=training_valid,
        anchor_valid=anchor,
        source_timestamp=target_timestamp,
        source_index=index,
        raw=raw,
    )


def action_mask(valid: np.ndarray, config: ConversionConfig) -> np.ndarray:
    """Wide per-frame action validity, in the repacked ``80D`` action order.

    Psi0's ``SonicRepackTransform`` fetches this field with the same 30-frame
    delta timestamps as the action and multiplies it into an
    ``(action_chunk_size, action_dim)`` loss mask, so it must be exactly
    ``action_dim`` wide. The two neck padding channels are never valid, and a
    frame whose future reference window was clamped upstream contributes
    nothing.
    """
    mask = np.zeros((valid.shape[0], ACTION_MODEL_DIM), dtype=np.float32)
    mask[:, : BODY_TOKEN_DIM + HAND_DIM] = np.asarray(valid, dtype=np.float32)[:, None]
    return mask


def read_raw_episode(config: ConversionConfig, collection: str, episode_index: int) -> RawEpisode:
    """Read one raw episode's measured state, timeline and camera segment."""
    import pyarrow.parquet as pq

    dataset = Path(config.raw_root) / collection
    info = json.loads((dataset / "meta/info.json").read_text(encoding="utf-8"))
    state_feature = info["features"]["observation.state"]
    state_names = tuple(state_feature["names"][0])
    if int(state_feature["shape"][0]) != SOURCE_STATE_DIM:
        raise ValueError(f"{collection}: source observation.state must be {SOURCE_STATE_DIM}D")

    rows = _episode_rows(dataset)
    if episode_index not in rows:
        raise ValueError(f"{collection}: episode {episode_index} has no metadata row")
    row = rows[episode_index]
    data_file = dataset / f"data/chunk-{int(row['data/chunk_index']):03d}/file-{int(row['data/file_index']):03d}.parquet"
    start, stop = int(row["dataset_from_index"]), int(row["dataset_to_index"])
    table = pq.read_table(data_file, columns=["episode_index", "observation.state", "timestamp"]).slice(start, stop - start)
    records = table.to_pylist()
    if not records or any(int(record["episode_index"]) != episode_index for record in records):
        raise ValueError(f"{collection}/{episode_index}: episode metadata does not match the data rows")
    if int(row["length"]) != len(records):
        raise ValueError(f"{collection}/{episode_index}: metadata length {row['length']} != {len(records)} rows")

    key = f"videos/{config.camera_source_key}"
    if f"{key}/chunk_index" not in row:
        raise ValueError(f"{collection}/{episode_index}: the metadata has no {config.camera_source_key} segment")
    measured = np.asarray([record["observation.state"] for record in records], dtype=np.float64)
    repaired, repair = repair_nonphysical_samples(measured, state_names)
    return RawEpisode(
        collection=collection,
        episode_index=episode_index,
        state=repaired.astype(np.float32),
        timestamp=np.asarray([record["timestamp"] for record in records], dtype=np.float64),
        state_names=state_names,
        data_file=str(data_file.relative_to(dataset)),
        video_file=dataset / f"{key}/chunk-{int(row[key + '/chunk_index']):03d}/file-{int(row[key + '/file_index']):03d}.mp4",
        video_start_s=float(row[key + "/from_timestamp"]),
        video_stop_s=float(row[key + "/to_timestamp"]),
        data_path=data_file,
        data_slice=(start, stop),
        repair=repair,
    )


def _episode_rows(dataset: Path) -> dict[int, dict[str, Any]]:
    import pyarrow.parquet as pq

    rows: dict[int, dict[str, Any]] = {}
    for path in sorted((dataset / "meta/episodes").glob("chunk-*/*.parquet")):
        for row in pq.read_table(path).to_pylist():
            index = int(row["episode_index"])
            if index in rows:
                raise ValueError(f"{dataset.name}: duplicate episode metadata row {index}")
            rows[index] = row
    return rows


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


__all__ = [
    "CANONICAL_HAND_NAMES",
    "CANONICAL_STATE_NAMES",
    "ConvertedEpisode",
    "HandRepair",
    "RawEpisode",
    "action_mask",
    "anchor_validity",
    "assert_mapping",
    "build_state",
    "convert_episode",
    "file_sha256",
    "mapping_error",
    "nearest_indices",
    "read_raw_episode",
    "repair_nonphysical_samples",
]
