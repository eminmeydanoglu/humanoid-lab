"""Per-episode conversion into the GR00T Unitree Dex3 / SONIC contract.

One converted episode is the frozen SONIC 50 Hz episode minus its clamped tail,
carrying

* ``observation.state`` — 43D, the synthetic standing lower body plus the
  measured arms and hands resampled from 30 Hz onto the exact SONIC timestamps,
* ``observation.projected_gravity`` — the synthetic upright ``[0, 0, -1]``,
* ``action.motion_token`` — the 64D SONIC v1.1 body latent, copied verbatim,
* ``teleop.left_hand_joints`` / ``teleop.right_hand_joints`` — rebuilt from the
  raw desired action into the official actuated order; the corpus's own hand
  block is never copied because its left side is stored in the source motor
  order.

Nothing here decides a layout: the orders and slices come from :mod:`.contract`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .contract import (
    CANONICAL_STATE_NAMES,
    HAND_CHANNELS,
    HAND_DIM,
    MOTION_TOKEN_DIM,
    PROJECTED_GRAVITY,
    SOURCE_STATE_DIM,
    STATE_BLOCKS,
    ConversionConfig,
    assert_source_names,
    corpus_hand_permutation,
    source_action_hand_permutation,
    source_state_permutation,
    standing_lower_body,
)

#: Canonical names of the 14 measured hand channels.
_HAND_CHANNEL_NAMES = frozenset(
    tuple(CANONICAL_STATE_NAMES[STATE_BLOCKS["left_hand"]]) + tuple(CANONICAL_STATE_NAMES[STATE_BLOCKS["right_hand"]])
)


class ConversionError(ValueError):
    """The source episode cannot be converted under the frozen contract."""


@dataclass(frozen=True)
class RawEpisode:
    """One raw 30 Hz episode: measured state, desired action and camera segment."""

    collection: str
    episode_index: int
    state: np.ndarray          # [T, 28] measured arm/hand joint positions
    action: np.ndarray         # [T, 28] desired arm/hand positions of the same row
    timestamp: np.ndarray      # [T] seconds, episode-relative
    state_names: tuple[str, ...]
    action_names: tuple[str, ...]
    data_file: str
    video_file: Path
    video_start_s: float
    video_stop_s: float

    @property
    def frames(self) -> int:
        return int(self.state.shape[0])

    @property
    def label(self) -> str:
        return f"{self.collection}/episode {self.episode_index}"


@dataclass(frozen=True)
class SonicEpisode:
    """One frozen 50 Hz SONIC corpus episode."""

    path: Path
    action: np.ndarray                # [T, 78]
    timestamp: np.ndarray             # [T] seconds, session-relative
    training_valid_mask: np.ndarray   # [T] the last 45 rows are upstream-clamped

    @property
    def frames(self) -> int:
        return int(self.action.shape[0])


@dataclass(frozen=True)
class HandRepair:
    """Per-channel repair ledger of one source episode, keyed by state channel."""

    threshold_rad: float
    channels: dict[str, int] = field(default_factory=dict)

    @property
    def samples(self) -> int:
        return int(sum(self.channels.values()))

    def payload(self) -> dict[str, Any]:
        return {
            "policy": "linear_interpolation_over_valid_source_samples",
            "threshold_rad": self.threshold_rad,
            "channels": dict(sorted(self.channels.items())),
            "invalid_source_samples": self.samples,
        }


@dataclass(frozen=True)
class ConvertedEpisode:
    """One episode in the GR00T training contract."""

    collection: str
    episode_index: int
    state: np.ndarray              # [N, 43]
    gravity: np.ndarray            # [N, 3], the synthetic upright constant
    motion_token: np.ndarray       # [N, 64]
    left_hand: np.ndarray          # [N, 7] official actuated order
    right_hand: np.ndarray         # [N, 7] official actuated order
    timestamp: np.ndarray          # [N] the exact SONIC timestamps of the retained rows
    repair: HandRepair
    corpus_rows: int
    trimmed_rows: int
    corpus_hand_max_error: float   # rebuilt vs name-permuted corpus hand target
    raw: RawEpisode
    sonic: SonicEpisode

    @property
    def frames(self) -> int:
        return int(self.state.shape[0])

    @property
    def hand_action(self) -> np.ndarray:
        """Both action hands as the 14D pair the action modality declares."""
        return np.concatenate([self.left_hand, self.right_hand], axis=1)


def measured_channel_map(names: tuple[str, ...]) -> dict[int, str]:
    """Source column -> canonical state channel name for the 28 measured channels.

    The map is derived from the source's declared names, never from position, so
    a re-ordered upstream release fails instead of moving a channel.
    """
    permutation = source_state_permutation(names)
    return {column: CANONICAL_STATE_NAMES[15 + index] for index, column in enumerate(permutation)}


def corrupt_hand_columns(values: np.ndarray, names: tuple[str, ...], threshold: float) -> dict[int, np.ndarray]:
    """Measured hand columns that carry gross corruption, with their sample mask.

    A sample is invalid when it is not finite or beyond ``threshold`` rad: a Dex3
    finger cannot reach that far, so the reading is a recording fault rather than
    a pose.
    """
    array = np.asarray(values, dtype=np.float64)
    assert_source_names(names)
    if array.ndim != 2 or array.shape[1] != SOURCE_STATE_DIM:
        raise ConversionError(f"measured values must be [T, {SOURCE_STATE_DIM}]")
    result: dict[int, np.ndarray] = {}
    for column, channel in measured_channel_map(names).items():
        if channel not in _HAND_CHANNEL_NAMES:
            continue
        samples = array[:, column]
        invalid = ~np.isfinite(samples) | (np.abs(samples) > threshold)
        if bool(invalid.any()):
            result[column] = invalid
    return result


def assert_measured_finite(values: np.ndarray, names: tuple[str, ...], label: str) -> None:
    """Fail closed on a non-finite arm reading or commanded sample.

    Only the measured hands have a declared repair; a non-finite arm or desired
    action value is not something this pipeline may invent a replacement for.
    """
    array = np.asarray(values, dtype=np.float64)
    bad = ~np.isfinite(array)
    if bool(bad.any()):
        columns = sorted({names[index] for index in np.flatnonzero(bad.any(axis=0))})
        raise ConversionError(f"{label}: non-finite samples in {columns}")


def repair_measured_hands(
    state: np.ndarray,
    names: tuple[str, ...],
    config: ConversionConfig,
    label: str,
) -> tuple[np.ndarray, HandRepair]:
    """Replace grossly invalid measured hand samples by temporal interpolation.

    The repair runs on the 30 Hz source episode, channel by channel, and only on
    the measured hands. Interpolation needs a valid sample on both sides, so an
    invalid first or last sample, a channel with no valid sample at all, a
    channel invalid for more than the declared fraction of the episode, or a gap
    longer than the declared bound all stop the conversion: under those
    conditions a repaired value would be an invention, and the value is never
    clipped.
    """
    array = np.asarray(state, dtype=np.float64)
    repaired = array.copy()
    ledger: dict[str, int] = {}
    channels = measured_channel_map(names)
    for column, invalid in corrupt_hand_columns(array, names, config.repair_invalid_abs_rad).items():
        channel = channels[column]
        count = int(invalid.sum())
        if bool(invalid.all()):
            raise ConversionError(f"{label}: {channel} has no valid measured sample to interpolate from")
        if bool(invalid[0]) or bool(invalid[-1]):
            raise ConversionError(
                f"{label}: {channel} is invalid at the episode boundary, where interpolation would have "
                "to extrapolate"
            )
        if count / array.shape[0] > config.repair_max_invalid_fraction:
            raise ConversionError(
                f"{label}: {channel} is invalid in {count}/{array.shape[0]} samples, above the declared "
                f"{config.repair_max_invalid_fraction} fraction"
            )
        valid = np.flatnonzero(~invalid)
        longest = int(np.diff(valid).max()) - 1
        if longest > config.repair_max_gap_frames:
            raise ConversionError(
                f"{label}: {channel} has an invalid run of {longest} samples, above the declared "
                f"{config.repair_max_gap_frames}-frame gap bound"
            )
        repaired[:, column] = np.interp(np.arange(array.shape[0]), valid, array[valid, column])
        ledger[channel] = count
    return repaired, HandRepair(threshold_rad=config.repair_invalid_abs_rad, channels=ledger)


def assert_action_hands_clean(
    action: np.ndarray,
    names: tuple[str, ...],
    config: ConversionConfig,
    label: str,
) -> None:
    """Fail closed if the desired action hands carry the measured-state corruption.

    The action target is a commanded position, so it is checked and never
    repaired: a repaired label would train the policy on a value no collector
    ever sent.
    """
    if not config.repair_fail_on_action:
        return
    channels = measured_channel_map(names)
    for column, invalid in corrupt_hand_columns(action, names, config.repair_invalid_abs_rad).items():
        raise ConversionError(
            f"{label}: desired action channel {channels[column]} has {int(invalid.sum())} invalid samples; "
            "action targets are declared clean"
        )


def linear_resample(
    source_timestamps: np.ndarray,
    values: np.ndarray,
    target_timestamps: np.ndarray,
    label: str,
) -> np.ndarray:
    """Interpolate every column onto the target timeline, fail-closed outside it.

    The target timeline is inside the source span by construction: a target
    timestamp beyond either end would extrapolate the first or last segment,
    which no source sample supports.
    """
    source = np.asarray(source_timestamps, dtype=np.float64)
    target = np.asarray(target_timestamps, dtype=np.float64)
    data = np.asarray(values, dtype=np.float64)
    if data.ndim != 2 or data.shape[0] != source.shape[0]:
        raise ConversionError(f"{label}: values must be [T, channels] matching the timeline")
    if source.ndim != 1 or source.size == 0 or not bool(np.isfinite(source).all()):
        raise ConversionError(f"{label}: the source timeline must be a finite non-empty vector")
    if source.size > 1 and not bool(np.all(np.diff(source) > 0)):
        raise ConversionError(f"{label}: the source timeline must be strictly increasing")
    if not bool(np.isfinite(target).all()):
        raise ConversionError(f"{label}: the target timeline must be finite")
    if target.size and (
        float(target.min()) < float(source[0]) - 1e-9 or float(target.max()) > float(source[-1]) + 1e-9
    ):
        raise ConversionError(
            f"{label}: target timestamps span [{float(target.min()):.6f}, {float(target.max()):.6f}]s, "
            f"outside the source span [{float(source[0]):.6f}, {float(source[-1]):.6f}]s"
        )
    if not bool(np.isfinite(data).all()):
        raise ConversionError(f"{label}: the source carries NaN or Inf")
    return np.stack([np.interp(target, source, data[:, index]) for index in range(data.shape[1])], axis=1)


def retained_rows(valid: np.ndarray, config: ConversionConfig, label: str) -> int:
    """Rows of a corpus episode that are training rows, or fail.

    The corpus keeps its last ``trailing_invalid_rows`` rows for replay; their
    future window is upstream last-frame clamping. Only that exact clamped tail
    may disappear, so a mask that is invalid anywhere else -- or a tail of
    another length -- stops the conversion instead of quietly shortening or
    silently keeping a clamped row.
    """
    mask = np.asarray(valid, dtype=bool)
    if mask.ndim != 1 or mask.size == 0:
        raise ConversionError(f"{label}: training_valid_mask must be a non-empty vector")
    invalid = np.flatnonzero(~mask)
    expected = np.arange(mask.size - config.trailing_invalid_rows, mask.size)
    if invalid.size != config.trailing_invalid_rows or not np.array_equal(invalid, expected):
        raise ConversionError(
            f"{label}: the corpus marks {invalid.size} rows invalid; the contract drops exactly the "
            f"trailing {config.trailing_invalid_rows} rows"
        )
    rows = int(mask.size - config.trailing_invalid_rows)
    if rows < config.min_rows:
        raise ConversionError(
            f"{label}: {rows} training rows remain after dropping the clamped tail, below the "
            f"{config.min_rows}-row horizon"
        )
    return rows


def build_state(measured: np.ndarray, permutation: tuple[int, ...], frames: int) -> np.ndarray:
    """43D state = constant standing legs/waist + the measured arms and hands."""
    if measured.ndim != 2 or measured.shape[1] != len(permutation) or len(permutation) != 28:
        raise ConversionError("the measured block must be 28D and map to 28 stored channels")
    if measured.shape[0] != frames:
        raise ConversionError("the measured block must already be resampled to the output rows")
    lower = np.tile(standing_lower_body(), (frames, 1))
    return np.concatenate([lower, measured[:, list(permutation)]], axis=1).astype(np.float32)


def convert_episode(raw: RawEpisode, sonic: SonicEpisode, config: ConversionConfig) -> ConvertedEpisode:
    """Map one raw episode and its frozen SONIC episode onto the contract."""
    label = raw.label
    assert_source_names(raw.state_names)
    assert_source_names(raw.action_names)
    timestamp = np.asarray(sonic.timestamp, dtype=np.float64)
    mask = np.asarray(sonic.training_valid_mask, dtype=bool)
    action = np.asarray(sonic.action, dtype=np.float32)
    if action.ndim != 2 or action.shape[1] != MOTION_TOKEN_DIM + HAND_CHANNELS:
        raise ConversionError(f"{label}: the frozen SONIC action must be [T, 78], got {action.shape}")
    if timestamp.shape != (action.shape[0],) or mask.shape != (action.shape[0],):
        raise ConversionError(f"{label}: the corpus action, timestamp and mask must share one frame axis")

    rows = retained_rows(mask, config, label)
    output_timestamp = timestamp[:rows]
    if not bool(np.all(np.diff(output_timestamp) > 0)):
        raise ConversionError(f"{label}: the retained corpus timestamps must be strictly increasing")
    # The camera segment is 30 Hz; the retained 50 Hz span must fit inside it.
    if raw.frames * 5 < rows * 3:
        raise ConversionError(
            f"{label}: {rows} retained 50 Hz rows do not fit inside the {raw.frames}-frame 30 Hz camera "
            "segment"
        )

    assert_measured_finite(raw.state[:, :14], raw.state_names[:14], f"{label} measured arms")
    assert_measured_finite(raw.action, raw.action_names, f"{label} desired action")
    repaired, repair = repair_measured_hands(raw.state, raw.state_names, config, label)
    assert_action_hands_clean(raw.action, raw.action_names, config, label)

    state = build_state(
        linear_resample(raw.timestamp, repaired, output_timestamp, f"{label} measured state"),
        source_state_permutation(raw.state_names),
        rows,
    )
    hands = linear_resample(
        raw.timestamp,
        raw.action[:, list(source_action_hand_permutation(raw.action_names))],
        output_timestamp,
        f"{label} desired action hands",
    )
    motion_token = np.ascontiguousarray(action[:rows, :MOTION_TOKEN_DIM])
    corpus_hands = np.asarray(action[:rows, MOTION_TOKEN_DIM:], dtype=np.float64)
    corpus_hand_error = float(np.abs(corpus_hands[:, list(corpus_hand_permutation())] - hands).max())
    if corpus_hand_error > config.corpus_hand_tolerance_rad:
        raise ConversionError(
            f"{label}: the rebuilt hand action differs from the name-permuted corpus block by "
            f"{corpus_hand_error:.6f} rad, above the declared {config.corpus_hand_tolerance_rad} rad; the "
            "corpus hand order or the resampling grid does not match this contract"
        )

    return ConvertedEpisode(
        collection=raw.collection,
        episode_index=raw.episode_index,
        state=state,
        gravity=np.tile(np.asarray(PROJECTED_GRAVITY, dtype=np.float32), (rows, 1)),
        motion_token=motion_token,
        left_hand=np.ascontiguousarray(hands[:, :HAND_DIM], dtype=np.float32),
        right_hand=np.ascontiguousarray(hands[:, HAND_DIM:], dtype=np.float32),
        timestamp=output_timestamp,
        repair=repair,
        corpus_rows=int(action.shape[0]),
        trimmed_rows=int(action.shape[0]) - rows,
        corpus_hand_max_error=corpus_hand_error,
        raw=raw,
        sonic=sonic,
    )


def load_sonic_episode(path: Path) -> SonicEpisode:
    """Read one frozen corpus episode without modifying it."""
    path = Path(path)
    with np.load(path, allow_pickle=False) as payload:
        required = {"action", "timestamp", "training_valid_mask"}
        missing = sorted(required - set(payload.files))
        if missing:
            raise ConversionError(f"{path}: frozen SONIC episode is missing {missing}")
        return SonicEpisode(
            path=path,
            action=np.asarray(payload["action"], dtype=np.float32),
            timestamp=np.asarray(payload["timestamp"], dtype=np.float64),
            training_valid_mask=np.asarray(payload["training_valid_mask"], dtype=bool),
        )


def sonic_episode_path(root: Path, collection: str, episode: int) -> Path:
    return Path(root) / collection / "episodes" / f"episode_{episode:06d}" / "action.npz"


def _episode_rows(dataset: Path) -> dict[int, dict[str, Any]]:
    import pyarrow.parquet as pq

    rows: dict[int, dict[str, Any]] = {}
    for path in sorted((dataset / "meta/episodes").glob("chunk-*/*.parquet")):
        for row in pq.read_table(path).to_pylist():
            index = int(row["episode_index"])
            if index in rows:
                raise ConversionError(f"{dataset.name}: duplicate episode metadata row {index}")
            rows[index] = row
    return rows


def read_raw_episode(config: ConversionConfig, collection: str, episode_index: int) -> RawEpisode:
    """Read one raw episode's measured state, desired action and camera segment."""
    import pyarrow.parquet as pq

    dataset = Path(config.raw_root) / collection
    info_path = dataset / "meta/info.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"missing source collection metadata: {info_path}")
    info = json.loads(info_path.read_text(encoding="utf-8"))
    if float(info["fps"]) != float(config.source_fps):
        raise ConversionError(f"{collection}: the source is {info['fps']} fps, expected {config.source_fps}")
    state_names = tuple(info["features"]["observation.state"]["names"][0])
    action_names = tuple(info["features"]["action"]["names"][0])
    for key, names in (("observation.state", state_names), ("action", action_names)):
        if int(info["features"][key]["shape"][0]) != SOURCE_STATE_DIM or len(names) != SOURCE_STATE_DIM:
            raise ConversionError(
                f"{collection}: {key} must be {SOURCE_STATE_DIM}D with {SOURCE_STATE_DIM} declared names"
            )

    rows = _episode_rows(dataset)
    if episode_index not in rows:
        raise ConversionError(f"{collection}: episode {episode_index} has no metadata row")
    row = rows[episode_index]
    data_file = dataset / f"data/chunk-{int(row['data/chunk_index']):03d}/file-{int(row['data/file_index']):03d}.parquet"
    start, stop = int(row["dataset_from_index"]), int(row["dataset_to_index"])
    table = pq.read_table(data_file, columns=["episode_index", "timestamp", "observation.state", "action"])
    records = table.slice(start, stop - start).to_pylist()
    if not records or any(int(record["episode_index"]) != episode_index for record in records):
        raise ConversionError(f"{collection}/{episode_index}: episode metadata does not match the data rows")
    if int(row["length"]) != len(records):
        raise ConversionError(f"{collection}/{episode_index}: metadata length {row['length']} != {len(records)} rows")

    key = f"videos/{config.camera_source_key}"
    if f"{key}/chunk_index" not in row:
        raise ConversionError(f"{collection}/{episode_index}: the metadata has no {config.camera_source_key} segment")
    return RawEpisode(
        collection=collection,
        episode_index=episode_index,
        state=np.asarray([record["observation.state"] for record in records], dtype=np.float32),
        action=np.asarray([record["action"] for record in records], dtype=np.float32),
        timestamp=np.asarray([record["timestamp"] for record in records], dtype=np.float64),
        state_names=state_names,
        action_names=action_names,
        data_file=str(data_file.relative_to(dataset)),
        video_file=dataset / f"{key}/chunk-{int(row[key + '/chunk_index']):03d}/file-{int(row[key + '/file_index']):03d}.mp4",
        video_start_s=float(row[key + "/from_timestamp"]),
        video_stop_s=float(row[key + "/to_timestamp"]),
    )


__all__ = [
    "ConversionError",
    "ConvertedEpisode",
    "HandRepair",
    "RawEpisode",
    "SonicEpisode",
    "assert_action_hands_clean",
    "assert_measured_finite",
    "build_state",
    "convert_episode",
    "corrupt_hand_columns",
    "linear_resample",
    "load_sonic_episode",
    "measured_channel_map",
    "read_raw_episode",
    "repair_measured_hands",
    "retained_rows",
    "sonic_episode_path",
]
