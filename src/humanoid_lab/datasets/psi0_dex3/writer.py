"""LeRobot v2.1 writer for the Psi0 Unitree Dex3 / SONIC v1.1 dataset.

The output layout is LeRobot v2.1, which is the codebase version the Psi0
training environment pins (``lerobot`` 0.3.3, ``CODEBASE_VERSION='v2.1'``), so
the split directories are consumed by the existing loader without a conversion
step. Episode files are written one at a time and metadata is appended as the
run proceeds, so the same writer serves the Gate 1 mini dataset and the full
corpus conversion.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .contract import (
    ACTION_MODEL_DIM,
    ANCHOR_MASK_KEY,
    BODY_TOKEN_DIM,
    CANONICAL_HAND_NAMES,
    CANONICAL_STATE_NAMES,
    HAND_DIM,
    INSTRUCTION_KEY,
    LEGS_WAIST_SLICE,
    MASK_KEY,
    PHYSICAL_HAND_SAMPLE_BOUND_RAD,
    STATE_MODEL_DIM,
    ConversionConfig,
)
from .convert import ConvertedEpisode, file_sha256

CODEBASE_VERSION = "v2.1"
ROBOT_TYPE = "Unitree_G1"
CHUNK_SIZE = 1000

#: Features carrying numeric statistics. Scalar flags and the instruction text
#: have no meaningful min/max/mean and are deliberately excluded.
STAT_FEATURES = ("observation.state", "action", "action.body_token_v1_1")

DEFAULT_FEATURE_SPECS: dict[str, dict[str, Any]] = {
    "timestamp": {"dtype": "float32", "shape": [1], "names": None},
    "frame_index": {"dtype": "int64", "shape": [1], "names": None},
    "episode_index": {"dtype": "int64", "shape": [1], "names": None},
    "index": {"dtype": "int64", "shape": [1], "names": None},
    "task_index": {"dtype": "int64", "shape": [1], "names": None},
}


def feature_specs(config: ConversionConfig) -> dict[str, dict[str, Any]]:
    """The exact `info.json` feature box, with joint names on every vector."""
    fps = float(config.dataset_fps)
    specs: dict[str, dict[str, Any]] = {
        config.state_field: {
            "dtype": "float32",
            "shape": [len(CANONICAL_STATE_NAMES)],
            "names": list(CANONICAL_STATE_NAMES),
            "fps": fps,
        },
        config.action_field: {
            "dtype": "float32",
            "shape": [HAND_DIM],
            "names": list(CANONICAL_HAND_NAMES),
            "fps": fps,
        },
        config.body_token_field: {
            "dtype": "float32",
            "shape": [BODY_TOKEN_DIM],
            "names": [f"body_token_{index:02d}" for index in range(BODY_TOKEN_DIM)],
            "fps": fps,
        },
        INSTRUCTION_KEY: {"dtype": "string", "shape": [1], "names": None},
        "next.done": {"dtype": "bool", "shape": [1], "names": None},
        MASK_KEY: {
            "dtype": "float32",
            "shape": [ACTION_MODEL_DIM],
            "names": [
                *(f"body_token_{index:02d}" for index in range(BODY_TOKEN_DIM)),
                *CANONICAL_HAND_NAMES,
                "neck_padding_0",
                "neck_padding_1",
            ],
            "fps": fps,
        },
        ANCHOR_MASK_KEY: {"dtype": "bool", "shape": [1], "names": None},
        config.camera_target_key: {
            "dtype": "video",
            "shape": [3, config.camera_height, config.camera_width],
            "names": ["channels", "height", "width"],
            "info": {
                "video.fps": fps,
                "video.height": config.camera_height,
                "video.width": config.camera_width,
                "video.channels": 3,
                "video.codec": str(config.video["codec"]),
                "video.pix_fmt": str(config.video["pix_fmt"]),
                "video.is_depth_map": False,
                "has_audio": False,
            },
        },
        **DEFAULT_FEATURE_SPECS,
    }
    return specs


def _parquet_schema(config: ConversionConfig) -> Any:
    import pyarrow as pa

    vector = lambda width: pa.list_(pa.float32())
    return pa.schema(
        [
            pa.field(config.state_field, vector(len(CANONICAL_STATE_NAMES))),
            pa.field(config.action_field, vector(HAND_DIM)),
            pa.field(config.body_token_field, vector(BODY_TOKEN_DIM)),
            pa.field(INSTRUCTION_KEY, pa.string()),
            pa.field("next.done", pa.bool_()),
            pa.field(MASK_KEY, vector(ACTION_MODEL_DIM)),
            pa.field(ANCHOR_MASK_KEY, pa.bool_()),
            pa.field("timestamp", pa.float32()),
            pa.field("frame_index", pa.int64()),
            pa.field("episode_index", pa.int64()),
            pa.field("index", pa.int64()),
            pa.field("task_index", pa.int64()),
        ]
    )


@dataclass
class _RunningStats:
    """Exact min/max/count with pooled mean and standard deviation."""

    count: int = 0
    min: np.ndarray | None = None
    max: np.ndarray | None = None
    mean: np.ndarray | None = None
    m2: np.ndarray | None = None

    def update(self, values: np.ndarray) -> None:
        array = np.asarray(values, dtype=np.float64)
        if array.ndim != 2:
            raise ValueError("stats accept one row set per call")
        block_min = array.min(axis=0)
        block_max = array.max(axis=0)
        block_mean = array.mean(axis=0)
        block_m2 = ((array - block_mean) ** 2).sum(axis=0)
        if self.count == 0:
            self.min, self.max = block_min, block_max
            self.mean, self.m2 = block_mean, block_m2
        else:
            total = self.count + array.shape[0]
            delta = block_mean - self.mean
            self.mean = self.mean + delta * (array.shape[0] / total)
            self.m2 = self.m2 + block_m2 + delta**2 * (self.count * array.shape[0] / total)
            self.min = np.minimum(self.min, block_min)
            self.max = np.maximum(self.max, block_max)
        self.count += int(array.shape[0])

    def resolve(self) -> dict[str, list[float]]:
        if self.count == 0 or self.mean is None:
            raise ValueError("no frames were accumulated")
        return {
            "min": self.min.tolist(),
            "max": self.max.tolist(),
            "mean": self.mean.tolist(),
            "std": np.sqrt(self.m2 / self.count).tolist(),
            "count": [self.count],
        }


@dataclass
class _EpisodeRecord:
    episode_index: int
    length: int
    task_index: int
    task: str
    source_collection: str
    source_episode_index: int
    source_data_file: str
    frames_valid: int
    anchors_valid: int
    video_start_s: float
    video_stop_s: float
    parquet: Path
    video: Path
    stats: dict[str, dict[str, Any]] = field(default_factory=dict)
    #: Record of any physically impossible measured sample that was replaced
    #: while reading this episode (see ``convert.repair_nonphysical_samples``).
    repair: dict[str, Any] = field(default_factory=dict)


class SplitWriter:
    """Writes one LeRobot v2.1 split directory (``train/`` or ``val/``)."""

    def __init__(self, root: Path, config: ConversionConfig, split: str, *, force: bool = False):
        self.root = Path(root)
        self.config = config
        self.split = split
        self.force = force
        self.episodes: list[_EpisodeRecord] = []
        self.total_frames = 0
        self.running = {name: _RunningStats() for name in STAT_FEATURES}
        self.schema = _parquet_schema(config)
        self.video_pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix=f"{split}-video")
        self.video_futures: list[Future[None]] = []
        for directory in ("data/chunk-000", f"videos/chunk-000/{config.camera_target_key}", "meta"):
            (self.root / directory).mkdir(parents=True, exist_ok=True)

    # -- writing -----------------------------------------------------------
    def add(self, episode: ConvertedEpisode) -> _EpisodeRecord:
        import pyarrow as pa
        import pyarrow.parquet as pq

        index = len(self.episodes)
        chunk = index // CHUNK_SIZE
        source = episode.raw
        frames = episode.frames
        video_path = self.root / f"videos/chunk-{chunk:03d}/{self.config.camera_target_key}/episode_{index:06d}.mp4"
        video_path.parent.mkdir(parents=True, exist_ok=True)
        self.video_futures.append(self.video_pool.submit(self._write_video, source, frames, video_path))

        table = pa.Table.from_pydict(
            {
                self.config.state_field: [row.tolist() for row in episode.state],
                self.config.action_field: [row.tolist() for row in episode.hand_action],
                self.config.body_token_field: [row.tolist() for row in episode.body_token],
                INSTRUCTION_KEY: [source_instruction(self.config, source.collection)] * frames,
                "next.done": [False] * (frames - 1) + [True],
                MASK_KEY: [row.tolist() for row in episode.action_mask],
                ANCHOR_MASK_KEY: episode.anchor_valid.tolist(),
                "timestamp": episode.timestamp.astype(np.float32).tolist(),
                "frame_index": list(range(frames)),
                "episode_index": [index] * frames,
                "index": list(range(self.total_frames, self.total_frames + frames)),
                "task_index": [self.task_index(source.collection)] * frames,
            },
            schema=self.schema,
        )
        parquet_path = self.root / f"data/chunk-{chunk:03d}/episode_{index:06d}.parquet"
        parquet_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = parquet_path.with_suffix(".parquet.partial")
        pq.write_table(table, temporary, compression="snappy")
        temporary.replace(parquet_path)

        episode_stats = self._episode_stats(episode)
        for name, values in enumerate_stats(self.config, episode):
            self.running[name].update(values)

        record = _EpisodeRecord(
            episode_index=index,
            length=frames,
            task_index=self.task_index(source.collection),
            task=source_instruction(self.config, source.collection),
            source_collection=source.collection,
            source_episode_index=source.episode_index,
            source_data_file=source.data_file,
            frames_valid=int(episode.training_valid_mask.sum()),
            anchors_valid=int(episode.anchor_valid.sum()),
            video_start_s=source.video_start_s,
            video_stop_s=source.video_stop_s,
            parquet=parquet_path,
            video=video_path,
            stats=episode_stats,
            repair=episode.raw.repair.as_dict(),
        )
        self.episodes.append(record)
        self.total_frames += frames
        return record

    def _episode_stats(self, episode: ConvertedEpisode) -> dict[str, dict[str, Any]]:
        stats = {}
        for name, values in enumerate_stats(self.config, episode):
            block = _RunningStats()
            block.update(values)
            stats[name] = block.resolve()
        return stats

    def _write_video(self, source, frames: int, target: Path) -> None:
        if target.is_file() and target.stat().st_size > 0 and not self.force:
            try:
                if _frame_count(target) == frames:
                    return
            except Exception:
                pass
            target.unlink(missing_ok=True)
        video = self.config.video
        # Every source segment starts on an exact 30 Hz frame boundary and spans
        # exactly ``frames`` frames, so a frame count is the exact and
        # deterministic cut; a duration in seconds would round up one frame.
        command = [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-ss", f"{source.video_start_s:.6f}",
            "-i", str(source.video_file),
            "-frames:v", str(frames),
            "-an",
            "-c:v", "libx264",
            "-preset", str(video["preset"]),
            "-crf", str(video["crf"]),
            "-pix_fmt", str(video["pix_fmt"]),
            "-movflags", "+faststart",
            "-y", str(target),
        ]
        if not source.video_file.is_file():
            raise FileNotFoundError(f"missing source video segment: {source.video_file}")
        process = subprocess.run(command, capture_output=True, text=True, timeout=1800)
        if process.returncode != 0:
            target.unlink(missing_ok=True)
            detail = (process.stderr or "").strip().splitlines()
            raise RuntimeError(f"ffmpeg failed for {source.collection}/{source.episode_index}: {detail[-1] if detail else process.returncode}")
        got = _frame_count(target)
        if got != frames:
            target.unlink(missing_ok=True)
            raise ValueError(
                f"{source.collection}/{source.episode_index}: clip has {got} frames, the episode has {frames}"
            )

    # -- metadata ----------------------------------------------------------
    def task_index(self, collection: str) -> int:
        return self.config.collection_names.index(collection)

    def finalize(self) -> dict[str, Any]:
        if not self.episodes:
            raise ValueError(f"{self.split}: refusing to write an empty split")
        self.video_pool.shutdown(wait=True)
        for future in self.video_futures:
            future.result()
        stats = {name: accumulator.resolve() for name, accumulator in self.running.items()}
        _write_jsonl(
            self.root / "meta/tasks.jsonl",
            [
                {"task_index": index, "task": self.config.tasks[name]}
                for index, name in enumerate(self.config.collection_names)
            ],
        )
        _write_jsonl(
            self.root / "meta/episodes.jsonl",
            [
                {
                    "episode_index": record.episode_index,
                    "tasks": [record.task],
                    "length": record.length,
                    "source_collection": record.source_collection,
                    "source_episode_index": record.source_episode_index,
                    "source_data_file": record.source_data_file,
                    "frames_valid": record.frames_valid,
                    "anchors_valid": record.anchors_valid,
                    "video_start_s": record.video_start_s,
                    "video_stop_s": record.video_stop_s,
                    "hand_repair": record.repair,
                }
                for record in self.episodes
            ],
        )
        _write_jsonl(
            self.root / "meta/episodes_stats.jsonl",
            [{"episode_index": record.episode_index, "stats": record.stats} for record in self.episodes],
        )
        psi0_stats = {name: stats[name] for name in STAT_FEATURES}
        physical = verify_stats_are_physical(psi0_stats, self.config)
        (self.root / "meta/stats_psi0.json").write_text(
            json.dumps(psi0_stats, indent=4) + "\n", encoding="utf-8"
        )
        shutil.copy(self.root / "meta/stats_psi0.json", self.root / "meta/stats.json")
        (self.root / "meta/modality.json").write_text(
            json.dumps(_modality(self.config), indent=2) + "\n", encoding="utf-8"
        )
        info = {
            "codebase_version": CODEBASE_VERSION,
            "robot_type": ROBOT_TYPE,
            "total_episodes": len(self.episodes),
            "total_frames": self.total_frames,
            "total_tasks": len(self.config.collections),
            "total_videos": len(self.episodes),
            "total_chunks": 1 + (len(self.episodes) - 1) // CHUNK_SIZE,
            "chunks_size": CHUNK_SIZE,
            "fps": self.config.dataset_fps,
            "splits": {self.split: f"0:{len(self.episodes)}"},
            "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
            "features": feature_specs(self.config),
        }
        (self.root / "meta/info.json").write_text(json.dumps(info, indent=4) + "\n", encoding="utf-8")
        return {
            "split": self.split,
            "root": str(self.root),
            "episodes": len(self.episodes),
            "frames": self.total_frames,
            "tasks": len(self.config.collections),
            "anchors_valid": int(sum(record.anchors_valid for record in self.episodes)),
            "episodes_per_task": {
                name: sum(1 for record in self.episodes if record.source_collection == name)
                for name in self.config.collection_names
            },
            "stats_source": "this split",
            "stats_physical": physical,
            "hand_repair": {
                "episodes_repaired": sum(
                    1 for record in self.episodes if record.repair.get("invalid_source_samples")
                ),
                "invalid_source_samples": sum(
                    int(record.repair.get("invalid_source_samples", 0)) for record in self.episodes
                ),
            },
        }

    def summary(self) -> dict[str, Any]:
        return {
            "split": self.split,
            "episodes": len(self.episodes),
            "frames": self.total_frames,
            "parquet_sha256": {record.parquet.name: file_sha256(record.parquet) for record in self.episodes},
            "video_sha256": {record.video.name: file_sha256(record.video) for record in self.episodes},
        }


def _modality(config: ConversionConfig) -> dict[str, Any]:
    """LeRobot modality boxes; every field the Psi0 repack reads is declared."""
    return {
        "state": {"joint_positions": {"start": 0, "end": len(CANONICAL_STATE_NAMES)}},
        "action": {
            "hand_joints": {"start": 0, "end": HAND_DIM, "original_key": config.action_field},
            "body_token": {"start": 0, "end": BODY_TOKEN_DIM, "original_key": config.body_token_field},
        },
        "video": {"egocentric": {"original_key": config.camera_target_key}},
        "annotation": {INSTRUCTION_KEY: {}},
        "validity": {
            "action_mask": {"original_key": MASK_KEY, "width": ACTION_MODEL_DIM},
            "anchor_mask": {"original_key": ANCHOR_MASK_KEY, "horizon": config.action_chunk_size},
        },
        "model_contract": {"state_dim": STATE_MODEL_DIM, "action_dim": ACTION_MODEL_DIM},
    }


def enumerate_stats(config: ConversionConfig, episode: ConvertedEpisode):
    """The numeric vectors that carry statistics, by feature name."""
    yield config.state_field, episode.state
    yield config.action_field, episode.hand_action
    yield config.body_token_field, episode.body_token


def verify_stats_are_physical(
    stats: dict[str, dict[str, Any]],
    config: ConversionConfig,
    *,
    bound_rad: float = PHYSICAL_HAND_SAMPLE_BOUND_RAD,
) -> dict[str, Any]:
    """Fail closed if a state statistic is not a pose the robot can hold.

    The state normaliser divides by ``max - min``, so a single non-physical
    sample silently rescales a whole channel towards a constant -- which is how
    upstream spikes of up to 3363 rad collapsed four Dex3 channels to ~0.02% of
    their usable range.  ``repair_nonphysical_samples`` removes those samples at
    read time; this check is the independent guarantee that no burst reaching
    the statistics escaped it (a new source file, a changed reader, a future
    collector regression).

    The bound is applied to ``observation.state`` only: the 64D body token is a
    latent action with no physical constraint, and the 14D hand action is
    already bounded by the SONIC encoder contract.
    """
    block = stats.get(config.state_field)
    if not isinstance(block, dict):
        raise ValueError(f"statistics carry no {config.state_field!r} block")
    low = np.asarray(block["min"], dtype=np.float64)
    high = np.asarray(block["max"], dtype=np.float64)
    if low.shape != (len(CANONICAL_STATE_NAMES),) or high.shape != low.shape:
        raise ValueError(
            f"{config.state_field} statistics are {low.shape}/{high.shape}, "
            f"expected ({len(CANONICAL_STATE_NAMES)},)"
        )
    # The 15 synthetic legs/waist channels are a constant standing pose; the
    # measured channels are the remaining 28.
    offset = LEGS_WAIST_SLICE.stop
    measured = slice(offset, len(CANONICAL_STATE_NAMES))
    low_measured, high_measured = low[measured], high[measured]
    worst = float(np.abs(np.concatenate([low_measured, high_measured])).max())
    if worst > bound_rad:
        # ``offset`` maps a position in either concatenated half back to its
        # canonical channel name.
        half = low_measured.size
        offenders = sorted(
            {
                CANONICAL_STATE_NAMES[offset + int(index) % half]
                for index in np.flatnonzero(
                    np.abs(np.concatenate([low_measured, high_measured])) > bound_rad
                )
            }
        )
        raise ValueError(
            f"{config.state_field} statistics reach {worst:.4f} rad, beyond the physical "
            f"bound {bound_rad} rad, on {offenders}; a non-physical sample reached the "
            "normaliser, which would collapse those channels"
        )
    return {"state_abs_max_rad": worst, "bound_rad": bound_rad}


def source_instruction(config: ConversionConfig, collection: str) -> str:
    return config.tasks[collection]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _frame_count(path: Path) -> int:
    import av

    with av.open(str(path)) as container:
        return int(container.streams.video[0].frames)
