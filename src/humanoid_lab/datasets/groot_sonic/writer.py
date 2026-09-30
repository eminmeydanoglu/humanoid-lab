"""LeRobot v2.1 writer for the GR00T Unitree Dex3 / SONIC training pack.

The split layout is the LeRobot v2.1 layout the pinned GR00T reads directly:
``meta/{info,tasks,episodes,episodes_stats,modality,stats_groot_sonic}.json``
plus one parquet and one mp4 per episode. Episodes are written one at a time and
metadata is composed at the end, so a partial run leaves data files but no
metadata, and re-running skips the clips that are already complete.

The ego video is re-encoded at the pack frame rate with a causal last-frame hold
(``fps=50:round=down``): every output frame is a real source frame, the newest
one at or before the row's timestamp, and the clip is required to hold exactly
as many frames as the episode has rows.

The loader-facing statistics are *not* written here. GR00T's official
``gr00t/data/stats.py`` produces ``meta/stats.json`` in the training image; this
writer publishes its own ``meta/stats_groot_sonic.json`` with the manifest's
provenance and records the official command instead of faking that file.
"""

from __future__ import annotations

import json
import subprocess
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .contract import (
    CANONICAL_ACTION_HAND_NAMES,
    CANONICAL_STATE_NAMES,
    GRAVITY_FIELD,
    HAND_DIM,
    LEFT_HAND_FIELD,
    MOTION_TOKEN_DIM,
    MOTION_TOKEN_FIELD,
    RIGHT_HAND_FIELD,
    STATE_DIM,
    STATE_FIELD,
    ConversionConfig,
    official_stats_command,
)
from .convert import ConvertedEpisode, RawEpisode
from .modality import modality_payload

CODEBASE_VERSION = "v2.1"
ROBOT_TYPE = "unitree_g1"
CHUNK_SIZE = 1000

#: Vector fields that carry statistics; the instruction and flags do not.
STAT_FEATURES = (STATE_FIELD, GRAVITY_FIELD, MOTION_TOKEN_FIELD, LEFT_HAND_FIELD, RIGHT_HAND_FIELD)

DEFAULT_FEATURE_SPECS: dict[str, dict[str, Any]] = {
    "timestamp": {"dtype": "float32", "shape": [1], "names": None},
    "frame_index": {"dtype": "int64", "shape": [1], "names": None},
    "episode_index": {"dtype": "int64", "shape": [1], "names": None},
    "index": {"dtype": "int64", "shape": [1], "names": None},
    "task_index": {"dtype": "int64", "shape": [1], "names": None},
    "next.done": {"dtype": "bool", "shape": [1], "names": None},
}


def feature_specs(config: ConversionConfig) -> dict[str, dict[str, Any]]:
    """The exact ``info.json`` feature box, with joint names on every vector."""
    fps = float(config.dataset_fps)
    return {
        STATE_FIELD: {
            "dtype": "float32",
            "shape": [STATE_DIM],
            "names": list(CANONICAL_STATE_NAMES),
            "fps": fps,
        },
        GRAVITY_FIELD: {"dtype": "float32", "shape": [3], "names": ["x", "y", "z"], "fps": fps},
        MOTION_TOKEN_FIELD: {
            "dtype": "float32",
            "shape": [MOTION_TOKEN_DIM],
            "names": [f"motion_token_{index:02d}" for index in range(MOTION_TOKEN_DIM)],
            "fps": fps,
        },
        LEFT_HAND_FIELD: {
            "dtype": "float32",
            "shape": [HAND_DIM],
            "names": list(CANONICAL_ACTION_HAND_NAMES[:HAND_DIM]),
            "fps": fps,
        },
        RIGHT_HAND_FIELD: {
            "dtype": "float32",
            "shape": [HAND_DIM],
            "names": list(CANONICAL_ACTION_HAND_NAMES[HAND_DIM:]),
            "fps": fps,
        },
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


def info_payload(config: ConversionConfig, split: str, episodes: int, frames: int) -> dict[str, Any]:
    """The LeRobot v2.1 ``info.json`` of one split."""
    return {
        "codebase_version": CODEBASE_VERSION,
        "robot_type": ROBOT_TYPE,
        "total_episodes": episodes,
        "total_frames": frames,
        "total_tasks": len(config.collections),
        "total_videos": episodes,
        "total_chunks": 1 + (episodes - 1) // CHUNK_SIZE,
        "chunks_size": CHUNK_SIZE,
        "fps": config.dataset_fps,
        "splits": {split: f"0:{episodes}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": feature_specs(config),
    }


def tasks_rows(config: ConversionConfig) -> list[dict[str, Any]]:
    return [
        {"task_index": index, "task": config.tasks[name]}
        for index, name in enumerate(config.collection_names)
    ]


def _parquet_schema(config: ConversionConfig) -> Any:
    import pyarrow as pa

    return pa.schema(
        [
            pa.field(STATE_FIELD, pa.list_(pa.float32())),
            pa.field(GRAVITY_FIELD, pa.list_(pa.float32())),
            pa.field(MOTION_TOKEN_FIELD, pa.list_(pa.float32())),
            pa.field(LEFT_HAND_FIELD, pa.list_(pa.float32())),
            pa.field(RIGHT_HAND_FIELD, pa.list_(pa.float32())),
            pa.field("timestamp", pa.float32()),
            pa.field("frame_index", pa.int64()),
            pa.field("episode_index", pa.int64()),
            pa.field("index", pa.int64()),
            pa.field("task_index", pa.int64()),
            pa.field("next.done", pa.bool_()),
        ]
    )


def video_command(config: ConversionConfig, raw: RawEpisode, rows: int, offset_s: float, target: Path) -> list[str]:
    """The ffmpeg argv for one ego clip.

    ``-ss`` seeks to the retained span's first source instant, ``fps=50:round=down``
    holds the newest source frame at or before each output row's timestamp, and
    ``-frames:v`` pins the exact row count that is verified afterwards.
    """
    video = config.video
    return [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-ss", f"{offset_s:.6f}",
        "-i", str(raw.video_file),
        "-vf", f"fps={config.dataset_fps}:round=down",
        "-frames:v", str(rows),
        "-an",
        "-c:v", "libx264",
        "-preset", str(video["preset"]),
        "-crf", str(video["crf"]),
        "-pix_fmt", str(video["pix_fmt"]),
        "-movflags", "+faststart",
        "-y", str(target),
    ]


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
        block_min, block_max = array.min(axis=0), array.max(axis=0)
        block_mean = array.mean(axis=0)
        block_m2 = ((array - block_mean) ** 2).sum(axis=0)
        if self.count == 0:
            self.min, self.max, self.mean, self.m2 = block_min, block_max, block_mean, block_m2
        else:
            total = self.count + array.shape[0]
            delta = block_mean - self.mean
            self.mean = self.mean + delta * (array.shape[0] / total)
            self.m2 = self.m2 + block_m2 + delta**2 * (self.count * array.shape[0] / total)
            self.min = np.minimum(self.min, block_min)
            self.max = np.maximum(self.max, block_max)
        self.count += int(array.shape[0])

    def resolve(self) -> dict[str, Any]:
        if self.count == 0 or self.mean is None:
            raise ValueError("no frames were accumulated")
        return {
            "min": self.min.tolist(),
            "max": self.max.tolist(),
            "mean": self.mean.tolist(),
            "std": np.sqrt(self.m2 / self.count).tolist(),
            "count": [self.count],
        }


def episode_stats(episode: ConvertedEpisode) -> dict[str, dict[str, Any]]:
    """Per-episode statistics of the numeric fields, in the v2.1 shape."""
    blocks = {
        STATE_FIELD: episode.state,
        GRAVITY_FIELD: episode.gravity,
        MOTION_TOKEN_FIELD: episode.motion_token,
        LEFT_HAND_FIELD: episode.left_hand,
        RIGHT_HAND_FIELD: episode.right_hand,
    }
    result: dict[str, dict[str, Any]] = {}
    for name, values in blocks.items():
        accumulator = _RunningStats()
        accumulator.update(values)
        result[name] = accumulator.resolve()
    return result


@dataclass
class EpisodeRecord:
    """Everything one written episode contributes to the pack metadata."""

    episode_index: int
    length: int
    task_index: int
    task: str
    source_collection: str
    source_episode_index: int
    source_data_file: str
    source_frames: int
    source_fps: float
    corpus_frames: int
    trimmed_rows: int
    corpus_hand_max_error: float
    video_start_s: float
    video_stop_s: float
    repair: dict[str, Any]
    parquet: Path
    video: Path
    stats: dict[str, dict[str, Any]] = field(default_factory=dict)

    def metadata_row(self) -> dict[str, Any]:
        """The ``meta/episodes.jsonl`` row: v2.1 fields plus repair provenance."""
        return {
            "episode_index": self.episode_index,
            "tasks": [self.task],
            "length": self.length,
            "source_collection": self.source_collection,
            "source_episode_index": self.source_episode_index,
            "source_data_file": self.source_data_file,
            "source_frames": self.source_frames,
            "source_fps": self.source_fps,
            "corpus_frames": self.corpus_frames,
            "trimmed_rows": self.trimmed_rows,
            "hand_repair": self.repair,
            "corpus_hand_max_error_rad": self.corpus_hand_max_error,
            "video_start_s": self.video_start_s,
            "video_stop_s": self.video_stop_s,
        }


class SplitWriter:
    """Writes one LeRobot v2.1 split directory (``train/`` or ``val/``)."""

    def __init__(
        self,
        root: Path,
        config: ConversionConfig,
        split: str,
        *,
        force: bool = False,
        provenance: dict[str, Any] | None = None,
    ):
        self.root = Path(root)
        self.config = config
        self.split = split
        self.force = force
        self.provenance = dict(provenance or {})
        self.episodes: list[EpisodeRecord] = []
        self.total_frames = 0
        self.running = {name: _RunningStats() for name in STAT_FEATURES}
        self.schema = _parquet_schema(config)
        self.video_pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix=f"{split}-video")
        self.video_futures: list[Future[None]] = []
        for directory in ("data/chunk-000", f"videos/chunk-000/{config.camera_target_key}", "meta"):
            (self.root / directory).mkdir(parents=True, exist_ok=True)

    # -- writing -----------------------------------------------------------
    def add(self, episode: ConvertedEpisode) -> EpisodeRecord:
        import pyarrow as pa
        import pyarrow.parquet as pq

        index = len(self.episodes)
        chunk = index // CHUNK_SIZE
        raw, frames = episode.raw, episode.frames
        video_path = (
            self.root / f"videos/chunk-{chunk:03d}/{self.config.camera_target_key}/episode_{index:06d}.mp4"
        )
        video_path.parent.mkdir(parents=True, exist_ok=True)
        self.video_futures.append(
            self.video_pool.submit(self._write_video, raw, episode.timestamp, frames, video_path)
        )

        table = pa.Table.from_pydict(
            {
                STATE_FIELD: [row.tolist() for row in episode.state],
                GRAVITY_FIELD: [row.tolist() for row in episode.gravity],
                MOTION_TOKEN_FIELD: [row.tolist() for row in episode.motion_token],
                LEFT_HAND_FIELD: [row.tolist() for row in episode.left_hand],
                RIGHT_HAND_FIELD: [row.tolist() for row in episode.right_hand],
                "timestamp": episode.timestamp.astype(np.float32).tolist(),
                "frame_index": list(range(frames)),
                "episode_index": [index] * frames,
                "index": list(range(self.total_frames, self.total_frames + frames)),
                "task_index": [self.task_index(raw.collection)] * frames,
                "next.done": [False] * (frames - 1) + [True],
            },
            schema=self.schema,
        )
        parquet_path = self.root / f"data/chunk-{chunk:03d}/episode_{index:06d}.parquet"
        parquet_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = parquet_path.with_suffix(".parquet.partial")
        pq.write_table(table, temporary, compression="snappy")
        temporary.replace(parquet_path)

        for name, values in enumerate_stats(episode):
            self.running[name].update(values)
        record = EpisodeRecord(
            episode_index=index,
            length=frames,
            task_index=self.task_index(raw.collection),
            task=self.instruction(raw.collection),
            source_collection=raw.collection,
            source_episode_index=raw.episode_index,
            source_data_file=raw.data_file,
            source_frames=raw.frames,
            source_fps=float(self.config.source_fps),
            corpus_frames=episode.corpus_rows,
            trimmed_rows=episode.trimmed_rows,
            corpus_hand_max_error=episode.corpus_hand_max_error,
            video_start_s=raw.video_start_s,
            video_stop_s=raw.video_stop_s,
            repair=episode.repair.payload(),
            parquet=parquet_path,
            video=video_path,
        )
        record.stats = episode_stats(episode)
        self.episodes.append(record)
        self.total_frames += frames
        return record

    def _write_video(self, raw: RawEpisode, timestamps: np.ndarray, frames: int, target: Path) -> None:
        if target.is_file() and target.stat().st_size > 0 and not self.force:
            try:
                if _video_info(target)["frames"] == frames:
                    return
            except Exception:
                pass
            target.unlink(missing_ok=True)
        if not raw.video_file.is_file():
            raise FileNotFoundError(f"missing source video segment: {raw.video_file}")
        offset = raw.video_start_s + float(timestamps[0])
        command = video_command(self.config, raw, frames, offset, target)
        process = subprocess.run(command, capture_output=True, text=True, timeout=3600)
        if process.returncode != 0:
            target.unlink(missing_ok=True)
            detail = (process.stderr or "").strip().splitlines()
            raise RuntimeError(
                f"ffmpeg failed for {raw.label}: {detail[-1] if detail else process.returncode}"
            )
        info = _video_info(target)
        if info["frames"] != frames:
            target.unlink(missing_ok=True)
            raise ValueError(f"{raw.label}: clip has {info['frames']} frames, the episode has {frames} rows")
        if (info["width"], info["height"]) != (self.config.camera_width, self.config.camera_height):
            target.unlink(missing_ok=True)
            raise ValueError(f"{raw.label}: clip is {info['width']}x{info['height']}, expected 640x480")

    # -- metadata ----------------------------------------------------------
    def task_index(self, collection: str) -> int:
        return self.config.collection_names.index(collection)

    def instruction(self, collection: str) -> str:
        return self.config.tasks[collection]

    def finalize(self) -> dict[str, Any]:
        if not self.episodes:
            raise ValueError(f"{self.split}: refusing to write an empty split")
        self.video_pool.shutdown(wait=True)
        for future in self.video_futures:
            future.result()
        meta = self.root / "meta"
        _write_jsonl(meta / "tasks.jsonl", tasks_rows(self.config))
        _write_jsonl(meta / "episodes.jsonl", [record.metadata_row() for record in self.episodes])
        _write_jsonl(
            meta / "episodes_stats.jsonl",
            [{"episode_index": record.episode_index, "stats": record.stats} for record in self.episodes],
        )
        stats = {name: accumulator.resolve() for name, accumulator in self.running.items()}
        (meta / self.config.stats_own_file).write_text(
            json.dumps(stats, indent=4, sort_keys=True) + "\n", encoding="utf-8"
        )
        (meta / "modality.json").write_text(
            json.dumps(modality_payload(self.config), indent=2) + "\n", encoding="utf-8"
        )
        info = info_payload(self.config, self.split, len(self.episodes), self.total_frames)
        (meta / "info.json").write_text(json.dumps(info, indent=4) + "\n", encoding="utf-8")
        provenance = {
            **self.provenance,
            "split": self.split,
            "root": str(self.root),
            "episodes": len(self.episodes),
            "frames": self.total_frames,
            "state_sources": {
                "legs_and_waist": self.config.raw["state"]["lower_body_semantics"],
                "arms": self.config.raw["state"]["semantics"]["arms"],
                "hands": self.config.raw["state"]["semantics"]["hands"],
                "projected_gravity": "synthetic upright [0, 0, -1]",
            },
            "hand_repair": repair_totals(self.episodes),
            "statistics": {
                "own_file": f"meta/{self.config.stats_own_file}",
                "producer": "humanoid_lab.datasets.groot_sonic (not the official GR00T tool)",
                "official_file": f"meta/{self.config.stats_official_file}",
                "official_command": official_stats_command(self.config, self.root),
                "official_produced": False,
            },
        }
        (meta / "provenance.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return {
            "split": self.split,
            "root": str(self.root),
            "episodes": len(self.episodes),
            "frames": self.total_frames,
            "tasks": len(self.config.collections),
            "episodes_per_task": {
                name: sum(1 for record in self.episodes if record.source_collection == name)
                for name in self.config.collection_names
            },
            "hand_repair": repair_totals(self.episodes),
            "corpus_hand_max_error_rad": max(
                (record.corpus_hand_max_error for record in self.episodes), default=0.0
            ),
            "stats_source": f"meta/{self.config.stats_own_file}",
        }


def repair_totals(records: list[EpisodeRecord]) -> dict[str, Any]:
    """Total repaired samples and per-channel counts over a split."""
    channels: dict[str, int] = {}
    for record in records:
        for channel, count in record.repair["channels"].items():
            channels[channel] = channels.get(channel, 0) + int(count)
    return {
        "invalid_source_samples": int(sum(channels.values())),
        "episodes_with_repair": sum(1 for record in records if record.repair["invalid_source_samples"]),
        "channels": dict(sorted(channels.items())),
    }


def enumerate_stats(episode: ConvertedEpisode):
    """The numeric vectors that carry statistics, by feature name."""
    yield STATE_FIELD, episode.state
    yield GRAVITY_FIELD, episode.gravity
    yield MOTION_TOKEN_FIELD, episode.motion_token
    yield LEFT_HAND_FIELD, episode.left_hand
    yield RIGHT_HAND_FIELD, episode.right_hand


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _video_info(path: Path, *, count_frames: bool = False) -> dict[str, Any]:
    """Container metadata, decoding only when the frame count is not declared."""
    import av

    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        frames = int(stream.frames)
        if count_frames or frames <= 0:
            frames = sum(1 for _ in container.decode(stream))
        return {
            "frames": frames,
            "width": int(stream.codec_context.width),
            "height": int(stream.codec_context.height),
            "fps": float(stream.average_rate) if stream.average_rate else None,
            "codec": str(stream.codec_context.name),
            "pix_fmt": str(stream.codec_context.format.name) if stream.codec_context.format else None,
        }


__all__ = [
    "CHUNK_SIZE",
    "CODEBASE_VERSION",
    "EpisodeRecord",
    "ROBOT_TYPE",
    "STAT_FEATURES",
    "SplitWriter",
    "feature_specs",
    "info_payload",
    "repair_totals",
    "tasks_rows",
    "video_command",
]
