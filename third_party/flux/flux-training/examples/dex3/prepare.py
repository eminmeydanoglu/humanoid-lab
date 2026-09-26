"""Index the local Unitree G1 Dex3 collection for a single left head camera."""

from __future__ import annotations

import argparse
import filecmp
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from flux_action.data.droid.index import ROWS_FILENAME, write_manifest
from flux_action.data.lerobot.index import (
    STATISTICS_FILENAME,
    build_manifest,
    build_rows,
    build_statistics,
    camera_hw,
    feature_names,
    read_info,
    write_statistics,
)

CAMERA = "observation.images.cam_left_high"
STREAM = "head"
DUPLICATE = "G1_Dex3_GraspSquare_Dataset"
ORIGINAL = "G1_Dex3_BlockStacking_Dataset"


def task_texts(root: Path) -> dict[int, str]:
    table = pq.read_table(root / "meta/tasks.parquet").to_pydict()
    value_key = "task" if "task" in table else "__index_level_0__"
    if value_key not in table or "task_index" not in table:
        raise ValueError(f"{root.name}: cannot resolve task text in meta/tasks.parquet")
    return {int(i): str(t).strip() for i, t in zip(table["task_index"], table[value_key], strict=True)}


def episode_tasks(root: Path) -> dict[int, str]:
    tasks = task_texts(root)
    captions = {}
    for path in sorted((root / "meta/episodes").rglob("*.parquet")):
        for row in pq.read_table(path, columns=["episode_index", "tasks"]).to_pylist():
            labels = list(row["tasks"] or [])
            if len(labels) != 1 or len(tasks) != 1:
                raise ValueError(f"{root.name}: expected one task per episode and dataset")
            captions[int(row["episode_index"])] = next(iter(tasks.values()))
    return captions


def verify_duplicate(root: Path) -> None:
    a, b = root / ORIGINAL, root / DUPLICATE
    for subdir, ext in (("data", "*.parquet"), (f"videos/{CAMERA}", "*.mp4")):
        first = sorted((a / subdir).rglob(ext))
        second = sorted((b / subdir).rglob(ext))
        if not first or [p.relative_to(a) for p in first] != [p.relative_to(b) for p in second]:
            raise ValueError(f"{DUPLICATE}: unexpected file set; review before indexing")
        if not all(filecmp.cmp(p, b / p.relative_to(a), shallow=False) for p in first):
            raise ValueError(f"{DUPLICATE}: differs from {ORIGINAL}; review before indexing")


def reconcile_data_references(root: Path, manifest: dict) -> list[dict]:
    """Check file and global-row offsets against actual per-episode parquet rows."""
    if len(list((root / "data").rglob("*.parquet"))) == 1:
        return []
    observed = {}
    for path in sorted((root / "data").rglob("*.parquet")):
        table = pq.read_table(path, columns=["episode_index", "index", "frame_index"])
        episode_ids = table["episode_index"].to_numpy()
        indices = table["index"].to_numpy()
        frames = table["frame_index"].to_numpy()
        for episode_id in np.unique(episode_ids):
            selected = episode_ids == episode_id
            if int(episode_id) in observed or not np.array_equal(frames[selected], np.arange(selected.sum())):
                raise ValueError(f"{root.name}: non-contiguous or split episode {episode_id}")
            observed[int(episode_id)] = (path.relative_to(root).as_posix(), indices[selected])
    repairs = []
    for episode in manifest["episodes"]:
        file, indices = observed[episode["episode_index"]]
        start = int(indices[0])
        if len(indices) != episode["n_frames"] or not np.array_equal(
            indices, np.arange(start, start + len(indices))
        ):
            raise ValueError(f"{root.name}: invalid row indices for {episode['episode_id']}")
        if (episode["data_file"], episode["from_index"]) != (file, start):
            repairs.append(
                {
                    "dataset": root.name,
                    "episode": episode["episode_index"],
                    "metadata_file": episode["data_file"],
                    "metadata_from_index": episode["from_index"],
                    "actual_file": file,
                    "actual_from_index": start,
                }
            )
            episode["data_file"], episode["from_index"] = file, start
    return repairs


def prepare(source: Path, output: Path, *, chunk_size: int = 32, val_fraction: float = 0.1) -> dict:
    sources = json.loads((source / "sources.json").read_text())
    if ORIGINAL not in sources or DUPLICATE not in sources or not 0 < val_fraction < 1:
        raise ValueError("invalid collection or validation fraction")
    verify_duplicate(source)
    names = sorted(set(sources) - {DUPLICATE})
    manifests, disagreements, repairs = [], [], []
    order = None
    output.mkdir(parents=True, exist_ok=True)
    for name in names:
        root = source / name
        info = read_info(root)
        features = info["features"]
        states = feature_names(features["observation.state"], 28)
        actions = feature_names(features["action"], 28)
        if (
            info["fps"] != 30
            or features["observation.state"]["shape"] != [28]
            or features["action"]["shape"] != [28]
            or not states
            or states != actions
            or (order is not None and states != order)
            or CAMERA not in features
            or camera_hw(features, CAMERA) != (480, 640)
        ):
            raise ValueError(f"{name}: fps, 28-channel ordering or head camera differs")
        order = states
        n_val = max(1, round(info["total_episodes"] * val_fraction))
        manifest = build_manifest(
            root,
            {STREAM: CAMERA},
            chunk_size=chunk_size,
            val_episodes=n_val,
            dataset_id=sources[name],
        )
        if manifest["counts"]["eligible"] != info["total_episodes"]:
            raise ValueError(f"{name}: dropped episodes: {manifest['counts']}")
        repairs.extend(reconcile_data_references(root, manifest))
        captions = episode_tasks(root)
        for episode in manifest["episodes"]:
            expected = captions[episode["episode_index"]]
            if not expected:
                raise ValueError(f"{name}: empty task for {episode['episode_id']}")
            recorded = episode["caption"]
            if recorded != expected:
                disagreements.append(
                    {
                        "dataset": name,
                        "episode": episode["episode_index"],
                        "episode_task": recorded,
                        "task_metadata": expected,
                    }
                )
            episode["caption"] = expected
        part = output / "parts" / name
        part.mkdir(parents=True, exist_ok=True)
        result = build_rows(root, manifest, part / ROWS_FILENAME)
        if result["rejected"] or len(manifest["episodes"]) != info["total_episodes"]:
            raise ValueError(f"{name}: row validation failed: {result}")
        manifests.append((name, manifest, part / ROWS_FILENAME))

    total = sum(m["total_frames"] for _, m, _ in manifests)
    rows = np.lib.format.open_memmap(output / ROWS_FILENAME, mode="w+", dtype=np.float32, shape=(total, 56))
    episodes, files, offsets = [], {}, {}
    offset = 0
    for name, part_manifest, path in manifests:
        block = np.load(path, mmap_mode="r")
        n = part_manifest["total_frames"]
        rows[offset : offset + n] = block
        offsets[name] = offset
        for e in part_manifest["episodes"]:
            episode = dict(e)
            episode["source_episode_index"] = e["episode_index"]
            episode["episode_id"] = f"{name}/{e['episode_id']}"
            episode["episode_index"] = len(episodes)
            episode["from_index"] = offset + e["from_index"]
            episode["data_file"] = f"{name}/{e['data_file']}"
            episode["videos"] = {
                STREAM: {**e["videos"][STREAM], "file": f"{name}/{e['videos'][STREAM]['file']}"}
            }
            episodes.append(episode)
        files.update({f"{name}/{file}": record for file, record in part_manifest["files"].items()})
        offset += n
    rows.flush()
    manifest = {
        "format_version": 1,
        "kind": "lerobot",
        "dataset_id": "Unitree G1 Dex3 collection",
        "fps": 30,
        "chunk_size": chunk_size,
        "min_start": 1,
        "action_prev": True,
        "total_frames": total,
        "state_dim": 28,
        "action_dim": 28,
        "state_names": order,
        "action_names": order,
        "state_key": "observation.state",
        "action_key": "action",
        "cameras": {STREAM: CAMERA},
        "camera_order": [STREAM],
        "camera_hw": {STREAM: [480, 640]},
        "episodes": episodes,
        "files": files,
        "counts": {
            "eligible": len(episodes),
            "train_episodes": sum(e["split"] == "train" for e in episodes),
            "val_episodes": sum(e["split"] == "val" for e in episodes),
        },
    }
    train_manifest = {**manifest, "episodes": [e for e in episodes if e["split"] == "train"]}
    stats = build_statistics(
        train_manifest, rows, action_parameterization="absolute", absolute_action_dims=()
    )
    write_statistics(stats, output / STATISTICS_FILENAME)
    write_manifest(manifest, output / "manifest.json")
    audit = {
        "sources": names,
        "excluded_duplicate": DUPLICATE,
        "row_offsets": offsets,
        "task_source": "meta/tasks.parquet",
        "task_disagreements": disagreements,
        "row_reference_repairs": repairs,
        "statistics_split": "train",
        "train_episodes": manifest["counts"]["train_episodes"],
        "val_episodes": manifest["counts"]["val_episodes"],
    }
    (output / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--chunk-size", type=int, default=32)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    args = parser.parse_args()
    result = prepare(
        args.source_root, args.output_dir, chunk_size=args.chunk_size, val_fraction=args.val_fraction
    )
    print(json.dumps({k: v for k, v in result.items() if k != "task_disagreements"}, indent=2))
    print(f"Task metadata disagreements: {len(result['task_disagreements'])}; see audit.json")


if __name__ == "__main__":
    main()
