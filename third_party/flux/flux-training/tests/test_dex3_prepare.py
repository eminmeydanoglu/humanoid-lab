"""Unitree Dex3 data-contract checks without a full dataset download."""

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from examples.dex3.prepare import episode_tasks, reconcile_data_references, task_texts, verify_duplicate
from flux_action.config import PolicyConfig
from flux_action.data.lerobot.index import feature_names
from flux_action.training.trainer import TrainConfig


def test_unitree_nested_joint_names():
    names = [f"joint_{i}" for i in range(28)]
    assert feature_names({"names": [names]}, 28) == names
    assert feature_names({"names": [names[:-1]]}, 28) is None


def test_v3_task_metadata_overrides_stale_episode_label(tmp_path):
    root = tmp_path / "dataset"
    (root / "meta/episodes/chunk-000").mkdir(parents=True)
    pq.write_table(
        pa.table({"task_index": [0], "__index_level_0__": ["Put bottle into plate"]}),
        root / "meta/tasks.parquet",
    )
    pq.write_table(
        pa.table({"episode_index": [0, 1], "tasks": [["Pick red cup"], ["Pick red cup"]]}),
        root / "meta/episodes/chunk-000/file-000.parquet",
    )
    assert task_texts(root) == {0: "Put bottle into plate"}
    assert episode_tasks(root) == {0: "Put bottle into plate", 1: "Put bottle into plate"}


def test_duplicate_requires_matching_rows_and_selected_camera(tmp_path):
    from examples.dex3.prepare import CAMERA, DUPLICATE, ORIGINAL

    for name in (ORIGINAL, DUPLICATE):
        for file in ("data/chunk-000/file-000.parquet", f"videos/{CAMERA}/chunk-000/file-000.mp4"):
            path = tmp_path / name / file
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"same recording")
    verify_duplicate(tmp_path)
    (tmp_path / DUPLICATE / f"videos/{CAMERA}/chunk-000/file-000.mp4").write_bytes(b"different")
    with pytest.raises(ValueError, match="differs"):
        verify_duplicate(tmp_path)


def test_file_relative_episode_offset_reconciles_to_global_rows(tmp_path):
    root = tmp_path / "source"
    file = root / "data/chunk-000/file-001.parquet"
    file.parent.mkdir(parents=True)
    pq.write_table(
        pa.table({"episode_index": [5, 5, 6, 6], "index": [100, 101, 102, 103], "frame_index": [0, 1, 0, 1]}),
        file,
    )
    manifest = {
        "episodes": [
            {
                "episode_index": 5,
                "episode_id": "episode_000005",
                "n_frames": 2,
                "data_file": "data/chunk-000/file-000.parquet",
                "from_index": 100,
            },
            {
                "episode_index": 6,
                "episode_id": "episode_000006",
                "n_frames": 2,
                "data_file": "data/chunk-000/file-001.parquet",
                "from_index": 2,
            },
        ]
    }
    pq.write_table(
        pa.table({"episode_index": [4], "index": [99], "frame_index": [0]}),
        root / "data/chunk-000/file-000.parquet",
    )
    repairs = reconcile_data_references(root, manifest)
    assert len(repairs) == 2
    assert [e["from_index"] for e in manifest["episodes"]] == [100, 102]
    assert {e["data_file"] for e in manifest["episodes"]} == {"data/chunk-000/file-001.parquet"}


def test_dex3_config_agrees_with_single_camera_absolute_index():
    path = Path(__file__).resolve().parents[1] / "configs/dex3/train.json"
    config = TrainConfig(**json.loads(path.read_text()))
    policy = PolicyConfig(**config.policy)
    policy.validate_training()
    assert (policy.action_dim, policy.fps, policy.chunk_size) == (28, 30, 32)
    assert (policy.camera_layout, policy.camera_keys) == ("single", ("images.head",))
    assert policy.action_parameterization == "absolute"
    assert policy.absolute_action_dims == policy.gripper_flip_dims == ()
    assert config.frame_hw == (256, 256) and config.decoder == "pyav"
