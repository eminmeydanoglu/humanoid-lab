"""Contract tests for the Psi0 Unitree Dex3 SONIC v1 pack.

The pack itself is produced by the dataset owner, so these build the metadata a
correct pack would have and mutate one thing at a time: a drift has to be caught
in the preflight, not in a forward pass.  The wide-mask requirement is the one
non-obvious rule -- Psi0's SonicRepackTransform reshapes the mask field to
``(action_chunk_size, action_dim)``, so a scalar ``(T,)`` ``training_valid_mask``
raises ``cannot reshape array of size T into shape (T, 80)``.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from humanoid_lab.datasets.psi0 import contract


def default_info(*, fps: float = contract.FPS, mask_key: str = contract.MASK_KEY) -> dict:
    return {
        "fps": fps,
        "total_episodes": 4,
        "total_frames": 400,
        "total_tasks": 13,
        "features": {
            contract.IMAGE_KEY: {"dtype": "video", "shape": [3, 480, 640]},
            contract.STATE_KEY: {"dtype": "float32", "shape": [43], "names": [["joint"] * 43]},
            contract.BODY_TOKEN_KEY: {"dtype": "float32", "shape": [64]},
            contract.HAND_KEY: {"dtype": "float32", "shape": [14]},
            mask_key: {"dtype": "float32", "shape": [80]},
            contract.ANCHOR_MASK_KEY: {"dtype": "bool", "shape": [1]},
            contract.INSTRUCTION_KEY: {"dtype": "string", "shape": [1]},
            "timestamp": {"dtype": "float32", "shape": [1]},
            "frame_index": {"dtype": "int64", "shape": [1]},
            "episode_index": {"dtype": "int64", "shape": [1]},
            "index": {"dtype": "int64", "shape": [1]},
            "task_index": {"dtype": "int64", "shape": [1]},
        },
    }


def default_stats(*, token_width: int = 64, hand_width: int = 14, state_width: int = 43) -> dict:
    def block(width: int, *, varying: bool = True) -> dict:
        return {"min": [0.0] * width, "max": [1.0 if varying else 0.0] * width}

    return {
        contract.BODY_TOKEN_KEY: block(token_width),
        contract.HAND_KEY: block(hand_width),
        contract.STATE_KEY: block(state_width, varying=False),
    }


def write_split(
    root: Path,
    split: str,
    *,
    info: dict | None = None,
    stats: dict | None = None,
    drop_meta: str | None = None,
) -> Path:
    repo = root / split
    meta = repo / "meta"
    meta.mkdir(parents=True)
    files = {
        "info.json": json.dumps(info if info is not None else default_info()),
        "modality.json": "{}",
        "tasks.jsonl": '{"task_index": 0, "task": "pick the doll up"}\n',
        "episodes.jsonl": '{"episode_index": 0, "tasks": ["pick the doll up"], "length": 100}\n',
        contract.STATS_FILENAME: json.dumps(stats if stats is not None else default_stats()),
    }
    for name, body in files.items():
        if name == drop_meta:
            continue
        (meta / name).write_text(body, encoding="utf-8")
    return repo


class ValidPackTest(unittest.TestCase):
    def test_pack_reports_what_the_loader_will_read(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / contract.DATASET_DIR
            write_split(root, contract.TRAIN_REPO_ID)
            write_split(root, contract.VAL_REPO_ID)
            train, val = contract.validate_pack(root)
            self.assertEqual(train.mask_key, contract.MASK_KEY)
            self.assertEqual(train.instruction_key, contract.INSTRUCTION_KEY)
            self.assertEqual(train.total_episodes, 4)
            self.assertEqual(train.task_count, 13)
            self.assertEqual(train.stats_path, root / "train" / contract.STATS_PATH)
            self.assertIn("4 episodes", train.summary())
            self.assertIn("4 episodes", val.summary())

    def test_plan_mask_name_is_accepted_when_it_is_wide(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / contract.DATASET_DIR
            for split in (contract.TRAIN_REPO_ID, contract.VAL_REPO_ID):
                write_split(root, split, info=default_info(mask_key="training_valid_mask"))
            train, _ = contract.validate_pack(root, mask_key="training_valid_mask")
            self.assertEqual(train.mask_key, "training_valid_mask")

    def test_padding_and_stats_widths_agree(self) -> None:
        self.assertEqual(contract.STATE_DIM + 2, contract.STATE_MODEL_DIM)
        self.assertEqual(contract.BODY_TOKEN_DIM + contract.HAND_DIM + 2, contract.ACTION_MODEL_DIM)
        self.assertEqual(contract.STATS_PATH, f"meta/{contract.STATS_FILENAME}")


class ContractViolationTest(unittest.TestCase):
    def assert_rejected(self, message: str, **kwargs) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / contract.DATASET_DIR
            write_split(root, contract.TRAIN_REPO_ID, **kwargs)
            with self.assertRaisesRegex(contract.DatasetContractError, message):
                contract.validate_dataset(root / contract.TRAIN_REPO_ID)

    def test_scalar_mask_is_rejected(self) -> None:
        """The loader reshapes the mask to (30, 80); a (T,) field cannot be read."""
        info = default_info()
        info["features"][contract.MASK_KEY] = {"dtype": "float32", "shape": [1]}
        self.assert_rejected("expected \\[80\\]", info=info)

    def test_mask_alias_with_the_plan_name_is_only_found_when_configured(self) -> None:
        """The loader reads one concrete field name, so the default stays strict."""
        info = default_info()
        del info["features"][contract.MASK_KEY]
        info["features"]["training_valid_mask"] = {"dtype": "float32", "shape": [80]}
        self.assert_rejected("declares no 'action.mask'", info=info)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / contract.DATASET_DIR
            write_split(root, contract.TRAIN_REPO_ID, info=info)
            report = contract.validate_dataset(root / contract.TRAIN_REPO_ID, mask_key="training_valid_mask")
            self.assertEqual(report.mask_key, "training_valid_mask")

    def test_other_fps_is_rejected(self) -> None:
        self.assert_rejected("expected 30.0", info=default_info(fps=50.0))

    def test_padded_hand_field_is_rejected(self) -> None:
        """The pack stores the 14D Dex3 targets, not the 78D or 80D vector."""
        info = default_info()
        info["features"][contract.HAND_KEY] = {"dtype": "float32", "shape": [80]}
        self.assert_rejected("expected \\[14\\]", info=info)

    def test_missing_state_feature_is_rejected(self) -> None:
        info = default_info()
        del info["features"][contract.STATE_KEY]
        self.assert_rejected("declares no feature", info=info)

    def test_already_padded_stats_are_rejected(self) -> None:
        self.assert_rejected("expected the raw width 64", stats=default_stats(token_width=80))

    def test_stats_without_a_block_is_rejected(self) -> None:
        stats = default_stats()
        del stats[contract.HAND_KEY]
        self.assert_rejected("no 'action' block", stats=stats)

    def test_non_varying_stats_are_rejected(self) -> None:
        stats = default_stats()
        stats[contract.BODY_TOKEN_KEY]["max"] = [0.0] * 64
        self.assert_rejected("no varying dimension", stats=stats)

    def test_non_finite_stats_are_rejected(self) -> None:
        stats = default_stats()
        stats[contract.BODY_TOKEN_KEY]["max"][0] = float("inf")
        self.assert_rejected("is not finite", stats=stats)

    def test_missing_anchor_mask_is_rejected(self) -> None:
        """The converter writes the strict anchor mask; a pack without it is not this contract."""
        info = default_info()
        del info["features"][contract.ANCHOR_MASK_KEY]
        self.assert_rejected("declares no feature 'anchor_valid'", info=info)

    def test_missing_instruction_feature_is_rejected(self) -> None:
        info = default_info()
        del info["features"][contract.INSTRUCTION_KEY]
        self.assert_rejected("declares no instruction feature", info=info)

    def test_missing_stats_file_is_rejected(self) -> None:
        self.assert_rejected("is missing", drop_meta=contract.STATS_FILENAME)

    def test_missing_split_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / contract.DATASET_DIR
            write_split(root, contract.TRAIN_REPO_ID)
            with self.assertRaisesRegex(contract.DatasetContractError, "dataset split is missing"):
                contract.validate_pack(root)

    def test_parquet_episode_tables_are_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / contract.DATASET_DIR
            repo = write_split(root, contract.TRAIN_REPO_ID, drop_meta=None)
            (repo / "meta" / "tasks.jsonl").unlink()
            (repo / "meta" / "tasks.parquet").write_bytes(b"")
            (repo / "meta" / "episodes").mkdir()
            (repo / "meta" / "episodes.jsonl").unlink()
            self.assertEqual(contract.validate_dataset(repo).total_episodes, 4)


if __name__ == "__main__":
    unittest.main()
