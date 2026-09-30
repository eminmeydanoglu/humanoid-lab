#!/usr/bin/env python3
"""Unit tests for scripts/groot-unitree-dex3-sonic-preflight.py.

The shell suite covers the launcher argv and the wrapper's refusals; these
tests pin the validator itself: the stock UNITREE_G1_SONIC contract, torn-write
markers (including the dot-prefixed tmp files the official stats writer leaves
behind), the statistics fingerprint gate, and the artifact validator.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import pathlib
import subprocess
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
PREFLIGHT = ROOT / "scripts" / "groot-unitree-dex3-sonic-preflight.py"

STATE_DIMS = {
    "left_leg": 6,
    "right_leg": 6,
    "waist": 3,
    "left_arm": 7,
    "right_arm": 7,
    "left_hand": 7,
    "right_hand": 7,
    "projected_gravity": 3,
}
# Official SONIC exporter storage order inside observation.state (hands next to
# their arms), while the modality groups iterate in the registered order.
STATE_STORAGE_SLICES = {
    "left_leg": (0, 6),
    "right_leg": (6, 12),
    "waist": (12, 15),
    "left_arm": (15, 22),
    "left_hand": (22, 29),
    "right_arm": (29, 36),
    "right_hand": (36, 43),
}
REGISTERED_STATE_KEYS = [
    "left_leg",
    "right_leg",
    "waist",
    "left_arm",
    "right_arm",
    "left_hand",
    "right_hand",
    "projected_gravity",
]
ACTION_FIELDS = {
    "motion_token": ("action.motion_token", 64),
    "left_hand_joints": ("teleop.left_hand_joints", 7),
    "right_hand_joints": ("teleop.right_hand_joints", 7),
}


def load_module():
    spec = importlib.util.spec_from_file_location("groot_dex3_preflight", PREFLIGHT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


PREFLIGHT_MODULE = load_module()


def write_split(
    root: pathlib.Path,
    *,
    fps: int = 50,
    state_dim: int = 46,
    episode_lengths: tuple[int, ...] = (60, 60),
    tasks: tuple[str, ...] = ("pick",),
) -> pathlib.Path:
    """Write a LeRobot v2.1 split in the official SONIC storage layout.

    ``observation.state`` carries the measured body in exporter order (43D),
    gravity lives in ``observation.projected_gravity`` and the action is the
    three separate SONIC fields; the GR00T loader reassembles the model-facing
    46D/78D through each modality group's slice and ``original_key``.
    """
    (root / "meta").mkdir(parents=True, exist_ok=True)
    (root / "data/chunk-000").mkdir(parents=True, exist_ok=True)
    (root / "videos/chunk-000/observation.images.ego_view").mkdir(parents=True, exist_ok=True)

    gravity_dim = state_dim - sum(end - start for start, end in STATE_STORAGE_SLICES.values())
    features = {
        "observation.state": {"dtype": "float32", "shape": [43]},
        "observation.projected_gravity": {"dtype": "float32", "shape": [gravity_dim]},
        "action.motion_token": {"dtype": "float32", "shape": [64]},
        "teleop.left_hand_joints": {"dtype": "float32", "shape": [7]},
        "teleop.right_hand_joints": {"dtype": "float32", "shape": [7]},
        "observation.images.ego_view": {"dtype": "video", "shape": [480, 640, 3]},
    }

    (root / "meta/info.json").write_text(
        json.dumps(
            {
                "codebase_version": "v2.1",
                "robot_type": "unitree_g1",
                "total_episodes": len(episode_lengths),
                "total_frames": sum(episode_lengths),
                "total_tasks": len(tasks),
                "chunks_size": 1000,
                "fps": fps,
                "splits": {"train": f"0:{len(episode_lengths)}"},
                "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
                "video_path": (
                    "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
                ),
                "total_chunks": 1,
                "total_videos": len(episode_lengths),
                "features": features,
            },
        )
    )
    modality = {
        "state": {},
        "action": {},
        "video": {"ego_view": {"original_key": "observation.images.ego_view"}},
        "annotation": {
            "human.task_description": {"original_key": "annotation.human.task_description"}
        },
    }
    for key in REGISTERED_STATE_KEYS:
        if key == "projected_gravity":
            modality["state"][key] = {
                "start": 0,
                "end": gravity_dim,
                "original_key": "observation.projected_gravity",
            }
        else:
            start, end = STATE_STORAGE_SLICES[key]
            modality["state"][key] = {"start": start, "end": end}
    for key, (column, dim) in ACTION_FIELDS.items():
        modality["action"][key] = {"start": 0, "end": dim, "original_key": column}
    (root / "meta/modality.json").write_text(json.dumps(modality, indent=2))

    stat_columns = {
        "observation.state": 43,
        "observation.projected_gravity": gravity_dim,
        "action.motion_token": 64,
        "teleop.left_hand_joints": 7,
        "teleop.right_hand_joints": 7,
    }
    stats = {
        column: {field: [0.0] * dim for field in ("mean", "std", "min", "max", "q01", "q99")}
        for column, dim in stat_columns.items()
    }
    stats["__fingerprints__"] = {column: "sha256:" + column for column in stat_columns}
    (root / "meta/stats.json").write_text(json.dumps(stats))
    (root / "meta/relative_stats.json").write_text(json.dumps({"__fingerprints__": {}}))
    with open(root / "meta/episodes.jsonl", "w") as handle:
        for index, length in enumerate(episode_lengths):
            handle.write(json.dumps({"episode_index": index, "length": length, "tasks": ["pick"]}) + "\n")
    with open(root / "meta/tasks.jsonl", "w") as handle:
        for index, task in enumerate(tasks):
            handle.write(json.dumps({"task_index": index, "task": task}) + "\n")
    for index in range(len(episode_lengths)):
        (root / f"data/chunk-000/episode_{index:06d}.parquet").write_bytes(b"PAR1payload")
        (
            root / f"videos/chunk-000/observation.images.ego_view/episode_{index:06d}.mp4"
        ).write_bytes(b"v")
    return root


def check_split(root: pathlib.Path, **kwargs):
    report = PREFLIGHT_MODULE.Report(quiet=True)
    with contextlib.redirect_stdout(io.StringIO()):
        PREFLIGHT_MODULE.check_split_root(
            root, report, "split", ego_key="ego_view", expected_fps=50, **kwargs
        )
    return report


def run_cli(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(PREFLIGHT), *args], capture_output=True, text=True, check=False
    )


class SplitContractTests(unittest.TestCase):
    def test_stock_pack_passes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = write_split(pathlib.Path(tmp) / "train")
            report = check_split(root)
            self.assertEqual(report.errors, [], report.errors)

    def test_official_storage_order_is_required(self) -> None:
        """Registered-order storage with the official slices swaps hand/arm channels."""
        with tempfile.TemporaryDirectory() as tmp:
            root = write_split(pathlib.Path(tmp) / "train")
            modality = json.loads((root / "meta/modality.json").read_text())
            for key, start, end in (
                ("left_arm", 22, 29),
                ("left_hand", 15, 22),
                ("right_arm", 36, 43),
                ("right_hand", 29, 36),
            ):
                modality["state"][key] = {"start": start, "end": end}
            (root / "meta/modality.json").write_text(json.dumps(modality))
            report = check_split(root)
            self.assertTrue(
                any("official storage slice" in error for error in report.errors), report.errors
            )

    def test_missing_gravity_column_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = write_split(pathlib.Path(tmp) / "train")
            modality = json.loads((root / "meta/modality.json").read_text())
            del modality["state"]["projected_gravity"]["original_key"]
            (root / "meta/modality.json").write_text(json.dumps(modality))
            report = check_split(root)
            self.assertTrue(
                any("observation.projected_gravity" in error for error in report.errors),
                report.errors,
            )

    def test_missing_action_original_key_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = write_split(pathlib.Path(tmp) / "train")
            modality = json.loads((root / "meta/modality.json").read_text())
            del modality["action"]["motion_token"]["original_key"]
            (root / "meta/modality.json").write_text(json.dumps(modality))
            report = check_split(root)
            self.assertTrue(
                any("reads 'action'" in error for error in report.errors), report.errors
            )

    def test_short_gravity_column_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = write_split(pathlib.Path(tmp) / "train")
            info = json.loads((root / "meta/info.json").read_text())
            info["features"]["observation.projected_gravity"]["shape"] = [2]
            (root / "meta/info.json").write_text(json.dumps(info))
            report = check_split(root)
            self.assertTrue(
                any("observation.projected_gravity is 2 wide" in error for error in report.errors),
                report.errors,
            )

    def test_missing_stats_for_a_sliced_column_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = write_split(pathlib.Path(tmp) / "train")
            stats = json.loads((root / "meta/stats.json").read_text())
            del stats["teleop.left_hand_joints"]
            (root / "meta/stats.json").write_text(json.dumps(stats))
            report = check_split(root)
            self.assertTrue(
                any("teleop.left_hand_joints" in error for error in report.errors), report.errors
            )

    def test_missing_column_for_a_group_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = write_split(pathlib.Path(tmp) / "train")
            info = json.loads((root / "meta/info.json").read_text())
            del info["features"]["action.motion_token"]
            (root / "meta/info.json").write_text(json.dumps(info))
            report = check_split(root)
            self.assertTrue(
                any("action.motion_token" in error for error in report.errors), report.errors
            )

    def test_state_key_order_must_remain_registered(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = write_split(pathlib.Path(tmp) / "train")
            modality = json.loads((root / "meta/modality.json").read_text())
            state = modality["state"]
            # Storage order would put the hands beside their arms; the model
            # reads the registered order, so that must not leak into the keys.
            modality["state"] = {
                key: state[key]
                for key in ("left_leg", "right_leg", "waist", "left_arm", "left_hand",
                            "right_arm", "right_hand", "projected_gravity")
            }
            (root / "meta/modality.json").write_text(json.dumps(modality))
            report = check_split(root)
            self.assertTrue(
                any("expected the registered order" in error for error in report.errors),
                report.errors,
            )

    def test_fps_must_be_fifty(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = write_split(pathlib.Path(tmp) / "train", fps=30)
            report = check_split(root)
            self.assertTrue(any("fps is 30" in error for error in report.errors), report.errors)

    def test_state_width_must_be_46(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = write_split(pathlib.Path(tmp) / "train", state_dim=45)
            report = check_split(root)
            self.assertTrue(
                any("must be 46D" in error for error in report.errors),
                report.errors,
            )

    def test_renamed_state_key_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = write_split(pathlib.Path(tmp) / "train")
            modality = json.loads((root / "meta/modality.json").read_text())
            modality["state"]["projected_gravity_xyz"] = modality["state"].pop("projected_gravity")
            (root / "meta/modality.json").write_text(json.dumps(modality))
            report = check_split(root)
            self.assertTrue(any("projected_gravity" in error for error in report.errors), report.errors)

    def test_second_camera_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = write_split(pathlib.Path(tmp) / "train")
            modality = json.loads((root / "meta/modality.json").read_text())
            modality["video"]["wrist_view"] = {"original_key": "observation.images.wrist_view"}
            (root / "meta/modality.json").write_text(json.dumps(modality))
            report = check_split(root)
            self.assertTrue(any("expected exactly" in error for error in report.errors), report.errors)

    def test_missing_language_key_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = write_split(pathlib.Path(tmp) / "train")
            modality = json.loads((root / "meta/modality.json").read_text())
            del modality["annotation"]["human.task_description"]
            (root / "meta/modality.json").write_text(json.dumps(modality))
            report = check_split(root)
            self.assertTrue(
                any("human.task_description" in error for error in report.errors), report.errors
            )

    def test_episode_shorter_than_horizon_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = write_split(pathlib.Path(tmp) / "train", episode_lengths=(12, 60))
            report = check_split(root)
            self.assertTrue(
                any("shorter than the 40-step action horizon" in error for error in report.errors),
                report.errors,
            )

    def test_episode_count_must_match_info(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = write_split(pathlib.Path(tmp) / "train")
            info = json.loads((root / "meta/info.json").read_text())
            info["total_episodes"] = 0
            info["total_frames"] = 0
            (root / "meta/info.json").write_text(json.dumps(info))
            report = check_split(root)
            self.assertTrue(
                any("total_episodes is 0" in error for error in report.errors), report.errors
            )

    def test_unresolved_lfs_pointer_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = write_split(pathlib.Path(tmp) / "train")
            pointer = root / "data/chunk-000/episode_000000.parquet"
            pointer.write_bytes(b"version https://git-lfs.github.com/spec/v1\noid sha256:x\nsize 1\n")
            report = check_split(root)
            self.assertTrue(any("Git LFS pointer" in error for error in report.errors), report.errors)

    def test_empty_tasks_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = write_split(pathlib.Path(tmp) / "train")
            (root / "meta/tasks.jsonl").write_text("")
            report = check_split(root)
            self.assertTrue(any("tasks.jsonl lists 0" in error for error in report.errors), report.errors)


class PartialMarkerTests(unittest.TestCase):
    def test_hidden_stats_tmp_file_is_detected(self) -> None:
        """stats.py flushes through dot-prefixed tmp files; globs would miss them."""
        with tempfile.TemporaryDirectory() as tmp:
            root = write_split(pathlib.Path(tmp) / "train")
            (root / "meta/.stats.json.k3fm2q.tmp").write_text("{}")
            report = check_split(root)
            self.assertTrue(any("stale partial markers" in error for error in report.errors), report.errors)

    def test_zero_byte_stats_is_detected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = write_split(pathlib.Path(tmp) / "train")
            (root / "meta/stats.json").write_text("")
            report = check_split(root)
            self.assertTrue(any("0 bytes" in error for error in report.errors), report.errors)

    def test_incomplete_marker_is_detected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = write_split(pathlib.Path(tmp) / "train")
            (root / "INCOMPLETE").write_text("interrupted conversion")
            report = check_split(root)
            self.assertTrue(any("stale partial markers" in error for error in report.errors), report.errors)


class StatsContractTests(unittest.TestCase):
    def test_missing_fingerprints_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = write_split(pathlib.Path(tmp) / "train")
            stats = json.loads((root / "meta/stats.json").read_text())
            del stats["__fingerprints__"]
            (root / "meta/stats.json").write_text(json.dumps(stats))
            report = check_split(root)
            self.assertTrue(
                any("gr00t/data/stats.py" in error for error in report.errors), report.errors
            )

    def test_wrong_vector_width_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = write_split(pathlib.Path(tmp) / "train")
            stats = json.loads((root / "meta/stats.json").read_text())
            stats["action.motion_token"]["q99"] = stats["action.motion_token"]["q99"][:-1]
            (root / "meta/stats.json").write_text(json.dumps(stats))
            report = check_split(root)
            self.assertTrue(any("stats.action.motion_token" in error for error in report.errors), report.errors)

    def test_missing_stat_field_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = write_split(pathlib.Path(tmp) / "train")
            stats = json.loads((root / "meta/stats.json").read_text())
            del stats["observation.state"]["std"]
            (root / "meta/stats.json").write_text(json.dumps(stats))
            report = check_split(root)
            self.assertTrue(any("missing fields" in error for error in report.errors), report.errors)

    def test_relative_stats_without_sidecar_is_only_a_warning(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = write_split(pathlib.Path(tmp) / "train")
            (root / "meta/relative_stats.json").write_text(json.dumps({}))
            report = check_split(root)
            self.assertEqual(report.errors, [], report.errors)
            self.assertTrue(report.warnings, "expected a warning about the missing sidecar")

    def test_missing_relative_stats_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = write_split(pathlib.Path(tmp) / "train")
            (root / "meta/relative_stats.json").unlink()
            report = check_split(root)
            self.assertTrue(
                any("relative_stats.json is missing" in error for error in report.errors), report.errors
            )

    def test_stats_scope_skips_contract_checks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = write_split(pathlib.Path(tmp) / "train")
            info = json.loads((root / "meta/info.json").read_text())
            info["fps"] = 30
            (root / "meta/info.json").write_text(json.dumps(info))
            report = check_split(root, want_contract=False, want_stats=True)
            self.assertEqual(report.errors, [], report.errors)


class ModelAndSourceTests(unittest.TestCase):
    def write_model(self, root: pathlib.Path, *, model_type: str = "Gr00tN1d7", revision: str = "abc") -> pathlib.Path:
        root.mkdir(parents=True, exist_ok=True)
        (root / "config.json").write_text(json.dumps({"model_type": model_type}))
        (root / "processor_config.json").write_text("{}")
        (root / "statistics.json").write_text("{}")
        (root / "embodiment_id.json").write_text("{}")
        (root / "model.safetensors").write_bytes(b"weights")
        (root / "MODEL_PROVENANCE.json").write_text(
            json.dumps({"repo": "nvidia/GR00T-N1.7-3B", "revision": revision})
        )
        return root

    def test_valid_model_passes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model = self.write_model(pathlib.Path(tmp) / "model", revision="abc")
            report = PREFLIGHT_MODULE.Report(quiet=True)
            PREFLIGHT_MODULE.check_model(
                model, report, expected_repo="nvidia/GR00T-N1.7-3B", expected_revision="abc"
            )
            self.assertEqual(report.errors, [], report.errors)

    def test_wrong_model_type_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model = self.write_model(pathlib.Path(tmp) / "model", model_type="Gr00tN1d6")
            report = PREFLIGHT_MODULE.Report(quiet=True)
            PREFLIGHT_MODULE.check_model(
                model, report, expected_repo="nvidia/GR00T-N1.7-3B", expected_revision="abc"
            )
            self.assertTrue(any("Gr00tN1d7" in error for error in report.errors), report.errors)

    def test_wrong_revision_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model = self.write_model(pathlib.Path(tmp) / "model", revision="other")
            report = PREFLIGHT_MODULE.Report(quiet=True)
            PREFLIGHT_MODULE.check_model(
                model, report, expected_repo="nvidia/GR00T-N1.7-3B", expected_revision="abc"
            )
            self.assertTrue(any("revision" in error for error in report.errors), report.errors)

    def test_source_commit_must_match(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = pathlib.Path(tmp) / "source"
            (source / "gr00t/experiment").mkdir(parents=True)
            (source / "gr00t/data").mkdir(parents=True)
            (source / "gr00t/experiment/launch_finetune.py").write_text("")
            (source / "gr00t/data/stats.py").write_text("")
            report = PREFLIGHT_MODULE.Report(quiet=True)
            PREFLIGHT_MODULE.check_source(source, report, expected_commit="0" * 40)
            self.assertTrue(any("cannot read the GR00T source revision" in error for error in report.errors), report.errors)


class ArtifactTests(unittest.TestCase):
    def write_run(self, root: pathlib.Path, *, step: int = 2, trainer_state: dict | None = None) -> pathlib.Path:
        (root / "experiment_cfg").mkdir(parents=True, exist_ok=True)
        (root / "processor").mkdir(parents=True, exist_ok=True)
        (root / "experiment_cfg/config.yaml").write_text("{}")
        (root / "processor/processor_config.json").write_text("{}")
        (root / "processor/statistics.json").write_text("{}")
        checkpoint = root / f"checkpoint-{step}"
        (checkpoint / "experiment_cfg").mkdir(parents=True, exist_ok=True)
        (checkpoint / "trainer_state.json").write_text(
            json.dumps(trainer_state if trainer_state is not None else {"global_step": step})
        )
        (checkpoint / "processor_config.json").write_text("{}")
        (checkpoint / "statistics.json").write_text("{}")
        (checkpoint / "model.safetensors").write_bytes(b"weights")
        return root

    def check(self, root: pathlib.Path, *, expect_steps: int | None = None, expect_optim: str | None = None):
        report = PREFLIGHT_MODULE.Report(quiet=True)
        PREFLIGHT_MODULE.check_artifacts(root, report, expect_steps=expect_steps, expect_optim=expect_optim)
        return report

    def test_two_step_smoke_checkpoint_passes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            report = self.check(self.write_run(pathlib.Path(tmp) / "run", step=2), expect_steps=2)
            self.assertEqual(report.errors, [], report.errors)

    def test_wrong_step_count_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            report = self.check(self.write_run(pathlib.Path(tmp) / "run", step=3), expect_steps=2)
            self.assertTrue(
                any("expected exactly 2 optimizer steps" in error for error in report.errors),
                report.errors,
            )

    def test_trainer_state_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run = self.write_run(pathlib.Path(tmp) / "run", step=2, trainer_state={"global_step": 5})
            report = self.check(run, expect_steps=2)
            self.assertTrue(any("global_step is 5" in error for error in report.errors), report.errors)

    def test_zero_byte_weights_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run = self.write_run(pathlib.Path(tmp) / "run", step=2)
            (run / "checkpoint-2/model.safetensors").write_bytes(b"")
            report = self.check(run, expect_steps=2)
            self.assertTrue(any("zero-byte weight" in error for error in report.errors), report.errors)

    def test_run_without_checkpoints_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run = pathlib.Path(tmp) / "run"
            (run / "experiment_cfg").mkdir(parents=True)
            report = self.check(run)
            self.assertTrue(any("no checkpoint-" in error for error in report.errors), report.errors)

    def test_missing_run_directory_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            report = self.check(pathlib.Path(tmp) / "absent")
            self.assertTrue(any("run directory is missing" in error for error in report.errors), report.errors)

    def test_effective_optimizer_matches_expectation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run = self.write_run(pathlib.Path(tmp) / "run", step=2)
            (run / "experiment_cfg/conf.yaml").write_text(
                "model:\n  tune_llm: false\ntraining:\n  optim: adafactor\n  lr: 0.0001\n"
            )
            report = self.check(run, expect_steps=2, expect_optim="adafactor")
            self.assertEqual(report.errors, [], report.errors)
            self.assertEqual(report.records.get("optimizer"), "adafactor")

    def test_other_effective_optimizer_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run = self.write_run(pathlib.Path(tmp) / "run", step=2)
            (run / "experiment_cfg/conf.yaml").write_text("training:\n  optim: adamw_torch\n")
            report = self.check(run, expect_steps=2, expect_optim="adafactor")
            self.assertTrue(
                any("effective optimizer is 'adamw_torch'" in error for error in report.errors),
                report.errors,
            )

    def test_missing_recorded_optimizer_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run = self.write_run(pathlib.Path(tmp) / "run", step=2)
            report = self.check(run, expect_steps=2, expect_optim="adafactor")
            self.assertTrue(
                any("records training.optim" in error for error in report.errors), report.errors
            )

    def test_optimizer_check_is_optional(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            report = self.check(self.write_run(pathlib.Path(tmp) / "run", step=2), expect_steps=2)
            self.assertEqual(report.errors, [], report.errors)
            self.assertNotIn("optimizer", report.records)


class CliTests(unittest.TestCase):
    def test_clean_pack_exits_zero(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dataset = pathlib.Path(tmp) / "pack"
            write_split(dataset / "train")
            write_split(dataset / "val")
            result = run_cli("--check", "dataset", "--check", "stats", "--dataset-root", str(dataset))
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_missing_split_exits_two(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dataset = pathlib.Path(tmp) / "pack"
            write_split(dataset / "train")
            result = run_cli("--check", "dataset", "--dataset-root", str(dataset))
            self.assertEqual(result.returncode, 2)
            self.assertIn("val split does not exist", result.stdout)

    def test_json_report_is_machine_readable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dataset = pathlib.Path(tmp) / "pack"
            write_split(dataset / "train")
            result = run_cli(
                "--check", "dataset", "--check", "hashes", "--dataset-root", str(dataset),
                "--splits", "train", "--json",
            )
            payload = json.loads(result.stdout)
            self.assertTrue(payload["ok"])
            hashes = payload["records"]["dataset_meta_sha256"]["train"]
            self.assertEqual(hashes["info.json"]["bytes"], (dataset / "train/meta/info.json").stat().st_size)
            self.assertEqual(len(hashes["stats.json"]["sha256"]), 64)

    def test_unknown_split_is_rejected(self) -> None:
        result = run_cli("--check", "dataset", "--dataset-root", "/tmp", "--splits", "test")
        self.assertEqual(result.returncode, 2)

    def test_hashes_are_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dataset = pathlib.Path(tmp) / "pack"
            write_split(dataset / "train")
            args = ("--check", "hashes", "--dataset-root", str(dataset), "--splits", "train", "--json")
            first = json.loads(run_cli(*args).stdout)
            second = json.loads(run_cli(*args).stdout)
            self.assertEqual(
                first["records"]["dataset_meta_sha256"], second["records"]["dataset_meta_sha256"]
            )


if __name__ == "__main__":
    unittest.main()
