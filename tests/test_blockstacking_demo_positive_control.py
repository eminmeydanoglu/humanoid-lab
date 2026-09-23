"""CPU-only contracts for the BlockStacking demo positive control."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/blockstacking-demo-positive-control.py"


def load_script():
    spec = importlib.util.spec_from_file_location("blockstacking_demo_positive_control", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


PC = load_script()


class ResamplingTest(unittest.TestCase):
    def test_zero_order_hold_uses_latest_dataset_frame(self) -> None:
        timestamps = np.array([0.0, 1.0 / 30.0, 2.0 / 30.0])
        ticks = np.array([0.0, 0.02, 0.04, 0.06])
        np.testing.assert_array_equal(PC.source_indices(timestamps, ticks), [0, 0, 1, 1])

    def test_ticks_before_first_and_after_last_are_clamped(self) -> None:
        timestamps = np.array([0.1, 0.2])
        ticks = np.array([0.0, 0.15, 0.3])
        np.testing.assert_array_equal(PC.source_indices(timestamps, ticks), [0, 0, 1])


class LauncherTest(unittest.TestCase):
    def test_positive_control_passes_one_clock_path_to_isaac_and_bridge(self) -> None:
        rollout = PC.load_module(ROOT / "scripts/blockstacking-rollout.py", "pc_rollout_test")
        clock = ROOT / "data/outputs/test/replay-clock.txt"
        args = type("Args", (), {
            "psi_checkpoint_dir": "/outputs/psi", "checkpoint_step": 1,
            "groot_checkpoint_dir": "/outputs/groot", "headless": True, "gui": False,
            "reset_pose_file": None, "reset_pose_hold": False,
            "warmstart_tokens": ROOT / "data/outputs/test/stream.json",
            "warmstart_sim_clock": clock, "warmstart_clock_timeout_s": 7.0,
            "warmstart_window_seconds": 2.0, "settle_seconds": 3.0,
            "initial_pose_handshake": False, "scene_profile": None,
            "gravity_feedforward": False, "policy_clock": "wall",
            "policy_clock_timeout_s": 5.0, "groot_capture_dir": None,
            "groot_capture_max_requests": 32, "groot_left_hand_contract": "compatibility",
            "psi0_neck_policy": "error", "psi0_rtc_off": False,
        })()
        paths = {key: "/outputs/test/" + key for key in (
            "telemetry", "video_raw", "video_timestamps", "isaac_samples", "isaac_tracking", "isaac_metrics"
        )}
        command = rollout.build_launcher_command(args, paths)
        expected = "/workspace/humanoid-lab/data/outputs/test/replay-clock.txt"
        self.assertEqual(command[command.index("--replay-clock-output") + 1], expected)
        self.assertEqual(command[command.index("--warmstart-sim-clock") + 1], expected)


class AnalysisTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.out = Path(self.tmp.name)
        (self.out / "raw/telemetry").mkdir(parents=True)
        PC.write_json(self.out / "preparation.json", {
            "ticks": 2,
            "duration_s": 0.04,
            "source_duration_s": 0.02,
            "geometry_contract": {
                "demo_object_pose_available": False,
                "demo_pelvis_to_object_transform_available": False,
            },
        })
        PC.write_json(self.out / "reset-diagnostic.json", {"passed": True})
        self.write_jsonl(self.out / "raw/isaac.tracking.jsonl", [
            {"body_target": [0.0] * 29, "body_measured": [0.0] * 29,
             "wall_time_ns": 1_000_000_000, "sim_s": 0.0},
            {"body_target": [0.0] * 29, "body_measured": [0.0] * 29,
             "wall_time_ns": 1_020_000_000, "sim_s": 0.02},
        ])

    def write_jsonl(self, path: Path, rows: list[dict]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    def telemetry(self, source: str, index: int, wall_ns: int) -> dict:
        return {
            "event": "applied_action",
            "source": source,
            "wall_time_ns": wall_ns,
            "fields": {"frame_index": [[index]]},
        }

    def test_no_lift_is_indeterminate_when_demo_scene_alignment_is_absent(self) -> None:
        self.write_jsonl(self.out / "raw/isaac.samples.jsonl", [
            {"scene": {"cubes": {"red": {"live_position_m": [0.45, -0.1, 0.86]}}}},
            {"scene": {"cubes": {"red": {"live_position_m": [0.45, -0.1, 0.86]}}}},
        ])
        self.write_jsonl(self.out / "raw/isaac.tracking.jsonl", [
            {"body_target": [0.1] * 29, "body_measured": [0.0] * 29,
             "wall_time_ns": 1_000_000_000, "sim_s": 0.0},
            {"body_target": [0.1] * 29, "body_measured": [0.0] * 29,
             "wall_time_ns": 1_020_000_000, "sim_s": 0.02},
        ])
        self.write_jsonl(self.out / "raw/telemetry/bridge-telemetry.jsonl", [
            self.telemetry("warmstart", 0, 1_000_000_000),
            self.telemetry("warmstart", 1, 1_020_000_000),
        ])
        result = PC.analyse_run(self.out)
        self.assertEqual(result["decision"], "INDETERMINATE_SCENE_ALIGNMENT")
        self.assertTrue(result["validity_gates"]["learned_policy_disabled"])
        self.assertTrue(result["validity_gates"]["exact_stream_delivered"])
        self.assertAlmostEqual(result["decoded_target_vs_measured"]["body_mae_rad"], 0.1)

    def test_a_measured_three_centimetre_lift_is_positive_evidence(self) -> None:
        self.write_jsonl(self.out / "raw/isaac.samples.jsonl", [
            {"scene": {"cubes": {"red": {"live_position_m": [0.45, -0.1, 0.86]}}}},
            {"scene": {"cubes": {"red": {"live_position_m": [0.45, -0.1, 0.895]}}}},
        ])
        self.write_jsonl(self.out / "raw/telemetry/bridge-telemetry.jsonl", [
            self.telemetry("warmstart", 0, 1_000_000_000),
            self.telemetry("warmstart", 1, 1_020_000_000),
        ])
        result = PC.analyse_run(self.out)
        self.assertEqual(result["decision"], "POSITIVE_CONTROL_PASS")
        self.assertTrue(result["any_cube_lifted"])

    def test_wall_paced_stream_is_invalid_when_isaac_sim_time_is_compressed(self) -> None:
        self.write_jsonl(self.out / "raw/isaac.tracking.jsonl", [
            {"body_target": [0.0] * 29, "body_measured": [0.0] * 29,
             "wall_time_ns": 1_000_000_000, "sim_s": 0.0},
            {"body_target": [0.0] * 29, "body_measured": [0.0] * 29,
             "wall_time_ns": 1_020_000_000, "sim_s": 0.01},
        ])
        self.write_jsonl(self.out / "raw/telemetry/bridge-telemetry.jsonl", [
            self.telemetry("warmstart", 0, 1_000_000_000),
            self.telemetry("warmstart", 1, 1_020_000_000),
        ])
        result = PC.analyse_run(self.out)
        self.assertEqual(result["decision"], "TEMPORALLY_INVALID")
        self.assertAlmostEqual(result["temporal_alignment"]["coverage_ratio"], 0.5)
        self.assertFalse(result["negative_outcome_interpretable"])

    def test_any_learned_policy_message_fails_the_disabled_gate(self) -> None:
        self.write_jsonl(self.out / "raw/telemetry/bridge-telemetry.jsonl", [
            self.telemetry("warmstart", 0, 1_000_000_000),
            self.telemetry("warmstart", 1, 1_020_000_000),
            self.telemetry("groot", 2, 1_040_000_000),
        ])
        result = PC.analyse_run(self.out)
        self.assertFalse(result["validity_gates"]["learned_policy_disabled"])


if __name__ == "__main__":
    unittest.main()
