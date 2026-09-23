"""The campaign driver's opt-in warm-start plumbing.

``scripts/blockstacking-rollout.py`` is the only place that turns a campaign's
flags into the launcher's argv, so this is where "the treatment differs from the
baseline in exactly one switch" has to be true: without the stream file the
command is byte-for-byte the canonical one, and with it the stream and its delay
travel to the bridge in the container's own path spelling.
"""

from __future__ import annotations

import hashlib
import importlib.util
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "blockstacking-rollout.py"


def load_driver():
    spec = importlib.util.spec_from_file_location("blockstacking_rollout", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def base_args(driver, extra: list[str]):
    with mock.patch.dict("os.environ", {"HUMANOID_DATA_ROOT": str(ROOT / "data")}):
        return driver.parse_args([
            "--psi-checkpoint-dir", "/outputs/psi/run",
            "--checkpoint-step", "40000",
            "--groot-checkpoint-dir", "/outputs/groot/checkpoint-40000",
            "--model", "groot", "--out-dir", "/outputs/blockstacking-debug/experiments/07/x/runs/W",
        ] + extra)


class WarmStartCommandTest(unittest.TestCase):
    def setUp(self) -> None:
        self.driver = load_driver()
        self.paths = {
            "telemetry": "/outputs/x/raw/telemetry",
            "video_raw": "/outputs/x/raw/video.raw.mp4",
            "video_timestamps": "/outputs/x/raw/video.timestamps.jsonl",
            "isaac_samples": "/outputs/x/raw/isaac.samples.jsonl",
            "isaac_tracking": "/outputs/x/raw/isaac.tracking.parquet",
            "isaac_metrics": "/outputs/x/raw/isaac.metrics.json",
            "policy_clock": "/outputs/x/raw/isaac-policy-clock.txt",
        }

    def test_the_canonical_command_carries_no_warm_start_flags(self) -> None:
        args = base_args(self.driver, [])
        command = self.driver.build_launcher_command(args, self.paths)
        self.assertNotIn("--warmstart-tokens", command)
        self.assertNotIn("--warmstart-delay-s", command)

    def test_the_stream_and_its_delay_travel_to_the_bridge(self) -> None:
        stream = ROOT / "data/outputs/blockstacking-debug/experiments/07/x/tokens.json"
        args = base_args(self.driver, ["--warmstart-tokens", str(stream),
                                       "--warmstart-window-seconds", "6",
                                       "--settle-seconds", "15"])
        command = self.driver.build_launcher_command(args, self.paths)
        inside = command[command.index("--warmstart-tokens") + 1]
        # The bridge reads it on its own side of the mount, exactly like the
        # reset-pose file: the checkout spelling of a data-root path.
        self.assertEqual(
            inside, "/workspace/humanoid-lab/data/outputs/blockstacking-debug/experiments/07/x/tokens.json"
        )
        self.assertEqual(command[command.index("--warmstart-delay-s") + 1], "9")

    def test_a_window_longer_than_the_settle_is_refused(self) -> None:
        args = base_args(self.driver, ["--warmstart-tokens", "/tmp/tokens.json",
                                       "--warmstart-window-seconds", "20",
                                       "--settle-seconds", "15"])
        with self.assertRaises(self.driver.DriverError):
            self.driver.warmstart_delay_seconds(args)

    def test_the_window_defaults_inside_the_canonical_settle(self) -> None:
        args = base_args(self.driver, ["--warmstart-tokens", "/tmp/tokens.json"])
        self.assertEqual(args.settle_seconds, 12.0)
        self.assertEqual(self.driver.warmstart_delay_seconds(args), 12.0 - 6.0)

    def test_the_initial_pose_handshake_is_opt_in_and_exclusive(self) -> None:
        canonical = self.driver.build_launcher_command(base_args(self.driver, []), self.paths)
        self.assertNotIn("--initial-pose-handshake", canonical)
        args = base_args(self.driver, ["--initial-pose-handshake"])
        self.assertIn("--initial-pose-handshake",
                      self.driver.build_launcher_command(args, self.paths))
        # Both settle commands cannot be configured at once: the launcher
        # refuses it and so does the driver, before anything is spawned.
        self.assertEqual(self.driver.main([
            "--initial-pose-handshake", "--warmstart-tokens", "/tmp/tokens.json",
            "--psi-checkpoint-dir", "/outputs/psi/run", "--checkpoint-step", "40000",
            "--groot-checkpoint-dir", "/outputs/groot/checkpoint-40000",
        ]), 2)

    def test_the_scene_variant_is_opt_in_and_recorded_with_its_digest(self) -> None:
        canonical = self.driver.build_launcher_command(base_args(self.driver, []), self.paths)
        self.assertNotIn("--scene-profile", canonical)
        self.assertIsNone(self.driver.scene_profile_record(None))
        variant = "configs/profiles/isaac-g1-sonic-blockstacking-dex3-green-third.json"
        args = base_args(self.driver, ["--scene-profile", variant])
        command = self.driver.build_launcher_command(args, self.paths)
        self.assertEqual(command[command.index("--scene-profile") + 1], variant)
        record = self.driver.scene_profile_record(args.scene_profile)
        # The session has to be readable as the variant it ran: the relative
        # spelling for the launcher, the host path and the file's own digest.
        self.assertEqual(record["file"], variant)
        self.assertEqual(record["host_path"], str(ROOT / variant))
        self.assertEqual(
            record["sha256"], hashlib.sha256((ROOT / variant).read_bytes()).hexdigest()
        )

    def test_the_gravity_feedforward_is_opt_in(self) -> None:
        canonical = self.driver.build_launcher_command(base_args(self.driver, []), self.paths)
        self.assertNotIn("--gravity-feedforward", canonical)
        args = base_args(self.driver, ["--gravity-feedforward"])
        command = self.driver.build_launcher_command(args, self.paths)
        # One switch: the flag travels to the simulator, which owns the torque
        # law; nothing else about the canonical command changes.
        self.assertIn("--gravity-feedforward", command)
        self.assertEqual(command.count("--gravity-feedforward"), 1)

    def test_policy_clock_and_left_hand_default_to_corrected_contract(self) -> None:
        command = self.driver.build_launcher_command(base_args(self.driver, []), self.paths)
        self.assertEqual(command[command.index("--policy-clock") + 1], "simulation")
        self.assertEqual(command[command.index("--policy-clock-file") + 1], self.paths["policy_clock"])
        self.assertEqual(command[command.index("--groot-left-hand-contract") + 1], "model-independent")
        self.assertEqual(command[command.index("--policy-clock-timeout-s") + 1], "5")

    def test_wall_policy_can_record_a_separate_sim_horizon_clock(self) -> None:
        args = base_args(self.driver, ["--policy-clock", "wall", "--rollout-sim-seconds", "30"])
        paths = {**self.paths, "run_clock": "/outputs/x/raw/isaac-run-clock.txt"}
        command = self.driver.build_launcher_command(args, paths)
        self.assertEqual(command[command.index("--replay-clock-output") + 1], paths["run_clock"])
        self.assertNotIn("--policy-clock-file", command)

    def test_simulation_clock_uses_one_absolute_container_path(self) -> None:
        args = base_args(self.driver, ["--policy-clock", "simulation"])
        paths = {**self.paths, "policy_clock": "/outputs/x/raw/isaac-policy-clock.txt"}
        command = self.driver.build_launcher_command(args, paths)
        self.assertEqual(command[command.index("--policy-clock-file") + 1], paths["policy_clock"])
        self.assertTrue(paths["policy_clock"].startswith("/"))

    def test_psi_simulation_clock_uses_the_shared_clock_path(self) -> None:
        args = base_args(self.driver, ["--model", "psi", "--policy-clock", "simulation"])
        paths = {**self.paths, "policy_clock": "/outputs/x/raw/isaac-policy-clock.txt"}
        command = self.driver.build_launcher_command(args, paths)
        self.assertEqual(command[command.index("--policy-clock-file") + 1], paths["policy_clock"])

    def test_masked_neck_discard_must_be_explicit_and_reaches_the_launcher(self) -> None:
        canonical = self.driver.build_launcher_command(base_args(self.driver, []), self.paths)
        self.assertNotIn("--psi0-neck-policy", canonical)
        args = base_args(self.driver, ["--psi0-neck-policy", "discard"])
        command = self.driver.build_launcher_command(args, self.paths)
        self.assertEqual(command[command.index("--psi0-neck-policy") + 1], "discard")

    def test_capture_hand_contract_and_psi_rtc_flags_reach_the_launcher(self) -> None:
        args = base_args(self.driver, [
            "--groot-capture-dir", "/outputs/x/capture",
            "--groot-capture-max-requests", "9",
            "--groot-left-hand-contract", "model-coupled",
            "--psi0-rtc-off",
        ])
        paths = {**self.paths, "groot_capture": "/outputs/x/capture"}
        command = self.driver.build_launcher_command(args, paths)
        self.assertEqual(command[command.index("--groot-capture-dir") + 1], paths["groot_capture"])
        self.assertEqual(command[command.index("--groot-capture-max-requests") + 1], "9")
        self.assertEqual(command[command.index("--groot-left-hand-contract") + 1], "model-coupled")
        self.assertIn("--psi0-rtc-off", command)


if __name__ == "__main__":
    unittest.main()
