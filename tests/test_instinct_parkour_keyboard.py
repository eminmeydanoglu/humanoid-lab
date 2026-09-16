"""Contract tests for the InstinctLab parkour keyboard driver.

These run without a simulator.  They cover the rule that matters for driving the
robot by hand: a command must stay inside the envelope the checkpoint was trained
on, because the policy degrades into a stand or a fall outside it.

The clamp values are cross-checked against the checkpoint's own training config
(`parkour_env_cfg.py`) where that file is available, so the test notices if the
policy's trained range and the driver's clamp ever drift apart.
"""

from __future__ import annotations

import re
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PLAY = REPO / "scripts" / "play-instinct-parkour.py"
DRIVER = REPO / "scripts" / "instinct-parkour-keyboard.sh"
DEV = REPO / "dev.sh"
ENV_CFG = (
    REPO.parent
    / "humanoid-lab-main"
    / "data"
    / "src"
    / "InstinctLab"
    / "source"
    / "instinctlab"
    / "instinctlab"
    / "tasks"
    / "parkour"
    / "config"
    / "parkour_env_cfg.py"
)


def _arg_default(text: str, flag: str) -> float | None:
    """Return the numeric default of an argparse flag, or None if absent."""
    match = re.search(
        re.escape(flag) + r'"[^)]*?default=(-?[0-9.]+)',
        text,
        re.DOTALL,
    )
    return float(match.group(1)) if match else None


class KeyboardClampTests(unittest.TestCase):
    """The playback script must bound every command axis."""

    @classmethod
    def setUpClass(cls):
        cls.src = PLAY.read_text()

    def test_clamp_flags_exist(self):
        for flag in (
            "--keyboard_linvel_max",
            "--keyboard_linvel_min",
            "--keyboard_latvel_max",
            "--keyboard_angvel_max",
        ):
            self.assertIn(flag, self.src, f"{flag} is missing from the playback script")

    def test_defaults_match_the_trained_envelope(self):
        self.assertEqual(_arg_default(self.src, "--keyboard_linvel_max"), 1.0)
        self.assertEqual(_arg_default(self.src, "--keyboard_linvel_min"), 0.0)
        self.assertEqual(_arg_default(self.src, "--keyboard_latvel_max"), 0.0)
        self.assertEqual(_arg_default(self.src, "--keyboard_angvel_max"), 1.0)

    def test_every_axis_is_clamped_in_the_handler(self):
        # All three command axes must be clamped, not just the forward one.
        self.assertIn("override_command[:, 0].clamp_", self.src)
        self.assertIn("override_command[:, 1].clamp_", self.src)
        self.assertIn("override_command[:, 2].clamp_", self.src)

    def test_clamp_runs_after_every_keypress(self):
        # A clamp placed inside a branch would be skipped by the other keys.
        handler = self.src.split("def on_keyboard_input", 1)[1].split(
            "app_window = omni.appwindow", 1
        )[0]
        # X returns early after zeroing; every other key must fall through to the clamps.
        self.assertLess(handler.index("clamp_(args_cli.keyboard_linvel_min"), len(handler))
        for key in ("KeyboardInput.W", "KeyboardInput.S", "KeyboardInput.F", "KeyboardInput.G"):
            self.assertIn(key, handler, f"{key} has no binding")

    def test_x_still_zeroes_the_command(self):
        handler = self.src.split("def on_keyboard_input", 1)[1]
        self.assertIn("KeyboardInput.X", handler)
        self.assertIn("override_command[:] = 0.0", handler)

    def test_the_envelope_is_printed_at_startup(self):
        self.assertIn("Command envelope:", self.src)


@unittest.skipUnless(ENV_CFG.is_file(), "InstinctLab checkout not present")
class TrainedRangeTests(unittest.TestCase):
    """The clamps must agree with the range the checkpoint was trained on."""

    @classmethod
    def setUpClass(cls):
        cls.cfg = ENV_CFG.read_text()

    def _ranges(self, terrain: str) -> tuple[float, float, float, float, float, float]:
        block = self.cfg.split(f'"{terrain}": {{', 1)[1].split("}", 1)[0]
        nums = [float(n) for n in re.findall(r"-?[0-9.]+", block)]
        return tuple(nums)  # lin_x_min, lin_x_max, lin_y_min, lin_y_max, wz_min, wz_max

    def test_forward_command_range_is_positive_and_within_the_clamp(self):
        # Every terrain in the training config samples only positive lin_vel_x.
        for terrain in ("pyramid_stairs", "boxes", "square_gaps"):
            vx_min, vx_max, _, _, _, _ = self._ranges(terrain)
            self.assertGreaterEqual(vx_min, 0.0, f"{terrain} samples a negative lin_vel_x")
            self.assertLessEqual(vx_max, 1.0, f"{terrain} exceeds the playback clamp of 1.0")

    def test_lateral_command_is_never_trained(self):
        for terrain in ("pyramid_stairs", "boxes", "perlin_rough"):
            _, _, vy_min, vy_max, _, _ = self._ranges(terrain)
            self.assertEqual((vy_min, vy_max), (0.0, 0.0), f"{terrain} trains a lateral command")

    def test_yaw_command_matches_the_clamp(self):
        for terrain in ("pyramid_stairs", "boxes"):
            _, _, _, _, wz_min, wz_max = self._ranges(terrain)
            self.assertAlmostEqual(wz_min, -1.0, places=6)
            self.assertAlmostEqual(wz_max, 1.0, places=6)

    def test_only_positive_forward_commands_are_configured(self):
        self.assertIn("only_positive_lin_vel_x=True", self.cfg)


class DriverScriptTests(unittest.TestCase):
    """The terminal driver must exist, be wired into dev.sh and stay key-compatible."""

    @classmethod
    def setUpClass(cls):
        cls.driver = DRIVER.read_text()
        cls.dev = DEV.read_text()

    def test_driver_is_executable(self):
        self.assertTrue(DRIVER.is_file(), "keyboard driver script is missing")
        self.assertTrue(DRIVER.stat().st_mode & 0o111, "keyboard driver is not executable")

    def test_driver_sends_the_same_keys_the_sim_listens_for(self):
        for key in ("w|s|a|d|f|g|x"):
            self.assertIn(key, self.driver, f"driver does not handle the '{key}' key group")

    def test_driver_targets_the_isaac_window_by_id(self):
        # Sending to the window id is what makes it work without focus.
        self.assertIn("xdotool search", self.driver)
        self.assertIn('--window "$WID"', self.driver)

    def test_driver_offers_status_and_send_modes(self):
        self.assertIn("--status", self.driver)
        self.assertIn("--send", self.driver)

    def test_dev_sh_exposes_the_driver(self):
        self.assertIn("instinct-parkour-drive)", self.dev)
        self.assertIn("scripts/instinct-parkour-keyboard.sh", self.dev)


class DeviceDefaultTests(unittest.TestCase):
    """The physics/sensor device default is a measured 6x on this scene."""

    def test_sim_device_defaults_to_cuda(self):
        src = PLAY.read_text()
        self.assertEqual(
            _arg_default(src, "--sim_device"),
            None,
            "--sim_device should not be numeric; check its default separately",
        )
        match = re.search(r'"--sim_device",\s*default="([^"]+)"', src)
        self.assertIsNotNone(match, "--sim_device has no explicit default")
        self.assertEqual(match.group(1), "cuda:0")


if __name__ == "__main__":
    unittest.main()
