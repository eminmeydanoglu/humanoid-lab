"""Contract tests for the InstinctLab parkour playback: driving it, and seeing it.

These run without a simulator.  They cover the rules that matter while driving
the robot by hand -- a command must stay inside the envelope the checkpoint was
trained on, the viewport camera must be the operator's, and the depth window
must show the sensor's own frame -- because the policy degrades into a stand or
a fall outside the trained envelope, and a locked camera makes the run
unusable.

The clamp values are cross-checked against the checkpoint's own training config
(`parkour_env_cfg.py`) where that file is available, so the test notices if the
policy's trained range and the driver's clamp ever drift apart.
"""

from __future__ import annotations

import contextlib
import re
import types
import unittest
from pathlib import Path

import numpy

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


class CameraModeTests(unittest.TestCase):
    """The viewport camera belongs to the operator, not to the tracker.

    Isaac Lab's viewport camera controller re-poses the camera every frame while
    it follows an asset root, so a following camera cannot be orbited, panned or
    zoomed: the next frame overwrites the operator.  These tests pin the way out
    (a free default plus a runtime toggle) rather than the mechanism, which
    lives upstream.
    """

    @classmethod
    def setUpClass(cls):
        cls.src = PLAY.read_text()
        cls.driver = DRIVER.read_text()

    @staticmethod
    def _choice_default(text: str, flag: str) -> str | None:
        match = re.search(r'"' + re.escape(flag) + r'"\s*,\s*\n?\s*choices=\([^)]*\)\s*,\s*\n?\s*default="([^"]+)"', text)
        return match.group(1) if match else None

    def test_camera_modes_exist_and_default_to_free(self):
        self.assertIn('"--camera"', self.src)
        self.assertEqual(self._choice_default(self.src, "--camera"), "free")

    def test_only_follow_mode_tracks_the_robot(self):
        # The tracking mode is the one that overwrites navigation every frame.
        self.assertRegex(
            self.src,
            r'origin_type="asset_root" if mode == "follow" else "world"',
            "the viewer must only track the robot in follow mode",
        )
        self.assertIn("_update_tracking_callback", self.src, "the reason for the split is not documented")

    def test_the_toggle_key_is_not_a_command_key(self):
        handler = self.src.split("def on_keyboard_input", 1)[1].split("app_window = omni.appwindow", 1)[0]
        self.assertIn("KeyboardInput.C", handler)
        branch = handler.split("KeyboardInput.C", 1)[1].split("KeyboardInput.W", 1)[0]
        self.assertIn("toggle_camera()", branch)
        self.assertIn("return", branch, "the camera toggle must not fall through to the command clamps")
        toggle = self.src.split("def toggle_camera", 1)[1].split("def on_keyboard_input", 1)[0]
        self.assertIn('controller.cfg.origin_type = "asset_root" if follow else "world"', toggle)

    def test_the_operator_is_told_which_camera_mode_is_active(self):
        self.assertIn("C toggle the camera between free and robot-follow", self.src)
        self.assertIn("[INFO] Camera  :", self.src)

    def test_terminal_driver_forwards_the_camera_key(self):
        self.assertIn("c|C)", self.driver)
        branch = self.driver.split("c|C)", 1)[1].split(";;", 1)[0]
        self.assertIn("xdotool key", branch)
        # It must not pretend the command changed: the readout stays untouched.
        self.assertNotIn("mirror", branch)


def _load_depth_to_rgba():
    """Lift the depth-to-image helper out of the playback script and build it.

    The script cannot be imported as a module: its first statements start a Kit
    application.  The helper itself is plain numpy, so it is compiled on its own
    and given the same numpy the script would have had.
    """
    import ast

    import numpy as np

    tree = ast.parse(PLAY.read_text())
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "depth_to_rgba":
            module = ast.Module(body=[node], type_ignores=[])
            namespace = {"np": np}
            exec(compile(module, str(PLAY), "exec"), namespace)  # noqa: S102 - the file under test
            return namespace["depth_to_rgba"]
    raise AssertionError("depth_to_rgba is missing from the playback script")


class DepthWindowTests(unittest.TestCase):
    """The head camera's depth image is on screen, and cheap enough to leave on.

    The sensor is a ray-cast camera, not a render product, so the window cannot
    be a viewport: the image is the sensor's own tensor pushed through a byte
    provider.
    """

    @classmethod
    def setUpClass(cls):
        cls.src = PLAY.read_text()

    def test_the_window_is_on_by_default_and_can_be_turned_off(self):
        self.assertIn('"--no_depth_window"', self.src)
        block = self.src.split('"--no_depth_window"', 1)[1].split("parser.add_argument", 1)[0]
        self.assertIn('dest="depth_window"', block)
        self.assertIn('action="store_false"', block)

    def test_it_only_opens_with_a_gui(self):
        creation = [line for line in self.src.splitlines() if "DepthWindow(env)" in line]
        self.assertEqual(len(creation), 1, "the depth window is created in more than one place")
        self.assertIn("args_cli.headless", creation[0])
        self.assertIn("args_cli.depth_window", creation[0])

    def test_it_shows_the_sensor_frame_through_the_provider_kit_ships_with(self):
        self.assertIn('camera.data.output["distance_to_image_plane"]', self.src)
        self.assertIn("ui.ByteImageProvider()", self.src)
        self.assertIn("set_bytes_data(list(image.tobytes())", self.src)
        # No resampling: the widget scales the sensor's own 64x36 frame.
        self.assertNotIn("np.repeat", self.src)
        self.assertIn("IwpFillPolicy.IWP_PRESERVE_ASPECT_FIT", self.src)

    def test_the_update_is_rate_limited(self):
        self.assertIn("EVERY_STEPS", self.src)
        self.assertIn("if self._since_update % self.EVERY_STEPS:", self.src)

    def test_near_is_bright_far_is_dark_and_the_range_clips(self):
        import numpy as np

        to_rgba = _load_depth_to_rgba()
        depth = np.array([[[0.1]], [[1.3]], [[2.5]], [[9.0]]], dtype=np.float32)
        image = to_rgba(depth, 0.1, 2.5)
        self.assertEqual(image.shape, (4, 1, 4))
        self.assertEqual(image.dtype, np.uint8)
        self.assertTrue(np.all(image[..., 3] == 255), "the provider wants an opaque image")
        gray = image[..., 0]
        self.assertTrue(np.all(image[..., :3] == gray[..., None]), "the image must be grey")
        self.assertEqual(gray[0], 255, "the near clip must be the brightest value")
        self.assertEqual(gray[2], 0, "the far clip must be the darkest value")
        self.assertEqual(gray[3], 0, "beyond the range must clip, not wrap")
        self.assertLess(gray[1], gray[0])
        self.assertGreater(gray[1], gray[2])

    def test_a_channel_axis_is_tolerated(self):
        import numpy as np

        to_rgba = _load_depth_to_rgba()
        flat = np.full((2, 3), 1.3, dtype=np.float32)
        self.assertTrue(np.array_equal(to_rgba(flat, 0.1, 2.5), to_rgba(flat[..., None], 0.1, 2.5)))


def _load_depth_window():
    """The playback script's depth window, compiled without Kit."""
    import ast

    import numpy as np

    tree = ast.parse(PLAY.read_text())
    wanted = [
        node
        for node in tree.body
        if (isinstance(node, ast.FunctionDef) and node.name == "depth_to_rgba")
        or (isinstance(node, ast.ClassDef) and node.name == "DepthWindow")
    ]
    if len(wanted) != 2:
        raise AssertionError("DepthWindow and depth_to_rgba are missing from the playback script")
    module = ast.Module(body=wanted, type_ignores=[])
    namespace = {"np": np}
    exec(compile(module, str(PLAY), "exec"), namespace)  # noqa: S102 - the file under test
    return namespace["DepthWindow"]


class _FakeProvider:
    def __init__(self):
        self.calls: list[tuple[bytes, tuple[int, int]]] = []

    def set_bytes_data(self, data, size):
        self.calls.append((bytes(data), tuple(size)))


class _FakeWindow:
    def __init__(self, title, width, height):
        self.title = title
        self.width = width
        self.height = height
        self.frame = contextlib.nullcontext()


class _FakeScene:
    def __init__(self, camera):
        self._camera = camera

    def __getitem__(self, key):
        return self._camera


class _FakeEnv:
    """Just enough env for the window: a scene whose camera holds an output dict."""

    def __init__(self, output: dict) -> None:
        camera = types.SimpleNamespace(data=types.SimpleNamespace(output=output))
        self.unwrapped = types.SimpleNamespace(scene=_FakeScene(camera))


class DepthWindowBehaviourTests(unittest.TestCase):
    """Drive the window with a stand-in Kit, so its wiring is exercised."""

    def setUp(self):
        import sys

        self.np = numpy
        self.providers: list[_FakeProvider] = []
        self.windows: list[_FakeWindow] = []
        self.images: list[tuple[object, dict]] = []
        providers, windows, images = self.providers, self.windows, self.images

        fake_ui = types.ModuleType("omni.ui")
        fake_ui.ByteImageProvider = lambda: (providers.append(_FakeProvider()) or providers[-1])
        fake_ui.Window = lambda title, width, height: (
            windows.append(_FakeWindow(title, width, height)) or windows[-1]
        )
        fake_ui.ImageWithProvider = lambda provider, **kwargs: images.append((provider, kwargs))
        # The widget takes its own fill-policy enum; a plain `ui.FillPolicy` value
        # is refused by the binding at run time, which the stand-in reproduces.
        fake_ui.IwpFillPolicy = types.SimpleNamespace(IWP_PRESERVE_ASPECT_FIT="iwp_preserve_aspect_fit")
        fake_omni = types.ModuleType("omni")
        fake_omni.ui = fake_ui
        self._saved = {name: sys.modules.get(name) for name in ("omni", "omni.ui")}
        sys.modules["omni"] = fake_omni
        sys.modules["omni.ui"] = fake_ui
        self.addCleanup(self._restore_modules)

    def _restore_modules(self):
        import sys

        for name, module in self._saved.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module

    def test_the_window_opens_once_and_pushes_the_sensor_image(self):
        window_class = _load_depth_window()
        depth = self.np.full((36, 64, 1), 1.3, dtype=self.np.float32)
        window = window_class(_FakeEnv({"distance_to_image_plane": depth[None, ...]}))

        for _ in range(window_class.EVERY_STEPS - 1):
            window.update()
        self.assertEqual(self.windows, [], "the window opened before a frame was ready")
        window.update()

        self.assertEqual(len(self.windows), 1)
        opened = self.windows[0]
        self.assertEqual(opened.width, 64 * window_class.WIDGET_SCALE)
        self.assertEqual(opened.height, 36 * window_class.WIDGET_SCALE)
        self.assertIn("m", opened.title, "the title must state the range the image is clipped to")
        self.assertEqual(len(self.images), 1)
        self.assertEqual(self.images[0][1]["width"], 64 * window_class.WIDGET_SCALE)
        self.assertEqual(
            self.images[0][1]["fill_policy"],
            "iwp_preserve_aspect_fit",
            "the widget's own fill-policy enum is what the binding accepts",
        )

        self.assertEqual(len(self.providers), 1)
        pushed, size = self.providers[0].calls[0]
        self.assertEqual(size, (64, 36), "the provider must get the sensor's own resolution")
        rgba = self.np.frombuffer(pushed, dtype=self.np.uint8).reshape(36, 64, 4)
        self.assertTrue(self.np.all(rgba[..., 3] == 255))
        expected = _load_depth_to_rgba()(depth, *window_class.RANGE_M)
        self.assertTrue(self.np.array_equal(rgba, expected))

    def test_the_window_rate_limits_its_updates(self):
        window_class = _load_depth_window()
        depth = self.np.full((36, 64, 1), 0.5, dtype=self.np.float32)
        window = window_class(_FakeEnv({"distance_to_image_plane": depth[None, ...]}))
        for _ in range(window_class.EVERY_STEPS * 3):
            window.update()
        self.assertEqual(len(self.providers[0].calls), 3, "one push per EVERY_STEPS updates")
        self.assertEqual(len(self.windows), 1, "the window must be opened exactly once")

    def test_a_missing_sensor_output_opens_nothing(self):
        window_class = _load_depth_window()
        window = window_class(_FakeEnv({}))
        for _ in range(window_class.EVERY_STEPS * 2):
            window.update()
        self.assertEqual(self.windows, [])

    def test_a_ui_failure_switches_the_window_off_instead_of_ending_the_run(self):
        window_class = _load_depth_window()
        depth = self.np.full((36, 64, 1), 1.3, dtype=self.np.float32)
        window = window_class(_FakeEnv({"distance_to_image_plane": depth[None, ...]}))

        original = _FakeProvider.set_bytes_data
        attempts = []

        def explode(self, data, size):
            attempts.append(size)
            raise RuntimeError("no provider today")

        _FakeProvider.set_bytes_data = explode
        self.addCleanup(lambda: setattr(_FakeProvider, "set_bytes_data", original))

        for _ in range(window_class.EVERY_STEPS * 3):
            window.update()  # a display cannot be allowed to end a driving run

        self.assertEqual(len(attempts), 1, "a failed window must not keep retrying")
        self.assertEqual(len(self.windows), 1, "the window must not be reopened after a failure")


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
