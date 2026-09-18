"""Contract tests for the InstinctLab parkour playback: driving it, and seeing it.

These run without a simulator.  They cover the rules that matter while driving
the robot by hand -- a command must stay inside the envelope the checkpoint was
trained on, the viewport camera must be the operator's, and the depth window
must show the sensor's own frame -- because the policy degrades into a stand or
a fall outside the trained envelope, and a locked camera makes the run
unusable.  The two tiny ONNX sessions are pinned there as well: with their
default pools they spin on every core and starve the step they are part of.
The opt-in env.step attribution is pinned too: it must cover the step's manager
calls, keep nested sensor updates out of the phase sum, and never force a GPU
sync that would change what it measures.

The playback-only scene trimming and the diagnostic controls are pinned as
well: the training-only sensors and Kit's own loop rate limiter go, the
--diagnostic_no_depth freeze stays diagnostic, and the policy observations, the
contact sensor and the fall terminations all survive it.

The clamp values are cross-checked against the checkpoint's own training config
(`parkour_env_cfg.py`) where that file is available, so the test notices if the
policy's trained range and the driver's clamp ever drift apart.
"""

from __future__ import annotations

import contextlib
import io
import re
import time
import types
import unittest
from pathlib import Path

import numpy

try:  # pragma: no cover - depends on the interpreter
    import onnxruntime  # noqa: F401

    HAVE_ONNXRUNTIME = True
except ImportError:  # pragma: no cover
    HAVE_ONNXRUNTIME = False

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

    def test_it_only_opens_with_a_displayed_ui(self):
        creation = [line for line in self.src.splitlines() if "DepthWindow(env)" in line]
        self.assertEqual(len(creation), 1, "the depth window is created in more than one place")
        self.assertIn("ui_displayed", creation[0])
        self.assertIn("args_cli.depth_window", creation[0])
        # A streamed run is headless on the host yet still draws the UI, so the
        # answer is taken before AppLauncher rewrites the flag.
        self.assertIn("ui_displayed = not args_cli.headless", self.src)
        before_launcher = self.src.split("app_launcher = AppLauncher(", 1)[0]
        self.assertIn("ui_displayed = not args_cli.headless", before_launcher)

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

    def test_playback_drops_training_only_sensors(self):
        src = PLAY.read_text()
        self.assertIn('env_cfg.scene.motion_reference = None', src)
        self.assertIn('env_cfg.terminations.dataset_exhausted = None', src)
        self.assertIn('(\"left_height_scanner\", \"right_height_scanner\", \"leg_volume_points\")', src)
        self.assertIn('env_cfg.events.register_virtual_obstacles = None', src)
        self.assertIn('unused AMP motion reference and dataset termination disabled', src)
        self.assertIn('reward-only sensors disabled', src)


def _load_make_fast():
    """The playback script's scene trimming, compiled without Isaac Lab."""
    import ast

    tree = ast.parse(PLAY.read_text())
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "make_fast":
            module = ast.Module(body=[node], type_ignores=[])
            namespace: dict[str, object] = {}
            exec(compile(module, str(PLAY), "exec"), namespace)  # noqa: S102 - the file under test
            return namespace["make_fast"]
    raise AssertionError("make_fast is missing from the playback script")


def _tuning_cfg():
    """A stand-in env_cfg carrying exactly the fields make_fast reads and writes."""
    return types.SimpleNamespace(
        scene=types.SimpleNamespace(
            terrain=types.SimpleNamespace(terrain_generator=None),
            camera=object(),
            contact_forces=object(),
            left_height_scanner=object(),
            right_height_scanner=object(),
            leg_volume_points=object(),
            motion_reference=object(),
        ),
        commands=None,
        rewards=object(),
        observations=types.SimpleNamespace(
            policy=object(), critic=object(), amp_policy=object(), amp_reference=object()
        ),
        terminations=types.SimpleNamespace(
            time_out=object(),
            terrain_out_bound=object(),
            base_contact=object(),
            bad_orientation=object(),
            root_height=object(),
            dataset_exhausted=object(),
        ),
        events=types.SimpleNamespace(register_virtual_obstacles=object(), reset_base=object()),
    )


class PlaybackTuningTests(unittest.TestCase):
    """The --play trimming drops training-only work; playback's inputs survive.

    ``make_fast`` runs against a stand-in config here, so the conditions inside
    it are exercised instead of only pattern-matched: the AMP motion reference
    and its dataset termination leave together, the reward-only sensors leave
    with the reward manager, and everything the exported actor, the contact
    sensor and the fall terminations read is kept.
    """

    def setUp(self):
        self.make_fast = _load_make_fast()

    def _tuned(self, cfg=None):
        cfg = _tuning_cfg() if cfg is None else cfg
        kept = {
            "contact_forces": cfg.scene.contact_forces,
            "camera": cfg.scene.camera,
            "policy": cfg.observations.policy,
            "terminations": cfg.terminations,
            "fall_terms": tuple(
                getattr(cfg.terminations, name)
                for name in ("time_out", "base_contact", "bad_orientation", "root_height")
            ),
        }
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            self.make_fast(cfg, interactive=True)
        return cfg, kept, stdout.getvalue()

    def test_the_amp_reference_and_its_dataset_termination_go_together(self):
        cfg, _, out = self._tuned()
        self.assertIsNone(cfg.scene.motion_reference)
        self.assertIsNone(cfg.terminations.dataset_exhausted)
        self.assertIn("unused AMP motion reference and dataset termination disabled", out)

    def test_without_the_reference_the_dataset_termination_is_left_alone(self):
        # The termination is only removed with the sensor it binds, so a task
        # without the AMP reference comes through here with its MDP intact.
        cfg = _tuning_cfg()
        cfg.scene.motion_reference = None
        cfg, _, out = self._tuned(cfg)
        self.assertIsNotNone(cfg.terminations.dataset_exhausted)
        self.assertNotIn("motion reference", out)

    def test_reward_only_sensors_and_their_startup_event_are_dropped(self):
        cfg, _, out = self._tuned()
        for name in ("left_height_scanner", "right_height_scanner", "leg_volume_points"):
            self.assertIsNone(getattr(cfg.scene, name))
        self.assertIsNone(cfg.events.register_virtual_obstacles)
        self.assertIn(
            "reward-only sensors disabled: left_height_scanner, right_height_scanner, leg_volume_points", out
        )

    def test_contact_sensor_and_camera_are_kept(self):
        cfg, kept, _ = self._tuned()
        # The contact sensor is what resets the robot after a torso hit, and the
        # camera frame is the depth observation the actor consumes.
        self.assertIs(cfg.scene.contact_forces, kept["contact_forces"])
        self.assertIs(cfg.scene.camera, kept["camera"])

    def test_fall_terminations_survive_untouched(self):
        cfg, kept, _ = self._tuned()
        self.assertIs(cfg.terminations, kept["terminations"], "the terminations manager must stay configured")
        for name, term in zip(("time_out", "base_contact", "bad_orientation", "root_height"), kept["fall_terms"]):
            self.assertIs(
                getattr(cfg.terminations, name), term, f"{name} is a fall recovery path, not a training artefact"
            )
        self.assertIsNone(cfg.terminations.dataset_exhausted)

    def test_policy_observations_survive_and_only_the_amp_groups_go(self):
        cfg, kept, out = self._tuned()
        self.assertIs(cfg.observations.policy, kept["policy"])
        for group in ("critic", "amp_policy", "amp_reference"):
            self.assertIsNone(getattr(cfg.observations, group))
        self.assertIn(
            "observation groups disabled: critic, amp_policy, amp_reference (playback reads 'policy')", out
        )

    def test_rewards_are_dropped_because_playback_discards_them(self):
        cfg, _, out = self._tuned()
        self.assertIsNone(cfg.rewards)
        self.assertIn("reward terms disabled (playback discards the reward)", out)

    def test_no_tuning_returns_before_touching_anything(self):
        cfg = _tuning_cfg()
        rewards = cfg.rewards
        scene = {
            name: getattr(cfg.scene, name)
            for name in ("motion_reference", "left_height_scanner", "leg_volume_points")
        }
        groups = {name: getattr(cfg.observations, name) for name in ("critic", "amp_policy", "amp_reference")}
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            self.make_fast(cfg, interactive=False)
        self.assertEqual(stdout.getvalue(), "", "--no_tuning must restore the raw training config")
        for name, entity in scene.items():
            self.assertIs(getattr(cfg.scene, name), entity)
        self.assertIs(cfg.rewards, rewards)
        for name, group in groups.items():
            self.assertIs(getattr(cfg.observations, name), group)


def _main_block(start: str, stop: str):
    """Compile the run of ``main()`` statements from the one mentioning ``start``
    through the one mentioning ``stop``, so those real lines can run without Kit.
    """
    import ast

    source = PLAY.read_text()
    main = next(
        (node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef) and node.name == "main"),
        None,
    )
    if main is None:
        raise AssertionError("main is missing from the playback script")
    segments = [ast.get_source_segment(source, statement) or "" for statement in main.body]
    starts = [index for index, segment in enumerate(segments) if start in segment]
    if len(starts) != 1:
        raise AssertionError(f"expected exactly one main() statement mentioning {start!r}")
    first = starts[0]
    stops = [index for index in range(first, len(segments)) if stop in segments[index]]
    if not stops:
        raise AssertionError(f"no main() statement mentioning {stop!r} after {start!r}")
    module = ast.Module(body=main.body[first : stops[0] + 1], type_ignores=[])
    return compile(module, str(PLAY), "exec")


class _FakeKitSettings:
    """The slice of carb.settings that the playback script touches."""

    def __init__(self, rate_limited: bool):
        self.values = {"/app/runLoops/main/rateLimitEnabled": rate_limited}
        self.set_calls: list[tuple[str, bool]] = []

    def get(self, path: str):
        return self.values.get(path)

    def set_bool(self, path: str, value: bool) -> None:
        self.set_calls.append((path, value))
        self.values[path] = value


class KitMainLoopRateLimitTests(unittest.TestCase):
    """Kit's own main-loop rate limiter must be off; the runner owns pacing.

    With both enabled, every rendered frame waited on Kit's limiter (about 20 ms
    here) even when the GPU was idle, and the runner's absolute-deadline pacing
    was applied on top.  The real block from ``main()`` runs against a stand-in
    carb.settings, so the read-before-write and the fixed ``False`` are pinned.
    """

    def _run(self, rate_limited: bool):
        settings = _FakeKitSettings(rate_limited)
        carb = types.SimpleNamespace(settings=types.SimpleNamespace(get_settings=lambda: settings))
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            exec(_main_block("carb.settings.get_settings()", "Kit main-loop rate limiter"), {"carb": carb})
        return settings, stdout.getvalue()

    def test_the_limiter_is_switched_off_and_the_old_value_reported(self):
        for previous in (True, False):
            settings, out = self._run(previous)
            self.assertEqual(settings.set_calls, [("/app/runLoops/main/rateLimitEnabled", False)])
            self.assertFalse(settings.values["/app/runLoops/main/rateLimitEnabled"])
            self.assertIn(f"[INFO] Kit main-loop rate limiter: {previous} -> False", out)

    def test_kit_is_disabled_before_anything_is_built(self):
        src = PLAY.read_text()
        self.assertIn("import carb.settings", src)
        self.assertEqual(src.count("set_bool("), 1, "only the limiter block may write Kit settings")
        block = src.index("/app/runLoops/main/rateLimitEnabled")
        self.assertLess(block, src.index("env = gym.make("), "the limiter's wait would land in scene setup too")
        self.assertLess(block, src.index("while simulation_app.is_running()"))


class DiagnosticNoDepthTests(unittest.TestCase):
    """--diagnostic_no_depth is a profiling control, not a playback mode.

    The flag freezes the ray-cast depth sensor so a profile can take it out of
    the loop; the policy then runs on a stale frame, so the run's behavior is
    meaningless and has to be announced as such.  It must default to off, and
    it must not be wired into anything else the playback does.
    """

    @classmethod
    def setUpClass(cls):
        cls.src = PLAY.read_text()

    def _run(self, enabled: bool):
        cfg = types.SimpleNamespace(scene=types.SimpleNamespace(camera=types.SimpleNamespace(update_period=0.02)))
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            exec(
                _main_block("args_cli.diagnostic_no_depth", "args_cli.diagnostic_no_depth"),
                {"args_cli": types.SimpleNamespace(diagnostic_no_depth=enabled), "env_cfg": cfg},
            )
        return cfg, stdout.getvalue()

    def test_the_flag_defaults_to_off_and_is_announced_as_invalid(self):
        block = self.src.split('"--diagnostic_no_depth"', 1)[1].split("parser.add_argument", 1)[0]
        self.assertIn('action="store_true"', block)
        self.assertIn("default=False", block)
        self.assertIn("freeze the policy depth sensor after initialization", block)
        self.assertIn("robot behavior is not valid", block)

    def test_it_freezes_only_the_depth_sensor(self):
        cfg, out = self._run(True)
        self.assertEqual(cfg.scene.camera.update_period, 1.0e9, "the sensor must stop refreshing")
        self.assertIn("depth sensor updates frozen", out)
        self.assertIn("policy behavior is not valid", out)

    def test_without_the_flag_the_camera_period_is_untouched(self):
        cfg, out = self._run(False)
        self.assertEqual(cfg.scene.camera.update_period, 0.02)
        self.assertEqual(out, "")

    def test_the_flag_is_declared_once_and_consumed_once(self):
        # A diagnostic switch must not become a default or change the command
        # handling, the scene or the report; it is only the sensor period.
        self.assertEqual(self.src.count("diagnostic_no_depth"), 2, "one declaration, one use")
        handler = self.src.index("if args_cli.diagnostic_no_depth:")
        self.assertLess(self.src.index("env_cfg = parse_env_cfg("), handler)
        self.assertLess(handler, self.src.index("env = gym.make("))


class OnnxBatchSizeTests(unittest.TestCase):
    """The shipped policy exports have a fixed batch dimension of one."""

    @staticmethod
    def _run(*, keyboard: bool, requested: int | None, configured: int):
        args = types.SimpleNamespace(keyboard_control=keyboard, num_envs=requested)
        cfg = types.SimpleNamespace(scene=types.SimpleNamespace(num_envs=configured), episode_length_s=20.0)
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            exec(
                _main_block("You are driving it, not measuring it", "The shipped ONNX exports require"),
                {"args_cli": args, "env_cfg": cfg},
            )
        return cfg, stdout.getvalue()

    def test_no_keyboard_defaults_to_one_environment(self):
        cfg, out = self._run(keyboard=False, requested=None, configured=4096)
        self.assertEqual(cfg.scene.num_envs, 1)
        self.assertIn("fixed batch=1", out)

    def test_an_explicit_multi_environment_run_fails_before_onnx_inference(self):
        with self.assertRaisesRegex(SystemExit, "require --num_envs 1; got 8"):
            self._run(keyboard=False, requested=8, configured=8)

    def test_keyboard_playback_stays_single_environment(self):
        cfg, _ = self._run(keyboard=True, requested=8, configured=8)
        self.assertEqual(cfg.scene.num_envs, 1)
        self.assertEqual(cfg.episode_length_s, 1e10)


def _load_session_helpers():
    """The playback script's ONNX session helpers, compiled without Kit or onnxruntime."""
    import ast

    tree = ast.parse(PLAY.read_text())
    wanted = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in ("cpu_session_options", "create_policy_sessions")
    ]
    if len(wanted) != 2:
        raise AssertionError("cpu_session_options/create_policy_sessions are missing from the playback script")
    module = ast.Module(body=wanted, type_ignores=[])
    namespace: dict[str, object] = {}
    exec(compile(module, str(PLAY), "exec"), namespace)  # noqa: S102 - the file under test
    return namespace["cpu_session_options"], namespace["create_policy_sessions"]


class _FakeSessionOptions:
    """The slice of onnxruntime's SessionOptions that the playback script touches."""

    def __init__(self):
        self.intra_op_num_threads = 0
        self.inter_op_num_threads = 0
        self.config_entries: dict[str, str] = {}

    def add_session_config_entry(self, key: str, value: str) -> None:
        self.config_entries[key] = value


class _FakeInferenceSession:
    """Records how a session was opened; the graphs themselves are never needed."""

    created: list["_FakeInferenceSession"] = []

    def __init__(self, path, sess_options=None, providers=None):
        self.path = path
        self.sess_options = sess_options
        self.providers = providers
        type(self).created.append(self)


def _fake_ort(execution_mode: bool = True) -> types.SimpleNamespace:
    ort = types.SimpleNamespace(SessionOptions=_FakeSessionOptions, InferenceSession=_FakeInferenceSession)
    if execution_mode:
        ort.ExecutionMode = types.SimpleNamespace(ORT_SEQUENTIAL=0, ORT_PARALLEL=1)
    return ort


class OnnxSessionThreadTests(unittest.TestCase):
    """The two tiny ONNX graphs must run on one non-spinning thread each.

    onnxruntime's defaults give every session a thread pool sized to the
    machine's core count and let those threads spin between calls; on this
    24-core host that starved the environment step and the depth ray-caster,
    which is most of the loop.  The playback opts out through explicit
    SessionOptions instead, for both the encoder and the actor.
    """

    def setUp(self):
        _FakeInferenceSession.created.clear()

    def test_options_are_single_threaded(self):
        cpu_session_options, _ = _load_session_helpers()
        options = cpu_session_options(_fake_ort())
        self.assertEqual(options.intra_op_num_threads, 1)
        self.assertEqual(options.inter_op_num_threads, 1)

    def test_execution_is_sequential_when_the_runtime_offers_the_enum(self):
        cpu_session_options, _ = _load_session_helpers()
        ort = _fake_ort()
        options = cpu_session_options(ort)
        self.assertEqual(options.execution_mode, ort.ExecutionMode.ORT_SEQUENTIAL)

    def test_a_runtime_without_the_execution_mode_enum_is_still_configured(self):
        cpu_session_options, _ = _load_session_helpers()
        options = cpu_session_options(_fake_ort(execution_mode=False))
        self.assertFalse(hasattr(options, "execution_mode"))
        self.assertEqual(options.intra_op_num_threads, 1)

    def test_spinning_is_disabled_through_the_documented_session_entries(self):
        cpu_session_options, _ = _load_session_helpers()
        options = cpu_session_options(_fake_ort())
        self.assertEqual(
            options.config_entries,
            {"session.intra_op.allow_spinning": "0", "session.inter_op.allow_spinning": "0"},
        )

    def test_both_graphs_are_opened_with_the_same_configured_options(self):
        _, create_policy_sessions = _load_session_helpers()
        encoder, actor = create_policy_sessions(_fake_ort(), "encoder.onnx", "actor.onnx")
        self.assertEqual([session.path for session in _FakeInferenceSession.created], ["encoder.onnx", "actor.onnx"])
        encoder_session, actor_session = _FakeInferenceSession.created
        self.assertIs(encoder_session, encoder)
        self.assertIs(actor_session, actor)
        for session in (encoder_session, actor_session):
            self.assertEqual(session.providers, ["CPUExecutionProvider"])
            self.assertEqual(session.sess_options.intra_op_num_threads, 1)
            self.assertEqual(session.sess_options.inter_op_num_threads, 1)
        self.assertIs(
            encoder_session.sess_options,
            actor_session.sess_options,
            "the encoder and the actor must not drift apart in their settings",
        )

    def test_the_thread_configuration_is_printed_exactly_once(self):
        _, create_policy_sessions = _load_session_helpers()
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            create_policy_sessions(_fake_ort(), "encoder.onnx", "actor.onnx")
        config_lines = [line for line in stdout.getvalue().splitlines() if "intra_op_num_threads" in line]
        self.assertEqual(len(config_lines), 1, "the effective thread configuration must be printed once")
        line = config_lines[0]
        self.assertIn("intra_op_num_threads=1", line)
        self.assertIn("inter_op_num_threads=1", line)
        self.assertIn("session.intra_op.allow_spinning=0", line)
        self.assertIn("session.inter_op.allow_spinning=0", line)
        self.assertIn("CPUExecutionProvider", line)

    def test_the_policy_uses_the_helpers_instead_of_the_default_sessions(self):
        src = PLAY.read_text()
        self.assertIn("create_policy_sessions(ort, encoder_path, actor_path)", src)
        # Two InferenceSession(...) calls, both handing the configured options to
        # the runtime; a bare call would silently get the default thread pools.
        self.assertEqual(src.count("InferenceSession("), 2)
        self.assertEqual(src.count("sess_options="), 2)


@unittest.skipUnless(HAVE_ONNXRUNTIME, "onnxruntime is not installed in this interpreter")
class OnnxSessionOptionsRuntimeTests(unittest.TestCase):
    """The helper against the real runtime: no model needed, but the entries stick."""

    def test_the_runtime_reports_the_configured_settings_back(self):
        import onnxruntime as ort  # the class-level skip already proved it imports

        cpu_session_options, _ = _load_session_helpers()
        options = cpu_session_options(ort)
        self.assertEqual(options.intra_op_num_threads, 1)
        self.assertEqual(options.inter_op_num_threads, 1)
        self.assertEqual(options.execution_mode, ort.ExecutionMode.ORT_SEQUENTIAL)
        if hasattr(options, "get_session_config_entry"):
            self.assertEqual(options.get_session_config_entry("session.intra_op.allow_spinning"), "0")
            self.assertEqual(options.get_session_config_entry("session.inter_op.allow_spinning"), "0")


def _load_step_attribution():
    """The playback script's step attribution, compiled without Kit or torch."""
    import ast

    tree = ast.parse(PLAY.read_text())
    wanted = [
        node
        for node in tree.body
        if (isinstance(node, ast.ClassDef) and node.name == "StepAttribution")
        or (isinstance(node, ast.FunctionDef) and node.name == "_resolve_attribute")
    ]
    if len(wanted) != 2:
        raise AssertionError("StepAttribution/_resolve_attribute are missing from the playback script")
    module = ast.Module(body=wanted, type_ignores=[])
    namespace = {"time": time}
    exec(compile(module, str(PLAY), "exec"), namespace)  # noqa: S102 - the file under test
    return namespace["StepAttribution"]


_PROBE_SENSORS = (
    "camera",
    "left_height_scanner",
    "right_height_scanner",
    "contact_forces",
    "leg_volume_points",
    "motion_reference",
)


class _ProbeClock:
    """A fake perf_counter; the probe methods advance it by fixed amounts."""

    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


def _probe_method(log, label, clock, seconds):
    def method(*args, **kwargs):
        log.append(label)
        clock.advance(seconds)
        return label

    return method


def _drop_path(root, path):
    """Remove a dotted attribute (or dict entry) from the fake env."""
    parts = path.split(".")
    owner = root
    for part in parts[:-1]:
        owner = owner[part] if isinstance(owner, dict) else getattr(owner, part)
    if isinstance(owner, dict):
        owner.pop(parts[-1], None)
    else:
        delattr(owner, parts[-1])


def _fake_step_env(log, clock, drop=()):
    """A stand-in env with the attribute layout ManagerBasedRLEnv.step uses.

    Each stub advances the injected clock by a fixed slice (sensors 2 ms, the
    rest 1 ms), so the reported numbers are exact.  The physics-side calls repeat
    twice, like the real decimation loop.  ``drop`` removes targets given as
    dotted paths, to model managers and sensors a different task does not have.
    """

    def stub(label, seconds):
        return _probe_method(log, label, clock, seconds)

    env = types.SimpleNamespace()
    env.action_manager = types.SimpleNamespace(
        process_action=stub("action_manager.process_action", 0.001),
        apply_action=stub("action_manager.apply_action", 0.001),
    )
    env.command_manager = types.SimpleNamespace(compute=stub("command_manager.compute", 0.001))
    env.termination_manager = types.SimpleNamespace(compute=stub("termination_manager.compute", 0.001))
    env.reward_manager = types.SimpleNamespace(compute=stub("reward_manager.compute", 0.001))
    env.observation_manager = types.SimpleNamespace(compute=stub("observation_manager.compute", 0.001))
    env.curriculum_manager = types.SimpleNamespace(compute=stub("curriculum_manager.compute", 0.001))
    env.event_manager = types.SimpleNamespace(apply=stub("event_manager.apply", 0.001))
    env.sim = types.SimpleNamespace(step=stub("sim.step", 0.001), render=stub("sim.render", 0.001))

    sensors = {name: types.SimpleNamespace(update=stub(f"{name}.update", 0.002)) for name in _PROBE_SENSORS}

    def scene_write():
        clock.advance(0.001)
        log.append("scene.write_data_to_sim")

    def scene_update(dt):
        clock.advance(0.001)  # the scene's own bookkeeping
        log.append("scene.update")
        for sensor in sensors.values():
            sensor.update(dt, force_recompute=True)

    env.scene = types.SimpleNamespace(write_data_to_sim=scene_write, update=scene_update, sensors=sensors)

    def core_step(action):
        log.append("env.step")
        env.action_manager.process_action(action)
        for _ in range(2):  # decimation: the physics-side calls repeat
            env.action_manager.apply_action()
            env.scene.write_data_to_sim()
            env.sim.step(render=False)
            env.scene.update(1.0 / 200.0)
        env.termination_manager.compute()
        env.reward_manager.compute(dt=0.02)
        env.command_manager.compute(dt=0.02)
        env.observation_manager.compute()

    env.step = core_step
    env.unwrapped = env
    for path in drop:
        _drop_path(env, path)
    return env


def _step_line_values(line):
    return {name: float(value) for name, value in re.findall(r"([a-z_]+)=([0-9.]+)", line)}


class StepAttributionTests(unittest.TestCase):
    """Where one env.step's milliseconds go, measured on a fake env.

    The fake advances a fake clock by fixed amounts per call, so the reported
    per-loop values are exact: 1 ms per manager call, 2 ms per sensor update
    inside a 1 ms scene update, everything twice for the decimation loop.
    """

    def setUp(self):
        self.clock = _ProbeClock()
        self.log = []
        self.attribution_class = _load_step_attribution()

    def _installed(self, drop=()):
        env = _fake_step_env(self.log, self.clock, drop=drop)
        attribution = self.attribution_class(clock=self.clock)
        attribution.install(env)
        return env, attribution

    def test_the_required_phases_and_sensors_are_declared(self):
        phases = {name for _, name in self.attribution_class.PHASES}
        for required in (
            "action_process",
            "action_apply",
            "scene_write",
            "sim_step",
            "scene_update",
            "render",
            "command",
            "termination",
            "observation",
            "curriculum",
        ):
            self.assertIn(required, phases, f"{required} is not attributed")
        self.assertEqual(self.attribution_class.SENSORS, _PROBE_SENSORS)

    def test_the_phase_row_is_ms_per_loop_and_sums_to_the_env_step(self):
        env, attribution = self._installed()
        env.step(None)
        lines, _ = attribution.report(None, 0.040, 1)
        self.assertTrue(lines[0].startswith("[perf] step ms/loop: "))
        values = _step_line_values(lines[0])
        self.assertEqual(values["action_process"], 1.00)
        self.assertEqual(values["action_apply"], 2.00)
        self.assertEqual(values["scene_write"], 2.00)
        self.assertEqual(values["sim_step"], 2.00)
        self.assertEqual(values["scene_update"], 26.00)
        self.assertEqual(values["render"], 0.00)
        self.assertEqual(values["command"], 1.00)
        self.assertEqual(values["termination"], 1.00)
        self.assertEqual(values["reward"], 1.00)
        self.assertEqual(values["observation"], 1.00)
        self.assertEqual(values["other"], 0.00)
        self.assertEqual(values["wrapper"], 3.00)
        # Phases plus both remainders are exactly the loop-side env step.
        self.assertAlmostEqual(sum(values.values()), 40.00, places=6)

    def test_sensor_updates_are_reported_separately_from_the_phase_sum(self):
        env, attribution = self._installed()
        env.step(None)
        lines, _ = attribution.report(None, 0.040, 1)
        self.assertEqual(len(lines), 2, "the nested sensors get their own line")
        self.assertTrue(lines[1].startswith("[perf] step nested ms/loop"))
        top = _step_line_values(lines[0])
        nested = _step_line_values(lines[1])
        self.assertEqual(set(nested), set(_PROBE_SENSORS))
        for value in nested.values():
            self.assertEqual(value, 4.00, "two sensor updates per decimation loop")
        # The sensors are inside scene_update: taking them out of it and adding
        # them again would overshoot the env step the phase row sums to.
        self.assertEqual(top["scene_update"], sum(nested.values()) + 2.00)
        self.assertAlmostEqual(sum(top.values()), 40.00, places=6)
        self.assertNotIn("camera", top)

    def test_the_report_window_is_diffed_against_the_previous_snapshot(self):
        env, attribution = self._installed()
        env.step(None)
        _, snapshot = attribution.report(None, 0.040, 1)
        for _ in range(2):
            env.step(None)
        lines, _ = attribution.report(snapshot, 0.080, 2)
        values = _step_line_values(lines[0])
        # Two steps in the window, two decimation ticks each: 4 ms per loop for
        # sim_step, not the 8 ms a cumulative report would give.
        self.assertEqual(values["sim_step"], 2.00, "the second window must not repeat the first")
        self.assertAlmostEqual(sum(values.values()), 40.00, places=6)

    def test_absent_targets_are_skipped_without_failing(self):
        env, attribution = self._installed(
            drop=(
                "curriculum_manager",
                "event_manager",
                "scene.sensors.contact_forces",
                "scene.sensors.motion_reference",
            )
        )
        env.step(None)  # the remaining managers and sensors still run
        lines, _ = attribution.report(None, 0.032, 1)
        top = _step_line_values(lines[0])
        nested = _step_line_values(lines[1])
        self.assertNotIn("curriculum", top)
        self.assertNotIn("events", top)
        self.assertNotIn("contact_forces", nested)
        self.assertNotIn("motion_reference", nested)
        self.assertIn("curriculum_manager", " ".join(attribution.skipped))
        self.assertIn("scene.sensors.contact_forces", attribution.skipped)
        self.assertAlmostEqual(sum(top.values()), 32.00, places=6)

    def test_a_window_shorter_than_the_phases_clamps_the_remainder_at_zero(self):
        env, attribution = self._installed()
        env.step(None)
        lines, _ = attribution.report(None, 0.0, 1)
        values = _step_line_values(lines[0])
        for name, value in values.items():
            self.assertGreaterEqual(value, 0.0, f"{name} went negative")
        self.assertEqual(values["wrapper"], 0.00)

    def test_installing_twice_does_not_wrap_the_wrappers(self):
        env, attribution = self._installed()
        attribution.install(env)
        env.step(None)
        lines, _ = attribution.report(None, 0.040, 1)
        values = _step_line_values(lines[0])
        self.assertEqual(values["sim_step"], 2.00, "a second install would double every phase")


class StepAttributionSourceTests(unittest.TestCase):
    """The attribution must stay opt-in, and honest about what its numbers are."""

    @classmethod
    def setUpClass(cls):
        cls.src = PLAY.read_text()

    def test_the_flag_is_opt_in_and_implies_perf_detail(self):
        self.assertIn('"--step_detail"', self.src)
        block = self.src.split('"--step_detail"', 1)[1].split("parser.add_argument", 1)[0]
        self.assertIn('action="store_true"', block)
        self.assertIn("default=False", block)
        self.assertIn("if args_cli.step_detail:", self.src)
        self.assertIn("args_cli.perf_detail = True", self.src)

    def test_wrapping_happens_after_the_env_is_ready_and_before_the_loop(self):
        install = self.src.index("attribution.install(env)")
        self.assertGreater(install, self.src.index("env = InstinctRlVecEnvWrapper(env)"))
        self.assertGreater(install, self.src.index("obs, _ = env.get_observations()"))
        self.assertLess(install, self.src.index("while simulation_app.is_running()"))
        self.assertIn("attribution = None", self.src)

    def test_both_report_sites_are_gated_on_the_flag(self):
        self.assertEqual(self.src.count("if attribution is not None"), 2)
        self.assertEqual(self.src.count("attribution.report("), 2)

    def test_no_gpu_synchronize_is_ever_forced(self):
        # A sync would serialize the loop and change the timing it reports.
        self.assertNotIn("cuda.synchronize", self.src)
        self.assertNotIn("torch.cuda", self.src)

    def test_the_wall_interval_semantics_are_documented_and_announced(self):
        self.assertIn("CPU wall interval", self.src)
        self.assertIn("No GPU sync is inserted", self.src)
        self.assertIn("ms/loop are CPU wall intervals", self.src)
        # The remainder must be explained, not just printed as a number.
        self.assertIn("other=resets+recorder+bookkeeping, wrapper=outside the unwrapped env", self.src)


ISAACLAB_STEP_SOURCE = Path("/opt/src/isaaclab/source/isaaclab/isaaclab/envs/manager_based_rl_env.py")
ISAACLAB_SCENE_SOURCE = Path("/opt/src/isaaclab/source/isaaclab/isaaclab/scene/interactive_scene.py")
PARKOUR_ENV_CFG = (
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


@unittest.skipUnless(
    ISAACLAB_STEP_SOURCE.is_file() and ISAACLAB_SCENE_SOURCE.is_file(), "Isaac Lab sources are not present"
)
class StepAttributionIsaacLabContractTests(unittest.TestCase):
    """The wrapped methods must still be the calls the upstream step makes."""

    def test_the_step_still_makes_every_call_we_attribute(self):
        step = ISAACLAB_STEP_SOURCE.read_text()
        for call in (
            "self.action_manager.process_action(",
            "self.action_manager.apply_action()",
            "self.scene.write_data_to_sim()",
            "self.sim.step(render=False)",
            "self.sim.render()",
            "self.scene.update(dt=self.physics_dt)",
            "self.termination_manager.compute()",
            "self.reward_manager.compute(dt=self.step_dt)",
            "self.command_manager.compute(dt=self.step_dt)",
            "self.observation_manager.compute(update_history=True)",
            "self.curriculum_manager.compute(env_ids=env_ids)",
            "self.event_manager.apply(",
        ):
            self.assertIn(call, step, f"ManagerBasedRLEnv no longer calls {call}")

    def test_sensors_are_driven_from_scene_update(self):
        self.assertIn("sensor.update(dt, force_recompute=", ISAACLAB_SCENE_SOURCE.read_text())


@unittest.skipUnless(PARKOUR_ENV_CFG.is_file(), "InstinctLab checkout not present")
class StepAttributionSensorNameContractTests(unittest.TestCase):
    """The sensor names must be the ones the parkour task actually registers."""

    def test_every_instrumented_sensor_is_a_parkour_sensor(self):
        cfg = PARKOUR_ENV_CFG.read_text()
        for name in _load_step_attribution().SENSORS:
            self.assertRegex(
                cfg, rf"(?m)^\s*{re.escape(name)}\s*[:=]", f"{name} is not a sensor of the parkour cfg"
            )


def _cfg_block(cfg: str, start: str, stop: str) -> str:
    """The text of one config class, so a claim can be scoped to its block."""
    if start not in cfg or stop not in cfg:
        raise AssertionError(f"the parkour cfg has no {start!r} / {stop!r} boundary any more")
    return cfg.split(start, 1)[1].split(stop, 1)[0]


@unittest.skipUnless(PARKOUR_ENV_CFG.is_file(), "InstinctLab checkout not present")
class PlaybackRetentionContractTests(unittest.TestCase):
    """What make_fast drops must be training-only in the task that was trained.

    The playback keeps the contact sensor and the fall terminations because they
    reset the robot when it falls, and keeps the policy observations because the
    exported actor consumes them.  This checks that claim against the parkour
    config: the dropped names are reward inputs, and the kept paths are the ones
    the terminations, the policy group and the depth observation reference.
    """

    @classmethod
    def setUpClass(cls):
        cfg = PARKOUR_ENV_CFG.read_text()
        cls.cfg = cfg
        cls.rewards = _cfg_block(cfg, "class G1Rewards", "class RewardsCfg")
        cls.terminations = _cfg_block(cfg, "class TerminationsCfg", "class EventCfg")
        cls.policy = _cfg_block(cfg, "class PolicyCfg", "class CriticCfg")

    def test_dropped_sensors_exist_and_feed_rewards(self):
        for name in ("left_height_scanner", "right_height_scanner", "leg_volume_points"):
            self.assertRegex(self.cfg, rf"(?m)^\s*{re.escape(name)}\s*[:=]", f"{name} is not in the parkour cfg")
            self.assertIn(name, self.rewards, f"{name} must be a reward input to be droppable")

    def test_dropped_names_reach_no_termination_or_policy_observation(self):
        for name in ("left_height_scanner", "right_height_scanner", "leg_volume_points"):
            self.assertNotIn(name, self.terminations, f"{name} resets the robot; it cannot be dropped")
            self.assertNotIn(name, self.policy, f"{name} feeds the actor; it cannot be dropped")
        # The AMP reference does bind one termination -- that is the
        # dataset-exhausted term make_fast removes together with the sensor --
        # but it never reaches the policy observation group.
        self.assertNotIn("motion_reference", self.policy)

    def test_the_only_termination_reading_the_amp_reference_is_the_one_dropped(self):
        names = re.findall(r"(?m)^    (\w+) = DoneTerm\(", self.terminations)
        bodies = re.split(r"(?m)^    \w+ = DoneTerm\(", self.terminations)[1:]
        self.assertEqual(len(names), len(bodies), "the termination scan lost a term")
        self.assertEqual(
            [name for name, body in zip(names, bodies) if "motion_reference" in body],
            ["dataset_exhausted"],
            "the AMP reference must only bind the termination make_fast removes with it",
        )

    def test_the_fall_recovery_path_is_contact_based_and_kept(self):
        # base_contact resets the robot when the torso hits; it reads the sensor
        # make_fast deliberately leaves in place.
        self.assertIn("illegal_contact", self.terminations)
        self.assertIn('SceneEntityCfg("contact_forces", body_names="torso_link")', self.terminations)

    def test_the_depth_observation_reads_the_camera_make_fast_keeps(self):
        self.assertIn('SceneEntityCfg("camera")', self.policy)


if __name__ == "__main__":
    unittest.main()
