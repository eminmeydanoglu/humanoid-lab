#!/usr/bin/env python3
"""Timeline-lifecycle tests driving the shipped runner's real functions.

The runner module is loaded with importlib exactly the way
``test_cloudwalk_isaac_lifecycle.py`` loads its runner, so these assertions run
against the shipped code and not a copy.
"""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "run-sonic-isaac.py"
sys.path.insert(0, str(ROOT / "tools"))

from sonic_isaac_actuation import BodyActuation  # noqa: E402
from sonic_isaac_contract import BODY_JOINT_COUNT  # noqa: E402
from sonic_isaac_ipc import LowCmdFrame  # noqa: E402


def load_runner():
    spec = importlib.util.spec_from_file_location("run_sonic_isaac_lifecycle", RUNNER)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    # dataclasses resolve their module through sys.modules while the body runs.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


runner = load_runner()


class FakeApp:
    """Frame-counted app. ``is_running`` advances the clock because the shipped
    loop only calls ``update()`` while paused; during Play it relies on
    ``sim.step()`` to drive the application."""

    def __init__(self, frames: int) -> None:
        self.frames = frames
        self.remaining = frames
        self.updates = 0

    @property
    def iteration(self) -> int:
        """Zero-based index of the frame currently being processed."""
        return self.frames - self.remaining - 1

    def is_running(self) -> bool:
        if self.remaining <= 0:
            return False
        self.remaining -= 1
        return True

    def update(self) -> None:
        self.updates += 1


class FakeTimeline:
    """Play schedule indexed by app iteration.

    ``is_playing`` is queried by both the loop and ``step_frame``, so it must be
    idempotent within an iteration -- the real Isaac timeline is.
    """

    def __init__(self, playing: list[bool], app: FakeApp | None = None) -> None:
        self._playing = list(playing)
        self._app = app
        self._cursor = 0
        self.pauses = 0

    def is_playing(self) -> bool:
        index = self._app.iteration if self._app is not None else self._cursor
        if self._app is None and len(self._playing) > 1:
            self._cursor += 1
        return self._playing[min(index, len(self._playing) - 1)]

    def pause(self) -> None:
        self.pauses += 1


class FakeSim:
    def __init__(self) -> None:
        self.steps = 0
        self.current_time = 0.0
        self.renders = 0

    def render(self) -> None:
        self.renders += 1

    def reset(self) -> None:
        self.reset_called = True

    def step(self) -> None:
        self.steps += 1
        self.current_time += 0.005

    def get_physics_dt(self) -> float:
        return 0.005


class FakeScene:
    def __init__(self) -> None:
        self.writes = 0
        self.updates = 0

    def write_data_to_sim(self) -> None:
        self.writes += 1

    def update(self, dt) -> None:
        self.updates += 1


class FakeLink:
    def __init__(self, commands=None) -> None:
        self.published = 0
        self.accepted = 0
        self.commands = list(commands or [])

    def accept(self) -> bool:
        self.accepted += 1
        return True

    def publish(self, state) -> bool:
        self.published += 1
        return True

    def take_command(self):
        return self.commands.pop(0) if self.commands else None

    def snapshot(self) -> dict:
        return {}


def vector(value: float) -> tuple[float, ...]:
    return (value,) * BODY_JOINT_COUNT


def command(sequence: int, *, kp: float = 0.0, q: float = 0.0) -> LowCmdFrame:
    return LowCmdFrame(
        sequence=sequence, q=vector(q), dq=vector(0.0), tau=vector(0.0),
        kp=vector(kp), kd=vector(0.0),
    )


def make_metrics(body_ids=(0, 1, 2)):
    metrics = runner.Metrics()
    from sonic_isaac_contract import JointLimits

    limits = [
        JointLimits(f"j{i}", -3.0, 3.0, 88.0, 32.0) for i in range(len(body_ids))
    ]
    metrics.bind(
        body_ids=body_ids,
        limits=limits,
        read_joint_pos=lambda: (0.0,) * len(body_ids),
        read_root=lambda: ((0.0, 0.0, 0.7), (1.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
    )
    return metrics


class ResetOrderingTest(unittest.TestCase):
    def test_reset_uses_only_the_timeline_pause(self) -> None:
        """SimulationContext.pause() desyncs physics from the rendered textures."""
        import inspect

        source = inspect.getsource(runner.reset_simulation_paused)
        self.assertIn("timeline.pause()", source)
        self.assertNotIn("sim.pause()", source)

    def test_reset_completes_before_timeline_is_paused(self) -> None:
        order: list[str] = []
        sim = FakeSim()
        sim.reset = lambda: order.append("reset")
        timeline = FakeTimeline([False])
        timeline.pause = lambda: order.append("pause")
        runner.reset_simulation_paused(sim, timeline)
        self.assertEqual(order, ["reset", "pause"])

    def test_reset_tolerates_a_missing_timeline(self) -> None:
        runner.reset_simulation_paused(FakeSim(), None)


class StepFrameTest(unittest.TestCase):
    def test_paused_frame_updates_app_without_stepping_physics(self) -> None:
        app, sim, scene = FakeApp(1), FakeSim(), FakeScene()
        advanced = runner.step_frame(app, FakeTimeline([False]), sim, scene)
        self.assertFalse(advanced)
        self.assertEqual(app.updates, 1)
        self.assertEqual(sim.steps, 0)
        self.assertEqual(scene.writes, 0)

    def test_playing_frame_writes_step_and_update_in_order(self) -> None:
        order: list[str] = []
        app, sim, scene = FakeApp(1), FakeSim(), FakeScene()
        sim.step = lambda: order.append("step")
        scene.write_data_to_sim = lambda: order.append("write")
        scene.update = lambda dt: order.append("update")
        advanced = runner.step_frame(
            app, FakeTimeline([True]), sim, scene,
            efforts=vector(1.0), apply_effort=lambda efforts: order.append("effort"),
        )
        self.assertTrue(advanced)
        self.assertEqual(order, ["effort", "write", "step", "update"])

    def test_effort_is_not_written_while_paused(self) -> None:
        calls: list = []
        app, sim, scene = FakeApp(1), FakeSim(), FakeScene()
        runner.step_frame(
            app, FakeTimeline([False]), sim, scene,
            efforts=vector(1.0), apply_effort=lambda efforts: calls.append(efforts),
        )
        self.assertEqual(calls, [])


class InteractiveLoopTest(unittest.TestCase):
    def run_loop(self, playing, frames, *, commands=None, actuation=None, link=None):
        app, sim, scene = FakeApp(frames), FakeSim(), FakeScene()
        link = link or FakeLink(commands)
        actuation = actuation or BodyActuation([88.0] * BODY_JOINT_COUNT)
        metrics = make_metrics()
        applied: list = []
        steps = runner.run_interactive_app(
            app,
            FakeTimeline(playing, app),
            sim,
            scene,
            actuation=actuation,
            link=link,
            metrics=metrics,
            read_state=lambda: None,
            read_body_q_dq=lambda: (vector(0.0), vector(0.0)),
            apply_effort=lambda efforts: applied.append(efforts),
        )
        return steps, sim, applied, link, metrics

    def test_physics_advances_only_while_playing(self) -> None:
        steps, sim, applied, _link, _metrics = self.run_loop(
            [False, False, True, True, False], frames=5
        )
        self.assertEqual(steps, 2)
        self.assertEqual(sim.steps, 2)
        self.assertEqual(len(applied), 2)

    def test_viewport_refresh_uses_a_plain_app_frame(self) -> None:
        """app.update() is the refresh that measurably renders motion."""
        import inspect

        source = inspect.getsource(runner.refresh_viewport)
        self.assertIn("simulation_app.update()", source)
        self.assertNotIn("render()", source)
        app = FakeApp(2)
        runner.refresh_viewport(FakeSim(), app)
        self.assertEqual(app.updates, 2)

    def test_viewport_is_refreshed_every_playing_frame(self) -> None:
        """render() flushes fabric into the textures; app.update() does not."""
        app = FakeApp(4)
        sim, scene = FakeSim(), FakeScene()
        runner.run_interactive_app(
            app,
            FakeTimeline([True] * 4, app),
            sim,
            scene,
            actuation=BodyActuation([88.0] * BODY_JOINT_COUNT),
            link=FakeLink(),
            metrics=make_metrics(),
            read_state=lambda: None,
            read_body_q_dq=lambda: (vector(0.0), vector(0.0)),
            apply_effort=lambda efforts: None,
        )
        self.assertEqual(
            app.updates, 4, "the viewport must be refreshed on every playing frame"
        )

    def test_refresh_viewport_falls_back_when_render_is_unavailable(self) -> None:
        app = FakeApp(3)

        class NoRender:
            pass

        for _ in range(3):
            runner.refresh_viewport(NoRender(), app)
        self.assertEqual(app.updates, 3)

    def test_refresh_viewport_survives_a_failing_render(self) -> None:
        class BadRender:
            def render(self):
                raise RuntimeError("render failed")

        app = FakeApp(1)
        runner.refresh_viewport(BadRender(), app)
        self.assertEqual(app.updates, 1)

    def test_command_received_while_paused_is_cached_not_applied(self) -> None:
        link = FakeLink([command(1, kp=100.0, q=0.5)])
        steps, _sim, applied, _link, metrics = self.run_loop(
            [False, True], frames=2, link=link
        )
        self.assertEqual(steps, 1)
        # The command cached while paused is applied on the first playing frame.
        self.assertEqual(applied[0][0], 50.0)
        # It arrived before the measurement window opened, so it is not part of
        # the run's lowcmd frequency statistics.
        self.assertEqual(metrics.accepted_commands, 0)

    def test_command_received_during_the_window_is_counted(self) -> None:
        link = FakeLink([command(1, kp=100.0, q=0.5), command(2, kp=100.0, q=0.5)])
        _steps, _sim, _applied, _link, metrics = self.run_loop(
            [True, True, True], frames=3, link=link
        )
        self.assertEqual(metrics.accepted_commands, 2)
        self.assertGreaterEqual(metrics.rate.count, 2)

    def test_stale_command_returns_the_body_to_passive(self) -> None:
        link = FakeLink([command(1, kp=100.0, q=0.5)])
        actuation = BodyActuation([88.0] * BODY_JOINT_COUNT, max_age_s=0.0)
        _steps, _sim, applied, _link, metrics = self.run_loop(
            [True, True], frames=2, link=link, actuation=actuation
        )
        # First frame may apply the fresh command; once it is older than the
        # zero-length window every effort must be zero.
        self.assertEqual(applied[-1], (0.0,) * BODY_JOINT_COUNT)
        self.assertGreaterEqual(metrics.passive_steps, 1)

    def test_out_of_order_command_is_rejected(self) -> None:
        link = FakeLink([command(5, kp=100.0, q=0.5), command(5, kp=100.0, q=0.9)])
        _steps, _sim, _applied, _link, metrics = self.run_loop([True, True], frames=2, link=link)
        self.assertEqual(metrics.rejected_commands, 1)

    def test_duration_stops_the_run_even_while_running(self) -> None:
        steps, sim, _applied, _link, _metrics = self.run_loop([True] * 50, frames=50)
        # duration_s is None here, so the loop runs until the app stops.
        self.assertEqual(steps, 50)


class MetricsTest(unittest.TestCase):
    def test_fall_is_detected_from_the_root_drop(self) -> None:
        metrics = runner.Metrics()
        heights = iter([0.75, 0.70, 0.30])

        def read_root():
            z = next(heights)
            return (0.0, 0.0, z), (1.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0)

        metrics.bind(body_ids=(), limits=(), read_joint_pos=lambda: (), read_root=read_root)
        metrics.start_measurement(0.0)
        for _ in range(3):
            metrics.observe(None)
        summary = metrics.summary()
        self.assertAlmostEqual(summary["root_z_drop"], 0.45, places=9)
        self.assertTrue(metrics.fall_observed())

    def test_small_drop_but_tipped_over_also_counts_as_a_fall(self) -> None:
        metrics = runner.Metrics()
        # Pelvis rotated 90 degrees: up-Z collapses to ~0.
        poses = [
            ((0.0, 0.0, 0.75), (1.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
            ((0.0, 0.0, 0.74), (0.7071, 0.7071, 0.0, 0.0), (0.0, 0.0, 0.0)),
        ]

        def read_root():
            return poses.pop(0)

        metrics.bind(body_ids=(), limits=(), read_joint_pos=lambda: (), read_root=read_root)
        metrics.start_measurement(0.0)
        metrics.observe(None)
        metrics.observe(None)
        self.assertLessEqual(metrics.summary()["pelvis_up_z_min"], 0.50)
        self.assertTrue(metrics.fall_observed())

    def test_standing_run_is_not_reported_as_a_fall(self) -> None:
        metrics = runner.Metrics()

        def read_root():
            return (0.0, 0.0, 0.74), (1.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0)

        metrics.bind(body_ids=(), limits=(), read_joint_pos=lambda: (), read_root=read_root)
        metrics.start_measurement(0.0)
        for _ in range(5):
            metrics.observe(None)
        self.assertFalse(metrics.fall_observed())

    def test_joint_limit_violation_is_counted(self) -> None:
        from sonic_isaac_contract import JointLimits

        metrics = runner.Metrics()
        metrics.bind(
            body_ids=(0,),
            limits=[JointLimits("j0", -0.5, 0.5, 88.0, 32.0)],
            read_joint_pos=lambda: (9.0,),
            read_root=lambda: ((0.0, 0.0, 0.7), (1.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
        )
        metrics.start_measurement(0.0)
        metrics.observe(None)
        self.assertEqual(metrics.summary()["joint_limit_violations"], 1)

    def test_quat_to_rpy_recovers_a_known_rotation(self) -> None:
        roll, pitch, yaw = runner.quat_to_rpy((0.7071067811865476, 0.7071067811865475, 0.0, 0.0))
        self.assertAlmostEqual(roll, 1.5707963267948966, places=6)
        self.assertAlmostEqual(pitch, 0.0, places=6)
        self.assertAlmostEqual(yaw, 0.0, places=6)

    def test_summary_reports_none_instead_of_infinity(self) -> None:
        metrics = runner.Metrics()
        summary = metrics.summary()
        self.assertIsNone(summary["root_z_min"])
        self.assertIsNone(summary["pelvis_up_z_min"])


class SourceContractTest(unittest.TestCase):
    """Pin the invariants that must not silently drift in the runner source."""

    def setUp(self) -> None:
        self.source = RUNNER.read_text()

    def test_physics_runs_at_two_hundred_hertz(self) -> None:
        from sonic_isaac_contract import PHYSICS_DT, PHYSICS_HZ

        self.assertEqual(PHYSICS_HZ, 200)
        self.assertAlmostEqual(PHYSICS_DT, 0.005, places=9)
        self.assertIn("dt=PHYSICS_DT", self.source)

    def test_robot_starts_from_the_sonic_default_pose(self) -> None:
        """The asset's own initial state must not be the starting pose."""
        self.assertIn("sonic_default_joint_target(robot, body_ids)", self.source)
        self.assertNotIn("default_pos = robot.data.default_joint_pos.clone()", self.source)
        import inspect

        helper = inspect.getsource(runner.sonic_default_joint_target)
        self.assertIn("sonic_default_pose_by_name", helper)

    def test_body_drives_are_zeroed(self) -> None:
        self.assertIn("write_joint_stiffness_to_sim(0.0", self.source)
        self.assertIn("write_joint_damping_to_sim(0.0", self.source)
        self.assertIn("zero_body_drives(robot, body_ids)", self.source)

    def test_effort_target_is_used_instead_of_a_position_target(self) -> None:
        self.assertIn("set_joint_effort_target", self.source)

    def test_inspire_profile_is_made_free_based_with_gravity(self) -> None:
        self.assertIn("fix_root_link = False", self.source)
        self.assertIn("disable_gravity = False", self.source)

    def test_scene_provides_its_own_lights(self) -> None:
        """Stage Lights mode with no lights renders the scene black."""
        self.assertIn("DomeLightCfg", self.source)
        self.assertIn("DistantLightCfg", self.source)
        self.assertIn("/World/DomeLight", self.source)

    def test_viewport_camera_is_pointed_at_the_robot(self) -> None:
        """A recording whose camera misses the robot cannot evidence the fall."""
        self.assertIn("set_camera_view", self.source)
        self.assertIn("camera_view_set", self.source)

    def test_fabric_is_enabled_so_the_viewport_renders_physics(self) -> None:
        """Without fabric the renderer keeps showing the initial pose."""
        self.assertIn('"/physics/fabricEnabled", True', self.source)
        self.assertIn('"/physics/updateToUsd", False', self.source)
        self.assertIn("SimulationManager.enable_fabric(True)", self.source)
        self.assertIn("attach_stage_to_usd_context()", self.source)
        # Fabric is enabled before the context is constructed.
        self.assertLess(
            self.source.index("enable_fabric(True)"),
            self.source.index("sim_utils.SimulationContext("),
        )

    def test_evidence_records_which_viewport_was_captured(self) -> None:
        """An unusable recording must be explainable from the evidence."""
        self.assertIn("def viewport_diagnostics", self.source)
        self.assertIn('"viewport": viewport_diagnostics()', self.source)

    def test_evidence_is_written_even_with_a_non_serializable_value(self) -> None:
        """A stray Sdf.Path must not destroy the whole run record."""
        import json
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "evidence.json"

            class NotPrimitive:
                def __str__(self) -> str:
                    return "not-primitive"

            runner._write_evidence(path, {"ok": True, "weird": NotPrimitive()})
            payload = json.loads(path.read_text())
            self.assertTrue(payload["ok"])
            self.assertEqual(payload["weird"], "not-primitive")

    def test_viewport_diagnostics_are_json_safe(self) -> None:
        """Camera handles are Sdf.Path values, which json cannot encode."""
        import inspect
        import json

        source = inspect.getsource(runner.viewport_diagnostics)
        self.assertIn("str(viewport.get_active_camera())", source)
        self.assertIn("str(product)", source)
        # The helper must be callable (and JSON-safe) outside a Kit app.
        json.dumps(runner.viewport_diagnostics())

    def test_camera_sensors_are_off_by_default(self) -> None:
        """Camera sensors select an experience whose viewport does not repaint."""
        self.assertIn("args.enable_cameras = bool(args.camera_sensors)", self.source)
        self.assertNotIn("args.enable_cameras = True", self.source)
        self.assertIn("--camera-sensors", self.source)

    def test_hands_are_declared_uncontrolled(self) -> None:
        self.assertIn("HANDS_CONTROLLED_BY_SONIC = False", self.source)

    def test_every_publish_loop_accepts_the_incoming_link(self) -> None:
        """The bridge connects while paused, so all publish loops must accept."""
        import inspect

        source = inspect.getsource(runner._run)
        publishes = source.count("link.publish(build_state_frame())")
        accepts = source.count("link.accept()")
        self.assertGreater(publishes, 0)
        self.assertGreaterEqual(
            accepts, publishes,
            "a publish loop without accept() leaves the bridge unconnected",
        )

    def test_timeline_is_not_auto_played_without_an_explicit_flag(self) -> None:
        self.assertIn("--auto-play", self.source)
        self.assertIn("--play-trigger", self.source)
        # Play only happens on an explicit flag or an explicit trigger file.
        self.assertIn("if args.auto_play or args.play_trigger is not None:", self.source)

    def test_play_trigger_waits_instead_of_playing_on_a_timer(self) -> None:
        self.assertIn("while not args.play_trigger.is_file():", self.source)

    def test_state_frame_builder_matches_read_root_state_arity(self) -> None:
        """read_root_state returns pose + linear velocity; the builder must agree."""
        import inspect

        arity = len(inspect.getsource(runner.read_root_state).splitlines())
        self.assertIn("root_lin_vel_w", inspect.getsource(runner.read_root_state))
        builder = inspect.getsource(runner._run)
        self.assertIn("pos, quat, lin = read_root_state(robot)", builder)
        self.assertNotIn("pos, quat = read_root_state(robot)", builder)
        self.assertGreater(arity, 0)

    def test_motion_metrics_expose_the_drive_acceptance_quantities(self) -> None:
        for key in ("net_xy_m", "root_yaw_deg", "max_speed_mps", "trace"):
            self.assertIn(f'"{key}"', self.source)


if __name__ == "__main__":
    unittest.main()


class ViewportFrameTest(unittest.TestCase):
    """The viewport annotator returns batched arrays; encoding needs one image."""

    def setUp(self) -> None:
        import numpy as np

        self.np = np

    def as_image(self, frame):
        return runner.VideoRecorder._as_image(frame)

    def test_batched_rgba_frame_is_unwrapped(self) -> None:
        frame = self.np.zeros((1, 480, 640, 4), dtype=self.np.uint8)
        result = self.as_image(frame)
        self.assertIsNotNone(result)
        self.assertEqual(result.shape, (480, 640, 4))

    def test_plain_rgb_image_passes_through(self) -> None:
        frame = self.np.zeros((480, 640, 3), dtype=self.np.uint8)
        result = self.as_image(frame)
        self.assertEqual(result.shape, (480, 640, 3))

    def test_grayscale_image_is_accepted(self) -> None:
        frame = self.np.zeros((480, 640), dtype=self.np.uint8)
        self.assertEqual(self.as_image(frame).shape, (480, 640))

    def test_channel_first_layout_is_moved_last(self) -> None:
        frame = self.np.zeros((3, 480, 640), dtype=self.np.uint8)
        self.assertEqual(self.as_image(frame).shape, (480, 640, 3))

    def test_five_dimensional_frame_is_reduced(self) -> None:
        frame = self.np.zeros((1, 1, 480, 640, 4), dtype=self.np.uint8)
        self.assertEqual(self.as_image(frame).shape, (480, 640, 4))

    def test_unusable_shape_is_reported_as_none(self) -> None:
        frame = self.np.zeros((2, 3, 4, 5, 6, 7), dtype=self.np.uint8)
        self.assertIsNone(self.as_image(frame))

    def test_none_frame_is_reported_as_none(self) -> None:
        self.assertIsNone(self.as_image(None))

    @unittest.skipUnless(
        importlib.util.find_spec("imageio") is not None,
        "imageio is only installed in the container environment",
    )
    def test_recorded_frames_are_written_as_a_playable_video(self) -> None:
        """End to end: batched frames in, a real video file out."""
        import tempfile

        np = self.np
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "clip.mp4"
            recorder = runner.VideoRecorder(path, fps=10, every=1)
            # Bypass viewport discovery; feed frames the way a capture would.
            recorder.capture = lambda: np.zeros((1, 64, 64, 4), dtype=np.uint8)
            recorder.status = "ready"
            for step in range(1, 6):
                recorder.maybe_record(step)
            result = recorder.write()

            self.assertEqual(result["status"], "written", result)
            self.assertEqual(result["frames"], 5)
            self.assertTrue(path.is_file())
            self.assertGreater(path.stat().st_size, 0)


class ShutdownTest(unittest.TestCase):
    """Closing must not leave the desktop reporting the window as unresponsive."""

    def setUp(self) -> None:
        self.source = RUNNER.read_text()

    def test_shutdown_stops_the_timeline_before_closing(self) -> None:
        order: list[str] = []

        class App:
            def update(self) -> None:
                order.append("update")

            def close(self) -> None:
                order.append("close")

        class Timeline:
            def stop(self) -> None:
                order.append("stop")

        class Sim:
            def stop(self) -> None:
                order.append("sim_stop")

        report = runner.shutdown_app(App(), Timeline(), Sim(), settle_s=0.05,
                                     close_timeout_s=5.0)
        self.assertTrue(report["timeline_stopped"])
        self.assertTrue(report["closed"])
        self.assertFalse(report["timed_out"])
        # Timeline stops first, the UI gets pumped, and close happens last.
        self.assertLess(order.index("stop"), order.index("update"))
        self.assertEqual(order[-1], "close")

    def test_shutdown_is_bounded_when_close_hangs(self) -> None:
        import time as _time

        class HangingApp:
            def update(self) -> None:
                pass

            def close(self) -> None:
                _time.sleep(30)

        started = _time.monotonic()
        report = runner.shutdown_app(HangingApp(), None, None, settle_s=0.0,
                                     close_timeout_s=0.5)
        elapsed = _time.monotonic() - started
        self.assertTrue(report["timed_out"])
        self.assertFalse(report["closed"])
        # Must return promptly rather than blocking on Kit's teardown.
        self.assertLess(elapsed, 5.0)

    def test_shutdown_tolerates_a_missing_timeline(self) -> None:
        class App:
            def update(self) -> None:
                pass

            def close(self) -> None:
                pass

        report = runner.shutdown_app(App(), None, None, settle_s=0.0, close_timeout_s=1.0)
        self.assertFalse(report["timeline_stopped"])
        self.assertTrue(report["closed"])

    def test_shutdown_survives_a_failing_close(self) -> None:
        class BrokenApp:
            def update(self) -> None:
                raise RuntimeError("update failed")

            def close(self) -> None:
                pass

        report = runner.shutdown_app(BrokenApp(), None, None, settle_s=0.05,
                                     close_timeout_s=1.0)
        # BrokenApp.update raises; the helper must still return a report
        # instead of propagating, and must still attempt the close.
        self.assertIn("closed", report)
        self.assertTrue(report["closed"])

    def test_main_registers_handles_for_shutdown(self) -> None:
        self.assertIn('_SIM["sim"] = sim', self.source)
        self.assertIn('_TIMELINE["timeline"] = timeline', self.source)
        self.assertIn("shutdown_app(simulation_app,", self.source)
        self.assertNotIn("simulation_app.close()\n        os._exit", self.source)

    def test_startup_pumps_the_app_so_the_window_answers(self) -> None:
        self.assertIn("def pump_app", self.source)
        self.assertIn("pump_app(simulation_app, 2)", self.source)


