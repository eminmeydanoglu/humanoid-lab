from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from humanoid_lab.simulators.isaac.contracts import ContractError, RunProfile  # noqa: E402

SERVICE = ROOT / "src/humanoid_lab/simulators/isaac/service.py"


class IsaacG1ProfileTests(unittest.TestCase):
    def test_all_shipped_profiles_are_valid_and_distinct(self) -> None:
        base_profiles = [
            "isaac-g1-no_hands.json",
            "isaac-g1-inspire-ftp.json",
            "isaac-g1-dex3.json",
        ]
        profiles = [
            RunProfile.load(ROOT / "configs/profiles" / name) for name in base_profiles
        ]
        self.assertEqual({profile.robot.hand.kind for profile in profiles}, {"no_hands", "inspire-ftp", "dex3"})
        self.assertTrue(all(profile.robot.body_dofs == 29 for profile in profiles))
        self.assertTrue(all(profile.camera.width == 640 and profile.camera.height == 480 for profile in profiles))
        self.assertEqual({profile.robot.hand.dofs for profile in profiles}, {0, 14, 24})
        # The base platform profiles declare no controller: that is what keeps
        # Gate 1 behaviour reproducible while control paths are added.
        self.assertTrue(all(profile.controller is None for profile in profiles))

    def test_rejects_non_29dof_body(self) -> None:
        source = json.loads((ROOT / "configs/profiles/isaac-g1-no_hands.json").read_text())
        source["robot"]["body_dofs"] = 27
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            path.write_text(json.dumps(source))
            with self.assertRaisesRegex(ContractError, "29DoF"):
                RunProfile.load(path)

    def test_rejects_non_normalized_camera_rotation(self) -> None:
        source = json.loads((ROOT / "configs/profiles/isaac-g1-no_hands.json").read_text())
        source["camera"]["rotation_wxyz"] = [1.0, 1.0, 0.0, 0.0]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            path.write_text(json.dumps(source))
            with self.assertRaisesRegex(ContractError, "normalized"):
                RunProfile.load(path)


class IsaacG1SimulationConfigTests(unittest.TestCase):
    def test_shipped_profiles_resolve_to_cpu_and_smooth_render_cadence(self) -> None:
        for path in sorted((ROOT / "configs/profiles").glob("isaac-g1-*.json")):
            profile = RunProfile.load(path)
            self.assertEqual(profile.device, "cpu")
            self.assertEqual(profile.render_interval, 8)
            self.assertAlmostEqual(profile.camera_update_period, 0.04)

    def test_device_override_only_applies_when_requested(self) -> None:
        profile = RunProfile.load(ROOT / "configs/profiles/isaac-g1-dex3.json")
        self.assertEqual(profile.with_device(None).device, "cpu")
        self.assertEqual(profile.with_device("cuda").device, "cuda")


class IsaacG1SourceInvariantTests(unittest.TestCase):
    def test_normal_runs_have_no_implicit_five_minute_timeout(self) -> None:
        cli = (ROOT / "src/humanoid_lab/simulators/isaac/cli.py").read_text()
        self.assertNotIn("else 300.0", cli)
        self.assertIn("if args.duration is None and args.test:", cli)
        service = SERVICE.read_text()
        self.assertIn("self.duration is None", service)

    def test_runner_has_no_controller_or_policy_imports(self) -> None:
        source = (ROOT / "src/humanoid_lab/simulators/isaac/service.py").read_text()
        for forbidden in ("sonic_isaac", "cloudwalk", "groot", "StateLink", "LowCmd"):
            self.assertNotIn(forbidden, source)

    def test_loop_decouples_physics_from_render(self) -> None:
        source = SERVICE.read_text()
        # Two steppers exist: the ordinary physics step and the frame-exact
        # kinematic replay.  Both must advance physics with render=False and
        # render separately, so the invariant is checked per stepper instead of
        # counting occurrences across the whole module (a shared count silently
        # breaks as soon as a second, correctly-written stepper is added).
        physics_body = source[source.index("    def _step_physics") : source.index("    def _step_kinematic")]
        kinematic_body = source[source.index("    def _step_kinematic") : source.index("    def _render_paused")]
        step_body = source[source.index("    def _step_physics") : source.index("    def _render_paused")]
        paused_body = source[
            source.index("    def _render_paused") : source.index("    def run(self)")
        ]
        run_body = source[source.index("    def run(self)") : source.index("    def _accounting")]
        for name, body in (("physics", physics_body), ("kinematic", kinematic_body)):
            self.assertEqual(
                body.count("self._sim.step(render=False)"),
                1,
                f"{name} stepper must advance physics exactly once with render=False",
            )
        self.assertNotIn("self._sim.step()", source)
        # The kinematic stepper must not integrate physics after authoring the
        # frame, and must refresh kinematics without stepping time.
        self.assertIn("self._sim.forward()", kinematic_body)
        self.assertNotIn("self._sim.render()", kinematic_body.split("self._sim.forward()")[0])
        self.assertEqual(physics_body.count("self._sim.render()"), 1)
        self.assertEqual(kinematic_body.count("self._sim.render()"), 1)
        # Render cadence comes from the profile outside a kinematic replay; the
        # replay overrides it with its own ticks-per-frame.
        self.assertIn("if self._kinematic is not None else self.profile.render_interval", physics_body)
        self.assertIn("rendered = self._is_rendering and self.tick % render_interval == 0", physics_body)
        self.assertIn("if rendered:", physics_body)
        self.assertIn("self._consume_head_camera_frame()", step_body)
        self.assertEqual(run_body.count("self._step_physics()"), 2)
        self.assertNotIn("self._sim.render()", run_body)
        self.assertIn("self._render_paused()", run_body)
        self.assertIn("1.0 / self.PAUSED_RENDER_HZ", paused_body)
        self.assertIn("time.monotonic()", paused_body)
        self.assertIn("time.sleep(delay)", paused_body)
        self.assertIn("if self._pending_ui_reset:", run_body)
        self.assertNotIn("app.update", run_body)
        self.assertIn("self._transition(TimelineState.STOPPED)", run_body)

    def test_gui_exposes_reset_and_head_camera(self) -> None:
        source = (ROOT / "src/humanoid_lab/simulators/isaac/service.py").read_text()
        self.assertIn('ui.Button("Reset Robot"', source)
        self.assertIn('"G1 Head Camera",', source)
        self.assertIn("create_viewport_window", source)
        self.assertIn("camera_path=camera_path", source)
        self.assertIn("if self.show_head_camera:", source)
        self.assertIn('if self.test_mode is not None or self.show_head_camera', source)
        self.assertNotIn("ui.ByteImageProvider()", source)
        self.assertNotIn("set_data_array", source)
        self.assertIn("clicked_fn=self._request_reset_from_ui", source)
        request_body = source[
            source.index("    def _request_reset_from_ui") : source.index("    def _open_ui")
        ]
        self.assertIn("self._pending_ui_reset = True", request_body)
        self.assertNotIn("self._timeline.pause()", request_body)
        self.assertNotIn("reset_episode", request_body)
        self.assertIn("self.reset_episode(resume=True)", source)
    def test_acceptance_modes_are_explicit_and_measured(self) -> None:
        cli = (ROOT / "src/humanoid_lab/simulators/isaac/cli.py").read_text()
        service = (ROOT / "src/humanoid_lab/simulators/isaac/service.py").read_text()
        self.assertIn(
            'parser.add_argument("--test", choices=("passive-fall", "controlled-hold", "controller-hold"))',
            cli,
        )
        self.assertIn('if self.test_mode == "passive-fall"', service)
        self.assertIn('if self.test_mode == "controlled-hold"', service)
        self.assertIn('if self.test_mode == "controller-hold"', service)
        self.assertIn('"result": "COMPLETED"', service)
        self.assertIn('"head_camera_shape": self._camera_shape', service)
        self.assertIn('"head_camera_flowing": self._camera_frames >= 2', service)
        self.assertNotIn('"clean_stop": True', service)

    def test_controller_override_can_only_disable_a_profile_controller(self) -> None:
        """Provider configuration belongs to the profile; a CLI override must
        not accidentally open an unconfigured DDS domain or partial controller."""
        cli = (ROOT / "src/humanoid_lab/simulators/isaac/cli.py").read_text()
        self.assertIn('choices=("none",)', cli)
        self.assertIn("safe passive fallback", cli)

    def test_commands_reach_the_solver_by_one_path(self) -> None:
        """The controller's torque is the torque; the asset's actuator models
        compute their own effort and would silently discard it."""
        source = SERVICE.read_text()
        self.assertEqual(source.count("set_dof_actuation_forces"), 1)
        self.assertIn("self._flush_effort()", source)
        # The staged effort is the only thing written per tick; the asset's own
        # effort path is not used for control.
        write_body = source[
            source.index("    def _write_effort") : source.index("    def _flush_effort")
        ]
        self.assertNotIn("set_joint_effort_target", write_body)
        self.assertIn("self._effort_target[0, list(indices)] = tensor", write_body)

    def test_passive_behaviour_is_untouched_without_a_controller(self) -> None:
        source = SERVICE.read_text()
        step_body = source[source.index("    def _step_physics") : source.index("    def _render_paused")]
        self.assertIn("self._apply_control()", step_body)
        control_body = source[
            source.index("    def _apply_control") : source.index("    def _apply_body_command")
        ]
        # No controller configured means the loop never even polls one.
        self.assertIn("if self._controller is None:\n            return", control_body)

    def test_command_timeout_returns_the_robot_to_passive(self) -> None:
        source = SERVICE.read_text()
        control_body = source[
            source.index("    def _apply_control") : source.index("    def _apply_body_command")
        ]
        self.assertIn("command.is_valid_at(self.tick)", control_body)
        self.assertIn("self._go_passive()", control_body)
        self.assertIn("self._control_mode = PASSIVE", control_body)

    def test_a_bad_command_cannot_kill_the_loop(self) -> None:
        """Undecodable, non-finite or mis-ordered output must degrade to passive,
        not raise out of the physics loop."""
        source = SERVICE.read_text()
        control_body = source[
            source.index("    def _apply_control") : source.index("    def _apply_body_command")
        ]
        self.assertIn("except CommandError as error:", control_body)
        self.assertIn("self._rejected_commands += 1", control_body)
        self.assertIn("self._go_passive()", control_body)
        status_body = source[
            source.index("    def _controller_status") : source.index("    def _controller_hold_acceptance")
        ]
        self.assertIn('"rejected_commands": self._rejected_commands', status_body)

    def test_reset_is_accepted_while_a_controller_is_driving_and_keeps_episodes_apart(self) -> None:
        """The client stops its stream and then resets, so the request lands
        inside the command TTL window while the controller is still the last
        writer.  Refusing it there lost the reset; the request is queued, and
        the episode bump is what keeps the two episodes apart."""
        source = SERVICE.read_text()
        body = source[
            source.index("    def _request_reset_from_ui") : source.index("    def _open_ui")
        ]
        self.assertNotIn("isaac_g1_ui_reset_refused", body)
        self.assertNotIn("return False", body)
        self.assertIn("self._pending_ui_reset = True", body)
        self.assertIn("return True", body)
        # The safety boundary is the episode id, not a refusal.
        reset_body = source[
            source.index("    def reset_episode") : source.index("    def _write_initial_state_to_sim")
        ]
        self.assertIn("self.episode_id += 1", reset_body)
        # Stale commands are dropped by the episode check, and the robot is
        # passive until the controller sends new ones under the new id.
        self.assertIn("self._control_mode = PASSIVE", reset_body)
        self.assertLess(
            reset_body.index("self.episode_id += 1"),
            reset_body.index("self._control_mode = PASSIVE"),
        )
        # The declared start-up support goes back up so a reset robot that no
        # controller has claimed yet is held instead of falling.
        self.assertIn("self._start_support()", reset_body)
        control_body = source[
            source.index("    def _apply_control") : source.index("    def _apply_body_command")
        ]
        self.assertIn("command.episode_id == self.episode_id", control_body)

    def test_a_reset_keeps_physics_running(self) -> None:
        source = SERVICE.read_text()
        run_body = source[source.index("    def run(self)") : source.index("    def _accounting")]
        reset_branch = run_body[
            run_body.index("                if self._pending_ui_reset:") :
            run_body.index("                if self._timeline.is_playing():")
        ]
        self.assertIn("self.reset_episode(resume=True)", reset_branch)
        self.assertLess(
            reset_branch.index("self.reset_episode(resume=True)"),
            reset_branch.index("self._step_physics()"),
        )
        # A paused timeline never advances again by itself and nothing in the
        # service restarts it, so the reset must not pause.
        self.assertNotIn("self.pause()", reset_branch)
        self.assertIn('"event": "isaac_g1_reset"', reset_branch)
        self.assertIn('"timeline": self.state.value', reset_branch)

    def test_a_new_episode_invalidates_old_commands(self) -> None:
        source = (ROOT / "src/humanoid_lab/controllers/sonic_dds.py").read_text()
        publish_body = source[
            source.index("    def publish_state") : source.index("    def _publish_loop")
        ]
        self.assertIn("if state.episode_id != self._episode_id:", publish_body)
        for cleared in ("self._low_command = None", "self._left_hand_command = None"):
            self.assertIn(cleared, publish_body)

    def test_fabric_is_conditional_on_the_physics_device(self) -> None:
        source = SERVICE.read_text()
        self.assertIn('use_fabric = self._device.startswith("cuda")', source)
        self.assertIn('settings.set_bool("/physics/fabricEnabled", use_fabric)', source)
        self.assertIn('settings.set_bool("/physics/updateToUsd", not use_fabric)', source)
        self.assertIn("SimulationManager.enable_fabric(use_fabric)", source)
        self.assertIn(
            'settings.set_int("/persistent/physics/numThreads", self.PHYSX_NUM_THREADS)',
            source,
        )
        self.assertIn("device=self._device", source)
        self.assertIn("render_interval=1", source)
        self.assertIn('rendering_mode="balanced"', source)
        self.assertIn("enable_dlssg=self.show_ui", source)
        self.assertIn("enable_dl_denoiser=True", source)
        self.assertIn("update_period=self.profile.camera_update_period", source)

class IsaacG1BlockStackingSceneTests(unittest.TestCase):
    """The optional scene schema plus the head-camera service wiring."""

    SONIC = ROOT / "configs/profiles/isaac-g1-sonic-dex3.json"
    BLOCKS = ROOT / "configs/profiles/isaac-g1-sonic-blockstacking-dex3.json"

    def test_sonic_dex3_controller_and_camera_are_unchanged(self) -> None:
        profile = RunProfile.load(self.SONIC)
        self.assertEqual((profile.camera.width, profile.camera.height), (640, 480))
        assert profile.controller is not None
        self.assertEqual(profile.controller["provider"], "sonic_dds")
        self.assertEqual(profile.controller["mass_alignment"], "sonic_mujoco")
        self.assertEqual(profile.controller["joint_dynamics_alignment"], "sonic_mujoco")
        # The scene schema is optional: an existing profile declares neither a
        # scene nor a camera service, so nothing about it changed.
        self.assertIsNone(profile.scene)
        self.assertFalse(profile.camera_service_enabled)

    def test_block_stacking_scene_declares_table_cubes_and_black_target(self) -> None:
        profile = RunProfile.load(self.BLOCKS)
        self.assertEqual((profile.camera.width, profile.camera.height), (640, 480))
        # The scene schema adds no controller of its own: the profile keeps the
        # existing SONIC body controller and start-up support, unchanged.
        assert profile.controller is not None
        self.assertEqual(profile.controller["provider"], "sonic_dds")
        self.assertEqual(profile.controller["mass_alignment"], "sonic_mujoco")
        self.assertEqual(profile.controller["joint_dynamics_alignment"], "sonic_mujoco")
        self.assertIsNotNone(profile.support)
        scene = profile.scene
        assert scene is not None
        self.assertTrue(scene.camera_enabled)
        self.assertTrue(profile.camera_service_enabled)

        table = scene.table
        self.assertTrue(table.asset_reference.startswith("/data/models/unitree-sim-assets/"))
        # Variant 2: the PackingTable and PackingTable_1 copies carry a
        # container_h20 whose top reaches 1.0829 m, above the 0.9941 m worktop,
        # which is the space the cubes and the target need (measured with
        # usd-core BBoxCache on the shipped USDs).
        self.assertTrue(table.asset_reference.endswith("/PackingTable_2/PackingTable.usd"))
        # The one measured number: the lowest Dex3 mesh point (palm bottom)
        # with both elbows at a measured 90.00 deg bend, minus the 0.05 m cube
        # (see the profile's surface_height_provenance and the elbow-flexion
        # probe record under outputs/isaac-blockstacking-calibration/elbow90).
        self.assertAlmostEqual(table.surface_height_m, 0.8357609, places=4)
        self.assertTrue(table.surface_height_provenance)
        # table_surface + cube_size = lowest hand/finger world-z, the relation
        # the height exists for: the cube top lands on the measured hand bottom.
        self.assertAlmostEqual(
            table.surface_height_m + scene.cubes[0].size_m[2], 0.8857609, places=4
        )
        # The table is placed so the measured 0.994051 m worktop offset lands on
        # the declared surface height.
        self.assertAlmostEqual(
            table.position_m[2] + 0.9940513, table.surface_height_m, places=5
        )

        self.assertEqual(len(scene.cubes), 3)
        self.assertEqual([cube.color for cube in scene.cubes], ["red", "yellow", "blue"])
        for cube in scene.cubes:
            self.assertGreater(cube.mass_kg, 0.0)
            self.assertAlmostEqual(
                cube.position_m[2], table.surface_height_m + cube.size_m[2] / 2.0, places=4
            )

        # The canonical row: one x line, equal 0.15 m spacing along y, centred
        # on the tape that sits beyond the middle cube.  On 2026-09-19 the row
        # and the tape were shifted together +0.05 m in y, because the head
        # camera cropped the red cube at the frame's right edge; moving them as
        # one piece keeps the row, the spacing and the row-to-tape alignment
        # canonical and only changes where the arrangement sits in front of the
        # robot.
        self.assertEqual({cube.position_m[0] for cube in scene.cubes}, {0.45})
        ys = [cube.position_m[1] for cube in scene.cubes]
        self.assertAlmostEqual(ys[1] - ys[0], 0.15, places=6)
        self.assertAlmostEqual(ys[2] - ys[1], 0.15, places=6)
        self.assertAlmostEqual((ys[0] + ys[2]) / 2.0, ys[1], places=6)
        self.assertAlmostEqual(scene.target.position_m[1], ys[1], places=6)
        self.assertGreater(scene.target.position_m[0], max(cube.position_m[0] for cube in scene.cubes))

        self.assertEqual(scene.target.kind, "tape")
        self.assertEqual(scene.target.color, "black")
        self.assertAlmostEqual(
            scene.target.position_m[2], table.surface_height_m + scene.target.size_m[2] / 2.0, places=4
        )

    def test_rejects_a_scene_with_the_wrong_cube_colors(self) -> None:
        source = json.loads(self.BLOCKS.read_text())
        source["scene"]["cubes"][0]["color"] = "green"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            path.write_text(json.dumps(source))
            with self.assertRaisesRegex(ContractError, "cube.color"):
                RunProfile.load(path)

    def test_rejects_a_cube_that_does_not_rest_on_the_declared_surface(self) -> None:
        source = json.loads(self.BLOCKS.read_text())
        source["scene"]["cubes"][0]["position_m"][2] = 0.9
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            path.write_text(json.dumps(source))
            with self.assertRaisesRegex(ContractError, "declared table surface"):
                RunProfile.load(path)

    def test_camera_service_is_independent_of_test_window_and_video(self) -> None:
        source = SERVICE.read_text()
        # The camera scene is created whenever the service is enabled, not only
        # for a test, the head-camera window, or video recording.
        self.assertIn("or self.camera_service", source)
        self.assertIn("def _start_camera_service", source)
        self.assertIn("self._publish_camera_frame()", source)
        self.assertIn("endpoint.publish_frame(image", source)
        # A reset drops the retained frame so a stale pose cannot be served.
        self.assertIn("self._head_camera.invalidate(episode_id=self.episode_id)", source)
        # The reset request is queued by the control thread and consumed at the
        # simulation-loop boundary, never applied from the service thread.
        self.assertIn("def _request_reset_from_service", source)
        self.assertIn("on_reset=self._request_reset_from_service", source)

    def test_camera_service_module_never_touches_the_simulation(self) -> None:
        module = (ROOT / "src/humanoid_lab/simulators/isaac/camera_service.py").read_text()
        for forbidden in ("import isaac", "from isaaclab", "omni.", "pxr", "self._sim", "self._robot"):
            self.assertNotIn(forbidden, module)
        # The endpoint accepts exactly get_frame and returns three frames.
        self.assertIn('GET_FRAME_REQUEST = b"get_frame"', module)
        self.assertIn("DEFAULT_CAMERA_ENDPOINT = \"tcp://*:5558\"", module)

    def test_the_live_scene_probe_reports_the_worktop_and_the_cubes(self) -> None:
        """The declared surface height is a derived number; the probe is what
        turns it into a reading an operator can check against the running
        stage instead of trusting the profile."""
        source = SERVICE.read_text()
        self.assertIn("def _asset_top_height_m", source)
        self.assertIn("def _scene_probe", source)
        for field in (
            "live_worktop_height_m",
            "live_worktop_minus_declared_m",
            "live_target_top_z_m",
            "live_target_minus_declared_m",
            "cubes",
            "displacement_xy_m",
        ):
            self.assertIn(field, source)
        # Emitted with the palm reading and carried into the run summary.
        self.assertIn('"scene": self._scene_probe()', source)
        self.assertIn('"scene_probe": self._scene_probe()', source)

    def test_the_black_tape_is_flattened_and_robot_table_clearance_is_logged(self) -> None:
        """The declared black tape used to render mid-gray (the dielectric
        specular of its own preview surface under the scene lights), and the
        standing hold needs a robot-versus-table reading next to the palms."""
        source = SERVICE.read_text()
        # The tape keeps the scene's own UsdPreviewSurface; only its specular
        # response is authored away, and the inputs are logged as evidence.
        self.assertIn("def flatten_tape_specular", source)
        self.assertIn("usespecularworkflow", source.lower())
        self.assertIn("isaac_g1_target_tape_material", source)
        # Clearance and penetration travel with the palm sample, so the log and
        # the run summary both carry them.
        self.assertIn("def _table_clearance_sample", source)
        self.assertIn("bodies_inside_table_box", source)
        self.assertIn('"table_clearance": self._table_clearance_sample()', source)
        # The probe builds the same target, so it flattens the same material
        # rather than reproducing the gray in its frames.
        probe = (ROOT / "src/humanoid_lab/simulators/isaac/pose_probe.py").read_text()
        self.assertIn("flatten_tape_specular", probe)


if __name__ == "__main__":
    unittest.main()
