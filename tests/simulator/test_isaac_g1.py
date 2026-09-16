from __future__ import annotations

import dataclasses
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from humanoid_lab.simulators.isaac.contracts import (  # noqa: E402
    ContractError,
    RunProfile,
    TerrainSpec,
)

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


class IsaacG1TerrainTests(unittest.TestCase):
    ROUGH = "isaac-g1-sonic-rough-dex3.json"
    FLAT = "isaac-g1-sonic-dex3.json"

    def _load(self, name: str) -> RunProfile:
        return RunProfile.load(ROOT / "configs/profiles" / name)

    def test_rough_profile_is_the_sonic_profile_on_other_ground(self) -> None:
        """The option must change the ground and nothing else."""
        rough = self._load(self.ROUGH)
        flat = self._load(self.FLAT)
        self.assertIsNotNone(rough.terrain)
        self.assertEqual(rough.terrain.preset, "instinct_parkour_rough")
        self.assertEqual(rough.terrain.max_init_terrain_level, 0)
        self.assertIsNone(flat.terrain)
        self.assertEqual(dataclasses.replace(rough, terrain=None, profile_id=flat.profile_id), flat)

    def test_flat_profiles_declare_no_terrain(self) -> None:
        for path in sorted((ROOT / "configs/profiles").glob("isaac-g1-*.json")):
            if path.name == self.ROUGH:
                continue
            self.assertIsNone(RunProfile.load(path).terrain, f"{path.name} declares a terrain")

    def test_manifest_records_the_world_the_run_used(self) -> None:
        manifest = self._load(self.ROUGH).as_manifest()
        self.assertEqual(
            manifest["terrain"],
            {"preset": "instinct_parkour_rough", "max_init_terrain_level": 0},
        )
        self.assertIsNone(self._load(self.FLAT).as_manifest()["terrain"])

    def test_start_level_is_a_declared_choice(self) -> None:
        source = json.loads((ROOT / "configs/profiles" / self.ROUGH).read_text())
        source["terrain"]["max_init_terrain_level"] = 5
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rough.json"
            path.write_text(json.dumps(source))
            self.assertEqual(RunProfile.load(path).terrain.max_init_terrain_level, 5)

    def test_rejects_unknown_preset_and_negative_level(self) -> None:
        source = json.loads((ROOT / "configs/profiles" / self.ROUGH).read_text())
        for terrain, expected in (
            ({"preset": "moon"}, "must be one of"),
            ({"preset": "instinct_parkour_rough", "max_init_terrain_level": -1}, "negative"),
            ({"max_init_terrain_level": 0}, "preset is required"),
            ("instinct_parkour_rough", "must be an object"),
        ):
            source["terrain"] = terrain
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "bad.json"
                path.write_text(json.dumps(source))
                with self.assertRaisesRegex(ContractError, expected):
                    RunProfile.load(path)

    def test_terrain_presets_live_in_one_place(self) -> None:
        """The preset resolves to the pinned instinct world, not to a copy here."""
        terrains = (ROOT / "src/humanoid_lab/simulators/isaac/terrains.py").read_text()
        self.assertIn('PARKOUR_CONFIG_MODULE = "instinctlab.tasks.parkour.config.parkour_env_cfg"', terrains)
        self.assertIn("ROUGH_TERRAINS_CFG", terrains)
        self.assertIn("SceneCfg.terrain", terrains)
        self.assertIn("wall_prob = [0.0, 0.0, 0.0, 0.0]", terrains)
        self.assertIn("importer.use_terrain_origins = True", terrains)
        # A run without a terrain must not import the instinct stack at all.
        self.assertIn("def _preset_source", terrains)
        module_body = terrains.split("def _preset_source", 1)[0]
        self.assertNotIn("import instinctlab", module_body)
        self.assertNotIn("from instinctlab", module_body)

    def test_the_option_trims_the_world_exactly_like_the_playback(self) -> None:
        """Both entry points have to build the same world to be comparable."""
        terrains = (ROOT / "src/humanoid_lab/simulators/isaac/terrains.py").read_text()
        playback = (ROOT / "scripts/play-instinct-parkour.py").read_text()
        self.assertIn("PLAY_ROWS = 4", terrains)
        self.assertIn("PLAY_COLS = 10", terrains)
        self.assertIn("min(generator.num_rows, PLAY_ROWS)", terrains)
        self.assertIn("min(generator.num_cols, PLAY_COLS)", terrains)
        for trim in ("gen.num_rows = min(gen.num_rows, 4)", "gen.num_cols = min(gen.num_cols, 10)"):
            self.assertIn(trim, playback, f"the playback no longer trims the grid: {trim}")
        self.assertIn("wall_prob = [0.0, 0.0, 0.0, 0.0]", playback)

    def test_a_missing_instinct_checkout_fails_as_a_preset_error(self) -> None:
        """The builder must import cleanly without Kit and fail readably."""
        from humanoid_lab.simulators.isaac import terrains

        original = terrains.PARKOUR_CONFIG_MODULE
        terrains.PARKOUR_CONFIG_MODULE = "no_such_module_here"
        try:
            with self.assertRaisesRegex(ContractError, "no_such_module_here"):
                terrains.build_terrain_importer(TerrainSpec(preset="instinct_parkour_rough"))
        finally:
            terrains.PARKOUR_CONFIG_MODULE = original

    def test_the_builder_trims_the_world_without_touching_it(self) -> None:
        """The builder's own work, with the upstream config objects stood in for.

        The real ones need a Kit process to import; what is being checked here is
        that the declared world is trimmed and re-pointed by copy, never edited,
        so the training config in the interpreter stays intact.
        """
        import types

        from humanoid_lab.simulators.isaac import terrains

        def walled(probability: float) -> types.SimpleNamespace:
            return types.SimpleNamespace(wall_prob=[probability] * 4)

        generator = types.SimpleNamespace(
            seed=0,
            size=(8.0, 8.0),
            curriculum=True,
            num_rows=10,
            num_cols=20,
            sub_terrains={
                "perlin_rough": walled(0.3),
                "pyramid_stairs": walled(0.3),
                # A sub-terrain that never had walls must not grow the attribute.
                "hf_pyramid_slope_inv": types.SimpleNamespace(),
            },
        )
        importer = types.SimpleNamespace(
            prim_path="/World/ground",
            terrain_type="generator",
            terrain_generator=generator,
            max_init_terrain_level=5,
            use_terrain_origins=False,
            virtual_obstacles={"edges": object()},
            physics_material=types.SimpleNamespace(static_friction=1.0, dynamic_friction=1.0),
        )
        module_name = "fake_parkour_config"
        fake = types.ModuleType(module_name)
        fake.ROUGH_TERRAINS_CFG = generator

        @dataclasses.dataclass
        class FakeSceneCfg:
            """The shape ``@configclass`` leaves behind.

            Every member becomes a factory-backed field and Python drops the
            class attribute of such a field, so ``SceneCfg.terrain`` raises
            AttributeError at run time.  The stand-in keeps that property, and
            hands out the shared object rather than a copy, so the builder's own
            copy discipline stays observable.
            """

            terrain: object = dataclasses.field(default_factory=lambda: importer)

        fake.SceneCfg = FakeSceneCfg
        original = terrains.PARKOUR_CONFIG_MODULE
        sys.modules[module_name] = fake
        terrains.PARKOUR_CONFIG_MODULE = module_name
        try:
            built, summary = terrains.build_terrain_importer(TerrainSpec(preset="instinct_parkour_rough"))
        finally:
            terrains.PARKOUR_CONFIG_MODULE = original
            del sys.modules[module_name]

        self.assertFalse(hasattr(FakeSceneCfg, "terrain"), "the stand-in lost its factory-backed shape")
        self.assertIsNot(built, importer, "the builder handed back the declared object itself")

        self.assertEqual((built.terrain_generator.num_rows, built.terrain_generator.num_cols), (4, 10))
        for name in ("perlin_rough", "pyramid_stairs"):
            self.assertEqual(built.terrain_generator.sub_terrains[name].wall_prob, [0.0, 0.0, 0.0, 0.0])
        self.assertFalse(hasattr(built.terrain_generator.sub_terrains["hf_pyramid_slope_inv"], "wall_prob"))
        self.assertEqual(built.max_init_terrain_level, 0)
        self.assertTrue(built.use_terrain_origins)
        self.assertEqual(built.prim_path, "/World/ground")
        self.assertEqual(built.terrain_type, "generator")
        self.assertEqual(list(built.virtual_obstacles), ["edges"])
        # The world the preset points at is untouched.
        self.assertEqual(generator.num_rows, 10)
        self.assertEqual(generator.sub_terrains["perlin_rough"].wall_prob, [0.3] * 4)
        self.assertEqual(importer.max_init_terrain_level, 5)
        self.assertEqual(summary["grid"], [4, 10])
        self.assertEqual(summary["tile_size_m"], [8.0, 8.0])
        self.assertEqual(summary["walls_removed"], ["perlin_rough", "pyramid_stairs"])
        self.assertEqual(summary["max_init_terrain_level"], 0)
        self.assertEqual(summary["static_friction"], 1.0)
        self.assertEqual(summary["virtual_obstacles"], ["edges"])

    def test_a_declared_default_that_cannot_be_read_fails_readably(self) -> None:
        from humanoid_lab.simulators.isaac import terrains

        @dataclasses.dataclass
        class NoDefault:
            other: object = dataclasses.field(default_factory=dict)

        with self.assertRaisesRegex(ContractError, "declares no field named 'terrain'"):
            terrains._declared_default(NoDefault, "terrain")

        @dataclasses.dataclass
        class Bare:
            terrain: object

        with self.assertRaisesRegex(ContractError, "without a default to copy"):
            terrains._declared_default(Bare, "terrain")

        with self.assertRaisesRegex(ContractError, "is not a dataclass"):
            terrains._declared_default(object, "terrain")

    def test_terrain_is_the_ground_and_rebases_world_points(self) -> None:
        """Everything the profile declares in world coordinates follows the
        environment origin, which is no longer the world origin once a terrain
        decides where the robot stands."""
        source = SERVICE.read_text()
        self.assertIn("ground_entity", source, "the scene has no single ground entity")
        self.assertIn("if self._terrain_cfg is None", source, "the terrain is not optional")
        self.assertIn(
            "sim_cfg.physics_material = copy.deepcopy(self._terrain_cfg.physics_material)",
            source,
            "the terrain's contact material does not reach the physics scene",
        )
        self.assertIn('{"event": "isaac_g1_terrain"', source, "the world is not reported at start-up")
        self.assertIn('"terrain": self._terrain_summary', source, "the world is missing from the run summary")
        origin_body = source[source.index("    def _env_origin") : source.index("    def _set_debug_camera")]
        self.assertIn("self._scene.env_origins[0]", origin_body, "the environment origin is not read")
        debug_body = source[source.index("    def _set_debug_camera") : source.index("    def _align_validation_camera")]
        self.assertIn("self._env_origin()", debug_body, "the debug view ignores the environment origin")
        support_body = source[source.index("    def _apply_support") : source.index("    def _resolve_band_body_id")]
        self.assertIn(
            "point = torch.tensor(support.point_m, device=self._robot.device) + origin",
            support_body,
            "the support band would pull the robot toward the world origin",
        )
        align_body = source[
            source.index("    def _align_validation_camera") : source.index("    def _start_support")
        ]
        self.assertIn("camera.set_world_poses(positions=position.unsqueeze(0))", align_body)
        self.assertIn("if self._terrain_cfg is None:\n            return", align_body)
        self.assertIn('self._scene.sensors.get("validation_camera")', align_body)

    def test_a_flat_run_keeps_the_plane_scene(self) -> None:
        source = SERVICE.read_text()
        self.assertIn('AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())', source)
        self.assertIn('self._sim.set_camera_view(eye=(2.6 + x, 2.4 + y, 1.6 + z), target=(x, y, 0.65 + z))', source)


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
        reset_branch = source[
            source.index("                if self._pending_ui_reset:") :
            source.index("                if self._timeline.is_playing():")
        ]
        self.assertLess(
            reset_branch.index("self.reset_episode(resume=True)"),
            reset_branch.index("self._step_physics()"),
        )
        self.assertLess(
            reset_branch.index("self._step_physics()"),
            reset_branch.index("self.pause()"),
        )
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

    def test_reset_is_refused_while_a_controller_is_driving(self) -> None:
        """A reset teleports the articulation while the controller still holds
        state from the previous episode; mixing the two is not allowed."""
        source = SERVICE.read_text()
        body = source[
            source.index("    def _request_reset_from_ui") : source.index("    def _open_ui")
        ]
        self.assertIn("if self._control_mode == CONTROLLED:", body)
        self.assertIn("isaac_g1_ui_reset_refused", body)
        # Only the non-controlled path may queue work for the loop.
        refused, _, rest = body.partition('"isaac_g1_ui_reset_refused"')
        self.assertIn("return", rest.split("self._pending_ui_reset = True")[0])
        self.assertIn("self._pending_ui_reset = True", rest)

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

    def test_the_service_ui_tracks_the_displayed_ui(self) -> None:
        """A streamed run is headless on the host but still draws the UI."""
        source = (ROOT / "src/humanoid_lab/simulators/isaac/cli.py").read_text()
        self.assertIn("args.ui_displayed = not args.headless", source)
        self.assertIn("show_ui=args.ui_displayed", source)
        self.assertNotIn("show_ui=not args.headless", source)
        before_launcher = source.split("launcher = AppLauncher(args)", 1)[0]
        self.assertIn(
            "args.ui_displayed = not args.headless",
            before_launcher,
            "AppLauncher forces headless on a livestreaming run",
        )


if __name__ == "__main__":
    unittest.main()
