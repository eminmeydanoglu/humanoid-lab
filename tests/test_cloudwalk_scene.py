import json
import math
import sys
import unittest
from pathlib import Path
from tempfile import NamedTemporaryFile
from types import SimpleNamespace
from unittest.mock import MagicMock, patch


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "cloudwalk_scene.json"
sys.path.insert(0, str(ROOT / "tools"))
from cloudwalk_scene import configure_rtx, head_camera_prim_path, load_scene_config  # noqa: E402


class CloudWalkSceneContractTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads(CONFIG_PATH.read_text())

    def test_dataset_calibrated_nominal_geometry_is_pinned(self):
        camera = self.config["camera"]
        self.assertEqual(camera["resolution"], [640, 480])
        horizontal_fov_deg = math.degrees(2 * math.atan(camera["horizontal_aperture_mm"] / (2 * camera["focal_length_mm"])))
        self.assertAlmostEqual(horizontal_fov_deg, 69.45476033821478)
        self.assertEqual(self.config["bottle"]["position_m"], [0.35, 0.25, 0.865])
        self.assertAlmostEqual(self.config["bottle"]["height_m"], 0.207)
        self.assertAlmostEqual(self.config["table"]["size_m"][2], 0.75)

    def test_head_camera_uses_unitree_d435_frame(self):
        camera = self.config["camera"]
        self.assertEqual(camera["parent_link"], "torso_link")
        self.assertEqual(camera["position_rel_parent_m"], [0.0576235, 0.01753, 0.41987])
        expected_pitch = 0.8307767239493009
        expected_rotation = [math.cos(expected_pitch / 2), 0.0, math.sin(expected_pitch / 2), 0.0]
        for actual, expected in zip(camera["rotation_wxyz_parent"], expected_rotation, strict=True):
            self.assertAlmostEqual(actual, expected, places=9)
        w, x, y, z = camera["rotation_wxyz_parent"]
        forward = (1 - 2 * (y * y + z * z), 2 * (x * y + w * z), 2 * (x * z - w * y))
        self.assertAlmostEqual(forward[1], 0.0, places=9)
        self.assertGreater(forward[0], 0.67)
        self.assertLess(forward[2], -0.73)
        self.assertEqual(head_camera_prim_path(self.config), "{ENV_REGEX_NS}/Robot/torso_link/head_camera")
        self.assertEqual(head_camera_prim_path(self.config, "/World/envs/env_0"), "/World/envs/env_0/Robot/torso_link/head_camera")

    def test_camera_mount_validation_rejects_invalid_parent_and_quaternion(self):
        for camera_update, message in (
            ({"parent_link": "torso_link/nested"}, "parent_link"),
            ({"rotation_wxyz_parent": [1.0, 1.0, 0.0, 0.0]}, "normalized"),
        ):
            invalid = json.loads(json.dumps(self.config))
            invalid["camera"].update(camera_update)
            with NamedTemporaryFile(mode="w+", suffix=".json") as config_file:
                json.dump(invalid, config_file)
                config_file.flush()
                with self.assertRaisesRegex(ValueError, message):
                    load_scene_config(Path(config_file.name))

    def test_robot_starts_outside_and_close_to_table(self):
        robot = self.config["robot"]
        table = self.config["table"]
        table_min_x = table["position_m"][0] - table["size_m"][0] / 2
        robot_max_x = robot["position_m"][0] + robot["clearance_radius_m"]
        self.assertLessEqual(robot_max_x, table_min_x)
        self.assertLessEqual(table_min_x - robot_max_x, 0.15)
        self.assertEqual(robot["rotation_wxyz"], [1.0, 0.0, 0.0, 0.0])
        self.assertTrue(robot["require_asset_collisions"])
        self.assertTrue(table["collision_enabled"])
        self.assertEqual(load_scene_config(CONFIG_PATH)["robot"], robot)

    def test_invalid_robot_table_collision_layout_is_rejected(self):
        for update, message in (
            ({"position_m": [0.0, 0.0, 0.74]}, "outside"),
            ({"position_m": [-1.0, 0.0, 0.74]}, "close"),
            ({"require_asset_collisions": False}, "collisions"),
        ):
            invalid = json.loads(json.dumps(self.config))
            invalid["robot"].update(update)
            with NamedTemporaryFile(mode="w+", suffix=".json") as config_file:
                json.dump(invalid, config_file)
                config_file.flush()
                with self.assertRaisesRegex(ValueError, message):
                    load_scene_config(Path(config_file.name))

    def test_ground_table_and_robot_articulation_use_instance_safe_collisions(self):
        source = (ROOT / "tools" / "cloudwalk_scene.py").read_text()
        self.assertIn("size=(6.0, 6.0, 0.05), collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True)", source)
        self.assertIn('collision_enabled=table_data["collision_enabled"]', source)
        self.assertIn("Usd.TraverseInstanceProxies()", source)
        self.assertIn("UsdPhysics.CollisionAPI(prim)", source)
        self.assertIn("GetCollisionEnabledAttr().Get() is False", source)
        self.assertIn("init_state=robot_cfg.init_state.replace(", source)
        self.assertNotIn("spawn=robot_cfg.spawn.replace(", source)
        self.assertNotIn('collision_enabled=robot_data["', source)

    def test_bottle_starts_on_table_and_has_physical_properties(self):
        table_top = self.config["table"]["position_m"][2] + self.config["table"]["size_m"][2] / 2
        bottle_bottom = self.config["bottle"]["position_m"][2] - self.config["bottle"]["height_m"] / 2
        self.assertLess(abs(table_top - bottle_bottom), 0.015)
        self.assertGreater(self.config["bottle"]["mass_kg"], 0.45)
        self.assertLess(self.config["bottle"]["mass_kg"], 0.60)
        self.assertGreater(self.config["bottle"]["static_friction"], self.config["bottle"]["dynamic_friction"])
        self.assertGreater(self.config["table"]["static_friction"], self.config["table"]["dynamic_friction"])

    def test_scene_uses_rtx_pbr_room_and_compound_bottle(self):
        source = (ROOT / "tools" / "cloudwalk_scene.py").read_text()
        self.assertEqual(self.config["render"]["renderer"], "RayTracedLighting")
        for token in ("UsdPreviewSurface", "BottleGlass", "BlueWater", "CapPlastic", "shell_opacity", "water_opacity", "texture_scale", "lighting_data", "DiskLightCfg", "DomeLightCfg", "BackWall", "WarmGlossWood", "root_joint", "GetJointEnabledAttr", "ArticulationRootAPI.Apply(pelvis)", "head_camera_prim_path(config)", "convention=\"world\""):
            self.assertIn(token, source)
        self.assertNotIn("GroundPlaneCfg", source)
        self.assertTrue((ROOT / "configs" / "cloudwalk_wood.png").is_file())
        self.assertIn("UsdUVTexture", source)

    def test_reference_appearance_is_configured_for_blue_pet_and_light_wood(self):
        bottle = self.config["bottle"]
        table = self.config["table"]
        self.assertGreater(bottle["shell_color_srgb"][2], bottle["shell_color_srgb"][0] * 8)
        self.assertGreaterEqual(bottle["shell_opacity"], 0.8)
        self.assertLess(bottle["shell_opacity"], 0.9)
        self.assertGreater(bottle["shell_opacity"], bottle["water_opacity"])
        self.assertGreater(bottle["water_opacity"], 0.4)
        self.assertLess(bottle["water_opacity"], 0.55)
        self.assertTrue(self.config["render"]["realtime_translucency"])
        self.assertLess(max(bottle["cap_color_srgb"]), 0.06)
        self.assertGreater(sum(table["wood_color_srgb"]), 1.2)
        self.assertLess(table["roughness"], 0.25)
        self.assertGreater(self.config["render"]["samples_per_pixel"], 16)

    def test_rtx_realtime_translucency_is_explicitly_enabled(self):
        settings = MagicMock()
        carb = SimpleNamespace(settings=SimpleNamespace(get_settings=lambda: settings))
        with patch.dict(sys.modules, {"carb": carb}):
            configure_rtx(self.config)
        settings.set.assert_any_call("/rtx/rendermode", "RayTracedLighting")
        settings.set_bool.assert_any_call("/rtx/translucency/enabled", True)

    def test_appearance_validation_rejects_invisible_shell_and_invalid_color(self):
        for path, value, message in (
            (("bottle", "shell_opacity"), 0.0, "shell_opacity"),
            (("bottle", "cap_color_srgb"), [0.0, 0.0, 1.2], "cap_color_srgb"),
            (("render", "realtime_translucency"), False, "realtime_translucency"),
        ):
            invalid = json.loads(json.dumps(self.config))
            invalid[path[0]][path[1]] = value
            with NamedTemporaryFile(mode="w+", suffix=".json") as config_file:
                json.dump(invalid, config_file)
                config_file.flush()
                with self.assertRaisesRegex(ValueError, message):
                    load_scene_config(Path(config_file.name))

    def test_runner_delegates_scene_ownership_and_uses_free_base_physics(self):
        runner = (ROOT / "scripts" / "run-cloudwalk-isaac.py").read_text()
        self.assertIn("make_scene_cfg(scene_config", runner)
        self.assertIn("decorate_scene(scene_config)", runner)
        self.assertIn("disable_gravity = False", runner)
        self.assertIn("fix_root_link = False", runner)
        self.assertIn("configure_g1_free_base_articulation()", runner)
        self.assertIn("--interactive", runner)
        self.assertIn("timeline.is_playing()", runner)
        self.assertIn("timeline.pause()", runner)
        self.assertNotIn("timeline.stop()", runner)
        self.assertIn("omni.anim.window.timeline", runner)
        self.assertIn("HEAD_CAMERA_PANEL_TITLE", runner)
        self.assertIn("head_camera_prim_path(scene_config, \"/World/envs/env_0\")", runner)
        self.assertNotIn("/Robot/torso_link/head_camera", runner)
        self.assertIn("immutable_after_reset", runner)
        self.assertNotIn("class CloudWalkSceneCfg", runner)


if __name__ == "__main__":
    unittest.main()
