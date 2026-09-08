import json
import math
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "cloudwalk_scene.json"


class CloudWalkSceneContractTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads(CONFIG_PATH.read_text())

    def test_dataset_calibrated_nominal_geometry_is_pinned(self):
        self.assertEqual(self.config["camera"]["resolution"], [640, 480])
        self.assertAlmostEqual(self.config["camera"]["horizontal_fov_deg"], 69.45476033821478)
        self.assertEqual(self.config["bottle"]["position_m"], [0.35, 0.25, 0.865])
        self.assertAlmostEqual(self.config["bottle"]["height_m"], 0.207)
        self.assertAlmostEqual(self.config["table"]["size_m"][2], 0.75)

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
        for token in ("UsdPreviewSurface", "BottleGlass", "BlueWater", "CapPlastic", "DiskLightCfg", "DomeLightCfg", "BackWall", "WarmGlossWood", "root_joint", "GetJointEnabledAttr", "ArticulationRootAPI.Apply(pelvis)"):
            self.assertIn(token, source)
        self.assertNotIn("GroundPlaneCfg", source)
        self.assertTrue((ROOT / "configs" / "cloudwalk_wood.png").is_file())
        self.assertIn("UsdUVTexture", source)

    def test_runner_delegates_scene_ownership_and_uses_free_base_physics(self):
        runner = (ROOT / "scripts" / "run-cloudwalk-isaac.py").read_text()
        self.assertIn("make_scene_cfg(scene_config", runner)
        self.assertIn("decorate_scene(scene_config)", runner)
        self.assertIn("disable_gravity = False", runner)
        self.assertIn("fix_root_link = False", runner)
        self.assertIn("configure_g1_free_base_articulation()", runner)
        self.assertIn("--interactive", runner)
        self.assertIn("timeline.is_playing()", runner)
        self.assertIn("immutable_after_reset", runner)
        self.assertNotIn("class CloudWalkSceneCfg", runner)


if __name__ == "__main__":
    unittest.main()
