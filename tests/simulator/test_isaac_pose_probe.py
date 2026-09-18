from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from humanoid_lab.simulators.isaac.pose_probe import (  # noqa: E402
    BEND_ANGLE_TOLERANCE_DEG,
    build_report,
    lowest_link,
    perpendicular_elbow_angle,
    with_elbow_flexion,
    arm_bend_angle_deg,
    hand_link_names,
)

PROBE = ROOT / "src/humanoid_lab/simulators/isaac/pose_probe.py"

# The G1 Dex3 elbow frames, as authored in the pinned asset USD: the elbow joint
# origin sits in the shoulder-yaw frame (the upper-arm vector) and the wrist-roll
# joint origin sits in the elbow frame (the forearm vector).
ASSET_UPPER_ARM_M = (0.015783000737428665, 0.0, -0.08051799982786179)
ASSET_FOREARM_M = (0.10000000149011612, 0.001887909951619804, -0.009999999776482582)


def rotate_about_y(vector: tuple[float, float, float], angle: float) -> tuple[float, float, float]:
    x, y, z = vector
    return (
        x * math.cos(angle) + z * math.sin(angle),
        y,
        -x * math.sin(angle) + z * math.cos(angle),
    )


class ElbowFlexionTests(unittest.TestCase):
    def test_the_asset_frames_solve_to_a_perpendicular_bend(self) -> None:
        angle = perpendicular_elbow_angle(ASSET_UPPER_ARM_M, ASSET_FOREARM_M)
        forearm = rotate_about_y(ASSET_FOREARM_M, angle)
        dot = sum(a * b for a, b in zip(ASSET_UPPER_ARM_M, forearm))
        scale = math.dist((0, 0, 0), ASSET_UPPER_ARM_M) * math.dist((0, 0, 0), ASSET_FOREARM_M)
        self.assertLess(abs(dot), 1e-9 * scale + 1e-12)
        # The solution that keeps the hand in front of the body, not behind it.
        self.assertLess(abs(angle), math.pi / 2.0)

    def test_a_forearm_already_perpendicular_needs_no_rotation(self) -> None:
        angle = perpendicular_elbow_angle((0.0, 0.0, -1.0), (1.0, 0.0, 0.0))
        self.assertAlmostEqual(angle, 0.0, places=12)

    def test_the_solution_generalizes_over_synthetic_frames(self) -> None:
        for forearm in ((0.1, 0.0, 0.0), (0.1, 0.02, -0.01), (0.08, -0.03, 0.004)):
            angle = perpendicular_elbow_angle(ASSET_UPPER_ARM_M, forearm)
            rotated = rotate_about_y(forearm, angle)
            dot = sum(a * b for a, b in zip(ASSET_UPPER_ARM_M, rotated))
            self.assertLess(abs(dot), 1e-12)

    def test_flexion_overrides_only_the_elbows(self) -> None:
        pose = {"left_elbow_joint": 0.6, "right_elbow_joint": 0.6, "waist_pitch_joint": 0.0}
        flexed = with_elbow_flexion(pose, -0.2932)
        self.assertAlmostEqual(flexed["left_elbow_joint"], -0.2932)
        self.assertAlmostEqual(flexed["right_elbow_joint"], -0.2932)
        self.assertAlmostEqual(flexed["waist_pitch_joint"], 0.0)
        self.assertAlmostEqual(pose["left_elbow_joint"], 0.6)
        with self.assertRaisesRegex(ValueError, "left_elbow_joint"):
            with_elbow_flexion({"right_elbow_joint": 0.0}, -0.2932, sides=("left",))

    def test_the_measured_bend_of_a_right_angle_arm_is_ninety_degrees(self) -> None:
        angle = arm_bend_angle_deg(
            shoulder=(0.0, 0.1, 1.1), elbow=(0.0, 0.1, 0.8), wrist=(0.3, 0.1, 0.8)
        )
        self.assertAlmostEqual(angle, 90.0, places=9)
        self.assertAlmostEqual(
            arm_bend_angle_deg(
                shoulder=(0.0, 0.1, 1.1), elbow=(0.0, 0.1, 0.8), wrist=(0.0, 0.1, 0.5)
            ),
            0.0,
            places=9,
        )

    def test_hands_cover_palm_and_three_fingers(self) -> None:
        for side in ("left", "right"):
            links = hand_link_names(side)
            self.assertEqual(len(links), 8)
            for finger in ("index_0", "index_1", "middle_0", "middle_1", "thumb_0", "thumb_1", "thumb_2"):
                self.assertIn(f"{side}_hand_{finger}_link", links)
        with self.assertRaisesRegex(ValueError, "side"):
            hand_link_names("middle")


class ProbeReportTests(unittest.TestCase):
    def test_the_lowest_hand_link_sets_the_reported_height(self) -> None:
        bounds = {
            "left_hand_palm_link": ((0.70, 0.0, 0.02), (0.9, 0.1, 0.12)),
            "left_hand_middle_1_link": ((0.68, 0.0, -0.01), (0.9, 0.1, 0.09)),
        }
        name, value = lowest_link(bounds)
        self.assertEqual(name, "left_hand_middle_1_link")
        self.assertAlmostEqual(value, -0.01)

    def test_the_report_states_the_surface_the_cubes_require(self) -> None:
        hands = {
            "left": {"min_z_m": 0.846},
            "right": {"min_z_m": 0.850},
        }
        report = build_report(
            profile_id="isaac-g1-29dof-dex3-sonic-blockstacking",
            asset_reference="/data/models/sonic-assets/g1_29dof_with_hand_rev_1_0.usd",
            base_pose="sonic_standing",
            elbow_joint_value_rad=-0.2932,
            elbow_joint_limit_rad=(-1.0472, 2.0944),
            elbow_axis="Y",
            joint_values={"left_elbow_joint": -0.2932},
            hands=hands,
            scene={
                "declared_surface_height_m": 0.795,
                "cube_size_m": 0.05,
                "live_worktop_top_z_m": 0.795,
                "live_cube_top_z_m": 0.845,
            },
            frames=["/outputs/frame.jpg"],
        )
        self.assertAlmostEqual(report["common_lowest_hand_z_m"], 0.846)
        # table_surface + cube_size = lowest hand: the formula the height exists for.
        self.assertAlmostEqual(report["surface_from_cube_top_m"], 0.796)
        self.assertAlmostEqual(report["hand_minus_cube_top_m"]["left"], 0.001)
        self.assertAlmostEqual(report["hand_minus_cube_top_m"]["right"], 0.005)
        self.assertEqual(report["frames"], ["/outputs/frame.jpg"])

    def test_a_robot_only_report_has_no_scene_numbers(self) -> None:
        report = build_report(
            profile_id="p",
            asset_reference="/a.usd",
            base_pose="sonic_standing",
            elbow_joint_value_rad=0.0,
            elbow_joint_limit_rad=(-1.0, 1.0),
            elbow_axis="Y",
            joint_values={},
            hands={"left": {"min_z_m": 0.8}, "right": {"min_z_m": 0.81}},
            scene=None,
            frames=[],
        )
        self.assertIsNone(report["table_surface_m"])
        self.assertIsNone(report["surface_from_cube_top_m"])
        self.assertNotIn("hand_minus_cube_top_m", report)


class ProbeSourceInvariantTests(unittest.TestCase):
    def test_the_probe_imports_isaac_only_behind_the_app_launcher(self) -> None:
        # Importing the module must not need Isaac: the module-level imports are
        # usable by tests and by callers that never open a stage.
        for forbidden in ("\nimport isaaclab", "\nfrom isaaclab", "\nimport omni", "\nfrom omni"):
            self.assertNotIn(forbidden, PROBE.read_text())

    def test_the_probe_declares_the_measurement_it_takes(self) -> None:
        source = PROBE.read_text()
        self.assertIn("BEND_ANGLE_TOLERANCE_DEG = ", source)
        self.assertIn('"lowest_hand_z_m"', source)
        self.assertIn("--include-scene", source)
        self.assertIn("--no-flexion", source)
        self.assertIn("disable_gravity=True", source)
        self.assertLess(BEND_ANGLE_TOLERANCE_DEG, 2.0)


if __name__ == "__main__":
    unittest.main()
