"""Tests for aligning the simulated body to the controller's MuJoCo model."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from humanoid_lab.controllers.sonic import (  # noqa: E402
    STANDING_ROOT_HEIGHT_M,
    named_pose,
)
from humanoid_lab.simulators.isaac.masses import (  # noqa: E402
    MASSLESS_BODY_MASS_KG,
    WELDED_BODY_MASS_KG,
    MassAlignmentError,
    load_mujoco_body_masses,
    load_urdf_body_masses,
    load_urdf_link_names,
    load_urdf_merged_body_masses,
    plan_alignment,
)
from humanoid_lab.simulators.isaac.training_dynamics import (  # noqa: E402
    TrainingDynamicsError,
    load_training_joint_armature,
    resolve_armature,
)

FIXTURE = """<mujoco model="g1">
  <worldbody>
    <body name="pelvis" pos="0 0 0.79">
      <freejoint/>
      <inertial pos="0 0 0" mass="3.813" diaginertia="0.01 0.01 0.01"/>
      <body name="left_hip_pitch_link">
        <joint name="left_hip_pitch_joint" axis="0 1 0"/>
        <inertial pos="0 0 0" mass="1.35" diaginertia="0.01 0.01 0.01"/>
      </body>
      <body name="torso_link">
        <joint name="waist_yaw_joint" axis="0 0 1"/>
        <inertial pos="0 0 0" mass="9.598" diaginertia="0.01 0.01 0.01"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""


class LoadMujocoMassesTests(unittest.TestCase):
    def _write(self, text: str) -> Path:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "model.xml"
        path.write_text(text)
        return path

    def test_reads_every_body_mass(self) -> None:
        masses = load_mujoco_body_masses(self._write(FIXTURE))
        self.assertEqual(
            masses, {"pelvis": 3.813, "left_hip_pitch_link": 1.35, "torso_link": 9.598}
        )

    def test_a_model_without_masses_is_rejected(self) -> None:
        with self.assertRaises(MassAlignmentError):
            load_mujoco_body_masses(self._write('<mujoco><worldbody><body name="x"/></worldbody></mujoco>'))

    def test_a_missing_file_is_reported_not_ignored(self) -> None:
        with self.assertRaises(MassAlignmentError):
            load_mujoco_body_masses("/nonexistent/g1_29dof_with_hand.xml")

    def test_malformed_xml_is_reported(self) -> None:
        with self.assertRaises(MassAlignmentError):
            load_mujoco_body_masses(self._write("<mujoco><worldbody>"))


class PlanAlignmentTests(unittest.TestCase):
    def test_maps_masses_by_body_name(self) -> None:
        indices, values, welded = plan_alignment(
            ["pelvis", "torso_link", "logo_link"], {"torso_link": 9.598, "pelvis": 3.813}
        )
        self.assertEqual(indices, [0, 1, 2])
        self.assertEqual(values[:2], [3.813, 9.598])
        self.assertEqual(welded, ["logo_link"])

    def test_a_body_the_model_welds_into_its_parent_is_negligible(self) -> None:
        _, values, welded = plan_alignment(["torso_link", "head_link"], {"torso_link": 9.598})
        self.assertEqual(welded, ["head_link"])
        self.assertEqual(values[1], WELDED_BODY_MASS_KG)
        self.assertGreater(WELDED_BODY_MASS_KG, 0.0)

    def test_reports_when_nothing_matches(self) -> None:
        with self.assertRaises(MassAlignmentError):
            plan_alignment(["a", "b"], {"c": 1.0})


class ServiceWiringTests(unittest.TestCase):
    def test_alignment_is_declared_and_fails_closed(self) -> None:
        service = (ROOT / "src/humanoid_lab/simulators/isaac/service.py").read_text()
        body = service[
            service.index("    def _align_body_masses") : service.index("    def _report_actuators")
        ]
        # Only declared modes are accepted, and the historic one keeps working.
        # The tuple itself lives at module scope, so check its content there.
        self.assertIn(
            'JOINT_DYNAMICS_ALIGNMENT_MODES: tuple[str, ...] = ("sonic_mujoco", "sonic_training")',
            service,
        )
        self.assertIn("mode not in MASS_ALIGNMENT_MODES", body)
        self.assertIn('training = mode == "sonic_training"', body)
        # A body the welded model folds into its parent must not be guessed at:
        # it is made negligible and reported, because the parent already carries
        # it.
        self.assertIn("bodies_welded_in_model", body)
        self.assertIn("WELDED_BODY_MASS_KG", service + (ROOT / "src/humanoid_lab/simulators/isaac/masses.py").read_text())
        # The masses come from the pinned models, never from a table in the code,
        # and the training mode compares against the MERGED plant (the training
        # rig converts with merge_fixed_joints=True).
        self.assertIn("load_mujoco_body_masses", body)
        self.assertIn("load_urdf_merged_body_masses", body)
        self.assertIn("set_masses(", body)
        self.assertNotIn("MUJOCO_BODY_MASS_KG", service)
        self.assertIn("self._align_body_masses()", service)

    def test_no_mass_table_is_duplicated_in_the_codebase(self) -> None:
        # The torso mass is the value a hand-typed table would most likely
        # contain; it is read from the pinned model instead.
        for path in (ROOT / "src").rglob("*.py"):
            with self.subTest(path=path.name):
                self.assertNotIn("9.598", path.read_text())

    def test_joint_dynamics_are_per_group_not_one_flat_value(self) -> None:
        # The bug this guards: writing one armature for every joint mis-sets the
        # leg chain by 2.5x.  The training values are read from the pinned
        # config, and a body joint the config does not cover fails the run.
        body = (ROOT / "src/humanoid_lab/simulators/isaac/service.py").read_text()
        section = body[body.index("    def _align_joint_dynamics") : body.index("    def _align_body_masses")]
        self.assertIn("load_training_joint_armature", section)
        self.assertIn("resolve_armature", section)
        self.assertIn("armature_by_joint", section)
        self.assertIn("refusing to claim a training-plant alignment", section)
        # The historic mode still writes its own flat MuJoCo values.
        self.assertIn("MUJOCO_JOINT_ARMATURE", section)


class TrainingPlantTests(unittest.TestCase):
    """The two pinned plants are not the same robot, and the mode says which."""

    def _write_urdf(self, text: str) -> Path:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "model.urdf"
        path.write_text(text)
        return path

    def test_training_urdf_is_parsed_from_its_child_mass_element(self) -> None:
        # URDF stores mass as <inertial><mass value=.../></inertial>, unlike
        # MJCF's attribute; a shared lookup silently returns an empty model.
        path = self._write_urdf(
            '<robot name="g1"><link name="pelvis"><inertial>'
            '<mass value="3.813"/></inertial></link></robot>'
        )
        self.assertEqual(load_urdf_body_masses(path), {"pelvis": 3.813})

    def test_a_urdf_without_masses_is_rejected(self) -> None:
        path = self._write_urdf('<robot name="g1"><link name="pelvis"/></robot>')
        with self.assertRaises(MassAlignmentError):
            load_urdf_body_masses(path)

    def test_non_welded_mode_refuses_a_body_the_model_does_not_declare(self) -> None:
        # A body the model does not declare AT ALL is a dropped mass claimed as
        # an aligned plant.  Refusing is the point.
        with self.assertRaises(MassAlignmentError):
            plan_alignment(
                ["pelvis", "mystery_link"],
                {"pelvis": 3.813},
                missing="refuse",
                declared=["pelvis"],
            )

    def test_a_declared_but_massless_frame_is_never_written_as_zero(self) -> None:
        # The training URDF declares the IMUs and camera mounts with no
        # <inertial>: the model has those bodies, it just gives them no inertia.
        # Treating "absent from the mass map" as "absent from the model" would
        # refuse the correct plant -- and PhysX refuses a literal zero mass, so
        # the gram floor is used instead of 0.0.
        _, values, welded = plan_alignment(
            ["pelvis", "imu_in_torso"],
            {"pelvis": 3.813},
            missing="refuse",
            declared=["pelvis", "imu_in_torso"],
        )
        self.assertEqual(values, [3.813, MASSLESS_BODY_MASS_KG])
        self.assertEqual(welded, ["imu_in_torso"])
        self.assertGreater(MASSLESS_BODY_MASS_KG, 0.0)

    def test_welded_mode_still_makes_unnamed_bodies_negligible(self) -> None:
        _, values, welded = plan_alignment(["pelvis", "head_link"], {"pelvis": 3.813})
        self.assertEqual(values, [3.813, WELDED_BODY_MASS_KG])
        self.assertEqual(welded, ["head_link"])

    def test_an_unknown_missing_policy_is_rejected(self) -> None:
        with self.assertRaises(MassAlignmentError):
            plan_alignment(["pelvis"], {"pelvis": 3.813}, missing="guess")

    def test_the_training_rig_merges_fixed_joints_before_spawning(self) -> None:
        # UrdfFileCfg converts with merge_fixed_joints=True, so the plant the
        # policy sees is the MERGED model, not the URDF's raw per-link table.
        # Aligning to the raw table would both invent a wrist/palm difference
        # and drop the merged-away mass (those links are not asset bodies).
        authored = load_urdf_body_masses()
        merged, merged_away = load_urdf_merged_body_masses()
        self.assertIn("left_hand_palm_link", authored)
        self.assertNotIn("left_hand_palm_link", merged)
        self.assertIn("left_hand_palm_link", merged_away)
        self.assertIn("head_link", merged_away)
        # The palm is folded into wrist_yaw, the head into the torso.
        self.assertAlmostEqual(
            merged["left_wrist_yaw_link"], authored["left_wrist_yaw_link"] + authored["left_hand_palm_link"], places=7
        )
        # torso gains the head AND the logo shell (both fixed-attached).
        self.assertAlmostEqual(
            merged["torso_link"],
            authored["torso_link"] + authored["head_link"] + authored["logo_link"],
            places=7,
        )
        # The merged plant has exactly the deployment model's body set.
        self.assertEqual(set(merged), set(load_mujoco_body_masses()))

    def test_the_merged_training_arm_matches_the_deployment_model_exactly(self) -> None:
        # This is the load the arm PD actually carries, and the two plants agree
        # on it.  An evaluation that reports "the arms are ~1.8 kg heavy in one
        # plant" is wrong; the arm chain is identical.
        merged, _ = load_urdf_merged_body_masses()
        mujoco = load_mujoco_body_masses()
        arm = ("shoulder", "elbow", "wrist", "hand_")
        merged_arm = sum(v for k, v in merged.items() if any(s in k for s in arm))
        mujoco_arm = sum(v for k, v in mujoco.items() if any(s in k for s in arm))
        self.assertAlmostEqual(merged_arm, mujoco_arm, places=4)
        self.assertAlmostEqual(
            merged["left_wrist_yaw_link"], mujoco["left_wrist_yaw_link"], places=5
        )

    def test_the_remaining_mass_difference_is_torso_and_waist_only(self) -> None:
        merged, _ = load_urdf_merged_body_masses()
        mujoco = load_mujoco_body_masses()
        differing = {
            name: round(mujoco[name] - merged[name], 6)
            for name in sorted(set(merged) & set(mujoco))
            if abs(mujoco[name] - merged[name]) > 1e-4
        }
        self.assertEqual(
            differing,
            {
                "pelvis": -0.001,
                "torso_link": 1.781,
                "waist_roll_link": -0.039,
                "waist_yaw_link": 0.03,
            },
        )
        self.assertAlmostEqual(sum(merged.values()), 34.3942, places=3)
        self.assertAlmostEqual(sum(mujoco.values()), 36.1652, places=3)

    def test_the_massless_frames_are_declared_by_the_training_urdf(self) -> None:
        # These are the bodies that made a naive "absent from the mass map ==
        # absent from the model" policy refuse the correct plant.
        declared = set(load_urdf_link_names())
        masses = load_urdf_body_masses()
        for frame in ("imu_in_pelvis", "imu_in_torso", "mid360_link", "d435_link"):
            with self.subTest(frame=frame):
                self.assertIn(frame, declared)
                self.assertNotIn(frame, masses)

    def test_the_training_plant_is_lighter_than_the_deployment_model(self) -> None:
        # The historic eval aligned to 36.1742 kg; the training plant is
        # 34.3942 kg (34.4032 kg after the gram floor on 9 merged bodies).
        training = load_urdf_body_masses()
        mujoco = load_mujoco_body_masses()
        self.assertAlmostEqual(sum(training.values()), 34.3942, places=3)
        self.assertAlmostEqual(sum(mujoco.values()), 36.1652, places=3)

    def test_pose_models_keep_the_historic_height_as_default(self) -> None:
        deploy_pose, deploy_height = named_pose("sonic_standing")
        self.assertEqual(deploy_height, STANDING_ROOT_HEIGHT_M)
        training_pose, training_height = named_pose("sonic_standing", "training")
        self.assertEqual(training_height, 0.76)
        # Only the height may differ: the pose itself is the same plant-agnostic
        # joint vector, so a pose-model switch can never move a joint value.
        self.assertEqual(deploy_pose, training_pose)

    def test_an_unknown_pose_model_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            named_pose("sonic_standing", "guessed")


class TrainingJointDynamicsTests(unittest.TestCase):
    """The training armature is per group; a flat value mis-sets the legs."""

    def _armature(self) -> dict[str, float]:
        from humanoid_lab.controllers.sonic import BODY_JOINT_ORDER

        entries, _constants = load_training_joint_armature()
        values, unset = resolve_armature(list(BODY_JOINT_ORDER), entries)
        self.assertEqual(unset, [])
        return dict(zip(BODY_JOINT_ORDER, values))

    def test_legs_are_not_the_deployment_flat_armature(self) -> None:
        # This is the bug a flat 0.01 write would introduce: the knee is 2.5x.
        armature = self._armature()
        for joint in ("left_knee_joint", "right_knee_joint"):
            self.assertAlmostEqual(armature[joint], 0.025101925, places=9)
        for joint in ("left_hip_pitch_joint", "left_hip_roll_joint", "right_hip_pitch_joint"):
            self.assertAlmostEqual(armature[joint], 0.025101925, places=9)
        for joint in ("left_hip_yaw_joint", "right_hip_yaw_joint"):
            self.assertAlmostEqual(armature[joint], 0.01017752, places=9)
        self.assertNotAlmostEqual(armature["left_knee_joint"], 0.01)

    def test_ankles_and_waist_use_the_doubled_5020_armature(self) -> None:
        armature = self._armature()
        for joint in (
            "left_ankle_pitch_joint",
            "left_ankle_roll_joint",
            "right_ankle_pitch_joint",
            "right_ankle_roll_joint",
            "waist_roll_joint",
            "waist_pitch_joint",
        ):
            with self.subTest(joint=joint):
                self.assertAlmostEqual(armature[joint], 0.00721945, places=9)
        self.assertAlmostEqual(armature["waist_yaw_joint"], 0.01017752, places=9)

    def test_arms_use_the_5020_and_4010_armatures(self) -> None:
        armature = self._armature()
        for joint in (
            "left_shoulder_pitch_joint",
            "left_shoulder_roll_joint",
            "left_shoulder_yaw_joint",
            "left_elbow_joint",
            "left_wrist_roll_joint",
            "right_elbow_joint",
        ):
            with self.subTest(joint=joint):
                self.assertAlmostEqual(armature[joint], 0.003609725, places=9)
        for joint in ("left_wrist_pitch_joint", "left_wrist_yaw_joint", "right_wrist_yaw_joint"):
            with self.subTest(joint=joint):
                self.assertAlmostEqual(armature[joint], 0.00425, places=9)

    def test_legs_arms_and_waist_are_all_distinct_values(self) -> None:
        # A guard against silently collapsing to one armature again.
        armature = self._armature()
        leg = armature["left_knee_joint"]
        arm = armature["left_elbow_joint"]
        ankle = armature["left_ankle_pitch_joint"]
        self.assertEqual(len({leg, arm, ankle}), 3)
        self.assertGreater(leg, arm)

    def test_the_training_config_declares_no_friction(self) -> None:
        # The training cfg sets no friction field at all, and the URDF authors
        # no <dynamics>, so the training plant's friction is genuinely zero.
        source = (
            ROOT / "src/humanoid_lab/simulators/isaac/training_dynamics.py"
        ).read_text()
        self.assertIn("no friction", source.lower())

    def test_a_config_without_the_articulation_fails_closed(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "g1.py"
        path.write_text("SOMETHING_ELSE = 1\n")
        with self.assertRaises(TrainingDynamicsError):
            load_training_joint_armature(path)

    def test_conflicting_group_armatures_fail_closed(self) -> None:
        with self.assertRaisesRegex(TrainingDynamicsError, "conflicting armatures"):
            resolve_armature(["left_knee_joint"], [(".*_knee_joint", 0.01), ("left_knee_joint", 0.02)])

    def test_a_joint_no_group_covers_is_reported_not_defaulted(self) -> None:
        entries, _ = load_training_joint_armature()
        values, unset = resolve_armature(
            ["left_knee_joint", "some_unknown_joint"], entries
        )
        self.assertEqual(unset, ["some_unknown_joint"])
        self.assertAlmostEqual(values[0], 0.025101925, places=9)


class ProfileParityTests(unittest.TestCase):
    """The profiles state which plant they mean, and the default is unchanged."""

    PROFILES = ROOT / "configs" / "profiles"

    def _load(self, name: str):
        from humanoid_lab.simulators.isaac.contracts import RunProfile

        return RunProfile.load(self.PROFILES / name)

    def test_the_historic_profile_is_untouched(self) -> None:
        # The uncommitted baseline every previous rollout used must keep its
        # exact alignment and must not have gained an explicit self-collision
        # setting (None means "leave the asset's authored value alone").
        profile = self._load("isaac-g1-sonic-blockstacking-dex3.json")
        self.assertEqual(profile.controller["mass_alignment"], "sonic_mujoco")
        self.assertEqual(profile.controller["joint_dynamics_alignment"], "sonic_mujoco")
        self.assertIsNone(profile.robot.self_collisions)
        self.assertNotIn("initial_pose_model", profile.controller)
        self.assertNotIn("mass_model_path", profile.controller)

    def test_the_training_plant_profile_selects_every_training_plant_delta(self) -> None:
        profile = self._load("isaac-g1-sonic-blockstacking-dex3-training-plant.json")
        self.assertEqual(profile.controller["mass_alignment"], "sonic_training")
        self.assertEqual(profile.controller["joint_dynamics_alignment"], "sonic_training")
        self.assertTrue(profile.robot.self_collisions)
        # The scene and camera must be identical to the baseline: a plant
        # experiment has to change the plant and nothing else.
        baseline = self._load("isaac-g1-sonic-blockstacking-dex3.json")
        self.assertEqual(profile.scene, baseline.scene)
        self.assertEqual(profile.camera, baseline.camera)
        self.assertEqual(profile.robot.initial_position_m, baseline.robot.initial_position_m)
        self.assertEqual(profile.robot.asset_reference, baseline.robot.asset_reference)
        self.assertEqual(profile.physics_dt, baseline.physics_dt)

    def test_an_undeclared_self_collision_setting_stays_none(self) -> None:
        from humanoid_lab.simulators.isaac.contracts import RobotSpec

        spec = RobotSpec.from_dict(
            {
                "name": "g1-29dof-dex3",
                "body_dofs": 29,
                "asset_kind": "usd",
                "asset_reference": "/tmp/x.usd",
                "asset_provenance": "test",
                "initial_position_m": [0.0, 0.0, 0.8],
                "hand": {
                    "kind": "dex3",
                    "dofs": 14,
                    "behavior": "passive",
                    "joint_name_patterns": ["_hand_"],
                },
            }
        )
        self.assertIsNone(spec.self_collisions)
        declared = RobotSpec.from_dict(
            {
                "name": "g1-29dof-dex3",
                "body_dofs": 29,
                "asset_kind": "usd",
                "asset_reference": "/tmp/x.usd",
                "asset_provenance": "test",
                "initial_position_m": [0.0, 0.0, 0.8],
                "self_collisions": True,
                "hand": {
                    "kind": "dex3",
                    "dofs": 14,
                    "behavior": "passive",
                    "joint_name_patterns": ["_hand_"],
                },
            }
        )
        self.assertTrue(declared.self_collisions)

    def test_the_service_only_overrides_self_collisions_when_declared(self) -> None:
        service = (ROOT / "src/humanoid_lab/simulators/isaac/service.py").read_text()
        self.assertIn("if robot_spec.self_collisions is not None:", service)
        self.assertIn(
            "robot_cfg.spawn.articulation_props.enabled_self_collisions = (", service
        )


if __name__ == "__main__":
    unittest.main()
