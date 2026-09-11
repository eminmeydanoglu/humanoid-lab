"""Tests for aligning the simulated body to the controller's MuJoCo model."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from humanoid_lab.simulators.isaac.masses import (  # noqa: E402
    WELDED_BODY_MASS_KG,
    MassAlignmentError,
    load_mujoco_body_masses,
    plan_alignment,
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
        # Only the declared mode is accepted.
        self.assertIn('if mode != "sonic_mujoco":', body)
        # A body the model welds into its parent must not be guessed at: it is
        # made negligible and reported, because the parent already carries it.
        self.assertIn("bodies_welded_in_model", body)
        self.assertIn("WELDED_BODY_MASS_KG", service + (ROOT / "src/humanoid_lab/simulators/isaac/masses.py").read_text())
        # The masses come from the pinned model, never from a table in the code.
        self.assertIn("load_mujoco_body_masses(path)", body)
        self.assertIn("set_masses(", body)
        self.assertNotIn("MUJOCO_BODY_MASS_KG", service)
        self.assertIn("self._align_body_masses()", service)

    def test_no_mass_table_is_duplicated_in_the_codebase(self) -> None:
        for path in (ROOT / "src").rglob("*.py"):
            with self.subTest(path=path.name):
                self.assertNotIn("9.598", path.read_text())


if __name__ == "__main__":
    unittest.main()
