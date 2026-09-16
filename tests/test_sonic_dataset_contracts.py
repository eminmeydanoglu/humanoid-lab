from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from humanoid_lab.controllers.sonic import BODY_JOINT_ORDER, SONIC_REFERENCE_JOINT_ORDER
from humanoid_lab.datasets.sonic.adapters.nvidia_apple_to_plate import require_hand_schema
from humanoid_lab.datasets.sonic.encoder_observation import build_g1_encoder_observation, quaternion_to_rotation_6d
from humanoid_lab.datasets.sonic.joints import LEFT_HAND_ORDER, RIGHT_HAND_ORDER, permutation, reorder
from humanoid_lab.datasets.sonic.pilot import PilotSpec, latest_pilot_dir, review_status
from humanoid_lab.datasets.sonic.provenance import assert_processed_destination
from humanoid_lab.datasets.sonic.protocol_v1 import LOOKAHEAD_FRAMES, pack_command_message, pack_pose_message, staged_windows
from humanoid_lab.datasets.sonic.quality import hand_range_decision, hand_range_report
from humanoid_lab.datasets.sonic.reference import compose_unitree_standing_reference
from humanoid_lab.datasets.sonic.schema import compose_action
from humanoid_lab.datasets.sonic.timeline import finite_difference, uniform_timeline


class SonicDatasetContractsTest(unittest.TestCase):
    @staticmethod
    def _compose(timestamps: np.ndarray, arms: np.ndarray, hands: np.ndarray):
        idle_timestamps = np.arange(101, dtype=np.float64) / 50.0
        idle_q = np.zeros((101, 29), dtype=np.float32)
        idle_q[:, :15] = idle_timestamps[:, None]
        idle_qd = np.zeros_like(idle_q)
        idle_qd[:, :15] = 1.0
        return compose_unitree_standing_reference(
            timestamps, arms, hands, hands,
            idle_timestamps=idle_timestamps,
            idle_joint_pos=idle_q,
            idle_joint_vel=idle_qd,
            idle_body_pos=np.column_stack((np.zeros(101), np.zeros(101), np.full(101, 0.8))),
            idle_body_quat_wxyz=np.tile([1, 0, 0, 0], (101, 1)),
        )

    def test_duplicate_and_missing_joints_fail_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "duplicates"):
            permutation(["a", "a"], ["a"])
        with self.assertRaisesRegex(ValueError, "missing"):
            permutation(["a"], ["a", "b"])

    def test_30_to_50_hz_and_stationary_tail(self) -> None:
        source = np.arange(31) / 30.0
        target = uniform_timeline(source)
        self.assertEqual(len(target), 51)
        qdot = finite_difference(np.ones((4, 2), dtype=np.float32))
        np.testing.assert_array_equal(qdot, 0)

    def test_reference_and_exact_encoder_layout(self) -> None:
        timestamps = np.arange(31) / 30.0
        arms = np.tile(np.arange(14, dtype=np.float32), (31, 1))
        hands = np.zeros((31, 7), dtype=np.float32)
        episode = self._compose(timestamps, arms, hands)
        # Lower body is the IDLE time series; arms are absolute replacements.
        np.testing.assert_allclose(episode.joint_pos[:, 0], episode.timestamps)
        arm_names = (
            "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
            "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint",
            "left_wrist_yaw_joint", "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
            "right_shoulder_yaw_joint", "right_elbow_joint", "right_wrist_roll_joint",
            "right_wrist_pitch_joint", "right_wrist_yaw_joint",
        )
        arm_ids = [SONIC_REFERENCE_JOINT_ORDER.index(name) for name in arm_names]
        np.testing.assert_allclose(episode.joint_pos[:, arm_ids], np.tile(arms[0], (len(episode.timestamps), 1)))
        np.testing.assert_allclose(episode.joint_vel[:, 0], 1.0)
        np.testing.assert_allclose(episode.joint_vel[:, arm_ids], 0.0)
        observation, clamp = build_g1_encoder_observation(episode)
        self.assertEqual(observation.shape, (51, 1751))
        np.testing.assert_array_equal(observation[:, :4], 0)
        np.testing.assert_array_equal(observation[:, 644:], 0)
        self.assertEqual(float(clamp[0]), 0.0)
        self.assertGreater(float(clamp[-1]), 0.0)

    def test_official_reference_and_hardware_orders_round_trip_by_name(self) -> None:
        self.assertNotEqual(BODY_JOINT_ORDER, SONIC_REFERENCE_JOINT_ORDER)
        marker = np.arange(29, dtype=np.float32)[None, :]
        hardware = reorder(marker, SONIC_REFERENCE_JOINT_ORDER, BODY_JOINT_ORDER)
        restored = reorder(hardware, BODY_JOINT_ORDER, SONIC_REFERENCE_JOINT_ORDER)
        np.testing.assert_array_equal(restored, marker)
        for name in SONIC_REFERENCE_JOINT_ORDER:
            self.assertEqual(
                hardware[0, BODY_JOINT_ORDER.index(name)],
                marker[0, SONIC_REFERENCE_JOINT_ORDER.index(name)],
                name,
            )

    def test_upright_rotation_6d(self) -> None:
        actual = quaternion_to_rotation_6d(np.array([[1, 0, 0, 0]], dtype=np.float32))
        np.testing.assert_array_equal(actual, [[1, 0, 0, 1, 0, 0]])

    def test_action_shape_and_apple_blocker(self) -> None:
        self.assertEqual(compose_action(np.zeros(64), np.zeros(7), np.zeros(7)).shape, (78,))
        with self.assertRaisesRegex(RuntimeError, "unresolved"):
            require_hand_schema()

    def test_raw_destination_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            source = Path(root) / "first_tur_ham"
            source.mkdir()
            with self.assertRaises(ValueError):
                assert_processed_destination(source, source / "bad")
            assert_processed_destination(source, Path(root) / "first_tur_processed")

    def test_protocol_v1_lookahead_and_tail_clamp(self) -> None:
        timestamps = np.arange(4) / 30.0
        episode = self._compose(timestamps, np.zeros((4, 14)), np.zeros((4, 7)))
        frame_index, window = next(staged_windows(episode))
        self.assertEqual(len(frame_index), LOOKAHEAD_FRAMES)
        np.testing.assert_array_equal(window.joint_pos[-1], episode.joint_pos[-1])
        self.assertTrue(pack_pose_message(window, frame_index).startswith(b"pose"))
        self.assertTrue(pack_command_message(start=True, stop=False, planner=False).startswith(b"command"))
        with self.assertRaisesRegex(ValueError, "contiguous"):
            pack_pose_message(window, frame_index * 2)


class BulkSourceNamingTest(unittest.TestCase):
    """A bulk episode must be filed under the dataset it actually came from.

    ``PilotSpec.pilot_name`` is derived from the dataset, so selecting a different
    bulk dataset and replacing only ``dataset`` filed one collection's episodes
    under another collection's name — and, because the pilot gate reads the latest
    directory of the *pilot* name, the mislabeled runs also shadowed the real
    pilot's review file.
    """

    CONFIG = Path(__file__).resolve().parents[1] / "configs/datasets/sonic/pilots.json"
    RAW = Path("/data/datasets/first_tur_ham")

    def _spec(self, dataset: str):
        return PilotSpec.bulk_from_config(
            self.CONFIG, "unitree", raw_root=self.RAW, dataset=dataset
        )

    def test_pilot_name_follows_the_selected_dataset(self) -> None:
        for dataset in (
            "unitree-g1-dex3/G1_Dex3_PickDoll_Dataset",
            "unitree-g1-dex3/G1_Dex3_ToastedBread_Dataset",
            "unitree-g1-dex3/G1_Dex3_ObjectPlacement_Dataset",
        ):
            spec = self._spec(dataset)
            self.assertEqual(spec.pilot_name, f"unitree_{Path(dataset).name}_ep000")
            self.assertEqual(spec.dataset, self.RAW / dataset)
            self.assertTrue(spec.pilot_name.endswith(Path(dataset).name + "_ep000"))

    def test_declared_bulk_datasets_are_all_addressable(self) -> None:
        declared = PilotSpec.bulk_datasets(self.CONFIG, "unitree")
        self.assertEqual(len(declared), 13)
        for dataset in declared:
            spec = self._spec(dataset)
            # The output directory must be unique per dataset, or two collections
            # would write into the same pilot tree.
            self.assertEqual(spec.pilot_name, f"unitree_{Path(dataset).name}_ep000")
        self.assertEqual(len({self._spec(d).pilot_name for d in declared}), len(declared))

    def test_unknown_dataset_is_refused(self) -> None:
        with self.assertRaisesRegex(ValueError, "does not declare bulk dataset"):
            self._spec("unitree-g1-dex3/Not_A_Real_Dataset")


class LatestPilotSelectionTest(unittest.TestCase):
    """The review gate must resolve the run a human actually reviewed.

    A later run that died before writing ``human_review.json`` used to become
    "the latest" and shadow the reviewed pilot, so the gate refused with
    "status is 'missing'" even though an accepted review existed.
    """

    def test_a_reviewless_newer_run_does_not_shadow_the_reviewed_pilot(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            runs = Path(root) / "pilots" / "unitree_demo_ep000"
            reviewed = runs / "20260101T000000Z"
            aborted = runs / "20260102T000000Z"
            for path in (reviewed, aborted):
                path.mkdir(parents=True)
            (reviewed / "human_review.json").write_text('{"status": "accepted"}', encoding="utf-8")
            (aborted / "run_manifest.json").write_text("{}", encoding="utf-8")

            chosen = latest_pilot_dir(Path(root), "unitree_demo_ep000")
            self.assertEqual(chosen, reviewed)
            self.assertEqual(review_status(chosen), "accepted")

    def test_newest_reviewed_run_wins_among_several(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            runs = Path(root) / "pilots" / "unitree_demo_ep000"
            older = runs / "20260101T000000Z"
            newer = runs / "20260103T000000Z"
            for path in (older, newer):
                path.mkdir(parents=True)
                (path / "human_review.json").write_text('{"status": "accepted"}', encoding="utf-8")
            self.assertEqual(latest_pilot_dir(Path(root), "unitree_demo_ep000"), newer)

    def test_missing_pilot_tree_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaises(FileNotFoundError):
                latest_pilot_dir(Path(root), "unitree_absent_ep000")


class HandRangeGateTest(unittest.TestCase):
    """The Dex3 hand channels must be range-checked, which they once were not.

    The body range check covers 29 joints and never looked at the 14 hand
    channels, so a hand trajectory could sit arbitrarily far outside the modelled
    Dex3 range while the episode still reported ``range.violation_count = 0``.
    """

    @staticmethod
    def _limits() -> dict[str, tuple[float, float]]:
        limits: dict[str, tuple[float, float]] = {}
        for side, order in (("left", LEFT_HAND_ORDER), ("right", RIGHT_HAND_ORDER)):
            for name in order:
                limits[f"{side}_hand_{name}"] = (-0.5, 0.5)
        return limits

    def test_clean_hands_report_no_violation(self) -> None:
        zeros = np.zeros((8, 7))
        report = hand_range_report(zeros, zeros.copy(), self._limits())
        self.assertEqual(report["violating_channels"], 0)
        self.assertEqual(report["max_excess_rad"], 0.0)
        self.assertEqual(hand_range_decision(report, allowed_channels=0)["result"], "PASS")

    def test_out_of_range_hand_is_reported_per_channel(self) -> None:
        left = np.zeros((8, 7))
        left[:, 3] = 1.25  # 0.75 rad past the synthetic upper limit
        report = hand_range_report(left, np.zeros((8, 7)), self._limits())
        self.assertEqual(report["violating_channels"], 1)
        self.assertAlmostEqual(report["max_excess_rad"], 0.75, places=6)
        self.assertIn("left_hand_middle_0_joint", report["sides"]["left"]["channels"])

    def test_revision_delta_passes_but_a_larger_error_fails(self) -> None:
        # 0.3491 rad is the systematic Unitree Dex3 revision delta (120 vs 100 deg);
        # the policy must tolerate exactly that and nothing worse.
        delta = 2.0944 - 1.7453
        at_delta = np.zeros((8, 7))
        at_delta[:, :2] = 0.5 + delta
        report = hand_range_report(at_delta, np.zeros((8, 7)), self._limits())
        self.assertEqual(
            hand_range_decision(report, allowed_channels=2, excess_tolerance_rad=0.35)["result"], "PASS"
        )
        worse = np.zeros((8, 7))
        worse[:, 0] = 0.5 + 0.9
        report = hand_range_report(worse, np.zeros((8, 7)), self._limits())
        self.assertEqual(
            hand_range_decision(report, allowed_channels=2, excess_tolerance_rad=0.35)["result"], "FAIL"
        )

    def test_channel_budget_is_enforced(self) -> None:
        many = np.zeros((8, 7))
        many[:, :5] = 0.5 + 0.2
        report = hand_range_report(many, np.zeros((8, 7)), self._limits())
        # Within the declared excess tolerance, only the channel budget decides.
        self.assertEqual(
            hand_range_decision(report, allowed_channels=4, excess_tolerance_rad=0.25)["result"], "FAIL"
        )
        self.assertEqual(
            hand_range_decision(report, allowed_channels=5, excess_tolerance_rad=0.25)["result"], "PASS"
        )
        # The budget is not a blank cheque: the same 5 channels fail when they
        # leave the declared excess tolerance.
        self.assertEqual(
            hand_range_decision(report, allowed_channels=5, excess_tolerance_rad=0.05)["result"], "FAIL"
        )

    def test_permutation_cannot_explain_the_violation(self) -> None:
        # A wrong hand order is the other candidate explanation for values outside
        # the modelled range.  Assigning one channel far outside and the rest
        # inside must still fail, so the gate cannot be satisfied by relabelling.
        left = np.zeros((8, 7))
        left[:, 0] = 3.0
        report = hand_range_report(left, np.zeros((8, 7)), self._limits())
        self.assertEqual(hand_range_decision(report, allowed_channels=0)["result"], "FAIL")


if __name__ == "__main__":
    unittest.main()
