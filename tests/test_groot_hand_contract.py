from __future__ import annotations

import unittest

import numpy as np

from humanoid_lab.psi0_bridge.groot_hand_contract import (
    ACTION_LEFT_HAND_NAMES,
    LIVE_LEFT_HAND_NAMES,
    TRAINING_OBSERVATION_LEFT_HAND_NAMES,
    left_hand_action_to_live,
    left_hand_observation_to_robot_model_actuated,
)


class GrootLeftHandContractTest(unittest.TestCase):
    def test_independent_observation_matches_robot_model_actuated_order(self) -> None:
        live = np.arange(7, dtype=np.float32)
        actuated = left_hand_observation_to_robot_model_actuated(live, "model-independent")
        self.assertEqual(tuple(actuated), (0, 1, 2, 5, 6, 3, 4))
        # Pinned SONIC RobotModel assigns thumb,index,middle and later extracts
        # index,middle,thumb for the policy state group.
        final_model = actuated[[3, 4, 5, 6, 0, 1, 2]]
        self.assertEqual(tuple(final_model), (5, 6, 3, 4, 0, 1, 2))
        self.assertEqual(
            TRAINING_OBSERVATION_LEFT_HAND_NAMES,
            tuple(LIVE_LEFT_HAND_NAMES[int(i)] for i in final_model),
        )

    def test_action_labels_map_independently_back_to_live_order(self) -> None:
        action = {"action.left_hand_joints": np.arange(7, dtype=np.float32)[None, None, :]}
        left_hand_action_to_live(action, "model-independent")
        np.testing.assert_array_equal(action["action.left_hand_joints"][0, 0], [0, 1, 2, 5, 6, 3, 4])
        self.assertEqual(
            LIVE_LEFT_HAND_NAMES,
            tuple(ACTION_LEFT_HAND_NAMES[i] for i in (0, 1, 2, 5, 6, 3, 4)),
        )

    def test_real_training_row_keeps_separately_measured_index_and_middle(self) -> None:
        # G1_Dex3_BlockStacking_Dataset episode 0, raw frame 504. Raw state is
        # thumb,middle,index; converted checkpoint input is index,middle,thumb.
        raw_live = np.array([
            -0.29605788, 0.77710658, 0.03198371,
            -0.22154425, -1.15354490,
            -0.29018801, -0.74398851,
        ], dtype=np.float32)
        expected_training = np.array([
            -0.29018801, -0.74398851,
            -0.22154425, -1.15354490,
            -0.29605788, 0.77710658, 0.03198371,
        ], dtype=np.float32)
        np.testing.assert_allclose(
            left_hand_observation_to_robot_model_actuated(raw_live, "model-independent")[
                [3, 4, 5, 6, 0, 1, 2]
            ],
            expected_training,
            rtol=0,
            atol=1e-7,
        )
        self.assertGreater(abs(float(expected_training[1] - expected_training[3])), 0.4)

    def test_coupled_option_is_not_the_training_observation(self) -> None:
        live = np.arange(7, dtype=np.float32)
        np.testing.assert_array_equal(
            left_hand_observation_to_robot_model_actuated(live, "model-coupled"),
            [0, 1, 2, 5, 6, 5, 6],
        )

    def test_compatibility_preserves_original_overwrite_and_output(self) -> None:
        live = np.arange(7, dtype=np.float32)
        np.testing.assert_array_equal(
            left_hand_observation_to_robot_model_actuated(live, "compatibility"),
            [0, 1, 2, 3, 4, 3, 4],
        )
        action = {"left_hand_joints": np.arange(7, dtype=np.float32)}
        left_hand_action_to_live(action, "compatibility")
        np.testing.assert_array_equal(action["left_hand_joints"], np.arange(7, dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
