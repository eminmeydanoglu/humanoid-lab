"""The 43D raw state the bridge sends, tied to the repo's own dataset contract.

The layout is fixed by ``configs/datasets/psi0/unitree_dex3_sonic_v1.yaml``
(``state.joint_layout``: legs+waist [0,15], arms [15,29], hands [29,43]) and the
Dex3 motor counts come from :mod:`humanoid_lab.datasets.sonic.joints`.  The
``g1_debug`` field names are the ones the pinned SONIC state logger publishes
(``body_q``, ``left_hand_q``, ``right_hand_q``, 29 + 7 + 7); nothing here touches
the SONIC deployment itself.
"""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import yaml

from humanoid_lab.datasets.sonic.joints import LEFT_HAND_ORDER, RIGHT_HAND_ORDER
from humanoid_lab.psi0_bridge.contracts import BODY_DIM, HAND_DIM, RAW_STATE_DIM
from humanoid_lab.psi0_bridge.state import BODY_FIELD, LEFT_HAND_FIELD, RIGHT_HAND_FIELD, build_raw_state

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "datasets" / "psi0" / "unitree_dex3_sonic_v1.yaml"


class DatasetContractAlignmentTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))

    def test_state_dim_matches_the_dataset_contract(self) -> None:
        state = self.config["state"]
        self.assertEqual(state["dim"], RAW_STATE_DIM)
        layout = state["joint_layout"]
        self.assertEqual(layout["legs_and_waist"], [0, 15])
        self.assertEqual(layout["arms"], [15, 29])
        self.assertEqual(layout["hands"], [29, 43])
        # body_q covers legs+waist then arms; the hand block is both Dex3 hands.
        self.assertEqual(layout["arms"][1] - layout["legs_and_waist"][0], BODY_DIM)
        self.assertEqual(layout["hands"][1] - layout["hands"][0], 2 * HAND_DIM)

    def test_hand_width_matches_the_canonical_dex3_motor_count(self) -> None:
        self.assertEqual(len(LEFT_HAND_ORDER), HAND_DIM)
        self.assertEqual(len(RIGHT_HAND_ORDER), HAND_DIM)

    def test_g1_debug_fields_are_the_ones_the_bridge_consumes(self) -> None:
        self.assertEqual((BODY_FIELD, LEFT_HAND_FIELD, RIGHT_HAND_FIELD),
                         ("body_q", "left_hand_q", "right_hand_q"))

    def test_raw_state_orders_body_then_left_then_right(self) -> None:
        payload = {
            BODY_FIELD: np.arange(BODY_DIM, dtype=np.float32) + 1,
            LEFT_HAND_FIELD: np.arange(HAND_DIM, dtype=np.float32) + 100,
            RIGHT_HAND_FIELD: np.arange(HAND_DIM, dtype=np.float32) + 200,
        }
        state = build_raw_state(payload)
        np.testing.assert_array_equal(state[:BODY_DIM], payload[BODY_FIELD])
        np.testing.assert_array_equal(state[BODY_DIM:BODY_DIM + HAND_DIM], payload[LEFT_HAND_FIELD])
        np.testing.assert_array_equal(state[BODY_DIM + HAND_DIM:], payload[RIGHT_HAND_FIELD])


if __name__ == "__main__":
    unittest.main()
