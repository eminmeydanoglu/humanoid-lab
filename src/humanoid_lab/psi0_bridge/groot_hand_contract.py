"""Named GR00T left-hand mappings between live SONIC and model order."""

from __future__ import annotations

from typing import Any, MutableMapping

import numpy as np

LIVE_LEFT_HAND_NAMES = (
    "thumb_0", "thumb_1", "thumb_2", "middle_0", "middle_1", "index_0", "index_1",
)
TRAINING_OBSERVATION_LEFT_HAND_NAMES = (
    "index_0", "index_1", "middle_0", "middle_1", "thumb_0", "thumb_1", "thumb_2",
)
ACTION_LEFT_HAND_NAMES = (
    "thumb_0", "thumb_1", "thumb_2", "index_0", "index_1", "middle_0", "middle_1",
)
# This permutation is applied before RobotModel assignment, not to final policy state.
# RobotModel accepts thumb,index,middle and extracts index,middle,thumb.
LIVE_TO_ROBOT_MODEL_ACTUATED = (0, 1, 2, 5, 6, 3, 4)
ACTION_TO_LIVE = (0, 1, 2, 5, 6, 3, 4)
MODES = ("compatibility", "model-independent", "model-coupled")


def _require(values: Any) -> np.ndarray:
    array = np.asarray(values)
    if array.shape[-1:] != (7,):
        raise ValueError(f"left hand must end in 7 channels, got {array.shape}")
    return array


def left_hand_observation_to_robot_model_actuated(values: Any, mode: str) -> np.ndarray:
    """Return RobotModel actuated input for the desired final model ordering.

    RobotModel assigns thumb,index,middle, then extracts index,middle,thumb.
    """
    source = _require(values)
    if mode == "compatibility":
        result = source.copy()
        result[..., 5:7] = source[..., 3:5]
        return result
    if mode not in MODES:
        raise ValueError(f"unknown GR00T left-hand contract {mode!r}")
    result = source[..., LIVE_TO_ROBOT_MODEL_ACTUATED].copy()
    if mode == "model-coupled":
        # Retained as an explicit hardware-coupled alternative; it is not train-matched.
        result[..., 5:7] = result[..., 3:5]
    return result


def left_hand_action_to_live(action: MutableMapping[str, Any], mode: str) -> None:
    """Convert the model's named left-hand output to live SONIC order in place."""
    if mode == "compatibility":
        return
    if mode not in MODES:
        raise ValueError(f"unknown GR00T left-hand contract {mode!r}")
    key = "left_hand_joints" if "left_hand_joints" in action else "action.left_hand_joints"
    if key not in action:
        raise KeyError("GR00T action has no left_hand_joints field")
    values = _require(action[key])
    action[key] = values[..., ACTION_TO_LIVE].copy()
