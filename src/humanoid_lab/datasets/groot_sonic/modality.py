"""The official ``unitree_g1_sonic`` modality boxes for this pack.

``meta/modality.json`` is the file GR00T reads to know which parquet column and
which slice of it backs every registered modality key. The boxes below are the
pre-registered stock schema, written once here and re-checked on every produced
split, so a slice typo fails in validation instead of silently feeding the model
a permuted state.

Two conventions the boxes follow:

* a state key whose channels live inside ``observation.state`` names them by
  position; a key stored in its own column (``projected_gravity``) addresses that
  column through ``original_key`` and indexes from 0;
* every action key is a separate field, so ``motion_token``,
  ``left_hand_joints`` and ``right_hand_joints`` each carry their own
  ``original_key`` and start at 0 of their own column.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .contract import (
    ACTION_MODEL_KEYS,
    GRAVITY_FIELD,
    HAND_DIM,
    INSTRUCTION_KEY,
    INSTRUCTION_ORIGINAL_KEY,
    LEFT_HAND_FIELD,
    MODEL_STATE_DIM,
    MOTION_TOKEN_DIM,
    MOTION_TOKEN_FIELD,
    RIGHT_HAND_FIELD,
    STATE_BLOCKS,
    STATE_DIM,
    STATE_MODEL_KEYS,
    ConversionConfig,
    EGO_VIEW_KEY,
)

#: The action keys in the order GR00T concatenates them: 64 + 7 + 7 = 78.
ACTION_FIELD_ORDER: tuple[tuple[str, str, int], ...] = (
    ("motion_token", MOTION_TOKEN_FIELD, MOTION_TOKEN_DIM),
    ("left_hand_joints", LEFT_HAND_FIELD, HAND_DIM),
    ("right_hand_joints", RIGHT_HAND_FIELD, HAND_DIM),
)

#: The projection keys whose own column holds 3 values.
STATE_FIELD_ORDER: tuple[tuple[str, str, int], ...] = (("projected_gravity", GRAVITY_FIELD, 3),)


class ModalityError(ValueError):
    """The declared modality box is not the stock ``unitree_g1_sonic`` schema."""


def modality_payload(config: ConversionConfig) -> dict[str, dict[str, Any]]:
    """The exact ``meta/modality.json`` object for both splits.

    The keys are listed in the registered order, not in storage order: the
    loader reads each key through its own slice, so the pack can keep the
    official exporter's block layout and still present arms-then-hands.
    """
    state: dict[str, Any] = {
        key: {"start": STATE_BLOCKS[key].start, "end": STATE_BLOCKS[key].stop}
        for key in STATE_MODEL_KEYS
        if key in STATE_BLOCKS
    }
    for key, original_key, width in STATE_FIELD_ORDER:
        state[key] = {"start": 0, "end": width, "original_key": original_key}
    action: dict[str, Any] = {
        key: {"start": 0, "end": width, "original_key": original_key}
        for key, original_key, width in ACTION_FIELD_ORDER
    }
    return {
        "state": state,
        "action": action,
        "video": {EGO_VIEW_KEY: {"original_key": config.camera_target_key}},
        "annotation": {INSTRUCTION_KEY: {"original_key": INSTRUCTION_ORIGINAL_KEY}},
    }


def assert_modality_payload(payload: Mapping[str, Any], config: ConversionConfig) -> None:
    """Fail closed unless a parsed ``modality.json`` is exactly the stock schema."""
    declared = tuple(payload)
    if declared != ("state", "action", "video", "annotation"):
        raise ModalityError(f"the modality boxes must be state/action/video/annotation, got {declared}")

    state = payload["state"]
    if tuple(state) != STATE_MODEL_KEYS:
        raise ModalityError(f"the state keys must be {STATE_MODEL_KEYS}, got {tuple(state)}")
    total = 0
    for key, block in STATE_BLOCKS.items():
        entry = state[key]
        if (int(entry["start"]), int(entry["end"])) != (block.start, block.stop):
            raise ModalityError(f"state.{key} must address {block.start}:{block.stop}, got {entry}")
        total += block.stop - block.start
    gravity = state["projected_gravity"]
    if gravity.get("original_key") != GRAVITY_FIELD:
        raise ModalityError(f"state.projected_gravity must read {GRAVITY_FIELD!r}, got {gravity!r}")
    if (int(gravity["start"]), int(gravity["end"])) != (0, 3):
        raise ModalityError("projected_gravity is a 3D column and indexes from its own 0")
    if total != STATE_DIM or total + 3 != MODEL_STATE_DIM:
        raise ModalityError(f"the state slices cover {total} stored channels, expected {STATE_DIM}")

    action = payload["action"]
    if tuple(action) != ACTION_MODEL_KEYS:
        raise ModalityError(f"the action keys must be {ACTION_MODEL_KEYS}, got {tuple(action)}")
    for key, original_key, width in ACTION_FIELD_ORDER:
        entry = action[key]
        if entry.get("original_key") != original_key:
            raise ModalityError(f"action.{key} must read {original_key!r}, got {entry.get('original_key')!r}")
        if (int(entry["start"]), int(entry["end"])) != (0, width):
            raise ModalityError(f"action.{key} is a {width}D column and indexes from its own 0")

    video = payload["video"]
    if tuple(video) != (EGO_VIEW_KEY,) or video[EGO_VIEW_KEY].get("original_key") != config.camera_target_key:
        raise ModalityError(f"the single video must be {EGO_VIEW_KEY!r} -> {config.camera_target_key!r}")
    annotation = payload["annotation"]
    if tuple(annotation) != (INSTRUCTION_KEY,) or annotation[INSTRUCTION_KEY].get("original_key") != (
        INSTRUCTION_ORIGINAL_KEY
    ):
        raise ModalityError(
            f"the language annotation must be {INSTRUCTION_KEY!r} via {INSTRUCTION_ORIGINAL_KEY!r}"
        )


def state_key_slices() -> dict[str, slice]:
    """The 43D state slices by key, in stored block order."""
    return {key: slice(block.start, block.stop) for key, block in STATE_BLOCKS.items()}


__all__ = [
    "ACTION_FIELD_ORDER",
    "STATE_FIELD_ORDER",
    "ModalityError",
    "assert_modality_payload",
    "modality_payload",
    "state_key_slices",
]
