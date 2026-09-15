"""Fail-closed, name-based body and Dex3 permutations."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from humanoid_lab.controllers.sonic import BODY_JOINT_ORDER, SONIC_REFERENCE_JOINT_ORDER

LEFT_HAND_ORDER = (
    "thumb_0_joint", "thumb_1_joint", "thumb_2_joint", "middle_0_joint",
    "middle_1_joint", "index_0_joint", "index_1_joint",
)
RIGHT_HAND_ORDER = (
    "thumb_0_joint", "thumb_1_joint", "thumb_2_joint", "index_0_joint",
    "index_1_joint", "middle_0_joint", "middle_1_joint",
)


def permutation(source_names: Sequence[str], target_names: Sequence[str]) -> tuple[int, ...]:
    if len(set(source_names)) != len(source_names):
        raise ValueError("source joint names contain duplicates")
    missing = [name for name in target_names if name not in source_names]
    if missing:
        raise ValueError(f"source is missing joints: {missing}")
    return tuple(source_names.index(name) for name in target_names)


def reorder(values: np.ndarray, source_names: Sequence[str], target_names: Sequence[str]) -> np.ndarray:
    array = np.asarray(values)
    if array.shape[-1] != len(source_names):
        raise ValueError("value width does not match source joint names")
    return array[..., permutation(source_names, target_names)]


def canonical_body(values: np.ndarray, source_names: Sequence[str]) -> np.ndarray:
    return reorder(values, source_names, BODY_JOINT_ORDER)


def sonic_reference_body(values: np.ndarray, source_names: Sequence[str]) -> np.ndarray:
    """Return the 29 body joints in SONIC's official reference-motion order."""
    return reorder(values, source_names, SONIC_REFERENCE_JOINT_ORDER)


def hand_order(side: str) -> tuple[str, ...]:
    """Canonical Dex3 motor order of one hand, as `left_hand_<name>`."""
    if side == "left":
        order = LEFT_HAND_ORDER
    elif side == "right":
        order = RIGHT_HAND_ORDER
    else:
        raise ValueError(f"unknown hand side {side!r}")
    return tuple(f"{side}_hand_{name}" for name in order)


def canonical_hand(values: np.ndarray, source_names: Sequence[str], side: str) -> np.ndarray:
    """Reorder one hand from source motor order to the canonical Dex3 order."""
    return reorder(values, source_names, hand_order(side))
