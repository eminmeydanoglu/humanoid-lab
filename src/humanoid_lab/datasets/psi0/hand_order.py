"""Right-hand joint-order contract for the Psi0 SONIC packs.

The released Psi0 SONIC post-trained checkpoints (``postpre.sonic1.0...`` and
the v1.1 init ``postpre.sonic1.1.unifolm.2609181726.40k``) were post-trained
with **both** hands in the order ``thumb, middle, index``: the right hand
mirrors the left exactly.  Upstream pins that in
``scripts/train/psi0/posttrain-psix-unifolm-g1-sonic1.{0,1}.sh`` (slot 39 must
be ``right_hand_middle_0_joint``, "right-hand swap not applied") and in
``scripts/data/merge_posttrain_sonic.py`` ("An earlier revision had the right
hand as index-before-middle; that was wrong and showed up in viser as the
right index/middle fingers moving as each other").

The Unitree Dex3 collection - and therefore the frozen 78D SONIC corpus and
the produced ``psi0-unitree-dex3-sonic-v1`` pack - stores the right hand in
the **hardware / SONIC motor order** ``thumb, index, middle`` (the left hand
is ``thumb, middle, index`` and agrees with the checkpoints).  Handing a
hardware-order pack to a v1.1-init run without an explicit decision would
silently train two swapped right-hand channels.  This module makes that
impossible:

* the pack must declare its hand joint names, and the right-hand block must be
  one of the two known orders;
* the requested mapping (``none`` / ``hardware2checkpoint``) must match the
  declared order - anything else fails closed with the exact knob to set;
* the mapping itself is expressed through Psi0's own slice-capable repack /
  statistics keys (:func:`action_keys`, :func:`state_keys`), so no dataset
  file is written and the mask field passes through unchanged (the pack's
  ``action.mask`` is per-frame uniform over the 78 supervised channels).

Nothing here imports Psi0, torch or numpy.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .contract import BODY_TOKEN_KEY, HAND_DIM, HAND_KEY, STATE_DIM, STATE_KEY

#: Left hand, identical in every order in play (SONIC motor order, the frozen
#: Unitree collection and both released post-trained checkpoints).
LEFT_HAND_ORDER: tuple[str, ...] = (
    "thumb_0",
    "thumb_1",
    "thumb_2",
    "middle_0",
    "middle_1",
    "index_0",
    "index_1",
)

#: Right hand as the released Psi0 SONIC checkpoints were post-trained: the
#: mirror of the left hand.
CHECKPOINT_RIGHT_HAND_ORDER: tuple[str, ...] = LEFT_HAND_ORDER

#: Right hand as recorded by the Unitree Dex3 collection, as carried by the
#: frozen 78D SONIC corpus and the produced v1 pack, and as the SONIC motor
#: interface (``action[64:78]`` written straight to motor IDs 0..6) expects.
HARDWARE_RIGHT_HAND_ORDER: tuple[str, ...] = (
    "thumb_0",
    "thumb_1",
    "thumb_2",
    "index_0",
    "index_1",
    "middle_0",
    "middle_1",
)

#: Classification of a pack's declared right-hand order.
CHECKPOINT = "checkpoint"
HARDWARE = "hardware"

#: Mapping modes: what the training run does with the right hand.
#:
#: ``none``                  the pack already stores the checkpoint order.
#: ``hardware2checkpoint``   the pack stores the SONIC hardware order; permute
#:                           it to the checkpoint order at load time.
MAP_NONE = "none"
MAP_HARDWARE_TO_CHECKPOINT = "hardware2checkpoint"
MAP_MODES: tuple[str, ...] = (MAP_NONE, MAP_HARDWARE_TO_CHECKPOINT)


class HandOrderError(ValueError):
    """The pack's declared hand order and the requested mapping disagree."""


def _block(side: str, order: Sequence[str]) -> tuple[str, ...]:
    return tuple(f"{side}_hand_{name}_joint" for name in order)


LEFT_HAND_NAMES: tuple[str, ...] = _block("left", LEFT_HAND_ORDER)
CHECKPOINT_HAND_NAMES: tuple[str, ...] = LEFT_HAND_NAMES + _block("right", CHECKPOINT_RIGHT_HAND_ORDER)
HARDWARE_HAND_NAMES: tuple[str, ...] = LEFT_HAND_NAMES + _block("right", HARDWARE_RIGHT_HAND_ORDER)

#: Right-hand blocks alone, used to classify a pack that declares the two
#: groups separately.
CHECKPOINT_RIGHT_NAMES: tuple[str, ...] = _block("right", CHECKPOINT_RIGHT_HAND_ORDER)
HARDWARE_RIGHT_NAMES: tuple[str, ...] = _block("right", HARDWARE_RIGHT_HAND_ORDER)

#: Permutation of a 14D hand block (left 0:7, right 7:14) from hardware order
#: to checkpoint order.  The swap is an involution, so the same tuple maps
#: checkpoint order back to hardware order.
HAND14_HARDWARE_TO_CHECKPOINT: tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 13, 10, 11)

#: The same swap inside the 43D state (legs+waist 0:15, arms 15:29, left hand
#: 29:36, right hand 36:43); also an involution.
STATE43_HARDWARE_TO_CHECKPOINT: tuple[int, ...] = tuple(range(36)) + (36, 37, 38, 41, 42, 39, 40)

#: Psi0 repack / statistics keys for the mapped action: token, then left hand,
#: then the right hand reordered thumb/middle/index.  ``parse_modality_key``
#: concatenates keys in the order given, so these four slices are the whole
#: load-time conversion.
_ACTION_SPLIT_KEYS: tuple[str, ...] = (
    BODY_TOKEN_KEY,
    "action[0:7]",
    "action[7:10]",
    "action[12:14]",
    "action[10:12]",
)
#: Psi0 repack / statistics keys for the mapped 43D state: body + left hand
#: unchanged, right hand reordered.
_STATE_SPLIT_KEYS: tuple[str, ...] = (
    "observation.state[0:36]",
    "observation.state[36:39]",
    "observation.state[41:43]",
    "observation.state[39:41]",
)


def _require_map(mapping: str) -> str:
    if mapping not in MAP_MODES:
        raise HandOrderError(f"unknown right-hand map {mapping!r}; use one of {MAP_MODES}")
    return mapping


def classify_hand_order(names: Sequence[str]) -> str | None:
    """Classify a full 14D hand block; ``None`` means an order we do not know."""
    declared = tuple(names)
    if declared == CHECKPOINT_HAND_NAMES:
        return CHECKPOINT
    if declared == HARDWARE_HAND_NAMES:
        return HARDWARE
    return None


def classify_right_hand(names: Sequence[str]) -> str | None:
    """Classify one 7D right-hand block; ``None`` means an order we do not know."""
    declared = tuple(names)
    if declared == CHECKPOINT_RIGHT_NAMES:
        return CHECKPOINT
    if declared == HARDWARE_RIGHT_NAMES:
        return HARDWARE
    return None


def _flat_names(feature: Mapping[str, Any] | None) -> tuple[str, ...] | None:
    """LeRobot stores names either flat or as one nested list; accept both."""
    if not isinstance(feature, Mapping):
        return None
    names = feature.get("names")
    if not isinstance(names, (list, tuple)) or not names:
        return None
    first = names[0]
    if isinstance(first, (list, tuple)):
        names = first
    return tuple(str(name) for name in names)


@dataclass(frozen=True)
class SplitHandOrder:
    """What one split's ``meta/info.json`` declares about the hands."""

    repo_dir: Path
    state: str
    action: str
    state_right: tuple[str, ...]
    action_right: tuple[str, ...]

    def summary(self) -> str:
        return (
            f"{self.repo_dir.name}: state right hand {self.state}, "
            f"action right hand {self.action}"
        )


def read_split_hand_order(repo_dir: Path | str) -> SplitHandOrder:
    """Read and classify one split's declared hand order.  Fails closed.

    The names are the pack's own statement of what each column means; a pack
    that does not state them cannot be checked, and the v1.1 init contract
    refuses to guess.
    """
    repo_dir = Path(repo_dir)
    path = repo_dir / "meta/info.json"
    try:
        info = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise HandOrderError(f"hand-order contract cannot be read: {path} is missing") from error
    except json.JSONDecodeError as error:
        raise HandOrderError(f"hand-order contract cannot be read: {path}: {error}") from error

    features = info.get("features")
    if not isinstance(features, Mapping):
        raise HandOrderError(f"{path} has no features table to read the hand order from")

    state_names = _flat_names(features.get(STATE_KEY))
    action_names = _flat_names(features.get(HAND_KEY))
    if state_names is None:
        raise HandOrderError(
            f"{path} declares no {STATE_KEY!r} joint names; the right-hand order cannot be "
            "verified from the pack alone and the v1.1 entrypoint refuses to guess. Rebuild "
            f"the pack with the {STATE_DIM} state names, or restore them in meta/info.json."
        )
    if action_names is None:
        raise HandOrderError(
            f"{path} declares no {HAND_KEY!r} joint names; the right-hand order cannot be "
            "verified from the pack alone and the v1.1 entrypoint refuses to guess. Rebuild "
            f"the pack with the {HAND_DIM} hand names, or restore them in meta/info.json."
        )
    if len(state_names) != STATE_DIM:
        raise HandOrderError(f"{path}: {STATE_KEY!r} declares {len(state_names)} names, expected {STATE_DIM}")
    if len(action_names) != HAND_DIM:
        raise HandOrderError(f"{path}: {HAND_KEY!r} declares {len(action_names)} names, expected {HAND_DIM}")

    if state_names[29:36] != LEFT_HAND_NAMES:
        raise HandOrderError(
            f"{path}: state channels 29:36 are not the canonical left hand "
            f"{LEFT_HAND_NAMES}; the left hand must be unchanged in every supported layout"
        )
    if action_names[:7] != LEFT_HAND_NAMES:
        raise HandOrderError(
            f"{path}: action channels 0:7 are not the canonical left hand "
            f"{LEFT_HAND_NAMES}; the left hand must be unchanged in every supported layout"
        )

    state_right = tuple(state_names[36:43])
    action_right = tuple(action_names[7:14])
    state_order = classify_right_hand(state_right)
    action_order = classify_right_hand(action_right)
    if state_order is None:
        raise HandOrderError(
            f"{path}: right-hand state channels are {state_right}; the v1.1 entrypoint only "
            f"understands the checkpoint mirror order {CHECKPOINT_RIGHT_NAMES} and the SONIC "
            f"hardware order {HARDWARE_RIGHT_NAMES}"
        )
    if action_order is None:
        raise HandOrderError(
            f"{path}: right-hand action channels are {action_right}; the v1.1 entrypoint only "
            f"understands the checkpoint mirror order {CHECKPOINT_RIGHT_NAMES} and the SONIC "
            f"hardware order {HARDWARE_RIGHT_NAMES}"
        )
    return SplitHandOrder(repo_dir=repo_dir, state=state_order, action=action_order,
                          state_right=state_right, action_right=action_right)


def plan_mapping(state_order: str, action_order: str, requested: str) -> str:
    """Reconcile the pack's declared order with the explicitly requested mapping."""
    _require_map(requested)
    if state_order != action_order:
        raise HandOrderError(
            f"the pack declares right-hand state order {state_order!r} but right-hand action "
            f"order {action_order!r}; a pack that disagrees with itself cannot be mapped "
            "coherently, so both fields must be rebuilt in one order"
        )
    if requested == MAP_NONE:
        if state_order == CHECKPOINT:
            return MAP_NONE
        raise HandOrderError(
            "the pack stores the right hand in the SONIC hardware order (thumb, index, middle) "
            "while the v1.1 init was post-trained on the mirrored order (thumb, middle, index); "
            "either rebuild the pack in the init order or rerun with "
            f"RIGHT_HAND_MAP={MAP_HARDWARE_TO_CHECKPOINT} to apply the documented load-time "
            "permutation"
        )
    if state_order == HARDWARE:
        return MAP_HARDWARE_TO_CHECKPOINT
    raise HandOrderError(
        f"RIGHT_HAND_MAP={MAP_HARDWARE_TO_CHECKPOINT} was requested but the pack already "
        "declares the v1.1-init mirror order; applying the permutation again would double-swap "
        f"the right index and middle fingers. Set RIGHT_HAND_MAP={MAP_NONE}"
    )


def action_keys(mapping: str) -> list[str]:
    """Psi0 ``repack.action_keys`` / ``field.stat_action_keys`` for this mapping."""
    if _require_map(mapping) == MAP_NONE:
        return [BODY_TOKEN_KEY, HAND_KEY]
    return list(_ACTION_SPLIT_KEYS)


def state_keys(mapping: str) -> list[str]:
    """Psi0 ``repack.state_keys`` / ``field.stat_state_keys`` for this mapping."""
    if _require_map(mapping) == MAP_NONE:
        return [STATE_KEY]
    return list(_STATE_SPLIT_KEYS)


def apply_hand14(values: Any, mapping: str) -> Any:
    """Permute the last axis of a 14D-hand array into the training order.

    ``values`` must support advanced indexing on the last axis (numpy array or
    torch tensor); ``none`` returns the input untouched.
    """
    if _require_map(mapping) == MAP_NONE:
        return values
    return values[..., list(HAND14_HARDWARE_TO_CHECKPOINT)]


def apply_state43(values: Any, mapping: str) -> Any:
    """Permute the last axis of a 43D state array into the training order."""
    if _require_map(mapping) == MAP_NONE:
        return values
    return values[..., list(STATE43_HARDWARE_TO_CHECKPOINT)]


__all__ = [
    "CHECKPOINT",
    "CHECKPOINT_HAND_NAMES",
    "CHECKPOINT_RIGHT_HAND_ORDER",
    "CHECKPOINT_RIGHT_NAMES",
    "HAND14_HARDWARE_TO_CHECKPOINT",
    "HARDWARE",
    "HARDWARE_HAND_NAMES",
    "HARDWARE_RIGHT_HAND_ORDER",
    "HARDWARE_RIGHT_NAMES",
    "HandOrderError",
    "LEFT_HAND_NAMES",
    "LEFT_HAND_ORDER",
    "MAP_HARDWARE_TO_CHECKPOINT",
    "MAP_MODES",
    "MAP_NONE",
    "STATE43_HARDWARE_TO_CHECKPOINT",
    "SplitHandOrder",
    "action_keys",
    "apply_hand14",
    "apply_state43",
    "classify_hand_order",
    "classify_right_hand",
    "plan_mapping",
    "read_split_hand_order",
    "state_keys",
]
