"""Build the raw 43D proprio condition from a SONIC ``g1_debug`` payload.

Canonical training order (``configs/datasets/psi0/unitree_dex3_sonic_v1.yaml``,
``state.joint_layout``): legs+waist (15) then arms (14) form the 29 body joints
``body_q``; the 14 Dex3 motor positions follow as ``left_hand_q`` (7) then
``right_hand_q`` (7).

The ``g1_debug`` topic is the only state source this bridge reads, and it is
msgpack-decoded elsewhere (``ZMQStateSubscriber`` strips the topic prefix and
unpacks).  This module works on the already-decoded mapping so it stays free of
zmq/msgpack imports and can be unit-tested directly.

If a payload cannot supply all 29 body + 14 hand values, :func:`build_raw_state`
raises: the bridge reports "state not ready" instead of inventing zeros for the
missing joints.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from .contracts import BODY_DIM, HAND_DIM, RAW_STATE_DIM

BODY_FIELD = "body_q"
LEFT_HAND_FIELD = "left_hand_q"
RIGHT_HAND_FIELD = "right_hand_q"

# Payload fields the SONIC g1_debug message carries that this bridge does not
# consume (declared so an operator can see why they are ignored).
UNUSED_FIELDS = ("body_dq", "base_quat", "actions", "left_hand_dq", "right_hand_dq")


class StateContractError(ValueError):
    """A ``g1_debug`` payload cannot produce the required 43D raw state."""


def _joint_vector(payload: Mapping[str, Any], field: str, dim: int) -> np.ndarray:
    if field not in payload:
        raise StateContractError(f"g1_debug payload is missing field {field!r}")
    values = np.asarray(payload[field], dtype=np.float32).reshape(-1)
    if values.shape != (dim,):
        raise StateContractError(
            f"g1_debug field {field!r} has {values.shape[0]} values, expected {dim}"
        )
    if not np.isfinite(values).all():
        raise StateContractError(f"g1_debug field {field!r} contains NaN or Inf")
    return values


def build_raw_state(payload: Any) -> np.ndarray:
    """Return the 43D raw state ``[body_q(29) | left_hand_q(7) | right_hand_q(7)]``.

    Raises :class:`StateContractError` when the payload cannot supply the full
    29 body + 14 hand layout, or carries a different hand split.
    """
    if not isinstance(payload, Mapping):
        raise StateContractError(
            f"g1_debug payload must be a mapping, got {type(payload).__name__}"
        )
    body = _joint_vector(payload, BODY_FIELD, BODY_DIM)
    left = _joint_vector(payload, LEFT_HAND_FIELD, HAND_DIM)
    right = _joint_vector(payload, RIGHT_HAND_FIELD, HAND_DIM)
    state = np.concatenate([body, left, right]).astype(np.float32, copy=False)
    if state.shape != (RAW_STATE_DIM,):  # defensive: the concat is already exact
        raise StateContractError(f"assembled state has shape {state.shape}")
    return state
