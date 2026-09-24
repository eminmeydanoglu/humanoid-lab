"""Action adapter: Ψ₀ model action -> Protocol v4 ``pose`` message.

The model emits either

* 78D = 64D SONIC motion token + 7D left Dex3 + 7D right Dex3, or
* 80D = the same 78D plus a trailing 2D neck block.

The 80D neck block has no field in the ``pose`` message.  Rather than truncate
to 78 silently, the adapter accepts an 80D action only when the neck block is
finite and inside :data:`~humanoid_lab.psi0_bridge.contracts.NECK_NOOP_TOLERANCE`
of zero (a genuine no-op), and otherwise raises so the session transitions to
ERROR and stops publishing.

That generic fail-closed default is wrong for this repo's own 80D pack:
``action.neck`` is unsupervised padding (``action.mask`` is zero on those two
columns for every frame), so the served model may emit arbitrary finite values
there. The SONIC evaluation launcher selects ``neck_policy="discard"`` by
default; the generic adapter retains ``error`` for unknown action contracts.
The discarded values are available in telemetry when enabled.

The first 64 dims are snapped onto the WBC's FSQ grid
(``[-0.625, 0.625]`` step ``0.0625``) before publishing, exactly like every live
SONIC client; the recorded dataset tokens already sit on that grid.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from humanoid_lab.datasets.sonic.protocol_v4 import pack_latent_action_message

from .contracts import (
    ACTION_DIMS,
    HAND14_DIM,
    HAND_DIM,
    NECK_DIM,
    NECK_NOOP_TOLERANCE,
    TOKEN_DIM,
)

FSQ_MIN, FSQ_MAX, FSQ_STEP = -0.625, 0.625, 0.0625

#: How an out-of-tolerance 80D neck block is treated.
#:
#: ``"error"`` (default) keeps the fail-closed behaviour.  ``"discard"`` drops
#: the block after recording it, which is only correct for a pack whose neck
#: columns were never supervised (see the module docstring).
NECK_POLICIES = ("error", "discard")


class ActionAdapterError(ValueError):
    """The model action cannot be turned into a valid Protocol v4 message."""


def fsq_quantize(values: np.ndarray) -> np.ndarray:
    """Snap the token channel onto the WBC's FSQ grid."""
    array = np.asarray(values, dtype=np.float32)
    quantized = np.round(np.clip(array, FSQ_MIN, FSQ_MAX) / FSQ_STEP) * FSQ_STEP
    return np.clip(quantized, FSQ_MIN, FSQ_MAX).astype(np.float32)


def _flatten(action: np.ndarray) -> np.ndarray:
    array = np.asarray(action, dtype=np.float32)
    if array.ndim == 2 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 1:
        raise ActionAdapterError(
            f"action must be a flat vector (or 1xD), got shape {array.shape}"
        )
    if array.shape[0] not in ACTION_DIMS:
        raise ActionAdapterError(
            f"action width {array.shape[0]} is neither 78 nor 80; refusing to guess"
        )
    return array


@dataclass(frozen=True)
class AdaptedAction:
    """One model action split into its protocol fields."""

    token: np.ndarray      # (64,) FSQ-quantized motion token
    left_hand: np.ndarray  # (7,)
    right_hand: np.ndarray  # (7,)
    neck: np.ndarray | None  # (2,) only for 80D input; validated no-op or discarded
    frame_index: int
    source_dim: int
    #: The raw 80D neck block when ``neck_policy="discard"`` dropped a value that
    #: was outside the no-op tolerance; ``None`` for a 78D action, an in-tolerance
    #: no-op, or the fail-closed default.  Carried so a caller can record exactly
    #: what was thrown away instead of the drop being invisible.
    discarded_neck: np.ndarray | None = None

    @property
    def neck_discarded(self) -> bool:
        return self.discarded_neck is not None


class ActionAdapter:
    """Validates, quantizes and packs model actions with monotonic frame indices.

    ``expected_dim`` pins the width the served checkpoint declared in ``/info``;
    an action of any other width is refused instead of silently re-split (a
    server that advertises 80 and sends 78, or the reverse, is a contract
    violation).  ``None`` keeps the generic 78-or-80 behaviour.

    The frame sequence restarts at 0 on every new Start/Reset.  That is safe for
    the pinned consumer: SONIC's Protocol v4 reader
    (``gear_sonic_deploy/.../input_interface/zmq_endpoint_interface.hpp``) only
    decodes ``frame_index`` for its own debug log line and never requires it to
    be monotonic; each packed message carries a complete action.
    """

    def __init__(
        self,
        neck_tolerance: float = NECK_NOOP_TOLERANCE,
        expected_dim: Optional[int] = None,
        neck_policy: str = "error",
    ) -> None:
        if expected_dim is not None and expected_dim not in ACTION_DIMS:
            raise ActionAdapterError(
                f"expected_dim {expected_dim} is neither 78 nor 80"
            )
        if neck_policy not in NECK_POLICIES:
            raise ActionAdapterError(
                f"unknown neck policy {neck_policy!r}; expected one of {NECK_POLICIES}"
            )
        self.neck_tolerance = float(neck_tolerance)
        self.expected_dim = expected_dim
        self.neck_policy = neck_policy
        self._next_index = 0
        self._last_sent: Optional[int] = None
        self._neck_discards = 0
        #: The most recently packed action, so a caller can report what the
        #: adapter had to drop without re-deriving it from the raw vector.
        self.last_adapted: Optional[AdaptedAction] = None

    def reset(self) -> None:
        """Clear the frame sequence so a new session starts from index 0."""
        self._next_index = 0
        self._last_sent = None

    @property
    def next_frame_index(self) -> int:
        return self._next_index

    @property
    def last_frame_index(self) -> Optional[int]:
        return self._last_sent

    @property
    def neck_discards(self) -> int:
        """How many out-of-tolerance neck blocks this adapter discarded."""
        return self._neck_discards

    def adapt(self, action: np.ndarray, frame_index: Optional[int] = None) -> AdaptedAction:
        array = _flatten(action)
        if not np.isfinite(array).all():
            raise ActionAdapterError("action contains NaN or Inf")

        if self.expected_dim is not None and array.shape[0] != self.expected_dim:
            raise ActionAdapterError(
                f"/info declares action_dim {self.expected_dim} but the server sent "
                f"{array.shape[0]} values; refusing to re-split the action"
            )

        token = array[:TOKEN_DIM]
        hands = array[TOKEN_DIM:TOKEN_DIM + HAND14_DIM]
        neck: np.ndarray | None = None
        discarded: np.ndarray | None = None
        if array.shape[0] == 78 + NECK_DIM:
            raw_neck = array[TOKEN_DIM + HAND14_DIM:]
            if np.abs(raw_neck).max() > self.neck_tolerance:
                if self.neck_policy == "error":
                    raise ActionAdapterError(
                        "80D neck block is "
                        f"{raw_neck.tolist()} but Protocol v4 has no neck field; only a "
                        f"no-op (|neck| <= {self.neck_tolerance}) is accepted "
                        "(pass neck_policy='discard' only for a pack whose neck "
                        "columns are unsupervised padding)"
                    )
                # Opt-in discard: keep the evidence, drop the command.
                discarded = raw_neck.astype(np.float32, copy=True)
                self._neck_discards += 1
            else:
                neck = raw_neck.astype(np.float32, copy=True)

        if frame_index is None:
            index = self._next_index
        else:
            if isinstance(frame_index, bool) or not isinstance(frame_index, (int, np.integer)):
                raise ActionAdapterError(f"frame_index must be an int, got {frame_index!r}")
            index = int(frame_index)
        if index < 0:
            raise ActionAdapterError(f"frame_index must be >= 0, got {index}")
        if self._last_sent is not None and index <= self._last_sent:
            raise ActionAdapterError(
                f"frame_index {index} is not monotonic (last sent {self._last_sent})"
            )

        return AdaptedAction(
            token=fsq_quantize(token),
            left_hand=hands[:HAND_DIM].astype(np.float32, copy=True),
            right_hand=hands[HAND_DIM:].astype(np.float32, copy=True),
            neck=neck,
            frame_index=index,
            source_dim=int(array.shape[0]),
            discarded_neck=discarded,
        )

    def pack(self, action: np.ndarray, frame_index: Optional[int] = None) -> bytes:
        """Adapt ``action`` and pack it for the ``pose`` PUB socket.

        The frame index advances only after a successful pack, so a rejected
        action never consumes an index.
        """
        adapted = self.adapt(action, frame_index)
        payload = pack_latent_action_message(
            motion_token=adapted.token,
            frame_index=adapted.frame_index,
            left_hand_joints=adapted.left_hand,
            right_hand_joints=adapted.right_hand,
        )
        self._last_sent = adapted.frame_index
        self._next_index = adapted.frame_index + 1
        self.last_adapted = adapted
        return payload
