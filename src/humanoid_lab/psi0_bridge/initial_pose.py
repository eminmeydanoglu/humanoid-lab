"""Opt-in delivery of the VLA client's own initial-pose command over the settle.

GR00T's pre-policy pose belongs to NVIDIA's VLA client: on the ``i`` key its
``publish_initial_pose()`` sends exactly one protocol v4 ``pose`` message holding
``LATENT_INITIAL_MOTION_TOKEN`` (the checkpoint-specific standing token from
``gear_sonic.utils.inference.initial_poses``) with the Dex3 hands open, and the
SONIC deployment re-decodes that held token every tick until the policy's first
action replaces it.

That handshake races the client's own startup.  The bridge sends ``k``/``i`` on a
fire-and-forget PUB socket and retries only while the *deployment* reports no
state -- which on every Reset after the first is already false, because the
deployment is never restarted.  Both keys are then published about a second after
a freshly spawned client, before that client's keyboard subscriber has joined, and
are dropped: no initial-pose message is published, and the deployment keeps
holding whatever token it last had (rollout 1: the token delivered at the model
switch; later rollouts: the previous rollout's last *policy* action).

This module removes the race instead of timing it.  The client's own constant is
read from the pinned SONIC tree, packed with the client's own protocol v4 packer
(so the quantization and the wire format are the client's, not a reconstruction)
and published through the router at the deployment's control rate from the
beginning of every settle until ``Start``; the router's single source switch then
hands the port to the GR00T stream.  Nothing is invented and nothing is written
into the robot: it is the same command the client would have sent, sent on time.

Opt-in: without ``--initial-pose-handshake`` this module is never imported.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Callable, Optional

import numpy as np

from .warmstart import TokenStream

#: The deployment's control rate, which is also the VLA client's own
#: ``action_publish_rate`` (50 Hz): the handshake drives the port at exactly the
#: cadence the client itself would, so the deployment's decoder sees no
#: difference between the two publishers.
DEFAULT_CONTROL_HZ = 50.0
TOKEN_DIM = 64
HAND_DIM = 7
#: Where the intended command comes from, and who shows it on the wire.
CLIENT_MODULE = "gear_sonic.utils.inference.initial_poses"
CLIENT_CONSTANT = "LATENT_INITIAL_MOTION_TOKEN"
CLIENT_PACKER = "gear_sonic.utils.teleop.zmq.zmq_planner_sender.pack_pose_message"


class InitialPoseError(RuntimeError):
    """The intended initial-pose command cannot be used; meant for the operator."""


@dataclass(frozen=True)
class InitialPoseCommand:
    """The client's initial-pose command, exactly as the client would send it."""

    token: np.ndarray
    left_hand_joints: np.ndarray
    right_hand_joints: np.ndarray
    control_hz: float = DEFAULT_CONTROL_HZ
    source: dict[str, Any] | None = None

    @property
    def token_sha256(self) -> str:
        return hashlib.sha256(np.asarray(self.token, dtype=np.float32).tobytes()).hexdigest()

    def summary(self) -> dict[str, Any]:
        return {
            "module": CLIENT_MODULE,
            "constant": CLIENT_CONSTANT,
            "token_dim": int(self.token.shape[0]),
            "token_sha256_f32": self.token_sha256,
            "token_head": [float(value) for value in self.token[:4]],
            "hand_joints": "open (zeros): the client's own default",
            "control_hz": float(self.control_hz),
            "packed_by": CLIENT_PACKER,
            "sent_by": "gear_sonic/scripts/run_vla_inference.py publish_initial_pose() ('i' key)",
            **dict(self.source or {}),
        }

    def stream(self) -> TokenStream:
        """The command as a one-tick :class:`TokenStream`.

        A one-tick stream re-sent at the control rate is the hold: the deployment
        gets the initial-pose token every tick from the start of the settle until
        the policy takes the port, which is the cadence a live client drives it
        with while it streams.
        """
        return TokenStream(
            tokens=np.asarray(self.token, dtype=np.float64).reshape(1, TOKEN_DIM),
            left_hand_joints=np.asarray(self.left_hand_joints, dtype=np.float64).reshape(1, HAND_DIM),
            right_hand_joints=np.asarray(self.right_hand_joints, dtype=np.float64).reshape(1, HAND_DIM),
            control_hz=float(self.control_hz),
            source=self.summary(),
        )


def _client_constant() -> np.ndarray:
    """Read the constant from the pinned SONIC tree (only the container has it)."""
    import importlib

    module = importlib.import_module(CLIENT_MODULE)
    return np.asarray(getattr(module, CLIENT_CONSTANT), dtype=np.float32)


def load_initial_pose_command(
    *,
    control_hz: float = DEFAULT_CONTROL_HZ,
    provider: Optional[Callable[[], Any]] = None,
) -> InitialPoseCommand:
    """Build the intended initial-pose command from the client's own constant.

    ``provider`` exists for tests and for a caller that already holds the token
    (the bridge reads it from the pinned SONIC tree by default).  The token's
    shape, finiteness and the hand width are checked here, so a wrong or missing
    constant fails before a session starts rather than during a settle.
    """
    read = provider if provider is not None else _client_constant
    try:
        token = np.asarray(read(), dtype=np.float32).reshape(-1)
    except Exception as exc:  # noqa: BLE001 - reported, never raised into a session
        raise InitialPoseError(
            f"cannot read {CLIENT_CONSTANT} from {CLIENT_MODULE}: {type(exc).__name__}: {exc}"
        ) from exc
    if token.shape[0] != TOKEN_DIM:
        raise InitialPoseError(
            f"{CLIENT_MODULE}.{CLIENT_CONSTANT} must be {TOKEN_DIM}-D, got {token.shape[0]}"
        )
    if not np.isfinite(token).all():
        raise InitialPoseError(f"{CLIENT_MODULE}.{CLIENT_CONSTANT} contains a non-finite value")
    if not (float(control_hz) > 0.0) or not np.isfinite(control_hz):
        raise InitialPoseError(f"control_hz must be a positive number, got {control_hz!r}")
    return InitialPoseCommand(
        token=token,
        left_hand_joints=np.zeros(HAND_DIM, dtype=np.float32),
        right_hand_joints=np.zeros(HAND_DIM, dtype=np.float32),
        control_hz=float(control_hz),
        source={},
    )


def initial_pose_handshake(
    router: Any,
    *,
    command: InitialPoseCommand | None = None,
    telemetry: Any | None = None,
    packer: Optional[Callable[[np.ndarray, np.ndarray, np.ndarray, int], bytes]] = None,
    log: Callable[[str], None] = lambda message: None,
) -> Any:
    """The stream that delivers the intended initial-pose command.

    It reuses the demonstration warm start's publisher -- same cadence, same hold
    of the last tick, same arming/halting and recording -- with the client's own
    command as its single tick and its own ``initial_pose`` source label, so the
    hand-off at ``Start`` is the router's one atomic generation change.
    """
    from .warmstart import WarmStartStream

    command = command if command is not None else load_initial_pose_command()
    return WarmStartStream(
        command.stream(),
        router,
        telemetry=telemetry,
        packer=packer,
        source="initial_pose",
        publish=router.submit_initial_pose,
        log=log,
    )
