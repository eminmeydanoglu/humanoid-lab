"""Thin Ψ₀–SONIC bridge: Prompt + camera + ``g1_debug`` state -> Protocol v4 pose.

The bridge owns exactly one control loop and one session state machine.  It does
not import Isaac, does not import the SONIC X controller, and never resizes the
policy camera: the Ψ₀ server applies the checkpoint's ``240x320`` transform.

Module map:

* :mod:`~humanoid_lab.psi0_bridge.prompt`      -- the one canonical instruction;
* :mod:`~humanoid_lab.psi0_bridge.state`       -- ``g1_debug`` -> raw 43D state;
* :mod:`~humanoid_lab.psi0_bridge.state_source`-- SUB on ``:5557``;
* :mod:`~humanoid_lab.psi0_bridge.camera`      -- REQ ``get_frame`` on ``:5558``;
* :mod:`~humanoid_lab.psi0_bridge.actions`     -- 78/80D action -> Protocol v4;
* :mod:`~humanoid_lab.psi0_bridge.publisher`   -- PUB ``pose`` on ``:5556``;
* :mod:`~humanoid_lab.psi0_bridge.psi0_client` -- ``/info`` + WebSocket client;
* :mod:`~humanoid_lab.psi0_bridge.monitor`     -- one ZMQ thread, cached snapshots;
* :mod:`~humanoid_lab.psi0_bridge.session`     -- IDLE/RUNNING/STOPPED/ERROR;
* :mod:`~humanoid_lab.psi0_bridge.reset_client`-- Isaac reset REQ on ``:5559``;
* :mod:`~humanoid_lab.psi0_bridge.app`         -- FastAPI UI + control API.
"""

from __future__ import annotations

from .prompt import CANONICAL_PROMPT, PromptMismatch, require_canonical

__all__ = ["CANONICAL_PROMPT", "PromptMismatch", "require_canonical"]
