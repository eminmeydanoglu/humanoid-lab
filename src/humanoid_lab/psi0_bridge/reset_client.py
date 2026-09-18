"""Narrow client for the Isaac simulator reset endpoint.

The simulation is owned by the Isaac service; the bridge never imports Isaac and
never touches the physics loop.  The Isaac service exposes its reset as a ZMQ
REP control endpoint (``ResetControlEndpoint`` in the Isaac camera service): one
``b"reset"`` request queues the reset on the simulation loop, which answers
``b"reset_queued"`` once it accepted the work or ``b"reset_refused"`` when no
loop is consuming it.  The endpoint is configurable because the port belongs to
the Isaac side.
"""

from __future__ import annotations

import os

DEFAULT_RESET_ENDPOINT = os.environ.get("ISAAC_CONTROL_ENDPOINT", "tcp://localhost:5559")

RESET_REQUEST = b"reset"
RESET_QUEUED = b"reset_queued"
RESET_REFUSED = b"reset_refused"
ERROR_PREFIX = b"error:"


class ResetError(RuntimeError):
    """The simulator reset request was refused or unreachable."""


class IsaacResetClient:
    def __init__(self, endpoint: str = DEFAULT_RESET_ENDPOINT, *, timeout_ms: int = 2000) -> None:
        self.endpoint = endpoint
        self.timeout_ms = int(timeout_ms)

    def request_reset(self) -> None:
        """Send one ``reset`` and require an accepted reply.

        Owns its own ZMQ context and closes it, so the socket never outlives the
        call and cannot leak into another thread.
        """
        import zmq

        context = zmq.Context()
        socket = context.socket(zmq.REQ)
        socket.setsockopt(zmq.LINGER, 0)
        try:
            socket.connect(self.endpoint)
            socket.send(RESET_REQUEST)
            if not socket.poll(self.timeout_ms):
                raise ResetError(
                    f"Isaac reset endpoint {self.endpoint} did not answer within {self.timeout_ms} ms"
                )
            reply = socket.recv()
        except zmq.ZMQError as exc:
            raise ResetError(f"Isaac reset endpoint {self.endpoint} error: {exc}") from exc
        finally:
            socket.close()
            context.term()

        if reply == RESET_QUEUED:
            return
        if reply == RESET_REFUSED:
            raise ResetError(f"Isaac reset endpoint {self.endpoint} refused the reset")
        raise ResetError(f"Isaac reset endpoint {self.endpoint} answered {reply[:64]!r}")
