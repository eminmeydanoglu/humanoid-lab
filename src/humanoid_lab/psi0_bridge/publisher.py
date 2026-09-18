"""Action output: PUB of Protocol v4 ``pose`` messages on ``tcp://*:5556``.

The bridge binds this socket itself (the SONIC deploy process connects).  The
bind happens when the session is constructed, so the port is owned for the whole
service lifetime: a busy port fails the UI service at startup instead of failing
later at Start, and Stop only closes the send gate — it never releases the
socket to another publisher mid-session.

Only fully-packed messages produced by
:class:`humanoid_lab.psi0_bridge.actions.ActionAdapter` are sent, and output can
be halted immediately so a stop never leaves a stale action on the wire.
"""

from __future__ import annotations

ACTION_ENDPOINT = "tcp://*:5556"


class PublisherError(RuntimeError):
    """The pose PUB socket cannot be bound; the endpoint belongs to someone else."""


class PosePublisher:
    def __init__(self, endpoint: str = ACTION_ENDPOINT) -> None:
        import zmq

        self.endpoint = endpoint
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.PUB)
        self._socket.setsockopt(zmq.LINGER, 0)
        try:
            self._socket.bind(endpoint)
        except zmq.ZMQError as exc:
            self._socket.close(linger=0)
            self._context.term()
            raise PublisherError(
                f"cannot bind the action endpoint {endpoint} ({exc}); another pilot or "
                "replay is probably publishing pose messages there"
            ) from exc
        self._halted = True  # the send gate opens only when a session starts
        self._closed = False

    @property
    def halted(self) -> bool:
        return self._halted

    @property
    def closed(self) -> bool:
        return self._closed

    def publish(self, payload: bytes) -> bool:
        """Send one packed action; a no-op once :meth:`halt` has been called."""
        if self._halted or self._closed:
            return False
        self._socket.send(payload)
        return True

    def halt(self) -> None:
        """Close the send gate immediately (safe to call more than once)."""
        self._halted = True

    def resume(self) -> None:
        """Open the send gate for a new session."""
        if self._closed:
            raise PublisherError("the action socket is closed; the service is shutting down")
        self._halted = False

    def close(self) -> None:
        """Release the socket; only the owning service calls this, never Stop."""
        if self._closed:
            return
        self._closed = True
        self._halted = True
        self._socket.close(linger=0)
        self._context.term()
