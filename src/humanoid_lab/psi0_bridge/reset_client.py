"""Narrow client for the Isaac simulator reset endpoint.

The simulation is owned by the Isaac service; the bridge never imports Isaac and
never touches the physics loop.  The Isaac service exposes its reset as a ZMQ
REP control endpoint (``ResetControlEndpoint`` in the Isaac camera service): one
``b"reset"`` request queues the reset on the simulation loop, which answers
``b"reset_queued"`` once it accepted the work or ``b"reset_refused"`` when no
loop is consuming it.  The endpoint is configurable because the port belongs to
the Isaac side.

``reset_queued`` is only a *queue* acknowledgement: the callback that answers it
sets the loop's pending-reset flag and returns, so the reply says "accepted",
never "applied".  The applied boundary is the simulation loop's own
``episode_id``, which it bumps inside ``reset_episode``; the same endpoint
answers ``b"status"`` with a JSON object carrying it.  :meth:`IsaacResetClient.
request_reset_applied` is the opt-in handshake built on that: it reads the
pre-reset ``episode_id``, sends the reset, and waits for the loop to publish a
strictly greater one, so a caller that must not act before the scene actually
moved can wait for the real boundary.  :meth:`request_reset` keeps the legacy
queue-acknowledged behaviour for every existing caller.
"""

from __future__ import annotations

import json
import os
import time

DEFAULT_RESET_ENDPOINT = os.environ.get("ISAAC_CONTROL_ENDPOINT", "tcp://localhost:5559")

RESET_REQUEST = b"reset"
RESET_QUEUED = b"reset_queued"
RESET_REFUSED = b"reset_refused"
STATUS_REQUEST = b"status"
ERROR_PREFIX = b"error:"

#: Polling cadence for the applied-reset handshake.  The simulation loop consumes
#: the flag at the top of its own iteration, so on a running simulator the wait
#: is one physics iteration; the poll only bounds how fast the caller notices.
STATUS_POLL_S = 0.005
#: Per-poll budget for the ``status`` round trip, deliberately independent of the
#: reset timeout: a status read is a local REP call, so a slow one is a stalled
#: endpoint to report rather than a reason to shorten the applied-reset wait.
STATUS_TIMEOUT_MS = 500


class ResetError(RuntimeError):
    """The simulator reset request was refused or unreachable."""


def parse_episode_id(reply: bytes) -> int:
    """The ``episode_id`` of one ``b"status"`` reply.

    Kept as a pure function of the payload, like the rest of the endpoint's
    contract, so the handshake's decision rule is testable without ZMQ.  The
    status object is written by the Isaac service and validated here rather than
    trusted: a reply that is not a JSON object with an integer ``episode_id`` is
    reported as such instead of being read as "no reset happened yet".
    """
    try:
        payload = json.loads(reply.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ResetError(f"Isaac status reply is not JSON: {reply[:64]!r}") from exc
    if not isinstance(payload, dict):
        raise ResetError(f"Isaac status reply is not a JSON object: {reply[:64]!r}")
    episode_id = payload.get("episode_id")
    if isinstance(episode_id, bool) or not isinstance(episode_id, int):
        raise ResetError(
            f"Isaac status reply has no integer episode_id: {reply[:64]!r}"
        )
    return episode_id


class IsaacResetClient:
    def __init__(self, endpoint: str = DEFAULT_RESET_ENDPOINT, *, timeout_ms: int = 2000) -> None:
        self.endpoint = endpoint
        self.timeout_ms = int(timeout_ms)

    def request_reset(self) -> None:
        """Send one ``reset`` and require an accepted reply.

        Owns its own ZMQ context and closes it, so the socket never outlives the
        call and cannot leak into another thread.
        """
        self._request(RESET_REQUEST, RESET_QUEUED, RESET_REFUSED, "the reset")

    def request_reset_applied(self, *, applied_timeout_ms: int | None = None) -> int:
        """Send one ``reset`` and wait for the simulator's applied boundary.

        Returns the new ``episode_id``.  The legacy :meth:`request_reset` returns
        as soon as the request is *queued*; this waits until the loop has
        actually run the reset, which is the only point at which the scene, the
        robot and the camera have moved.  A caller that publishes settle commands
        (the opt-in initial-pose handshake or warm start) must use this, because
        anything sent in that window is decoded against the pre-reset state.

        ``applied_timeout_ms`` bounds the whole call and defaults to
        ``timeout_ms``.  A refused request, a missing status reply or a loop that
        never bumps the episode id raises :class:`ResetError`, so the caller
        fails closed instead of settling on a scene that never moved.
        """
        deadline = time.monotonic() + (
            self.timeout_ms if applied_timeout_ms is None else int(applied_timeout_ms)
        ) / 1000.0
        # Read the boundary before asking for the reset: the loop may consume the
        # flag and bump the id before this call could read it afterwards, and a
        # post-hoc read would then look like "no reset happened".
        before = self._episode_id()
        self.request_reset()
        while True:
            current = self._episode_id()
            if current > before:
                return current
            if time.monotonic() >= deadline:
                raise ResetError(
                    f"Isaac reset endpoint {self.endpoint} queued the reset but the simulation "
                    f"loop did not apply it within the timeout (episode_id stayed {before})"
                )
            time.sleep(STATUS_POLL_S)

    # -- transport ---------------------------------------------------------

    def _episode_id(self) -> int:
        """One ``status`` round trip, decoded to the loop's episode id."""
        reply = self._request(
            STATUS_REQUEST, None, None, "the status", timeout_ms=STATUS_TIMEOUT_MS
        )
        return parse_episode_id(reply)

    def _request(
        self,
        request: bytes,
        expected: bytes | None,
        refused: bytes | None,
        what: str,
        *,
        timeout_ms: int | None = None,
    ) -> bytes:
        """Send one request and return its reply.

        Owns its own ZMQ context and closes it, so the socket never outlives the
        call and cannot leak into another thread.  ``expected``/``refused`` are
        the exact-match replies the caller accepts or rejects.
        """
        import zmq

        timeout = self.timeout_ms if timeout_ms is None else int(timeout_ms)
        context = zmq.Context()
        socket = context.socket(zmq.REQ)
        socket.setsockopt(zmq.LINGER, 0)
        try:
            socket.connect(self.endpoint)
            socket.send(request)
            if not socket.poll(timeout):
                raise ResetError(
                    f"Isaac reset endpoint {self.endpoint} did not answer {what} within "
                    f"{timeout} ms"
                )
            reply = socket.recv()
        except zmq.ZMQError as exc:
            raise ResetError(f"Isaac reset endpoint {self.endpoint} error: {exc}") from exc
        finally:
            socket.close()
            context.term()

        if expected is not None and reply == expected:
            return reply
        if refused is not None and reply == refused:
            raise ResetError(f"Isaac reset endpoint {self.endpoint} refused {what}")
        if expected is not None:
            raise ResetError(f"Isaac reset endpoint {self.endpoint} answered {reply[:64]!r}")
        return reply
