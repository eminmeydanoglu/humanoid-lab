"""Robot state source: SUB on SONIC's ``g1_debug`` topic.

Mirrors ``ZMQStateSubscriber`` from the pinned SONIC tree (CONFLATE, strip the
topic prefix, msgpack decode) so the payload shape matches what the SONIC
deploy process publishes.  The decoded mapping is handed to
:func:`humanoid_lab.psi0_bridge.state.build_raw_state`; no other state source is
read.
"""

from __future__ import annotations

from typing import Any

STATE_TOPIC = "g1_debug"
STATE_ENDPOINT = "tcp://localhost:5557"


def _to_numpy(data: Any) -> Any:
    import numpy as np

    if isinstance(data, dict):
        return {key: _to_numpy(value) for key, value in data.items()}
    if isinstance(data, (list, tuple)):
        if data and all(isinstance(item, (int, float)) for item in data):
            return np.asarray(data, dtype=np.float32)
        return [_to_numpy(item) for item in data]
    return data


class StateSubscriber:
    """Non-blocking, conflating SUB on ``g1_debug``."""

    def __init__(
        self,
        endpoint: str = STATE_ENDPOINT,
        *,
        topic: str = STATE_TOPIC,
    ) -> None:
        import zmq

        self.endpoint = endpoint
        self.topic = topic
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.SUB)
        self._socket.setsockopt_string(zmq.SUBSCRIBE, topic)
        self._socket.setsockopt(zmq.CONFLATE, 1)
        self._socket.setsockopt(zmq.RCVTIMEO, 0)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.connect(endpoint)

    def latest(self) -> dict[str, Any] | None:
        """Return the newest state payload, or ``None`` when nothing arrived."""
        import msgpack
        import zmq

        try:
            raw = self._socket.recv(zmq.NOBLOCK)
        except zmq.Again:
            return None
        payload = raw[len(self.topic):]
        decoded = msgpack.unpackb(payload, raw=False)
        if not isinstance(decoded, dict):
            return None
        return _to_numpy(decoded)

    def close(self) -> None:
        self._socket.close()
        self._context.term()
