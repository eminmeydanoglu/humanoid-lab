"""Minimal client for the served Ψ₀ policy: ``/info`` plus the WebSocket stream.

The request/response wire format is the one ``psi.deploy.helpers`` defines
(``RequestMessage`` / ``ResponseMessage``): a JSON object whose ``image`` and
``state`` blocks carry numpy arrays as ``{"__numpy__": <base64>,
"dtype": <descr>, "shape": [...]}``.  That tiny codec is re-implemented here
instead of imported, so the bridge has no runtime dependency on the pinned Psi0
tree; a test decodes our request with the server's own decoder shape.

One tick is: send one observation, then read the newest action strictly newer
than the last one consumed.  Actions buffered from an earlier observation are
dropped, never re-sent, so a stalled or refused link cannot drive the robot with
a repeated action.
"""

from __future__ import annotations

import base64
import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Mapping
from urllib.parse import urlsplit, urlunsplit

import numpy as np

from .contracts import ServerInfo, validate_info

DEFAULT_WS_URL = "ws://localhost:8014/ws"

#: One 640x480 RGB frame base64-encodes to ~1.2 MB; the response carries a
#: 30x80 action, so a generous cap keeps a first frame from being rejected.
DEFAULT_MAX_SIZE = 32 * 1024 * 1024


class Psi0ClientError(RuntimeError):
    """The policy server is unreachable or answered something unparseable."""


def _dtype_descr(dtype: np.dtype) -> str:
    try:
        from numpy.lib.format import dtype_to_descr

        return dtype_to_descr(dtype)
    except Exception:  # pragma: no cover - only if numpy drops the helper
        return dtype.str


def numpy_serialize(value: Any) -> dict[str, Any]:
    """Encode one numpy array the way ``psi.deploy.helpers.numpy_serialize`` does."""
    if isinstance(value, (np.ndarray, np.generic)):
        array = np.asarray(value)
        data = array.data if array.flags["C_CONTIGUOUS"] else array.tobytes()
        return {
            "__numpy__": base64.b64encode(bytes(data)).decode("ascii"),
            "dtype": _dtype_descr(array.dtype),
            "shape": list(array.shape),
        }
    raise TypeError(f"Object of type {value.__class__.__name__} is not JSON serializable")


def numpy_deserialize(value: Mapping[str, Any]) -> Any:
    """Decode one ``__numpy__`` envelope; non-envelopes pass through unchanged."""
    if "__numpy__" not in value:
        return value
    buffer = np.frombuffer(base64.b64decode(value["__numpy__"]), dtype=np.dtype(value["dtype"]))
    shape = value.get("shape") or []
    return buffer.reshape(tuple(int(dim) for dim in shape)) if shape else buffer[0]


def _convert(value: Any, func: Any) -> Any:
    if isinstance(value, Mapping):
        if "__numpy__" in value:
            return func(value)
        return {key: _convert(item, func) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_convert(item, func) for item in value]
    if isinstance(value, (np.ndarray, np.generic)):
        return func(value)
    return value


def serialize_request(
    *,
    image: Mapping[str, np.ndarray],
    state: np.ndarray,
    instruction: str,
    dataset_name: str,
    timestamp: str | None = None,
) -> str:
    """Serialize one observation into the server's ``RequestMessage`` JSON.

    ``state`` is the RAW 43D vector; the server normalizes and pads it itself.
    ``history`` is empty because the checkpoint consumes a single frame.
    """
    raw_state = np.asarray(state, dtype=np.float32).reshape(-1)
    images = {
        str(key): np.asarray(value, dtype=np.uint8)
        for key, value in image.items()
    }
    message = {
        "image": images,
        "instruction": str(instruction),
        "history": {},
        "state": {"states": raw_state},
        "condition": {},
        "gt_action": [],
        "dataset_name": str(dataset_name),
        "timestamp": str(timestamp if timestamp is not None else ""),
    }
    return json.dumps(_convert(message, numpy_serialize))


@dataclass(frozen=True)
class ActionReply:
    """One decoded ``ResponseMessage`` plus its monotonic ``version``."""

    version: int
    action: np.ndarray
    err: float


def decode_response(payload: str | bytes | Mapping[str, Any]) -> ActionReply:
    """Decode one server action message into a flat ``(action_dim,)`` vector."""
    if isinstance(payload, (str, bytes, bytearray)):
        try:
            data = json.loads(payload)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise Psi0ClientError(f"policy server sent unparseable JSON: {exc}") from exc
    else:
        data = payload
    if not isinstance(data, Mapping):
        raise Psi0ClientError(f"policy server sent a {type(data).__name__}, expected an object")
    if "action" not in data:
        raise Psi0ClientError("policy server response has no 'action' field")
    decoded = _convert(data["action"], numpy_deserialize)
    action = np.asarray(decoded, dtype=np.float32).reshape(-1)
    version = int(data.get("version", 0))
    try:
        err = float(data.get("err", 0.0))
    except (TypeError, ValueError):
        err = 0.0
    return ActionReply(version=version, action=action, err=err)


def http_base_from_ws(url: str) -> str:
    """``ws://host:port/ws`` -> ``http://host:port`` (scheme and host only)."""
    parts = urlsplit(url)
    scheme = {"ws": "http", "wss": "https"}.get(parts.scheme, parts.scheme)
    return urlunsplit((scheme, parts.netloc, "", "", ""))


def fetch_info(ws_url: str = DEFAULT_WS_URL, *, timeout_s: float = 10.0) -> ServerInfo:
    """GET ``/info`` and fail closed unless it matches the bridge contract."""
    url = http_base_from_ws(ws_url).rstrip("/") + "/info"
    try:
        with urllib.request.urlopen(url, timeout=timeout_s) as response:
            body = response.read()
    except (urllib.error.URLError, OSError, ValueError) as exc:
        raise Psi0ClientError(f"policy server /info is unreachable at {url}: {exc}") from exc
    try:
        decoded = json.loads(body.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise Psi0ClientError(f"policy server /info is not JSON: {exc}") from exc
    return validate_info(decoded)


class Psi0Connection:
    """Async WebSocket connection to the served policy."""

    def __init__(
        self,
        url: str = DEFAULT_WS_URL,
        *,
        max_size: int = DEFAULT_MAX_SIZE,
        open_timeout_s: float = 10.0,
    ) -> None:
        self.url = url
        self.max_size = int(max_size)
        self.open_timeout_s = float(open_timeout_s)
        self._ws: Any = None

    async def __aenter__(self) -> "Psi0Connection":
        import websockets

        try:
            self._ws = await websockets.connect(
                self.url, max_size=self.max_size, open_timeout=self.open_timeout_s
            )
        except Exception as exc:
            raise Psi0ClientError(f"cannot connect to {self.url}: {exc}") from exc
        return self

    async def __aexit__(self, *_exc: Any) -> None:
        await self.close()

    @property
    def connected(self) -> bool:
        return self._ws is not None

    async def send(self, payload: str) -> None:
        await self._ws.send(payload)

    async def recv(self) -> str:
        return await self._ws.recv()

    async def close(self) -> None:
        socket, self._ws = self._ws, None
        if socket is None:
            return
        try:
            await socket.close()
        except Exception:
            pass
