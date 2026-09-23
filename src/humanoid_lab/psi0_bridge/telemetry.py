"""Timestamped policy telemetry for one evaluation session.

The bridge already sees everything needed to explain a rollout: the observation
it handed to the policy (raw state vector plus head-camera frame), the raw
action the policy answered with, the Protocol v4 message the router actually
published to SONIC, and the session/backend transitions around them.  This
module writes that to JSONL so a rollout can be analysed -- and compared against
the training dataset -- without a screen recording or a guess.

It is opt-in: with no ``--telemetry-dir`` the launcher does not build one and
the session behaves exactly as before.  Recording must never break a session, so
a write failure disables the recorder and is reported once.

For a PSI selection the observation is exactly the payload the session sent
(``Observation`` events carry that state vector and the hash of that frame).  A
GR00T selection is served by NVIDIA's own VLA client, which reads the camera and
state streams itself, so for that backend the recorder reports the closest
observable proxy: the same head-camera frame and the applied `pose` messages
from the router, marked ``source: "groot"``.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np

PROTOCOL_V4_HEADER_SIZE = 1280
PROTOCOL_V4_TOPIC = b"pose"
_DTYPES = {"f32": "<f4", "i64": "<i8"}


class TelemetryError(RuntimeError):
    """The recorder could not be created."""


def decode_protocol_v4(payload: bytes) -> dict[str, Any]:
    """Decode one Protocol v4 message into its named fields.

    Only the fields the message actually carries are returned; an unknown
    descriptor dtype or a truncated payload raises instead of producing a
    half-decoded action that later reads as measurement.
    """
    if not payload.startswith(PROTOCOL_V4_TOPIC):
        raise TelemetryError(f"not a pose message: starts with {payload[:4]!r}")
    header_end = 4 + PROTOCOL_V4_HEADER_SIZE
    if len(payload) < header_end:
        raise TelemetryError(f"payload shorter than the fixed header: {len(payload)} bytes")
    header = payload[4:header_end].rstrip(b"\0")
    specification = json.loads(header.decode("utf-8"))
    offset = header_end
    fields: dict[str, Any] = {}
    for descriptor in specification.get("fields", []):
        dtype = _DTYPES.get(str(descriptor.get("dtype")))
        if dtype is None:
            raise TelemetryError(f"unsupported field dtype {descriptor.get('dtype')!r}")
        shape = [int(value) for value in descriptor.get("shape", [])]
        count = 1
        for value in shape:
            count *= value
        array = np.frombuffer(payload, dtype=dtype, count=count, offset=offset)
        offset += count * array.dtype.itemsize
        fields[str(descriptor["name"])] = array.reshape(shape).tolist()
    return {"topic": "pose", "fields": fields, "bytes": len(payload)}


def unpack_frame(frame: Any) -> tuple[Optional[np.ndarray], Optional[float]]:
    """A camera snapshot as ``(HxWx3 array, timestamp)``, or ``(None, None)``.

    The bridge hands over whatever its own reader produced -- the monitor's
    ``FrameSnapshot`` (whose image is the ``.frame`` attribute), a bare array, or
    a ``(frame, timestamp)`` pair -- so the shape is checked here instead of
    being assumed by the caller that is writing the log.
    """
    timestamp: Optional[float] = None
    image = frame
    if hasattr(frame, "frame"):  # the monitor's FrameSnapshot
        timestamp = getattr(frame, "timestamp_s", None)
        image = getattr(frame, "frame")
    elif isinstance(frame, tuple) and len(frame) == 2:
        image, candidate = frame
        timestamp = candidate if isinstance(candidate, (int, float)) else None
    try:
        array = np.asarray(image)
    except (TypeError, ValueError):
        return None, timestamp
    if array.ndim != 3 or array.shape[2] < 3 or array.size == 0:
        return None, timestamp
    return array, timestamp


class SessionTelemetry:
    """Append-only JSONL recorder for one evaluation session.

    Every line is one event with ``wall_time_ns`` (and ``wall_time`` in UTC
    seconds) so the bridge-side stream can be aligned with the Isaac samples,
    the tracking parquet and the video frame map, none of which share a clock.
    """

    def __init__(
        self,
        directory: Path,
        *,
        camera_every: int = 5,
        frame_interval_s: float = 0.2,
        session_tag: str = "",
    ) -> None:
        if camera_every < 0:
            raise TelemetryError(f"camera_every must be >= 0, got {camera_every}")
        self.directory = Path(directory)
        self.camera_every = int(camera_every)
        self.frame_interval_s = float(frame_interval_s)
        self.session_tag = session_tag
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
            self.camera_dir.mkdir(parents=True, exist_ok=True)
            self.events_path = self.directory / "bridge-telemetry.jsonl"
            self._handle = self.events_path.open("a", encoding="utf-8")
        except OSError as exc:
            raise TelemetryError(f"cannot open the telemetry directory {directory}: {exc}") from exc
        self._lock = threading.Lock()
        self._disabled_reason: Optional[str] = None
        self._observation_index = 0
        self._action_index = 0
        self._saved_frames = 0
        self._frame_provider: Any = None
        # Never sampled: ``time.monotonic()`` counts from boot, so a 0.0 start
        # would throttle the first frame whenever the process starts within one
        # frame interval of the machine coming up.
        self._last_frame_time = float("-inf")

    @property
    def camera_dir(self) -> Path:
        return self.directory / "head_camera"

    @property
    def disabled_reason(self) -> Optional[str]:
        return self._disabled_reason

    def set_frame_provider(self, provider: Any) -> None:
        """Install the callable whose snapshot the GR00T proxy frames come from.

        A GR00T selection is driven by NVIDIA's VLA client, so the only camera
        reading the bridge can honestly report is the one its own monitor holds
        (the same service frame the UI previews); ``sample_frame`` time-throttles
        those snapshots into the same event stream.
        """
        self._frame_provider = provider

    def close(self) -> None:
        with self._lock:
            handle, self._handle = self._handle, None
        if handle is not None:
            try:
                handle.close()
            except OSError:
                pass

    # -- events ------------------------------------------------------------

    def event(self, kind: str, **fields: Any) -> Optional[dict[str, Any]]:
        now_ns = time.time_ns()
        record: dict[str, Any] = {
            "kind": kind,
            "wall_time_ns": now_ns,
            "wall_time": now_ns / 1e9,
        }
        if self.session_tag:
            record["session"] = self.session_tag
        record.update(fields)
        self._write(record)
        return record

    def transition(self, state: str, **fields: Any) -> Optional[dict[str, Any]]:
        """One session state change (``RUNNING``/``STOPPED``/``IDLE``/``ERROR``)."""
        return self.event("session", state=state, **fields)

    def observation(
        self,
        *,
        state: Any,
        frame: Any,
        state_time_s: Optional[float] = None,
        frame_time_s: Optional[float] = None,
        **fields: Any,
    ) -> Optional[dict[str, Any]]:
        """The observation handed to the policy, with the frame it carried."""
        payload = np.asarray(state, dtype=np.float32).reshape(-1)
        record_fields: dict[str, Any] = {
            "index": self._observation_index,
            "state": [float(value) for value in payload.tolist()],
            "state_dim": int(payload.shape[0]),
            "state_source_time": state_time_s,
            "frame_source_time": frame_time_s,
            **fields,
        }
        image = np.asarray(frame) if frame is not None else None
        if image is None:
            record_fields.update({"frame_sha1": None, "frame_shape": None, "camera_jpeg": None})
        else:
            contiguous = np.ascontiguousarray(image[..., :3].astype(np.uint8, copy=False))
            record_fields.update(
                {
                    "frame_sha1": hashlib.sha1(contiguous.tobytes()).hexdigest(),
                    "frame_shape": list(contiguous.shape),
                    "camera_jpeg": self._save_frame(contiguous, index=record_fields["index"]),
                }
            )
        record = self.event("observation", **record_fields)
        self._observation_index += 1
        return record

    def target_action(self, action: Any, **fields: Any) -> Optional[dict[str, Any]]:
        """The raw action the policy returned, before it becomes a message."""
        values = np.asarray(action, dtype=np.float32).reshape(-1)
        record = self.event(
            "target_action",
            index=self._action_index,
            action=[float(value) for value in values.tolist()],
            action_dim=int(values.shape[0]),
            **fields,
        )
        self._action_index += 1
        return record

    def frame(self, frame: Any, **fields: Any) -> Optional[dict[str, Any]]:
        """A head-camera frame sampled outside a PSI observation (GR00T path).

        Anything that is not a decodable image is reported as such and skipped:
        recording must never be able to stop the action path it observes.
        """
        image, timestamp = unpack_frame(frame)
        if image is None:
            return self.event(
                "frame",
                index=self._saved_frames,
                frame_sha1=None,
                frame_shape=None,
                camera_jpeg=None,
                frame_error=f"not an image: {type(frame).__name__}",
                **fields,
            )
        contiguous = np.ascontiguousarray(image[..., :3].astype(np.uint8, copy=False))
        return self.event(
            "frame",
            index=self._saved_frames,
            frame_sha1=hashlib.sha1(contiguous.tobytes()).hexdigest(),
            frame_shape=list(contiguous.shape),
            camera_jpeg=self._save_frame(contiguous, index=self._saved_frames, always=True),
            frame_source_time=timestamp,
            **{key: value for key, value in fields.items() if key != "frame_source_time"},
        )

    def sample_frame(self, source: str, **fields: Any) -> Optional[dict[str, Any]]:
        """Time-throttled proxy frame for a backend the bridge does not drive."""
        if self._frame_provider is None:
            return None
        now = time.monotonic()
        if now - self._last_frame_time < self.frame_interval_s:
            return None
        self._last_frame_time = now
        try:
            snapshot = self._frame_provider()
        except Exception as exc:  # noqa: BLE001 - sampling must never break the router
            self._disable(f"frame provider failed: {type(exc).__name__}: {exc}")
            return None
        if snapshot is None:
            return None
        return self.frame(snapshot, source=source, **fields)

    def applied_action(self, source: str, payload: bytes, **fields: Any) -> Optional[dict[str, Any]]:
        """One ``pose`` message the router published to the SONIC action port.

        The message is what the robot is actually commanded with, so it is kept
        verbatim (decoded fields plus the topic and the payload size); a decode
        failure is recorded as such instead of dropping the event.
        """
        record_fields: dict[str, Any] = {"source": source, "payload_bytes": len(payload), **fields}
        try:
            record_fields.update(decode_protocol_v4(payload))
        except (TelemetryError, ValueError, KeyError) as exc:
            record_fields.update({"decode_error": f"{type(exc).__name__}: {exc}"})
        return self.event("applied_action", **record_fields)

    # -- internals ---------------------------------------------------------

    def _save_frame(self, image: np.ndarray, *, index: int, always: bool = False) -> Optional[str]:
        """JPEG for the sampled frame, or ``None`` when this sample is not kept.

        PSI observations keep every ``camera_every``-th observation (the policy
        runs at 30 Hz, the frames are 640x480, and keeping all of them would
        dwarf the series they belong to); frames sampled on the GR00T path are
        already time-throttled and are always written.  The file name counts the
        frames actually written, the modulo the observations seen.
        """
        if not always and (self.camera_every <= 0 or index % self.camera_every != 0):
            return None
        from .camera import CameraError, encode_jpeg

        path = self.camera_dir / f"head_camera_{self._saved_frames:06d}.jpg"
        try:
            path.write_bytes(encode_jpeg(image))
        except (CameraError, OSError) as exc:
            self._disable(f"cannot write {path}: {type(exc).__name__}: {exc}")
            return None
        self._saved_frames += 1
        return str(path)

    def _write(self, record: Mapping[str, Any]) -> bool:
        with self._lock:
            handle = self._handle
            if handle is None:
                if self._disabled_reason is None:
                    self._disabled_reason = "recorder is closed"
                return False
            try:
                handle.write(json.dumps(record, separators=(",", ":")) + "\n")
                handle.flush()
            except (OSError, TypeError, ValueError) as exc:
                self._handle = None
                self._disabled_reason = f"write failed: {type(exc).__name__}: {exc}"
                try:
                    handle.close()
                except OSError:
                    pass
                return False
        return True

    def _disable(self, reason: str) -> None:
        with self._lock:
            self._disabled_reason = reason
