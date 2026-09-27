"""SONIC ``ego_view`` frame handling, independent of ROS.

The Isaac simulator publishes the head camera in NVIDIA SONIC's native wire
format (``src/humanoid_lab/simulators/isaac/sonic_camera.py``): one msgpack map
with the wall-clock capture time and JPEG bytes per view.  This module decodes
that payload and decides which frames may be published, so the ROS node in
``camera_bridge`` only forwards frames the simulator actually produced.
"""

from __future__ import annotations

import io
from typing import NamedTuple

import numpy as np

EGO_VIEW = "ego_view"
RGB8_SHAPE = (480, 640, 3)
#: A stamp this far behind the last published one is a clock step (for example
#: the host clock being corrected), not a replayed frame; the next genuine
#: frame after it is accepted so the bridge cannot wedge on a backwards jump.
DEFAULT_RESYNC_S = 5.0


class FrameError(ValueError):
    """A payload that cannot be published as a stamped RGB frame."""


class Frame(NamedTuple):
    stamp_s: float
    rgb: np.ndarray


def decode_frame(payload: bytes) -> Frame:
    """Decode one SONIC ``ego_view`` msgpack payload; reject anything else."""
    import msgpack
    from PIL import Image

    try:
        message = msgpack.unpackb(payload, raw=False)
    except Exception as exc:  # msgpack raises several concrete error types
        raise FrameError("invalid msgpack payload") from exc
    if not isinstance(message, dict):
        raise FrameError("payload is not a map")
    timestamps = message.get("timestamps")
    images = message.get("images")
    if not isinstance(timestamps, dict) or not isinstance(images, dict):
        raise FrameError("payload lacks the timestamps/images maps")
    if EGO_VIEW not in timestamps or EGO_VIEW not in images:
        raise FrameError("payload lacks ego_view entries")
    stamp = timestamps[EGO_VIEW]
    if type(stamp) not in (int, float) or not np.isfinite(stamp) or stamp <= 0:
        raise FrameError("invalid ego_view capture timestamp")
    jpeg = images[EGO_VIEW]
    if not isinstance(jpeg, (bytes, bytearray)):
        raise FrameError("invalid ego_view image payload")
    try:
        image = Image.open(io.BytesIO(jpeg))
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    except Exception as exc:
        raise FrameError("undecodable ego_view JPEG") from exc
    if rgb.shape != RGB8_SHAPE:
        raise FrameError("expected %dx%d RGB, got %s" % (RGB8_SHAPE[1], RGB8_SHAPE[0], rgb.shape))
    return Frame(float(stamp), np.ascontiguousarray(rgb))


class FrameGate:
    """Publish only frames whose simulator timestamp strictly advances.

    The stamp travels from the simulator unchanged; the bridge never stamps a
    frame with its own clock.  A missing frame therefore publishes nothing, and
    a repeated stamp is dropped rather than being re-dated.
    """

    def __init__(self, resync_s: float = DEFAULT_RESYNC_S) -> None:
        if resync_s <= 0:
            raise ValueError("resync_s must be positive")
        self.resync_s = float(resync_s)
        self.last_stamp: float | None = None
        self.published = 0
        self.skipped = 0
        self.resyncs = 0

    def accept(self, stamp_s: float) -> bool:
        """Return True when this frame may be published."""
        stamp_s = float(stamp_s)
        last = self.last_stamp
        if last is not None and stamp_s <= last:
            if last - stamp_s > self.resync_s:
                # Clock stepped backwards (host clock correction): accept and
                # resynchronise instead of rejecting frames forever.
                self.resyncs += 1
            else:
                self.skipped += 1
                return False
        self.last_stamp = stamp_s
        self.published += 1
        return True

    def age_s(self, now_s: float) -> float | None:
        """Seconds since the last accepted frame, or None before the first."""
        if self.last_stamp is None:
            return None
        return now_s - self.last_stamp
