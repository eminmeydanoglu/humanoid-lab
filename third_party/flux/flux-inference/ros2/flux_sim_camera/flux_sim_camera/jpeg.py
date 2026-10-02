"""JPEG encoding and parameter validation independent of ROS."""

import io
import math

import numpy as np
from PIL import Image


def validate_quality(quality: int) -> None:
    if type(quality) is not int or not 1 <= quality <= 95:
        raise ValueError("quality must be an integer in 1..95")


def validate_parameters(max_fps: float, stale_s: float, quality: int) -> None:
    if not all(math.isfinite(value) and value > 0 for value in (max_fps, stale_s)):
        raise ValueError("max_fps and stale_s must be positive and finite")
    validate_quality(quality)


def encode_jpeg(data, width: int, height: int, step: int, encoding: str,
                quality: int = 75) -> bytes:
    """Encode packed or row-padded rgb8/bgr8 pixels as an RGB JPEG."""
    validate_quality(quality)
    if encoding not in ("rgb8", "bgr8"):
        raise ValueError("unsupported image encoding: %s" % encoding)
    if width <= 0 or height <= 0 or step < width * 3:
        raise ValueError("invalid image dimensions or step")
    pixels = np.frombuffer(data, dtype=np.uint8)
    if pixels.size != height * step:
        raise ValueError("image data length does not match height * step")
    rgb = pixels.reshape(height, step)[:, :width * 3].reshape(height, width, 3)
    if encoding == "bgr8":
        rgb = rgb[:, :, ::-1]
    buffer = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(rgb)).save(buffer, format="JPEG", quality=quality)
    return buffer.getvalue()
