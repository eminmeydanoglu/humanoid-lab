"""Radial barrel distortion for the head camera (numpy only, OpenCV if present).

Isaac renders an ideal pinhole.  The real G1 head camera is a wide lens with
visible barrel distortion, so the pinhole frame is rendered larger and wider
than the output and resampled with the radial model

    r_u = r_d * (1 + k1 * r_d**2 + k2 * r_d**4)

where ``r_d`` is the normalised radius in the output (distorted) image and
``r_u`` the normalised radius in the rendered pinhole image.  The output focal
length is chosen so the output corners land on the rendered corners.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


def _undistorted_radius(r_d: Any, k1: float, k2: float) -> Any:
    r2 = r_d * r_d
    return r_d * (1.0 + k1 * r2 + k2 * r2 * r2)


def distorted_radius(r_u: float, k1: float, k2: float) -> float:
    """Invert the monotonic radial model by bisection."""
    lo, hi = 0.0, max(r_u, 1e-9)
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if _undistorted_radius(mid, k1, k2) < r_u:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


@dataclass(frozen=True)
class LensModel:
    width: int
    height: int
    render_width: int
    render_height: int
    render_focal_px: float
    output_focal_px: float
    k1: float
    k2: float

    @classmethod
    def build(cls, width: int, height: int, render_width: int, render_height: int,
              focal_length_mm: float, horizontal_aperture_mm: float, k1: float, k2: float) -> "LensModel":
        render_focal = render_width * focal_length_mm / horizontal_aperture_mm
        corner_u = math.hypot(render_width / 2.0, render_height / 2.0) / render_focal
        corner_d = distorted_radius(corner_u, k1, k2)
        output_focal = math.hypot(width / 2.0, height / 2.0) / corner_d
        return cls(width, height, render_width, render_height, render_focal, output_focal, k1, k2)

    def maps(self) -> tuple[Any, Any]:
        """Per output pixel, the rendered-image sample position (float32)."""
        import numpy as np

        u = (np.arange(self.width, dtype=np.float64) + 0.5 - self.width / 2.0) / self.output_focal_px
        v = (np.arange(self.height, dtype=np.float64) + 0.5 - self.height / 2.0) / self.output_focal_px
        xd, yd = np.meshgrid(u, v)
        r2 = xd * xd + yd * yd
        scale = 1.0 + self.k1 * r2 + self.k2 * r2 * r2
        map_x = xd * scale * self.render_focal_px + self.render_width / 2.0 - 0.5
        map_y = yd * scale * self.render_focal_px + self.render_height / 2.0 - 0.5
        return map_x.astype(np.float32), map_y.astype(np.float32)

    def project(self, x_u: float, y_u: float) -> tuple[float, float]:
        """Pinhole-normalised coordinates to output pixel coordinates."""
        r_u = math.hypot(x_u, y_u)
        factor = 1.0 if r_u == 0.0 else distorted_radius(r_u, self.k1, self.k2) / r_u
        return (x_u * factor * self.output_focal_px + self.width / 2.0,
                y_u * factor * self.output_focal_px + self.height / 2.0)


class Distorter:
    """Resample rendered frames into the distorted output frame."""

    def __init__(self, lens: LensModel) -> None:
        self.lens = lens
        self._map_x, self._map_y = lens.maps()
        try:
            import cv2
        except ImportError:
            self._cv2 = None
        else:
            self._cv2 = cv2

    def __call__(self, image: Any) -> Any:
        import numpy as np

        if self._cv2 is not None:
            return self._cv2.remap(image, self._map_x, self._map_y, self._cv2.INTER_LINEAR,
                                   borderMode=self._cv2.BORDER_REPLICATE)
        h, w = image.shape[:2]
        x = np.clip(self._map_x, 0.0, w - 1.001)
        y = np.clip(self._map_y, 0.0, h - 1.001)
        x0 = x.astype(np.int32)
        y0 = y.astype(np.int32)
        fx = (x - x0)[..., None]
        fy = (y - y0)[..., None]
        src = image.astype(np.float32)
        top = src[y0, x0] * (1 - fx) + src[y0, x0 + 1] * fx
        bottom = src[y0 + 1, x0] * (1 - fx) + src[y0 + 1, x0 + 1] * fx
        return np.clip(top * (1 - fy) + bottom * fy + 0.5, 0, 255).astype(np.uint8)
