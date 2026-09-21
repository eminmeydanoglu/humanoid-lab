from __future__ import annotations

import io
import unittest

import numpy as np

from humanoid_lab.simulators.isaac.sonic_camera import _pack


class SonicCameraPayloadTest(unittest.TestCase):
    def test_payload_matches_nvidia_camera_schema(self) -> None:
        try:
            import msgpack
            from PIL import Image
        except ImportError as exc:
            self.skipTest(str(exc))

        rgb = np.zeros((4, 6, 3), dtype=np.uint8)
        rgb[..., 0] = 255
        jpeg = io.BytesIO()
        Image.fromarray(rgb).save(jpeg, format="JPEG")

        payload = _pack(
            {
                "timestamps": {"ego_view": 12.5},
                "images": {"ego_view": jpeg.getvalue()},
            }
        )
        decoded = msgpack.unpackb(payload, raw=False)

        self.assertEqual(decoded["timestamps"], {"ego_view": 12.5})
        self.assertEqual(decoded["images"]["ego_view"], jpeg.getvalue())

    def test_rejects_values_outside_camera_schema(self) -> None:
        with self.assertRaisesRegex(TypeError, "unsupported msgpack value"):
            _pack({"ego_view": 1})


if __name__ == "__main__":
    unittest.main()
