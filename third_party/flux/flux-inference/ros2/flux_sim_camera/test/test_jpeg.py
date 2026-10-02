"""Pure JPEG tests: colors, row padding and parameter rejection."""

import io

import numpy as np
import pytest
from PIL import Image

from flux_sim_camera.jpeg import encode_jpeg, validate_parameters


@pytest.mark.parametrize("encoding", ["rgb8", "bgr8"])
@pytest.mark.parametrize("padding", [0, 5])
def test_encode_colors_with_row_padding(encoding, padding):
    rgb = np.full((16, 16, 3), [210, 70, 25], dtype=np.uint8)
    pixels = rgb if encoding == "rgb8" else rgb[:, :, ::-1]
    rows = np.full((16, 48 + padding), 255, dtype=np.uint8)
    rows[:, :48] = pixels.reshape(16, 48)
    payload = encode_jpeg(rows.tobytes(), 16, 16, 48 + padding, encoding)
    assert payload[:2] == b"\xff\xd8"
    decoded = Image.open(io.BytesIO(payload))
    assert decoded.format == "JPEG"
    assert decoded.size == (16, 16)
    assert np.asarray(decoded)[8, 8].tolist() == pytest.approx([210, 70, 25], abs=4)


@pytest.mark.parametrize("width,height,step,encoding,data", [
    (0, 1, 3, "rgb8", b"123"),
    (1, 0, 3, "rgb8", b""),
    (1, 1, 2, "rgb8", b"12"),
    (1, 1, 3, "mono8", b"123"),
    (1, 2, 3, "rgb8", b"123"),
    (1, 1, 3, "rgb8", b"1234"),
])
def test_reject_invalid_images(width, height, step, encoding, data):
    with pytest.raises(ValueError):
        encode_jpeg(data, width, height, step, encoding)


@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize("parameter", ["max_fps", "stale_s"])
def test_reject_invalid_timing(parameter, value):
    params = dict(max_fps=15.0, stale_s=0.5, quality=75)
    params[parameter] = value
    with pytest.raises(ValueError):
        validate_parameters(**params)


@pytest.mark.parametrize("quality", [0, 96, -1, 75.5, True])
def test_reject_invalid_quality(quality):
    with pytest.raises(ValueError):
        validate_parameters(15.0, 0.5, quality)
    with pytest.raises(ValueError):
        encode_jpeg(b"123", 1, 1, 3, "rgb8", quality)


@pytest.mark.parametrize("quality", [1, 75, 95])
def test_accept_valid_parameters(quality):
    validate_parameters(15.0, 0.5, quality)
    assert encode_jpeg(b"123", 1, 1, 3, "rgb8", quality)
