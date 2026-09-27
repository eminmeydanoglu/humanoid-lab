"""Unit tests for the pure SONIC ego_view frame logic (no ROS, no simulator)."""

import io

import msgpack
import numpy as np
import pytest
from flux_sim_camera.frame import FrameError, FrameGate, decode_frame
from PIL import Image


def pack_ego_view(stamp_s=1.0, shape=(480, 640, 3), value=127):
    image = np.full(shape, value, dtype=np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(image).save(buffer, format="JPEG")
    return msgpack.packb(
        {"timestamps": {"ego_view": stamp_s}, "images": {"ego_view": buffer.getvalue()}},
        use_bin_type=True,
    )


def test_decode_returns_stamp_and_rgb():
    frame = decode_frame(pack_ego_view(1234.5))
    assert frame.stamp_s == 1234.5
    assert frame.rgb.dtype == np.uint8
    assert frame.rgb.shape == (480, 640, 3)
    assert frame.rgb.flags["C_CONTIGUOUS"]


def test_decode_rejects_malformed_payloads():
    with pytest.raises(FrameError):
        decode_frame(b"\x91\x01")
    with pytest.raises(FrameError):
        decode_frame(b"not msgpack at all")
    # Wrong shape: a JPEG that is not the head camera's 640x480.
    with pytest.raises(FrameError):
        decode_frame(pack_ego_view(shape=(16, 16, 3)))
    with pytest.raises(FrameError):
        decode_frame(pack_ego_view(stamp_s=0.0))
    with pytest.raises(FrameError):
        decode_frame(pack_ego_view(stamp_s=float("nan")))
    # Missing view entries.
    payload = msgpack.packb({"timestamps": {}, "images": {}}, use_bin_type=True)
    with pytest.raises(FrameError):
        decode_frame(payload)


def test_gate_publishes_only_advancing_stamps():
    gate = FrameGate()
    assert gate.accept(10.0) is True
    assert gate.accept(10.0) is False  # duplicate stamp never re-dated
    assert gate.accept(9.9) is False  # an older stamp is not a newer frame
    assert gate.accept(10.1) is True
    assert gate.published == 2
    assert gate.skipped == 2
    assert gate.resyncs == 0
    assert gate.last_stamp == pytest.approx(10.1)
    assert gate.age_s(10.6) == pytest.approx(0.5)
    assert FrameGate().age_s(1.0) is None


def test_gate_resyncs_after_a_backwards_clock_step():
    gate = FrameGate(resync_s=5.0)
    assert gate.accept(100.0) is True
    assert gate.accept(80.0) is True  # clock stepped back: resynchronise
    assert gate.resyncs == 1
    assert gate.accept(80.0) is False  # ...but duplicates are still duplicates
    assert gate.accept(80.2) is True


def test_gate_rejects_nonpositive_resync():
    with pytest.raises(ValueError):
        FrameGate(resync_s=0)
