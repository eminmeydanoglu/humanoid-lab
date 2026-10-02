"""JPEG node tests in the ROS container, without a simulator."""

import io
import time
from unittest.mock import Mock

import pytest
import rclpy
from PIL import Image as PilImage
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, ReliabilityPolicy
from sensor_msgs.msg import CompressedImage, Image

import flux_sim_camera.camera_jpeg as jpeg_node
from flux_sim_camera.camera_jpeg import CameraJpeg


@pytest.fixture
def node():
    rclpy.init(args=[])
    instance = CameraJpeg()
    try:
        yield instance
    finally:
        instance.destroy_node()
        rclpy.shutdown()


def frame(node, value=90, age_s=0.0):
    msg = Image()
    msg.header.stamp = rclpy.time.Time(
        nanoseconds=node.get_clock().now().nanoseconds - int(age_s * 1e9)
    ).to_msg()
    msg.header.frame_id = "head_camera_optical"
    msg.height, msg.width, msg.step = 16, 16, 48
    msg.encoding = "rgb8"
    msg.data = bytes([value]) * (16 * 48)
    return msg


def test_default_parameters_and_qos(node):
    assert (node.max_fps, node.stale_s, node.quality) == (15.0, 0.5, 75)
    assert node.timer.timer_period_ns == int(1e9 / 15)
    assert node.subscription.topic_name == "/camera/color/image_raw"
    assert node.publisher.topic_name == "/camera/color/image_raw/compressed"
    for endpoint in (node.subscription, node.publisher):
        qos = endpoint.qos_profile
        assert qos.depth == 1
        assert qos.history == HistoryPolicy.KEEP_LAST
        assert qos.reliability == ReliabilityPolicy.BEST_EFFORT
        assert qos.durability == DurabilityPolicy.VOLATILE


def test_latest_only_and_header_copy(node):
    node.publisher = Mock()
    node._on_image(frame(node, 10))
    latest = frame(node, 180)
    node._on_image(latest)
    node._publish_latest()
    node._publish_latest()
    node.publisher.publish.assert_called_once()
    msg = node.publisher.publish.call_args.args[0]
    assert msg.header == latest.header
    assert msg.header is not latest.header
    assert msg.format == "rgb8; jpeg compressed bgr8"
    decoded = PilImage.open(io.BytesIO(bytes(msg.data)))
    assert decoded.getpixel((8, 8)) == (180, 180, 180)
    assert node._latest is None


def test_stale_capture_and_stale_slot_are_consumed(node, monkeypatch):
    node.publisher = Mock()
    node._on_image(frame(node, age_s=0.6))
    node._publish_latest()
    msg = frame(node)
    node._on_image(msg)
    received = node._latest[1]
    monkeypatch.setattr(jpeg_node.time, "monotonic", lambda: received + 0.6)
    node._publish_latest()
    assert node._latest is None
    node.publisher.publish.assert_not_called()


def test_frame_aging_during_encode_is_skipped(node, monkeypatch):
    node.publisher = Mock()
    node._on_image(frame(node))
    received = node._latest[1]
    encode = jpeg_node.encode_jpeg

    def slow_encode(*args):
        payload = encode(*args)
        monkeypatch.setattr(jpeg_node.time, "monotonic", lambda: received + 0.6)
        return payload

    monkeypatch.setattr(jpeg_node, "encode_jpeg", slow_encode)
    node._publish_latest()
    node.publisher.publish.assert_not_called()


def test_failures_are_rate_limited_and_recover(node, monkeypatch):
    node.publisher = Mock()
    logger = Mock()
    monkeypatch.setattr(node, "get_logger", lambda: logger)
    for encoding in ("mono8", "rgba8"):
        msg = frame(node)
        msg.encoding = encoding
        node._on_image(msg)
        node._publish_latest()
    logger.warn.assert_called_once()
    node.publisher.publish.assert_not_called()
    node._last_warning_at -= 11
    node._on_image(frame(node))
    node.publisher.publish.side_effect = RuntimeError("publish failed")
    node._publish_latest()
    assert logger.warn.call_count == 2
    node.publisher.publish.side_effect = None
    node._on_image(frame(node))
    node._publish_latest()
    assert node.publisher.publish.call_count == 2


def test_ros_transport_publishes_jpeg(node):
    peer = Node("camera_jpeg_test_peer")
    received = []
    publisher = peer.create_publisher(Image, node.subscription.topic_name, 1)
    peer.create_subscription(CompressedImage, node.publisher.topic_name,
                             received.append, node.publisher.qos_profile)
    try:
        deadline = time.monotonic() + 3.0
        while not received and time.monotonic() < deadline:
            publisher.publish(frame(node, 120))
            rclpy.spin_once(node, timeout_sec=0.02)
            rclpy.spin_once(peer, timeout_sec=0.02)
        assert received
        assert received[0].header.frame_id == "head_camera_optical"
        assert PilImage.open(io.BytesIO(bytes(received[0].data))).format == "JPEG"
    finally:
        peer.destroy_node()
