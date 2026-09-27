"""Node-level tests for the SONIC ego_view -> ROS Image bridge.

Runs in the flux-ros container (rclpy + pyzmq + Pillow): a fake simulator PUB
socket feeds the real node, and a ROS subscriber checks what was published.
"""

import io
import time

import msgpack
import numpy as np
import pytest
import rclpy
import zmq
from flux_sim_camera.camera_bridge import CameraBridge
from PIL import Image
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image as RosImage

TOPIC = "/flux_sim_camera/test/image"
INFO_TOPIC = "/flux_sim_camera/test/camera_info"


def pack(stamp_s, value=90):
    image = np.full((480, 640, 3), value, dtype=np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(image).save(buffer, format="JPEG", quality=95)
    return msgpack.packb(
        {"timestamps": {"ego_view": stamp_s}, "images": {"ego_view": buffer.getvalue()}},
        use_bin_type=True,
    )


class Collector(Node):
    def __init__(self):
        super().__init__("flux_sim_camera_test_collector")
        self.frames = []
        self.camera_info = []
        self.create_subscription(RosImage, TOPIC, self._on_image, 10)
        self.create_subscription(CameraInfo, INFO_TOPIC, self.camera_info.append, 10)

    def _on_image(self, msg):
        self.frames.append(msg)


@pytest.fixture
def simulator_pub():
    context = zmq.Context()
    socket = context.socket(zmq.PUB)
    socket.setsockopt(zmq.LINGER, 0)
    socket.bind("tcp://127.0.0.1:*")
    endpoint = socket.getsockopt_string(zmq.LAST_ENDPOINT)
    try:
        yield socket, endpoint
    finally:
        socket.close(linger=0)
        context.term()


def _spin(nodes, seconds):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        for node in nodes:
            rclpy.spin_once(node, timeout_sec=0.02)


def test_bridge_publishes_only_new_frames(simulator_pub):
    socket, endpoint = simulator_pub
    rclpy.init(args=["--ros-args", "-p", "endpoint:=" + endpoint,
                     "-p", "topic:=" + TOPIC, "-p", "camera_info_topic:=" + INFO_TOPIC,
                     "-p", "frame_id:=head_camera"])
    bridge = CameraBridge()
    collector = Collector()
    try:
        # Let the SUB connection settle: everything published before it is
        # connected is dropped by design.
        _spin([bridge, collector], 0.5)

        # A repeated stamp is the same frame: publish it many times and require
        # exactly one ROS message (duplicates are never re-dated).
        for _ in range(20):
            socket.send(pack(100.0))
            _spin([bridge, collector], 0.05)
        _spin([bridge, collector], 0.3)
        assert len(collector.frames) == 1
        first = collector.frames[0]
        assert first.encoding == "rgb8"
        assert (first.height, first.width) == (480, 640)
        assert first.step == 640 * 3
        assert (first.header.stamp.sec, first.header.stamp.nanosec) == (100, 0)
        assert first.header.frame_id == "head_camera"
        assert len(collector.camera_info) == 1
        info = collector.camera_info[0]
        assert info.header == first.header
        assert (info.width, info.height) == (640, 480)
        assert info.distortion_model == "plumb_bob"
        assert list(info.d) == [0.0] * 5
        focal_px = 15.1159925892 * 640 / 20.955
        assert info.k == pytest.approx([focal_px, 0, 320, 0, focal_px, 240, 0, 0, 1])
        assert info.p == pytest.approx([focal_px, 0, 320, 0, 0, focal_px, 240, 0, 0, 0, 1, 0])

        # An older stamp must not be published as a newer frame.
        for _ in range(10):
            socket.send(pack(99.5))
            _spin([bridge, collector], 0.05)
        _spin([bridge, collector], 0.3)
        assert len(collector.frames) == 1
        assert len(collector.camera_info) == 1

        # With no frame arriving nothing is published, so no freshness is faked.
        _spin([bridge, collector], 1.0)
        assert len(collector.frames) == 1
        assert len(collector.camera_info) == 1

        # The next genuine frame is published with its own stamp.
        for _ in range(20):
            socket.send(pack(100.25))
            _spin([bridge, collector], 0.05)
        _spin([bridge, collector], 0.3)
        assert len(collector.frames) == 2
        assert collector.frames[1].header.stamp.sec == 100
        assert collector.frames[1].header.stamp.nanosec == 250_000_000
        assert len(collector.camera_info) == 2
        assert collector.camera_info[1].header == collector.frames[1].header
    finally:
        bridge.destroy_node()
        collector.destroy_node()
        rclpy.shutdown()
