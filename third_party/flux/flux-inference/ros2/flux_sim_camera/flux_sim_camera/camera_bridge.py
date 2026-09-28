"""ROS 2 bridge: SONIC ``ego_view`` ZMQ frames -> stamped rgb8 Image messages.

The simulation-only counterpart of the robot's RealSense stream.  It subscribes
to the Isaac simulator's SONIC camera PUB socket and republishes each frame on
the ordinary ROS camera topic with the simulator's own capture timestamp, so the
``flux_dex3`` node's freshness gate measures the simulator, not the bridge.

Images carry the camera's optical frame (forward ``+Z``, REP 103), like the
robot's RealSense stream: the simulator authors the mount in Isaac's world
convention (forward ``+X``), and the TF stack in ``flux_sim_viz`` publishes the
fixed ``head_camera -> head_camera_optical`` rotation.  This node only names the
frame; it publishes no transforms.
"""

from __future__ import annotations

import array
import math
import threading
import time

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image

from flux_sim_camera.frame import FrameError, FrameGate, decode_frame

#: Warnings about a broken stream are rate-limited like the node's own sensor
#: warnings, so a stopped simulator cannot flood the log.
WARNING_INTERVAL_S = 10.0


class CameraBridge(Node):
    def __init__(self) -> None:
        super().__init__("flux_sim_camera")
        self.declare_parameter("endpoint", "tcp://127.0.0.1:5555")
        self.declare_parameter("topic", "/camera/color/image_raw")
        self.declare_parameter("camera_info_topic", "/camera/color/camera_info")
        self.declare_parameter("frame_id", "head_camera_optical")
        self.declare_parameter("focal_length_mm", 15.1159925892)
        self.declare_parameter("horizontal_aperture_mm", 20.955)
        self.declare_parameter("resync_s", 5.0)
        self.endpoint = str(self.get_parameter("endpoint").value)
        self.frame_id = str(self.get_parameter("frame_id").value)
        self.focal_length_mm = float(self.get_parameter("focal_length_mm").value)
        self.horizontal_aperture_mm = float(self.get_parameter("horizontal_aperture_mm").value)
        if not all(math.isfinite(value) and value > 0 for value in (
            self.focal_length_mm, self.horizontal_aperture_mm
        )):
            raise ValueError("camera focal length and horizontal aperture must be positive and finite")
        self.publisher = self.create_publisher(Image, str(self.get_parameter("topic").value), 2)
        self.info_publisher = self.create_publisher(
            CameraInfo, str(self.get_parameter("camera_info_topic").value), 2
        )
        self.gate = FrameGate(float(self.get_parameter("resync_s").value))
        self.received = 0
        self.decode_errors = 0
        self._stop = threading.Event()
        self._last_warning = ""
        self._last_warning_at = 0.0
        self._last_logged_count = 0
        self._first_frame_logged = False
        self._thread = threading.Thread(target=self._receive_loop, name="sonic-camera-sub", daemon=True)
        self._thread.start()
        self.create_timer(10.0, self._heartbeat)
        self.get_logger().info(
            "Camera bridge started: endpoint=%s topic=%s frame=%s"
            % (self.endpoint, self.publisher.topic_name, self.frame_id)
        )

    def _warn(self, reason: str) -> None:
        now = time.monotonic()
        if reason != self._last_warning or now - self._last_warning_at >= WARNING_INTERVAL_S:
            self.get_logger().warn("Camera frame rejected: %s" % reason)
            self._last_warning, self._last_warning_at = reason, now

    def _heartbeat(self) -> None:
        age = self.gate.age_s(time.time())
        if self.gate.published == 0:
            if self.received == 0:
                self.get_logger().info(
                    "Waiting for simulator camera frames on %s (received=0)" % self.endpoint
                )
            return
        if self.gate.published != self._last_logged_count:
            self._last_logged_count = self.gate.published
            return
        # Nothing new was published in the last window: say so rather than
        # letting a stalled simulator look alive.
        self.get_logger().warn(
            "No new camera frame for %.1fs (received=%d published=%d skipped=%d)"
            % (age if age is not None else -1.0, self.received, self.gate.published, self.gate.skipped)
        )

    def _receive_loop(self) -> None:
        import zmq

        context = zmq.Context()
        socket = context.socket(zmq.SUB)
        socket.setsockopt(zmq.SUBSCRIBE, b"")
        socket.setsockopt(zmq.RCVHWM, 2)
        socket.setsockopt(zmq.LINGER, 0)
        socket.connect(self.endpoint)
        poller = zmq.Poller()
        poller.register(socket, zmq.POLLIN)
        try:
            while not self._stop.is_set():
                if not poller.poll(100):
                    continue
                payload = socket.recv()
                try:
                    frame = decode_frame(payload)
                except FrameError as exc:
                    self.decode_errors += 1
                    self._warn(str(exc))
                    continue
                self.received += 1
                if not self.gate.accept(frame.stamp_s):
                    continue
                if not self._first_frame_logged:
                    self._first_frame_logged = True
                    self.get_logger().info(
                        "First SONIC ego_view frame: %dx%d stamp=%.3f"
                        % (frame.rgb.shape[1], frame.rgb.shape[0], frame.stamp_s)
                    )
                image = self._image(frame.rgb, frame.stamp_s)
                self.info_publisher.publish(self._camera_info(image))
                self.publisher.publish(image)
        finally:
            socket.close(linger=0)
            context.term()

    def _image(self, rgb, stamp_s: float):
        message = Image()
        if self.frame_id:
            message.header.frame_id = self.frame_id
        whole = int(stamp_s)
        message.header.stamp.sec = whole
        message.header.stamp.nanosec = int(round((stamp_s - whole) * 1e9)) % 1_000_000_000
        message.height, message.width = int(rgb.shape[0]), int(rgb.shape[1])
        message.encoding = "rgb8"
        message.is_bigendian = 0
        message.step = message.width * 3
        # array.array assignment is a bulk copy of the exact byte image.
        message.data = array.array("B", rgb.tobytes())
        return message

    def _camera_info(self, image: Image) -> CameraInfo:
        info = CameraInfo()
        info.header = image.header
        info.width, info.height = image.width, image.height
        info.distortion_model = "plumb_bob"
        info.d = [0.0] * 5
        focal_px = self.focal_length_mm * image.width / self.horizontal_aperture_mm
        cx, cy = image.width / 2.0, image.height / 2.0
        info.k = [focal_px, 0.0, cx, 0.0, focal_px, cy, 0.0, 0.0, 1.0]
        info.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        info.p = [focal_px, 0.0, cx, 0.0, 0.0, focal_px, cy, 0.0, 0.0, 0.0, 1.0, 0.0]
        return info

    def destroy_node(self) -> None:
        self._stop.set()
        self._thread.join(timeout=3.0)
        super().destroy_node()


def main() -> None:
    rclpy.init()
    node = None
    try:
        node = CameraBridge()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # The signal that ended spin shuts the context down and can land again
        # during teardown; that is a normal stop, not a crash worth a traceback.
        try:
            if node is not None:
                node.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()
        except (KeyboardInterrupt, RuntimeError):
            pass
