"""SONIC ZMQ Protocol v3 (SMPL) message packing + publisher.

Wire format (see gear_sonic/utils/teleop/zmq/zmq_planner_sender.py):
    [topic_bytes][1280-byte JSON header][concatenated little-endian binary fields]

We prefer SONIC's own `pack_pose_message` when `gear_sonic` is importable
(run with `--sonic-root /opt/src/sonic`); otherwise we fall back to a
byte-identical local implementation so the bridge can also run standalone.

v3 fields (encoder mode 2, `smpl`):
    smpl_joints (1,24,3) f32   root-local, Z-up
    smpl_pose   (1,21,3) f32   zeros (not used by the SMPL encoder)
    joint_pos   (1,29)   f32   zeros except [23:29] = 6 wrist values
    joint_vel   (1,29)   f32   zeros
    body_quat   (1,4)    f32   root orientation (SONIC convention)
    frame_index (1,)     i64
"""

from __future__ import annotations

import json
import sys

import numpy as np

HEADER_SIZE = 1280  # SONIC ZMQ header size

_DTYPE_STR = {
    np.dtype(np.float32): "f32",
    np.dtype(np.float64): "f64",
    np.dtype(np.int32): "i32",
    np.dtype(np.int64): "i64",
    np.dtype(bool): "bool",
}


def _fallback_pack_pose_message(pose_data: dict, topic: str = "pose",
                                version: int = 3) -> bytes:
    fields, binary = [], []
    for key, value in pose_data.items():
        if not isinstance(value, np.ndarray):
            continue
        dtype_str = _DTYPE_STR.get(value.dtype, "f32")
        if dtype_str == "f32" and value.dtype != np.float32:
            value = value.astype(np.float32)
        if not value.flags["C_CONTIGUOUS"]:
            value = np.ascontiguousarray(value)
        fields.append({"name": key, "dtype": dtype_str, "shape": list(value.shape)})
        binary.append(value.tobytes())
    header = {"v": version, "endian": "le", "count": 1, "fields": fields}
    header_json = json.dumps(header, separators=(",", ":")).encode("utf-8")
    if len(header_json) > HEADER_SIZE:
        raise ValueError(f"Header too large: {len(header_json)} > {HEADER_SIZE}")
    return (topic.encode("utf-8")
            + header_json.ljust(HEADER_SIZE, b"\x00")
            + b"".join(binary))


def get_packer(sonic_root: str | None = None):
    """Return SONIC's pack_pose_message if importable, else the fallback."""
    if sonic_root and sonic_root not in sys.path:
        sys.path.insert(0, sonic_root)
    try:
        from gear_sonic.utils.teleop.zmq.zmq_planner_sender import pack_pose_message
        print("[protocol] using SONIC pack_pose_message (exact wire format)")
        return pack_pose_message
    except Exception as exc:  # noqa: BLE001
        print(f"[protocol] gear_sonic not importable ({exc}); using local packer")
        return _fallback_pack_pose_message


def build_v3_pose_data(smpl_joints: np.ndarray, body_quat: np.ndarray,
                       frame_index: int, wrists: np.ndarray | None = None) -> dict:
    """Assemble the v3 field dict (arrays only, all per-frame batch=1)."""
    smpl_joints = np.asarray(smpl_joints, dtype=np.float32).reshape(1, 24, 3)
    body_quat = np.asarray(body_quat, dtype=np.float32).reshape(1, 4)

    joint_pos = np.zeros((1, 29), dtype=np.float32)
    if wrists is not None:
        joint_pos[:, 23:29] = np.asarray(wrists, dtype=np.float32).reshape(1, 6)

    return {
        "smpl_joints": smpl_joints,
        "smpl_pose": np.zeros((1, 21, 3), dtype=np.float32),
        "joint_pos": joint_pos,
        "joint_vel": np.zeros((1, 29), dtype=np.float32),
        "body_quat": body_quat,
        "frame_index": np.array([int(frame_index)], dtype=np.int64),
    }


class V3Publisher:
    """ZMQ PUB publisher for SONIC Protocol v3."""

    def __init__(self, port: int = 5556, topic: str = "pose",
                 sonic_root: str | None = None):
        import zmq
        self.pack = get_packer(sonic_root)
        self.topic = topic
        self.frame_index = 0
        self.ctx = zmq.Context()
        self.sock = self.ctx.socket(zmq.PUB)
        self.sock.bind(f"tcp://*:{port}")
        print(f"[protocol] publishing '{topic}' v3 on tcp://*:{port}")

    def publish(self, smpl_joints, body_quat, wrists=None) -> int:
        data = build_v3_pose_data(smpl_joints, body_quat, self.frame_index, wrists)
        self.sock.send(self.pack(data, topic=self.topic, version=3))
        self.frame_index += 1
        return self.frame_index - 1

    def close(self) -> None:
        self.sock.close(0)
        self.ctx.term()
