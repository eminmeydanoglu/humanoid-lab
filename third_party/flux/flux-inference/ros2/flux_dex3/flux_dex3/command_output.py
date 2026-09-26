"""Explicitly gated Unitree arm/hand position command output."""

import json
import math
import struct
from pathlib import Path

import numpy as np

from flux_dex3.mapping import split_action


def lowcmd_crc(cmd):
    """Unitree hg LowCmd CRC32 over little-endian 32-bit words, excluding crc."""
    values = [cmd.mode_pr, cmd.mode_machine]
    for motor in cmd.motor_cmd:
        values.extend((motor.mode, motor.q, motor.dq, motor.tau, motor.kp, motor.kd, motor.reserve))
    values.extend(cmd.reserve)
    values.append(0)
    packed = struct.pack("<2B2x" + "B3x5fI" * 35 + "5I", *values)
    crc = 0xFFFFFFFF
    for offset in range(0, len(packed) - 4, 4):
        word = struct.unpack_from("<I", packed, offset)[0]
        bit = 1 << 31
        for _ in range(32):
            if crc & 0x80000000:
                crc = ((crc << 1) & 0xFFFFFFFF) ^ 0x04C11DB7
            else:
                crc = (crc << 1) & 0xFFFFFFFF
            if word & bit:
                crc ^= 0x04C11DB7
            bit >>= 1
    return crc


def load_motor_config(path):
    """Require a complete, hardware-specific motor contract before creating publishers."""
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    expected = {"control_authority_confirmed", "hand_revision_confirmed", "joint_limits_rad",
                "max_tracking_error_rad", "arm_kp", "arm_kd", "hand_kp", "hand_kd",
                "hand_timeout_enabled", "arm_mode_machine"}
    if not isinstance(config, dict) or set(config) != expected:
        raise ValueError("motor config fields do not match the required contract")
    if config["control_authority_confirmed"] is not True or config["hand_revision_confirmed"] is not True:
        raise ValueError("control ownership and hand revision must be confirmed")
    for name, size in (("joint_limits_rad", 28), ("arm_kp", 14), ("arm_kd", 14),
                       ("hand_kp", 14), ("hand_kd", 14)):
        values = config[name]
        if not isinstance(values, list) or len(values) != size:
            raise ValueError("invalid %s" % name)
        if name == "joint_limits_rad":
            if any(not isinstance(pair, list) or len(pair) != 2 or
                   not all(type(value) in (int, float) and math.isfinite(value) for value in pair) or
                   pair[0] >= pair[1] for pair in values):
                raise ValueError("invalid joint limits")
        elif any(type(value) not in (int, float) or not math.isfinite(value) or value <= 0 for value in values):
            raise ValueError("invalid %s" % name)
    maximum = config["max_tracking_error_rad"]
    if type(maximum) not in (int, float) or not math.isfinite(maximum) or maximum <= 0:
        raise ValueError("invalid max_tracking_error_rad")
    if type(config["hand_timeout_enabled"]) is not bool:
        raise ValueError("hand_timeout_enabled must be boolean")
    mode = config["arm_mode_machine"]
    if type(mode) is not int or not 0 <= mode <= 255:
        raise ValueError("invalid arm_mode_machine")
    return config


class CommandOutput:
    def __init__(self, node, config_path):
        from unitree_hg.msg import HandCmd, LowCmd, MotorCmd

        self.config = load_motor_config(config_path)
        self.hand_type, self.arm_type, self.motor_type = HandCmd, LowCmd, MotorCmd
        self.arm_pub = node.create_publisher(LowCmd, "/arm_sdk", 10)
        self.left_pub = node.create_publisher(HandCmd, "/dex3/left/cmd", 10)
        self.right_pub = node.create_publisher(HandCmd, "/dex3/right/cmd", 10)

    def publish(self, target, measured):
        target = np.asarray(target, dtype=np.float32)
        measured = np.asarray(measured, dtype=np.float32)
        if target.shape != (28,) or measured.shape != (28,) or not np.isfinite(target).all() or not np.isfinite(measured).all():
            raise ValueError("invalid command or measured state")
        limits = np.asarray(self.config["joint_limits_rad"], dtype=np.float32)
        if np.any(target < limits[:, 0]) or np.any(target > limits[:, 1]):
            raise ValueError("target exceeds verified hardware limits")
        if np.any(np.abs(target - measured) > self.config["max_tracking_error_rad"]):
            raise ValueError("target exceeds measured tracking-error bound")
        arm_targets, left_targets, right_targets = split_action(target)
        arm = self.arm_type()
        if len(arm.motor_cmd) != 35:
            raise ValueError("expected 35 arm command slots")
        arm.mode_machine = self.config["arm_mode_machine"]
        arm.motor_cmd[29].q = 1.0
        for offset, (index, q) in enumerate(arm_targets.items()):
            motor = arm.motor_cmd[index]
            motor.q = float(q)
            motor.dq = motor.tau = 0.0
            motor.kp = float(self.config["arm_kp"][offset])
            motor.kd = float(self.config["arm_kd"][offset])
        arm.crc = lowcmd_crc(arm)

        hands = []
        for side, targets, base in (("left", left_targets, 0), ("right", right_targets, 7)):
            hand = self.hand_type()
            hand.motor_cmd = [self.motor_type() for _ in range(7)]
            if len(hand.motor_cmd) != 7:
                raise ValueError("expected seven %s hand command slots" % side)
            for index, q in targets.items():
                motor = hand.motor_cmd[index]
                motor.mode = index | 0x10 | (0x80 if self.config["hand_timeout_enabled"] else 0)
                motor.q = float(q)
                motor.dq = motor.tau = 0.0
                motor.kp = float(self.config["hand_kp"][base + index])
                motor.kd = float(self.config["hand_kd"][base + index])
            hands.append(hand)
        self.arm_pub.publish(arm)
        self.left_pub.publish(hands[0])
        self.right_pub.publish(hands[1])
