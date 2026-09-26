"""Offline Unitree packet construction; publishing is tested with fake ROS handles."""

import json
import sys
from types import ModuleType

import numpy as np
import pytest
from flux_dex3.command_output import CommandOutput, load_motor_config, lowcmd_crc


class Motor:
    def __init__(self):
        self.mode = self.reserve = 0
        self.q = self.dq = self.tau = self.kp = self.kd = 0.0


class LowCmd:
    def __init__(self):
        self.mode_pr = self.mode_machine = self.crc = 0
        self.motor_cmd = [Motor() for _ in range(35)]
        self.reserve = [0] * 4


class HandCmd:
    def __init__(self):
        self.motor_cmd = []


class FakePublisher:
    def __init__(self):
        self.messages = []

    def publish(self, msg):
        self.messages.append(msg)


class FakeNode:
    def __init__(self):
        self.publishers = {}

    def create_publisher(self, msg_type, topic, qos):
        assert qos == 10
        self.publishers[topic] = FakePublisher()
        return self.publishers[topic]


@pytest.fixture
def config(tmp_path):
    path = tmp_path / "verified_config.json"
    path.write_text(json.dumps({
        "control_authority_confirmed": True,
        "hand_revision_confirmed": True,
        "joint_limits_rad": [[-2.5, 2.5]] * 28,
        "max_tracking_error_rad": 0.3,
        "arm_kp": [60.0] * 14,
        "arm_kd": [1.5] * 14,
        "hand_kp": [1.5] * 14,
        "hand_kd": [0.2] * 14,
        "hand_timeout_enabled": False,
        "arm_mode_machine": 0,
    }))
    return path


def test_crc_matches_robot_vendor_python_reference():
    cmd = LowCmd()
    cmd.motor_cmd[29].q = 1.0
    cmd.motor_cmd[15].q = 0.25
    cmd.motor_cmd[15].kp = 60
    cmd.motor_cmd[15].kd = 1.5
    assert lowcmd_crc(cmd) == 0xCEE03A29


def test_verified_config_required_before_publishers(config, monkeypatch):
    package = ModuleType("unitree_hg")
    messages = ModuleType("unitree_hg.msg")
    messages.LowCmd, messages.HandCmd, messages.MotorCmd = LowCmd, HandCmd, Motor
    monkeypatch.setitem(sys.modules, "unitree_hg", package)
    monkeypatch.setitem(sys.modules, "unitree_hg.msg", messages)
    node = FakeNode()
    output = CommandOutput(node, config)
    assert set(node.publishers) == {"/arm_sdk", "/dex3/left/cmd", "/dex3/right/cmd"}
    target = np.zeros(28, dtype=np.float32)
    target[0], target[14], target[24] = 0.1, -0.2, 0.2
    output.publish(target, np.zeros(28, dtype=np.float32))
    arm = node.publishers["/arm_sdk"].messages[0]
    assert arm.motor_cmd[15].q == pytest.approx(0.1)
    assert arm.motor_cmd[29].q == 1.0 and arm.crc == lowcmd_crc(arm)
    assert node.publishers["/dex3/left/cmd"].messages[0].motor_cmd[0].mode == 0x10
    assert node.publishers["/dex3/right/cmd"].messages[0].motor_cmd[3].mode == 0x13
    with pytest.raises(ValueError, match="tracking-error"):
        output.publish(np.ones(28, dtype=np.float32), np.zeros(28, dtype=np.float32))
    assert len(node.publishers["/arm_sdk"].messages) == 1


def test_invalid_hardware_contract_fails_closed(config):
    data = json.loads(config.read_text())
    data["control_authority_confirmed"] = False
    config.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="ownership"):
        load_motor_config(config)
    data["control_authority_confirmed"] = True
    data["joint_limits_rad"][0] = [0, 0]
    config.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="limits"):
        load_motor_config(config)
