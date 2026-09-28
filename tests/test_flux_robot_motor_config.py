"""Keep the gated robot motor settings aligned with the pinned SONIC sources."""

import json
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "third_party/flux/flux-inference/ros2/flux_dex3"))

from flux_dex3.command_output import load_motor_config
from flux_dex3.mapping import JOINT_NAMES
from humanoid_lab.controllers.sonic import deploy_gains


def test_robot_config_matches_sonic_and_loads():
    config_path = ROOT / "configs/flux/flux-dex3-robot-motor-config.json"
    config = json.loads(config_path.read_text())
    joints = json.loads((ROOT / "configs/datasets/sonic/g1_joint_limits.json").read_text())["joints"]

    def sonic_name(name):
        return re.sub(r"(?<=[a-z])(?=\d)", "_",
                      re.sub(r"(?<=[a-z])(?=[A-Z])", "_", name[1:])).lower() + "_joint"

    assert config["joint_limits_rad"] == [
        [joints[sonic_name(name)]["lower"], joints[sonic_name(name)]["upper"]]
        for name in JOINT_NAMES
    ]
    kp, kd = deploy_gains()
    assert config["arm_kp"] == list(kp[15:29])
    assert config["arm_kd"] == list(kd[15:29])
    assert config["hand_kp"] == [1.5] * 14
    assert config["hand_kd"] == [0.1] * 14
    assert config["hand_timeout_enabled"] is False
    assert config["arm_mode_machine"] == 5
    assert config["max_tracking_error_rad"] == 4.0
    assert load_motor_config(config_path) == config
