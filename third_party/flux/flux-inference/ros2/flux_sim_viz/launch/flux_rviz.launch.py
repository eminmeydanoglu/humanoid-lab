"""Simulation-only RViz view: the fixed-root G1, its joint state and the head camera.

The robot is the simulator's: measured joint state arrives over the same
unitree_hg topics the robot's own node reads, and the fixed chest mount is
published as a static `world -> pelvis` transform at the profile's own root
height.  Nothing here commands the robot.
"""

import json
import os
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from flux_sim_viz.urdf import DEFAULT_URDF, load_robot_description


DEFAULT_PROFILE = Path("/workspace/humanoid-lab/configs/profiles/isaac-g1-flux-dex3-pickapple.json")


def camera_mount(profile_path: Path = DEFAULT_PROFILE):
    """Resolve the canonical scene's inherited head-camera mount."""
    path = profile_path
    while True:
        profile = json.loads(path.read_text(encoding="utf-8"))
        if "camera" in profile:
            return profile["camera"]
        path = path.with_name(profile["base_profile"])


def generate_launch_description():
    share = get_package_share_directory("flux_sim_viz")
    camera = camera_mount()
    w, x, y, z = camera["rotation_wxyz"]
    root_height_m = LaunchConfiguration("root_height_m")
    rviz_config = LaunchConfiguration("rviz_config")
    rviz = LaunchConfiguration("rviz")

    return LaunchDescription([
        # The canonical profiles pin the root at this height; override it for a
        # scene whose profile places the robot elsewhere.
        DeclareLaunchArgument("root_height_m", default_value="0.792563"),
        DeclareLaunchArgument(
            "rviz_config", default_value=os.path.join(share, "rviz", "flux_sim.rviz")
        ),
        DeclareLaunchArgument("rviz", default_value="true"),
        Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            name="flux_sim_robot_state_publisher",
            output="screen",
            parameters=[{"robot_description": load_robot_description(DEFAULT_URDF)}],
        ),
        Node(
            package="tf2_ros",
            executable="static_transform_publisher",
            name="flux_sim_world_to_pelvis",
            arguments=[
                "--x", "0", "--y", "0", "--z", root_height_m,
                "--roll", "0", "--pitch", "0", "--yaw", "0",
                "--frame-id", "world", "--child-frame-id", "pelvis",
            ],
        ),
        Node(
            package="tf2_ros",
            executable="static_transform_publisher",
            name="flux_sim_torso_to_head_camera",
            arguments=[
                "--x", str(camera["position_m"][0]),
                "--y", str(camera["position_m"][1]),
                "--z", str(camera["position_m"][2]),
                "--qx", str(x), "--qy", str(y), "--qz", str(z), "--qw", str(w),
                "--frame-id", camera["parent_link"],
                "--child-frame-id", camera["name"],
            ],
        ),
        Node(
            package="flux_sim_viz",
            executable="joint_state_bridge",
            name="flux_sim_joint_state_bridge",
            output="screen",
        ),
        Node(
            package="rviz2",
            executable="rviz2",
            name="flux_sim_rviz",
            output="screen",
            arguments=["-d", rviz_config],
            condition=IfCondition(rviz),
        ),
    ])
