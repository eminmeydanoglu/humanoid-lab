"""Canonical simulation-only launch: TF stack + SONIC camera bridge + flux_dex3.

The node code is the robot's own package; only this configuration is
simulation-specific (loopback endpoints, the simulator's camera wire format,
and the optional command-enabling motor config).  Robot defaults are untouched:
the model endpoint here is explicit at 5561 because the server default 5557 is
the SONIC state port in this environment.

The TF stack (robot model, measured joint state, world root and head-camera
statics) is started here by default, so a running simulator is fully visible
without the optional RViz launch.  `profile` must be the profile the simulator
runs, so the fixed-root pose and camera mount match the scene on screen.  The
RViz launch is a pure RViz client of these frames and starts no second stack.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from flux_sim_viz.tf_args import DEFAULT_PROFILE


def generate_launch_description():
    model_endpoint = LaunchConfiguration("model_endpoint")
    camera_endpoint = LaunchConfiguration("camera_endpoint")
    camera_topic = LaunchConfiguration("camera_topic")
    motor_output_config = LaunchConfiguration("motor_output_config")
    enable_motor_commands = LaunchConfiguration("enable_motor_commands")
    freshness_s = LaunchConfiguration("freshness_s")
    network_timeout_s = LaunchConfiguration("network_timeout_s")
    max_chunk_age_s = LaunchConfiguration("max_chunk_age_s")
    pair_tolerance_s = LaunchConfiguration("pair_tolerance_s")
    camera_bridge = LaunchConfiguration("camera_bridge")
    frame_id = LaunchConfiguration("frame_id")
    tf = LaunchConfiguration("tf")
    profile = LaunchConfiguration("profile")
    root_height_m = LaunchConfiguration("root_height_m")
    tf_share = get_package_share_directory("flux_sim_viz")

    return LaunchDescription([
        DeclareLaunchArgument("model_endpoint", default_value="tcp://127.0.0.1:5561"),
        DeclareLaunchArgument("camera_endpoint", default_value="tcp://127.0.0.1:5555"),
        DeclareLaunchArgument("camera_topic", default_value="/camera/color/image_raw"),
        # Images carry the camera's optical child (forward +Z, REP 103), like the
        # robot's RealSense stream; the TF stack publishes that frame.
        DeclareLaunchArgument("frame_id", default_value="head_camera_optical"),
        # The robot model, joint state and statics are part of this launch; the
        # optional RViz launch attaches to them and must not publish its own.
        DeclareLaunchArgument("tf", default_value="true"),
        # The simulator's profile: the fixed-root pose and camera mount come from
        # it.  Relative paths are repository-relative.
        DeclareLaunchArgument("profile", default_value=str(DEFAULT_PROFILE)),
        DeclareLaunchArgument("root_height_m", default_value=""),
        # The command gate stays exactly what it is on the robot: publishers
        # exist only with an explicit, complete motor config.  An empty path is
        # the command-disabled dry run.
        DeclareLaunchArgument("motor_output_config", default_value=""),
        DeclareLaunchArgument("enable_motor_commands", default_value="false"),
        # Isaac render contention occasionally delays a frame past 0.25 s;
        # this remains bounded and leaves the robot node's own default intact.
        DeclareLaunchArgument("freshness_s", default_value="0.5"),
        # V2 sampling takes about 2.4 seconds while Isaac renders.
        DeclareLaunchArgument("network_timeout_s", default_value="3.0"),
        DeclareLaunchArgument("max_chunk_age_s", default_value="3.0"),
        DeclareLaunchArgument("pair_tolerance_s", default_value="0.1"),
        DeclareLaunchArgument("camera_bridge", default_value="true"),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(tf_share, "launch", "flux_tf.launch.py")
            ),
            launch_arguments={
                "profile": profile,
                "root_height_m": root_height_m,
            }.items(),
            condition=IfCondition(tf),
        ),
        Node(
            package="flux_sim_camera",
            executable="camera_bridge",
            name="flux_sim_camera",
            output="screen",
            condition=IfCondition(camera_bridge),
            parameters=[{
                "endpoint": camera_endpoint,
                "topic": camera_topic,
                "frame_id": frame_id,
            }],
        ),
        Node(
            package="flux_sim_camera",
            executable="camera_jpeg",
            name="flux_sim_camera_jpeg",
            output="screen",
            parameters=[{"raw_topic": camera_topic}],
        ),
        Node(
            package="flux_dex3",
            executable="dex3_node",
            name="flux_dex3",
            output="screen",
            remappings=[("/lowcmd", "/arm_sdk")],
            parameters=[{
                "endpoint": model_endpoint,
                "camera_topic": camera_topic,
                "freshness_s": freshness_s,
                "network_timeout_s": network_timeout_s,
                "max_chunk_age_s": max_chunk_age_s,
                "pair_tolerance_s": pair_tolerance_s,
                "enable_motor_commands": enable_motor_commands,
                "motor_output_config": motor_output_config,
            }],
        ),
    ])
