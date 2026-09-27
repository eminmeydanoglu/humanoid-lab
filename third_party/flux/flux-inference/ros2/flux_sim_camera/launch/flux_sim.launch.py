"""Simulation-only launch: SONIC camera bridge + the (same) flux_dex3 node.

The node code is the robot's own package; only this configuration is
simulation-specific (loopback endpoints, the simulator's camera wire format,
and the optional command-enabling motor config).  Robot defaults are untouched:
the model endpoint here is explicit at 5561 because the server default 5557 is
the SONIC state port in this environment.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    model_endpoint = LaunchConfiguration("model_endpoint")
    camera_endpoint = LaunchConfiguration("camera_endpoint")
    camera_topic = LaunchConfiguration("camera_topic")
    motor_output_config = LaunchConfiguration("motor_output_config")
    enable_motor_commands = LaunchConfiguration("enable_motor_commands")
    freshness_s = LaunchConfiguration("freshness_s")
    max_chunk_age_s = LaunchConfiguration("max_chunk_age_s")
    pair_tolerance_s = LaunchConfiguration("pair_tolerance_s")
    camera_bridge = LaunchConfiguration("camera_bridge")
    frame_id = LaunchConfiguration("frame_id")

    return LaunchDescription([
        DeclareLaunchArgument("model_endpoint", default_value="tcp://127.0.0.1:5561"),
        DeclareLaunchArgument("camera_endpoint", default_value="tcp://127.0.0.1:5555"),
        DeclareLaunchArgument("camera_topic", default_value="/camera/color/image_raw"),
        DeclareLaunchArgument("frame_id", default_value="head_camera"),
        # The command gate stays exactly what it is on the robot: publishers
        # exist only with an explicit, complete motor config.  An empty path is
        # the command-disabled dry run.
        DeclareLaunchArgument("motor_output_config", default_value=""),
        DeclareLaunchArgument("enable_motor_commands", default_value="false"),
        # Isaac render contention occasionally delays a frame past 0.25 s;
        # this remains bounded and leaves the robot node's own default intact.
        DeclareLaunchArgument("freshness_s", default_value="0.5"),
        # Observation age reached 1.5 s during the livestream run.
        DeclareLaunchArgument("max_chunk_age_s", default_value="1.6"),
        DeclareLaunchArgument("pair_tolerance_s", default_value="0.1"),
        DeclareLaunchArgument("camera_bridge", default_value="true"),
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
            package="flux_dex3",
            executable="dex3_node",
            name="flux_dex3",
            output="screen",
            parameters=[{
                "endpoint": model_endpoint,
                "camera_topic": camera_topic,
                "freshness_s": freshness_s,
                "max_chunk_age_s": max_chunk_age_s,
                "pair_tolerance_s": pair_tolerance_s,
                "enable_motor_commands": enable_motor_commands,
                "motor_output_config": motor_output_config,
            }],
        ),
    ])
