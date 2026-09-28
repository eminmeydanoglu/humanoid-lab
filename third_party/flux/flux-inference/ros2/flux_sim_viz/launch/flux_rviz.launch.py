"""RViz window for the simulation view.

RViz only: the frames it draws -- `world`, the URDF links, the head camera and
its optical child -- come from the TF stack that the canonical ROS launch
starts (`flux_sim_camera/flux_sim.launch.py`, `tf:=true` by default).  Each
frame has exactly one publisher, so this launch does not start a second copy;
``tf:=true`` here is only for opening RViz when that launch is not running.

Nothing here commands the robot.
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
    share = get_package_share_directory("flux_sim_viz")
    rviz_config = LaunchConfiguration("rviz_config")
    rviz = LaunchConfiguration("rviz")
    tf = LaunchConfiguration("tf")
    profile = LaunchConfiguration("profile")
    root_height_m = LaunchConfiguration("root_height_m")

    return LaunchDescription([
        DeclareLaunchArgument(
            "rviz_config", default_value=os.path.join(share, "rviz", "flux_sim.rviz")
        ),
        DeclareLaunchArgument("rviz", default_value="true"),
        # Off by default: flux_sim.launch.py owns the TF stack in the canonical
        # workflow, and a second stack would duplicate every publisher.
        DeclareLaunchArgument(
            "tf",
            default_value="false",
            description="start the TF stack here; only without the canonical launch's",
        ),
        DeclareLaunchArgument("profile", default_value=str(DEFAULT_PROFILE)),
        DeclareLaunchArgument("root_height_m", default_value=""),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(share, "launch", "flux_tf.launch.py")
            ),
            launch_arguments={
                "profile": profile,
                "root_height_m": root_height_m,
            }.items(),
            condition=IfCondition(tf),
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
