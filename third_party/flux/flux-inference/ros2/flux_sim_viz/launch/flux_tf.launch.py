"""The simulation's TF authority: robot model, measured joint state, statics.

Started by the canonical ROS launch (`flux_sim_camera/flux_sim.launch.py`,
`tf:=true` by default) and available to the optional RViz launch, which
otherwise only opens RViz.  Run it once: every frame it owns -- `world`, the
URDF links, `head_camera` and its optical child -- has exactly one publisher
here, so it is not started a second time alongside another copy.

The profile argument is the same file the simulator runs, so the fixed-root
pose and the head-camera mount always describe the scene on screen.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from flux_sim_viz.tf_args import (
    DEFAULT_PROFILE,
    camera_mount,
    camera_mount_arguments,
    camera_optical_arguments,
    resolve_profile,
    world_to_pelvis_arguments,
)
from flux_sim_viz.urdf import DEFAULT_URDF, load_robot_description


def _nodes(context):
    profile = resolve_profile(
        context.perform_substitution(LaunchConfiguration("profile"))
    )
    height = context.perform_substitution(LaunchConfiguration("root_height_m")).strip()
    camera = camera_mount(profile)
    return [
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
            arguments=world_to_pelvis_arguments(height or None, profile_path=profile),
        ),
        Node(
            package="tf2_ros",
            executable="static_transform_publisher",
            name="flux_sim_torso_to_head_camera",
            arguments=camera_mount_arguments(camera),
        ),
        Node(
            package="tf2_ros",
            executable="static_transform_publisher",
            name="flux_sim_head_camera_to_optical",
            arguments=camera_optical_arguments(camera),
        ),
        Node(
            package="flux_sim_viz",
            executable="joint_state_bridge",
            name="flux_sim_joint_state_bridge",
            output="screen",
        ),
    ]


def generate_launch_description():
    return LaunchDescription([
        # The file the simulator runs; relative paths are repository-relative.
        DeclareLaunchArgument("profile", default_value=str(DEFAULT_PROFILE)),
        # Empty means the profile's own root height; x, y and the rotation are
        # always the profile's, because that is the pose the simulator pins.
        DeclareLaunchArgument("root_height_m", default_value=""),
        OpaqueFunction(function=_nodes),
    ])
