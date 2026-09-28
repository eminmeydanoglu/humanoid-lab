"""The RViz view must describe the same robot the simulator drives.

The bridge publishes names and a child-frame layout, so this test compares them
with the simulator's own definition (`humanoid_lab.controllers.sonic`), checks
that the committed URDF carries every one of those joints, and covers the two
edits `robot_state_publisher` needs (absolute mesh URLs, no `<mujoco>` block).
The static transforms are resolved from the scene profile, so they are compared
with the profile the simulator itself loads.
"""

import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "third_party/flux/flux-inference/ros2/flux_sim_viz"))

from flux_sim_viz import tf_args, urdf  # noqa: E402
from flux_sim_viz.joints import (  # noqa: E402
    BODY_JOINT_ORDER,
    HAND_MOTOR_SLOTS,
    hand_joint_names,
    measured_joint_state,
)
from humanoid_lab.controllers.sonic import (  # noqa: E402
    BODY_JOINT_ORDER as SIMULATOR_BODY_JOINT_ORDER,
    hand_joint_names as simulator_hand_joint_names,
)

URDF_PATH = ROOT / "third_party/Psi0/real/assets/g1/g1_body29_hand14.urdf"
FLUX_ROS = ROOT / "third_party/flux/flux-inference/ros2"
TF_LAUNCH_PATH = FLUX_ROS / "flux_sim_viz/launch/flux_tf.launch.py"
RVIZ_LAUNCH_PATH = FLUX_ROS / "flux_sim_viz/launch/flux_rviz.launch.py"
CANONICAL_LAUNCH_PATH = FLUX_ROS / "flux_sim_camera/launch/flux_sim.launch.py"
CAMERA_BRIDGE_PATH = FLUX_ROS / "flux_sim_camera/flux_sim_camera/camera_bridge.py"
CANONICAL_PROFILE = ROOT / "configs/profiles/pick-apple-askida.json"
GUM_PROFILE = ROOT / "configs/profiles/pick-gum-askida.json"


def _rotate(quaternion_xyzw, vector):
    """Apply a quaternion to a vector, for checking the axis mapping."""
    x, y, z, w = quaternion_xyzw
    vx, vy, vz = vector
    ix = y * vz - z * vy + w * vx
    iy = z * vx - x * vz + w * vy
    iz = x * vy - y * vx + w * vz
    return (
        vx + 2.0 * (y * iz - z * iy),
        vy + 2.0 * (z * ix - x * iz),
        vz + 2.0 * (x * iy - y * ix),
    )


def urdf_joint_names() -> set[str]:
    return set(re.findall(r'<joint name="([^"]+)"', URDF_PATH.read_text()))


def test_joint_orders_are_the_simulators_own():
    assert BODY_JOINT_ORDER == SIMULATOR_BODY_JOINT_ORDER
    for side in ("left", "right"):
        assert hand_joint_names(side) == simulator_hand_joint_names(side)
        assert len(hand_joint_names(side)) == HAND_MOTOR_SLOTS


def test_every_published_joint_exists_in_the_robot_model():
    names = set(BODY_JOINT_ORDER) | set(hand_joint_names("left")) | set(
        hand_joint_names("right")
    )
    missing = sorted(names - urdf_joint_names())
    assert not missing, f"URDF has no such joints: {missing}"


def test_robot_description_is_loadable_by_robot_state_publisher():
    description = urdf.load_robot_description(str(URDF_PATH))
    assert "<mujoco>" not in description
    references = re.findall(r'filename="([^"]+)"', description)
    assert references, "the URDF must reference meshes"
    assert all(reference.startswith("file:///") for reference in references)
    assert all(Path(reference[len("file://") :]).is_file() for reference in references)
    assert set(BODY_JOINT_ORDER) <= set(re.findall(r'<joint name="([^"]+)"', description))


def test_measured_state_follows_the_wire_slots():
    body_q = [float(index) for index in range(len(BODY_JOINT_ORDER))]
    left_q = [10.0 + index for index in range(HAND_MOTOR_SLOTS)]
    right_q = [20.0 + index for index in range(HAND_MOTOR_SLOTS)]

    names, positions = measured_joint_state(body_q, left_q, right_q)

    assert names == list(BODY_JOINT_ORDER) + list(hand_joint_names("left")) + list(
        hand_joint_names("right")
    )
    assert positions == body_q + left_q + right_q


def test_missing_hands_are_omitted_rather_than_guessed():
    body_q = [0.0] * len(BODY_JOINT_ORDER)

    names, positions = measured_joint_state(body_q)

    assert names == list(BODY_JOINT_ORDER)
    assert positions == body_q


def _flags(arguments):
    """Flag/value pairs of a static_transform_publisher argument list."""
    pairs = iter(arguments)
    return {flag.lstrip("-"): value for flag, value in zip(pairs, pairs)}


def test_world_to_pelvis_follows_the_profiles_full_root_pose():
    profile = json.loads(CANONICAL_PROFILE.read_text())["robot"]
    position, rotation = tf_args.root_pose(CANONICAL_PROFILE)

    assert list(position) == profile["initial_position_m"]
    assert list(rotation) == profile["initial_rotation_wxyz"]

    flags = _flags(tf_args.world_to_pelvis_arguments(profile_path=CANONICAL_PROFILE))

    assert flags["frame-id"] == "world"
    assert flags["child-frame-id"] == "pelvis"
    assert float(flags["x"]) == position[0] == 0.08
    assert float(flags["y"]) == position[1]
    assert float(flags["z"]) == position[2]
    w, x, y, z = rotation
    assert [float(flags[key]) for key in ("qx", "qy", "qz", "qw")] == [x, y, z, w]


def test_root_height_is_the_only_axis_override():
    flags = _flags(
        tf_args.world_to_pelvis_arguments("1.25", profile_path=CANONICAL_PROFILE)
    )

    assert flags["z"] == "1.25"
    assert float(flags["x"]) == 0.08
    assert float(flags["y"]) == 0.0


def test_each_task_profile_resolves_its_own_root_pose():
    apple, apple_rotation = tf_args.root_pose(CANONICAL_PROFILE)
    gum, gum_rotation = tf_args.root_pose(GUM_PROFILE)

    assert apple == (0.08, 0.0, 0.792563)
    assert gum == (0.0, 0.0, 0.792563)
    # A profile that declares no rotation keeps the pinned asset's identity,
    # which is what the simulator leaves in place when none is given.
    assert apple_rotation == (1.0, 0.0, 0.0, 0.0)
    assert gum_rotation == tf_args.DEFAULT_ROOT_ROTATION_WXYZ


def test_rviz_head_camera_transform_uses_the_inherited_profile_mount():
    camera = tf_args.camera_mount(CANONICAL_PROFILE)
    inherited_chain = json.loads(
        (CANONICAL_PROFILE.parent / "isaac-g1-sonic-blockstacking-dex3.json").read_text()
    )["camera"]
    assert camera == inherited_chain
    assert camera["parent_link"] == "torso_link"
    assert camera["name"] == "head_camera"

    flags = _flags(tf_args.camera_mount_arguments(camera))
    w, x, y, z = camera["rotation_wxyz"]
    assert flags["frame-id"] == "torso_link"
    assert flags["child-frame-id"] == "head_camera"
    assert [float(flags[key]) for key in ("qx", "qy", "qz", "qw")] == [x, y, z, w]
    assert [float(flags[key]) for key in ("x", "y", "z")] == camera["position_m"]


def test_camera_optical_static_is_the_ros_convention():
    camera = tf_args.camera_mount(CANONICAL_PROFILE)
    flags = _flags(tf_args.camera_optical_arguments(camera))

    assert flags["frame-id"] == "head_camera"
    assert flags["child-frame-id"] == "head_camera_optical"
    assert [float(flags[key]) for key in ("x", "y", "z")] == [0.0, 0.0, 0.0]
    published = [float(flags[key]) for key in ("qx", "qy", "qz", "qw")]
    assert published == list(tf_args.BODY_TO_OPTICAL_XYZW)
    # As a tf parent (body) -> child (optical) rotation: optical forward is the
    # body's forward, optical right is the body's left, optical down is the
    # body's up -- and, read the other way, the camera's own forward (+X body)
    # is forward in the optical frame the images carry.
    assert _rotate(published, (0.0, 0.0, 1.0)) == pytest.approx((1.0, 0.0, 0.0))
    assert _rotate(published, (1.0, 0.0, 0.0)) == pytest.approx((0.0, -1.0, 0.0))
    assert _rotate(published, (0.0, 1.0, 0.0)) == pytest.approx((0.0, 0.0, -1.0))
    inverse = (-published[0], -published[1], -published[2], published[3])
    assert _rotate(inverse, (1.0, 0.0, 0.0)) == pytest.approx((0.0, 0.0, 1.0))


def test_resolve_profile_is_repository_relative():
    relative = tf_args.resolve_profile("configs/profiles/pick-gum-askida.json")
    absolute = tf_args.resolve_profile("/tmp/other.json")

    assert relative == tf_args.REPO_ROOT / "configs/profiles/pick-gum-askida.json"
    assert str(absolute) == "/tmp/other.json"


def test_the_tf_launch_owns_every_frame_once():
    tf_source = TF_LAUNCH_PATH.read_text()
    rviz_source = RVIZ_LAUNCH_PATH.read_text()
    canonical_source = CANONICAL_LAUNCH_PATH.read_text()

    # One authority: the TF launch publishes the robot model, the joint state
    # and all three statics, from the profile the simulator runs.
    for builder in (
        "world_to_pelvis_arguments",
        "camera_mount_arguments",
        "camera_optical_arguments",
    ):
        assert builder in tf_source
    assert "joint_state_bridge" in tf_source
    assert "robot_state_publisher" in tf_source
    assert '"profile"' in tf_source

    # The optional RViz launch is RViz-only by default; a second stack would
    # duplicate every publisher.
    assert "static_transform_publisher" not in rviz_source
    assert "robot_state_publisher" not in rviz_source
    assert "joint_state_bridge" not in rviz_source
    assert "flux_tf.launch.py" in rviz_source
    assert re.search(r'"tf",\s*default_value="false"', rviz_source)

    # The canonical launch starts that stack by default.
    assert "flux_tf.launch.py" in canonical_source
    assert 'DeclareLaunchArgument("tf", default_value="true")' in canonical_source
    assert re.search(r'DeclareLaunchArgument\(\s*"profile",', canonical_source)


def test_the_canonical_launch_and_bridge_agree_on_the_optical_frame():
    camera = tf_args.camera_mount(CANONICAL_PROFILE)
    expected = tf_args.camera_optical_name(camera)

    bridge_default = re.search(
        r'declare_parameter\("frame_id", "([^"]+)"\)', CAMERA_BRIDGE_PATH.read_text()
    )
    launch_default = re.search(
        r'DeclareLaunchArgument\("frame_id", default_value="([^"]+)"\)',
        CANONICAL_LAUNCH_PATH.read_text(),
    )

    assert bridge_default, "the bridge must declare its image frame"
    assert launch_default, "the canonical launch must declare its image frame"
    assert bridge_default.group(1) == expected
    assert launch_default.group(1) == expected


def test_tf_launch_uses_the_profile_derived_static_arguments():
    source = TF_LAUNCH_PATH.read_text()

    # The regression this guards: the root x/y hard-coded to the world origin
    # while the profile places the robot elsewhere.
    assert '"--x", "0"' not in source
    assert '"--y", "0"' not in source
