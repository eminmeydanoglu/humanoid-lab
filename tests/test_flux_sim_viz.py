"""The RViz view must describe the same robot the simulator drives.

The bridge publishes names and a child-frame layout, so this test compares them
with the simulator's own definition (`humanoid_lab.controllers.sonic`), checks
that the committed URDF carries every one of those joints, and covers the two
edits `robot_state_publisher` needs (absolute mesh URLs, no `<mujoco>` block).
"""

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "third_party/flux/flux-inference/ros2/flux_sim_viz"))

from flux_sim_viz import urdf  # noqa: E402
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
LAUNCH_PATH = (
    ROOT
    / "third_party/flux/flux-inference/ros2/flux_sim_viz/launch/flux_rviz.launch.py"
)
CANONICAL_PROFILE = ROOT / "configs/profiles/pick-apple-askida.json"


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


def test_rviz_root_height_matches_the_canonical_profile():
    profile = json.loads(CANONICAL_PROFILE.read_text())
    root_height_m = profile["robot"]["initial_position_m"][2]
    match = re.search(
        r'"root_height_m",\s*default_value="([0-9.]+)"', LAUNCH_PATH.read_text()
    )
    assert match, "the launch file must declare root_height_m with a default"
    assert float(match.group(1)) == root_height_m


def test_rviz_head_camera_transform_uses_the_inherited_profile_mount():
    import ast

    source = LAUNCH_PATH.read_text()
    function = next(
        node for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef) and node.name == "camera_mount"
    )
    namespace = {"json": json, "Path": Path, "DEFAULT_PROFILE": CANONICAL_PROFILE}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(LAUNCH_PATH), "exec"), namespace)
    camera = namespace["camera_mount"]()
    inherited = json.loads((CANONICAL_PROFILE.parent / "isaac-g1-sonic-blockstacking-dex3.json").read_text())["camera"]
    assert camera == inherited
    assert camera["parent_link"] == "torso_link"
    assert camera["name"] == "head_camera"
    assert '"--frame-id", camera["parent_link"]' in source
    assert '"--child-frame-id", camera["name"]' in source
    assert '"--qx", str(x), "--qy", str(y), "--qz", str(z), "--qw", str(w)' in source
