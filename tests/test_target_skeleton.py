"""The target overlay must preserve URDF joint and root-frame geometry."""

from pathlib import Path

import numpy as np
import pytest

from humanoid_lab.simulators.isaac.target_skeleton import UrdfKinematics


def test_revolute_joint_and_root_rotation(tmp_path: Path) -> None:
    urdf = tmp_path / "two_link.urdf"
    urdf.write_text("""<robot name="test">
      <link name="pelvis"/><link name="arm"/><link name="tip"/>
      <joint name="shoulder" type="revolute"><parent link="pelvis"/>
        <child link="arm"/><origin xyz="1 0 0"/><axis xyz="0 0 1"/></joint>
      <joint name="tip_fixed" type="fixed"><parent link="arm"/>
        <child link="tip"/><origin xyz="1 0 0"/></joint>
    </robot>""")
    model = UrdfKinematics(urdf)
    # The revolute child frame is at its axis.  A fixed tip frame is omitted
    # from the visual, but downstream joints still inherit its transform.
    points, edges = model.points({"shoulder": 0.5}, (0, 0, 1), (1, 0, 0, 0))
    assert edges == [(0, 1)]
    np.testing.assert_allclose(points[1], (1, 0, 1))
    points, _ = model.points({"shoulder": 0.5}, (0, 0, 1), (0, 0, 0, 1))
    np.testing.assert_allclose(points[1], (-1, 0, 1))
    with pytest.raises(ValueError, match="absent from URDF"):
        model.points({"wrong_joint": 1}, (0, 0, 1), (1, 0, 0, 0))


def test_revolute_motion_moves_descendant(tmp_path: Path) -> None:
    urdf = tmp_path / "chain.urdf"
    urdf.write_text("""<robot name="test">
      <link name="pelvis"/><link name="arm"/><link name="forearm"/>
      <joint name="shoulder" type="revolute"><parent link="pelvis"/>
        <child link="arm"/><axis xyz="0 0 1"/></joint>
      <joint name="elbow" type="revolute"><parent link="arm"/>
        <child link="forearm"/><origin xyz="1 0 0"/><axis xyz="0 1 0"/></joint>
    </robot>""")
    points, _ = UrdfKinematics(urdf).points(
        {"shoulder": np.pi / 2, "elbow": 0}, (0, 0, 0), (1, 0, 0, 0)
    )
    np.testing.assert_allclose(points[2], (0, 1, 0), atol=1e-12)


def test_body_only_mode_omits_dex3_palm(tmp_path: Path) -> None:
    urdf = tmp_path / "palm.urdf"
    urdf.write_text("""<robot name="test">
      <link name="pelvis"/><link name="arm"/><link name="left_hand_palm_link"/>
      <joint name="shoulder" type="revolute"><parent link="pelvis"/>
        <child link="arm"/><axis xyz="0 0 1"/></joint>
      <joint name="palm_fixed" type="fixed"><parent link="arm"/>
        <child link="left_hand_palm_link"/><origin xyz="1 0 0"/></joint>
    </robot>""")
    model = UrdfKinematics(urdf)
    body, _ = model.points({"shoulder": 0.0}, (0, 0, 0), (1, 0, 0, 0), include_palms=False)
    dex3, _ = model.points({"shoulder": 0.0}, (0, 0, 0), (1, 0, 0, 0))
    assert len(body) == 2
    assert len(dex3) == 3
