"""Small, simulator-free checks for the two plate-placement evaluations."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from humanoid_lab.psi0_bridge.prompt import TASK_PROMPTS
from humanoid_lab.simulators.isaac.contracts import ContractError, RunProfile, SceneSpec


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    ("task", "slug", "object_name", "plate_color"),
    [("PickApple", "pickapple", "apple", "pink"),
     ("PickGum", "pickgum", "gum", "teal")],
)
def test_plate_task_profile_matches_training_caption_and_camera_order(
    task: str, slug: str, object_name: str, plate_color: str
) -> None:
    captions_path = ROOT / "data/datasets/psi0-unitree-dex3-sonic-v1-mini/train/meta/tasks.jsonl"
    captions = (
        {row["task"] for row in map(json.loads, captions_path.read_text().splitlines())}
        if captions_path.is_file() else set(TASK_PROMPTS.values())
    )
    profile = RunProfile.load(ROOT / f"configs/profiles/isaac-g1-sonic-{slug}-dex3.json")
    assert TASK_PROMPTS[task] in captions
    assert profile.scene is not None and profile.scene.object is not None
    assert profile.scene.object.name == object_name
    assert profile.scene.target.color == plate_color
    assert profile.scene.object.position_m[0] > profile.scene.target.position_m[0]
    assert profile.scene.camera_enabled


@pytest.mark.parametrize("slug", ["pickapple", "pickgum"])
def test_plate_task_inherits_the_shared_robot_and_camera(slug: str) -> None:
    profile_path = ROOT / f"configs/profiles/isaac-g1-sonic-{slug}-dex3.json"
    raw = json.loads(profile_path.read_text())
    assert raw["base_profile"] == "isaac-g1-sonic-blockstacking-dex3-green-third.json"
    assert not ({"robot", "camera", "controller", "simulation", "support"} & raw.keys())

    base = RunProfile.load(profile_path.parent / raw["base_profile"])
    canonical = RunProfile.load(profile_path.parent / "isaac-g1-sonic-blockstacking-dex3.json")
    task = RunProfile.load(profile_path)
    assert base.robot == canonical.robot
    assert base.camera == canonical.camera
    assert task.robot == base.robot
    assert task.camera == base.camera
    assert task.controller == base.controller
    assert task.physics_dt == base.physics_dt
    assert task.initial_pose == base.initial_pose
    assert task.support == base.support


def test_plate_scene_rejects_a_mismatched_object_height() -> None:
    raw = json.loads(
        (ROOT / "configs/profiles/isaac-g1-sonic-pickapple-dex3.json").read_text()
    )["scene"]
    raw["object"]["position_m"][2] += 0.04
    with pytest.raises(ContractError, match="rest on"):
        SceneSpec.from_dict(raw)


def test_eval_launcher_rejects_an_unknown_task_before_startup() -> None:
    result = subprocess.run(
        ["bash", str(ROOT / "dev.sh"), "psi0-isaac-eval", "--task", "Unknown"],
        cwd=ROOT, capture_output=True, text=True, timeout=10,
    )
    assert result.returncode == 2
    assert "--task must be BlockStacking, PickApple or PickGum" in result.stderr
