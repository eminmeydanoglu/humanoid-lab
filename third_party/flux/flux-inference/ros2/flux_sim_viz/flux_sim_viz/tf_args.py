"""Static TF arguments derived from the canonical scene profile.

The TF stack (``flux_tf.launch.py``) publishes three static transforms:
``world -> pelvis`` at the profile's own fixed-root pose (position and rotation,
the same pose the simulator spawns and pins the robot at), ``torso_link ->
head_camera`` at the inherited head-camera mount, and the camera's optical child
``head_camera -> head_camera_optical`` that ROS image consumers read.  Resolving
the profile and building the argument lists lives here rather than in the launch
file so the mapping can be tested without a ROS install, and so the transform
values cannot drift from the profile again.

A profile is merged exactly as ``RunProfile`` merges it -- a child key replaces
the inherited one -- so the child-most profile that declares a key wins.
"""

import json
from pathlib import Path
from typing import Any, Iterator

#: The checkout as mounted in the flux-ros container, used to resolve a
#: profile argument that is relative to the repository root.
REPO_ROOT = Path("/workspace/humanoid-lab")

DEFAULT_PROFILE = REPO_ROOT / "configs/profiles/pick-apple-askida.json"

#: The pinned asset's root orientation when a profile declares none.  The
#: simulator only rewrites the fixed joint's rotation when the profile names
#: one, so an absent rotation means this identity, not "unknown".
DEFAULT_ROOT_ROTATION_WXYZ = (1.0, 0.0, 0.0, 0.0)

#: The fixed rotation from the simulator's camera frame (forward ``+X``, up
#: ``+Z``) to the ROS optical frame (forward ``+Z``, right ``+X``, down ``+Y``),
#: as ``(x, y, z, w)``.  Read as a tf parent -> child rotation it maps the
#: optical axes onto the body frame: optical forward ``+Z`` is body ``+X``,
#: optical right ``+X`` is body ``-Y`` (left), optical down ``+Y`` is body
#: ``-Z`` (up).
BODY_TO_OPTICAL_XYZW = (-0.5, 0.5, -0.5, 0.5)


def resolve_profile(profile: str) -> Path:
    """A launch ``profile`` value as a path: absolute, or relative to the repo."""
    path = Path(profile)
    return path if path.is_absolute() else REPO_ROOT / path


def profile_chain(profile_path: Path) -> Iterator[dict]:
    """Yield the profile and its base profiles, child first, then base."""
    seen: set[Path] = set()
    path = Path(profile_path)
    while True:
        resolved = path.resolve()
        if resolved in seen:
            raise ValueError(f"cyclic profile inheritance at {resolved}")
        seen.add(resolved)
        profile = json.loads(resolved.read_text(encoding="utf-8"))
        yield profile
        base = profile.get("base_profile")
        if base is None:
            return
        path = resolved.with_name(str(base))


def _inherited(profile_path: Path, key: str) -> Any:
    for profile in profile_chain(profile_path):
        if key in profile:
            return profile[key]
    raise ValueError(f"no {key!r} in {profile_path} or its base profiles")


def camera_mount(profile_path: Path = DEFAULT_PROFILE) -> dict:
    """The canonical scene's inherited head-camera mount."""
    return _inherited(profile_path, "camera")


def root_pose(
    profile_path: Path = DEFAULT_PROFILE,
) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
    """The profile's fixed-root pose: position metres and rotation ``wxyz``."""
    robot = _inherited(profile_path, "robot")
    position = tuple(float(value) for value in robot["initial_position_m"])
    rotation = tuple(
        float(value)
        for value in robot.get("initial_rotation_wxyz", DEFAULT_ROOT_ROTATION_WXYZ)
    )
    if len(position) != 3 or len(rotation) != 4:
        raise ValueError(f"malformed root pose in {profile_path}: {position} {rotation}")
    return (
        (position[0], position[1], position[2]),
        (rotation[0], rotation[1], rotation[2], rotation[3]),
    )


def _value(value: float) -> str:
    return repr(float(value))


def world_to_pelvis_arguments(
    root_height_m: Any = None, profile_path: Path = DEFAULT_PROFILE
) -> list:
    """``static_transform_publisher`` arguments for ``world -> pelvis``.

    ``root_height_m`` overrides the profile's z (a launch substitution); x, y
    and the rotation always come from the profile, because the simulator pins
    the pelvis at exactly that pose.
    """
    position, rotation = root_pose(profile_path)
    w, x, y, z = rotation
    height = _value(position[2]) if root_height_m is None else root_height_m
    return [
        "--x", _value(position[0]),
        "--y", _value(position[1]),
        "--z", height,
        "--qx", _value(x), "--qy", _value(y), "--qz", _value(z), "--qw", _value(w),
        "--frame-id", "world",
        "--child-frame-id", "pelvis",
    ]


def camera_mount_arguments(camera: dict) -> list:
    """``static_transform_publisher`` arguments for the head-camera body frame.

    The frame is published in the profile's own convention -- forward ``+X``,
    the convention the simulator authors the mount in.  Its optical child
    (:func:`camera_optical_arguments`) is what ROS image consumers read.
    """
    px, py, pz = (float(value) for value in camera["position_m"])
    w, x, y, z = (float(value) for value in camera["rotation_wxyz"])
    return [
        "--x", _value(px), "--y", _value(py), "--z", _value(pz),
        "--qx", _value(x), "--qy", _value(y), "--qz", _value(z), "--qw", _value(w),
        "--frame-id", str(camera["parent_link"]),
        "--child-frame-id", str(camera["name"]),
    ]


def camera_optical_name(camera: dict) -> str:
    """The optical frame of the profile's head camera (what images carry)."""
    return f"{camera['name']}_optical"


def camera_optical_arguments(camera: dict) -> list:
    """``static_transform_publisher`` arguments for the camera optical frame.

    Zero translation, the fixed body -> optical rotation: the child differs from
    the simulated body frame only by the ROS camera convention.
    """
    x, y, z, w = BODY_TO_OPTICAL_XYZW
    return [
        "--x", "0", "--y", "0", "--z", "0",
        "--qx", _value(x), "--qy", _value(y), "--qz", _value(z), "--qw", _value(w),
        "--frame-id", str(camera["name"]),
        "--child-frame-id", camera_optical_name(camera),
    ]
