"""The committed G1 URDF, prepared for `robot_state_publisher`.

The model is Psi0's `g1_body29_hand14.urdf`: the same 29 body joints and 14
Dex3 hand joints as the simulator.  Two edits make it loadable by plain ROS 2
tools -- mesh references become absolute `file://` URLs (they are written
relative to the asset directory, which the process working directory does not
match) and the `<mujoco>` element is dropped, since the URDF parser used by
`robot_state_publisher` accepts neither.  Geometry, joints and masses are left
untouched.
"""

import os
import re

DEFAULT_URDF = (
    "/workspace/humanoid-lab/third_party/Psi0/real/assets/g1/g1_body29_hand14.urdf"
)

_MESH_REFERENCE = re.compile(r'filename="([^"]+)"')
_MUJOCO_BLOCK = re.compile(r"\s*<mujoco>.*?</mujoco>", re.DOTALL)


def load_robot_description(path: str = DEFAULT_URDF) -> str:
    """Return the URDF text with absolute mesh URLs and no `<mujoco>` block."""
    with open(path, encoding="utf-8") as handle:
        text = handle.read()
    asset_dir = os.path.dirname(os.path.abspath(path))

    def absolute(match: "re.Match[str]") -> str:
        target = match.group(1)
        if target.startswith(("file://", "package://", "/")):
            return match.group(0)
        resolved = os.path.normpath(os.path.join(asset_dir, target))
        return f'filename="file://{resolved}"'

    prepared = _MESH_REFERENCE.sub(absolute, text)
    return _MUJOCO_BLOCK.sub("", prepared)
