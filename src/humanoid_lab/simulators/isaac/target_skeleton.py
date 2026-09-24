"""Render the commanded G1 pose as non-physical USD points and line segments.

The kinematics come from the same pinned URDF as the Isaac asset.  Each point
is a link frame, so this draws joint locations rather than mesh centres.
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np

DEFAULT_URDF = Path("/opt/src/sonic/gear_sonic/data/robots/g1/g1_29dof_with_hand_rev_1_0.urdf")


def _rotation_rpy(rpy: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ])


def _rotation_axis(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    c, s = math.cos(angle), math.sin(angle)
    return c * np.eye(3) + (1 - c) * np.outer(axis, axis) + s * np.array([
        [0, -z, y], [z, 0, -x], [-y, x, 0],
    ])


def _root_transform(position: tuple[float, ...], quaternion_wxyz: tuple[float, ...]) -> np.ndarray:
    w, x, y, z = np.asarray(quaternion_wxyz, dtype=float) / np.linalg.norm(quaternion_wxyz)
    transform = np.eye(4)
    transform[:3, :3] = np.array([
        [1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)],
        [2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)],
        [2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)],
    ])
    transform[:3, 3] = position
    return transform


@dataclass(frozen=True)
class Joint:
    name: str
    kind: str
    parent: str
    child: str
    origin: np.ndarray
    axis: np.ndarray


class UrdfKinematics:
    def __init__(self, urdf: Path = DEFAULT_URDF) -> None:
        root = ET.parse(urdf).getroot()
        joints = []
        for element in root.findall("joint"):
            parent = element.find("parent")
            child = element.find("child")
            if parent is None or child is None:
                raise ValueError("URDF joint lacks parent or child")
            origin_element = element.find("origin")
            axis_element = element.find("axis")
            xyz = np.fromstring(origin_element.get("xyz", "0 0 0") if origin_element is not None else "0 0 0", sep=" ")
            rpy = np.fromstring(origin_element.get("rpy", "0 0 0") if origin_element is not None else "0 0 0", sep=" ")
            origin = np.eye(4)
            origin[:3, :3] = _rotation_rpy(rpy)
            origin[:3, 3] = xyz
            axis = np.fromstring(axis_element.get("xyz", "1 0 0") if axis_element is not None else "1 0 0", sep=" ")
            joints.append(Joint(element.attrib["name"], element.attrib["type"], parent.attrib["link"], child.attrib["link"], origin, axis))
        self.joints = tuple(joints)
        self.actuated = frozenset(j.name for j in joints if j.kind in {"revolute", "continuous", "prismatic"})

    def points(self, q: Mapping[str, float], root_position: tuple[float, ...], root_quaternion_wxyz: tuple[float, ...], *, include_palms: bool = True) -> tuple[list[tuple[float, float, float]], list[tuple[int, int]]]:
        unknown = set(q) - self.actuated
        if unknown:
            raise ValueError(f"target joints absent from URDF: {sorted(unknown)}")
        poses = {"pelvis": _root_transform(root_position, root_quaternion_wxyz)}
        points = [tuple(float(v) for v in root_position)]
        indices = {"pelvis": 0}
        edges = []
        remaining = list(self.joints)
        while remaining:
            pending = []
            for joint in remaining:
                if joint.parent not in poses:
                    pending.append(joint)
                    continue
                motion = np.eye(4)
                if joint.kind in {"revolute", "continuous"}:
                    motion[:3, :3] = _rotation_axis(joint.axis, float(q.get(joint.name, 0.0)))
                elif joint.kind == "prismatic":
                    motion[:3, 3] = joint.axis * float(q.get(joint.name, 0.0))
                elif joint.kind != "fixed":
                    raise ValueError(f"unsupported URDF joint type: {joint.kind}")
                pose = poses[joint.parent] @ joint.origin @ motion
                poses[joint.child] = pose
                # Include actuated joints and palms.  Fixed decorative links
                # would create a dense web around the robot's mesh.
                if joint.name in q or (include_palms and joint.child.endswith("hand_palm_link")):
                    parent = joint.parent
                    while parent not in indices:
                        predecessor = next((j.parent for j in self.joints if j.child == parent), None)
                        if predecessor is None:
                            raise ValueError(f"no visible ancestor for {joint.name}")
                        parent = predecessor
                    indices[joint.child] = len(points)
                    points.append(tuple(float(v) for v in pose[:3, 3]))
                    edges.append((indices[parent], indices[joint.child]))
            if len(pending) == len(remaining):
                raise ValueError("URDF joint graph is disconnected from pelvis")
            remaining = pending
        return points, edges


class TargetSkeletonOverlay:
    """USD geometry is visual only and is updated on Isaac's render thread."""

    def __init__(self, stage: object, urdf: Path = DEFAULT_URDF, *, include_palms: bool = True) -> None:
        from pxr import Gf, UsdGeom

        self.kinematics = UrdfKinematics(urdf)
        self.include_palms = include_palms
        parent = "/World/SonicTargetSkeleton"
        root = UsdGeom.Xform.Define(stage, parent)
        self._dots = UsdGeom.Points.Define(stage, parent + "/Joints")
        self._lines = UsdGeom.BasisCurves.Define(stage, parent + "/Bones")
        self._dots.CreatePointsAttr()
        self._dots.CreateWidthsAttr()
        self._dots.CreateExtentAttr()
        self._lines.CreatePointsAttr()
        self._lines.CreateWidthsAttr()
        self._lines.CreateCurveVertexCountsAttr()
        self._lines.CreateExtentAttr()
        self._lines.CreateTypeAttr(UsdGeom.Tokens.linear)
        self._lines.CreateWrapAttr(UsdGeom.Tokens.nonperiodic)
        self._dots.CreateDisplayColorAttr([Gf.Vec3f(1.0, 0.78, 0.05)])
        self._lines.CreateDisplayColorAttr([Gf.Vec3f(0.05, 0.85, 1.0)])
        # One Stage eye controls both children.  Updates never author visibility.
        root.MakeInvisible()

    def clear(self) -> None:
        """Remove stale targets without changing the Stage eye settings."""
        from pxr import Vt

        self._dots.GetPointsAttr().Set(Vt.Vec3fArray())
        self._lines.GetPointsAttr().Set(Vt.Vec3fArray())
        self._lines.GetCurveVertexCountsAttr().Set(Vt.IntArray())

    def update(self, q: Mapping[str, float], root_position: tuple[float, ...], root_quaternion_wxyz: tuple[float, ...]) -> None:
        from pxr import Gf, Vt

        points, edges = self.kinematics.points(
            q, root_position, root_quaternion_wxyz, include_palms=self.include_palms
        )
        self._dots.GetPointsAttr().Set(Vt.Vec3fArray([Gf.Vec3f(*point) for point in points]))
        self._dots.GetWidthsAttr().Set(Vt.FloatArray([0.025] * len(points)))
        self._lines.GetPointsAttr().Set(Vt.Vec3fArray([Gf.Vec3f(*points[index]) for edge in edges for index in edge]))
        self._lines.GetWidthsAttr().Set(Vt.FloatArray([0.007] * (2 * len(edges))))
        self._lines.GetCurveVertexCountsAttr().Set(Vt.IntArray([2] * len(edges)))
        bounds = np.asarray(points)
        extent = Vt.Vec3fArray([
            Gf.Vec3f(*(bounds.min(axis=0) - 0.03)),
            Gf.Vec3f(*(bounds.max(axis=0) + 0.03)),
        ])
        self._dots.GetExtentAttr().Set(extent)
        self._lines.GetExtentAttr().Set(extent)
