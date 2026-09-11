"""Align the simulated body to the model the controller was tuned against.

The policy inside the deployment was trained against the MuJoCo G1 model that
ships with the pinned SONIC source.  The asset used here shares that model's
joint names and axes, but not its masses: the MuJoCo file already carries the
links it welds to their parents (the torso includes the head, cameras and torso
IMU, the wrist includes the hand), while the USD declares each body separately.

The masses are therefore read from that same pinned model at start-up rather
than transcribed, so there is exactly one source of truth and no hand-typed
numbers to drift.
"""

from __future__ import annotations

import xml.etree.ElementTree as ElementTree
from pathlib import Path

DEFAULT_MUJOCO_MODEL = (
    "/opt/src/sonic/gear_sonic/data/robot_model/model_data/g1/g1_29dof_with_hand.xml"
)

# Bodies the USD declares separately but the MuJoCo model welds into their
# parent (head, cameras, IMUs, logo, hand palms).  Their mass is already inside
# the parent's value there, so leaving them weighted here double-counts it; they
# are made negligible instead.  PhysX requires a positive mass, hence a gram.
WELDED_BODY_MASS_KG = 1e-3


class MassAlignmentError(ValueError):
    """The MuJoCo model or the asset cannot supply the masses to align with."""


def load_mujoco_body_masses(path: str | Path = DEFAULT_MUJOCO_MODEL) -> dict[str, float]:
    """Body masses of the pinned MuJoCo G1 model, keyed by body name."""
    model_path = Path(path)
    try:
        root = ElementTree.parse(model_path).getroot()
    except (OSError, ElementTree.ParseError) as error:
        raise MassAlignmentError(f"cannot read the MuJoCo model {model_path}: {error}") from error
    masses: dict[str, float] = {}
    for body in root.iter("body"):
        name = body.get("name")
        inertial = body.find("inertial")
        if not name or inertial is None or inertial.get("mass") is None:
            continue
        masses[name] = float(inertial.get("mass"))
    if not masses:
        raise MassAlignmentError(f"the MuJoCo model {model_path} declares no body masses")
    return masses


def plan_alignment(
    body_names: list[str], masses: dict[str, float]
) -> tuple[list[int], list[float], list[str]]:
    """Map MuJoCo masses onto asset body indices.

    Returns the indices to set, the mass for each, and the asset bodies the
    MuJoCo model does not name.  Those extra bodies are the welded links
    described above and are given a negligible mass, so the parent's total
    matches the model instead of doubling it.
    """
    indices: list[int] = []
    values: list[float] = []
    welded: list[str] = []
    for index, name in enumerate(body_names):
        if name in masses:
            indices.append(index)
            values.append(masses[name])
        else:
            indices.append(index)
            values.append(WELDED_BODY_MASS_KG)
            welded.append(name)
    if len(welded) == len(body_names):
        raise MassAlignmentError("no asset body matches the MuJoCo model")
    return indices, values, welded
