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
from typing import Sequence

DEFAULT_MUJOCO_MODEL = (
    "/opt/src/sonic/gear_sonic/data/robot_model/model_data/g1/g1_29dof_with_hand.xml"
)

# The model the *training* rig compiles.  ``gear_sonic`` trains on this URDF
# (``gear_sonic/envs/manager_env/robots/g1.py``, ``UrdfFileCfg`` with zero drive
# gains; the arms are then driven by the policy through explicit torque), so its
# link masses are the plant the policy learned against.  The deployed MuJoCo
# model above is welded -- it carries the head, palm and torso electronics
# inside their parents -- so aligning the evaluation asset to it is a *different*
# plant, not a stricter version of the same one.
#
# Measured deltas, both totals read from the files (36.1652 kg welded vs
# 34.3942 kg training).  Note the arm-chain *total* is identical on both sides
# (4.04562 kg per arm): the difference is +2.818 kg on ``torso_link`` (the welded
# model folds in the head, contour shell and logo) plus the 0.3728 kg palm, which
# training carries as its own link but the welded model folds into ``wrist_yaw``.
# So the arm mass is the same but sits one joint further out in training -- do not
# describe this as "the evaluation arms are 1.8 kg heavy".
DEFAULT_TRAINING_URDF = (
    "/opt/src/sonic/gear_sonic/data/robots/g1/g1_29dof_with_hand_rev_1_0.urdf"
)

#: Body masses can come from either of the two pinned plants.  ``sonic_mujoco``
#: is the deployment simulator's welded model, ``sonic_training`` the training
#: rig's URDF.  The names are the profile's ``controller.mass_alignment`` values.
MASS_ALIGNMENT_MODES: tuple[str, ...] = ("sonic_mujoco", "sonic_training")

# Bodies the USD declares separately but the MuJoCo model welds into their
# parent (head, cameras, IMUs, logo, hand palms).  Their mass is already inside
# the parent's value there, so leaving them weighted here double-counts it; they
# are made negligible instead.  PhysX requires a positive mass, hence a gram.
WELDED_BODY_MASS_KG = 1e-3

#: The same floor for a link a non-welded model declares *without* an
#: ``<inertial>`` (the training URDF's IMUs, mid360 and d435).  The model's own
#: value is zero, and PhysX refuses a literal zero, so a gram is used.
MASSLESS_BODY_MASS_KG = WELDED_BODY_MASS_KG


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


def load_urdf_body_masses(path: str | Path = DEFAULT_TRAINING_URDF) -> dict[str, float]:
    """Link masses of the G1 URDF, keyed by link name, exactly as authored.

    Unlike the MuJoCo model this URDF is not welded: the head, the palms and the
    contour/logo shells are declared as their own links.

    These are the *authored* link masses, which the training rig does not use
    directly -- its ``UrdfFileCfg`` converts the URDF with
    ``merge_fixed_joints=True`` (the converter default,
    ``isaaclab/sim/converters/urdf_converter_cfg.py:104``).  The plant the policy
    actually trains on is :func:`load_urdf_merged_body_masses`.
    """
    return _parse_urdf_links(path)[0]


def _parse_urdf_links(path: str | Path) -> tuple[dict[str, float], dict[str, str]]:
    """Authored link masses plus the ``fixed-joint child -> parent`` map."""
    model_path = Path(path)
    try:
        root = ElementTree.parse(model_path).getroot()
    except (OSError, ElementTree.ParseError) as error:
        raise MassAlignmentError(f"cannot read the URDF {model_path}: {error}") from error
    masses: dict[str, float] = {}
    for link in root.iter("link"):
        name = link.get("name")
        if not name:
            continue
        # URDF keeps the mass in a <mass value="..."/> *child* of <inertial>,
        # unlike MJCF's attribute, so the two loaders cannot share one lookup.
        mass = link.find("inertial/mass")
        if mass is None or mass.get("value") is None:
            continue
        masses[name] = float(mass.get("value"))
    if not masses:
        raise MassAlignmentError(f"the URDF {model_path} declares no link masses")
    fixed: dict[str, str] = {}
    for joint in root.iter("joint"):
        if joint.get("type") != "fixed":
            continue
        child = joint.find("child")
        parent = joint.find("parent")
        if child is None or parent is None:
            continue
        child_name, parent_name = child.get("link"), parent.get("link")
        if child_name and parent_name:
            fixed[child_name] = parent_name
    return masses, fixed


def load_urdf_merged_body_masses(
    path: str | Path = DEFAULT_TRAINING_URDF,
) -> tuple[dict[str, float], list[str]]:
    """The plant the training rig really spawns, and the links it merges away.

    ``UrdfFileCfg`` converts the training URDF with ``merge_fixed_joints=True``,
    so every link attached by a fixed joint is folded into its parent: the mass
    is added to the parent's and the link stops being a body.  That is why the
    head's 1.036 kg sits *inside* the training torso and the palm's 0.37284 kg
    *inside* ``wrist_yaw``.

    This matters for more than bookkeeping.  Aligning against the URDF's raw
    per-link table instead reports a palm/wrist difference that the trained plant
    does not have, and -- worse -- drops the fixed-child mass entirely, because
    the merged-away links are not bodies of the asset at all.  The merged model
    reproduces the MuJoCo ``wrist_yaw`` exactly (0.457415 both), which is the
    evidence that the two lineages describe the same arm.

    Returns ``(merged_masses, merged_away)``.
    """
    masses, fixed = _parse_urdf_links(path)
    merged = dict(masses)
    for child, parent in fixed.items():
        merged[parent] = merged.get(parent, 0.0) + masses.get(child, 0.0)
    for child in fixed:
        merged.pop(child, None)
    return merged, sorted(fixed)


def load_urdf_link_names(path: str | Path = DEFAULT_TRAINING_URDF) -> list[str]:
    """Every link the URDF declares, mass-bearing or not.

    The training URDF carries massless frames (``imu_in_pelvis``,
    ``imu_in_torso``, ``mid360_link``, ``d435_link``): the model does have those
    bodies, it just gives them no inertia.  A body missing from *this* list is
    the real mismatch; a body missing only from the mass map is a zero-mass link.
    """
    model_path = Path(path)
    try:
        root = ElementTree.parse(model_path).getroot()
    except (OSError, ElementTree.ParseError) as error:
        raise MassAlignmentError(f"cannot read the URDF {model_path}: {error}") from error
    names = [name for name in (link.get("name") for link in root.iter("link")) if name]
    if not names:
        raise MassAlignmentError(f"the URDF {model_path} declares no links")
    return names


def plan_alignment(
    body_names: list[str],
    masses: dict[str, float],
    *,
    missing: str = "weld",
    declared: Sequence[str] | None = None,
) -> tuple[list[int], list[float], list[str]]:
    """Map model masses onto asset body indices.

    Returns the indices to set, the mass for each, and the asset bodies the model
    does not supply a mass for.

    ``missing="weld"`` is the MuJoCo contract: that model welds the head, palms
    and shells into their parents, so an asset body it has no entry for is given
    a negligible mass to avoid double-counting the parent's value.

    ``missing="refuse"`` is the non-welded contract, and it needs ``declared``
    (the model's full link list) to be sound:

    * body in ``masses``            -> that mass;
    * body in ``declared`` only     -> a massless frame the model really has (an
      IMU or sensor mount).  PhysX refuses a literal zero mass, so it is set to
      :data:`MASSLESS_BODY_MASS_KG`, the same gram the welded path uses -- the
      model's own value is zero, and a gram is the smallest mass PhysX accepts;
    * body in neither               -> the model has no such link, so its mass
      would be silently dropped and the result claimed as an aligned plant.
      Raise.
    """
    if missing not in ("weld", "refuse"):
        raise MassAlignmentError(f"missing policy must be 'weld' or 'refuse', got {missing!r}")
    declared_names = set(declared) if declared is not None else set()
    indices: list[int] = []
    values: list[float] = []
    welded: list[str] = []
    absent: list[str] = []
    for index, name in enumerate(body_names):
        indices.append(index)
        if name in masses:
            values.append(masses[name])
            continue
        if missing == "refuse":
            if name not in declared_names:
                absent.append(name)
            # A declared-but-massless link is a real zero in the model.  PhysX
            # rejects a zero mass, so the gram floor is used instead -- the same
            # value and the same reason as the welded path's placeholder.
            values.append(MASSLESS_BODY_MASS_KG)
        else:
            values.append(WELDED_BODY_MASS_KG)
        welded.append(name)
    if len(welded) == len(body_names):
        raise MassAlignmentError("no asset body matches the model")
    if absent:
        raise MassAlignmentError(
            f"the model does not declare these asset bodies at all: {absent}"
        )
    return indices, values, welded
