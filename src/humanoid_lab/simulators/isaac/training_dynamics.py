"""Read the joint armature the training rig configures, from the pinned source.

The evaluation aligned every joint to one armature (0.01) and one viscous
friction (0.05), mirroring the deployment MuJoCo model.  The training rig does
not: ``gear_sonic/envs/manager_env/robots/g1.py`` sets a *per-family* armature on
the legs, feet, waist and arms, and sets no friction at all.  Writing a single
value therefore does not mis-set one joint, it mis-sets the leg chain by up to
2.5x (0.0251 vs 0.01).

Like the masses, the values are read from the pinned file at start-up rather
than transcribed: the module evaluates the same ``ARMATURE_*`` constants and the
same per-group ``armature=`` mapping the training rig itself builds, so the two
cannot drift apart by someone retyping a table.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

DEFAULT_TRAINING_CFG = "/opt/src/sonic/gear_sonic/envs/manager_env/robots/g1.py"

#: The articulation config the training rig spawns.
TRAINING_ARTICULATION = "G1_CYLINDER_MODEL_12_DEX_CFG"


class TrainingDynamicsError(ValueError):
    """The training config cannot supply the joint dynamics to align with."""


def _literal(node: ast.AST, names: dict[str, float]) -> float:
    """Evaluate the arithmetic the training config uses for its gains.

    Only what that file actually contains: a number, a known ``ARMATURE_*``
    name, or a product of the two (``2.0 * ARMATURE_5020``).  Anything else is
    refused rather than guessed -- a silently wrong armature is exactly the
    failure this module exists to prevent.
    """
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.Name):
        if node.id not in names:
            raise TrainingDynamicsError(f"unknown armature constant {node.id!r}")
        return float(names[node.id])
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
        return _literal(node.left, names) * _literal(node.right, names)
    raise TrainingDynamicsError(f"unsupported armature expression: {ast.dump(node)}")


def _strings(node: ast.AST) -> list[str]:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return [node.value]
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        out: list[str] = []
        for element in node.elts:
            out.extend(_strings(element))
        return out
    raise TrainingDynamicsError("expected a string or a list of strings")


def _armature_entries(node: ast.AST, names: dict[str, float]) -> list[tuple[str, float]]:
    """One group's ``armature=`` value as (joint pattern, armature) pairs."""
    if isinstance(node, ast.Dict):
        entries: list[tuple[str, float]] = []
        for key, value in zip(node.keys, node.values):
            if key is None:
                raise TrainingDynamicsError("armature dict has a ** expansion")
            for pattern in _strings(key):
                entries.append((pattern, _literal(value, names)))
        return entries
    # A scalar armature applies to every joint of the group.
    return [(".*", _literal(node, names))]


def load_training_joint_armature(
    path: str | Path = DEFAULT_TRAINING_CFG,
) -> tuple[list[tuple[str, float]], dict[str, float]]:
    """The training rig's per-group armature, and its module-level constants.

    Returns ``(entries, constants)`` where ``entries`` preserves the file's own
    group order. A joint may match duplicate patterns within its group, but the
    pinned config gives all such matches the same value. The resolver takes the
    first match; each entry is a joint-name regex and its armature.
    """
    cfg_path = Path(path)
    try:
        tree = ast.parse(cfg_path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError) as error:
        raise TrainingDynamicsError(f"cannot read the training config {cfg_path}: {error}") from error

    constants: dict[str, float] = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name) or not target.id.startswith("ARMATURE_"):
            continue
        try:
            constants[target.id] = _literal(node.value, constants)
        except TrainingDynamicsError:
            # A derived constant (e.g. 2.0 * ARMATURE_5020) is resolved here
            # only if its own dependencies are already known.
            continue

    articulation = None
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if isinstance(target, ast.Name) and target.id == TRAINING_ARTICULATION:
            articulation = node.value
            break
    if articulation is None:
        raise TrainingDynamicsError(f"{TRAINING_ARTICULATION} not found in {cfg_path}")

    actuators = None
    for node in ast.walk(articulation):
        if isinstance(node, ast.keyword) and node.arg == "actuators":
            actuators = node.value
            break
    if actuators is None or not isinstance(actuators, ast.Dict):
        raise TrainingDynamicsError(f"{TRAINING_ARTICULATION} declares no actuators mapping")

    entries: list[tuple[str, float]] = []
    for value in actuators.values:
        if not isinstance(value, ast.Call):
            raise TrainingDynamicsError("actuator group is not a config call")
        patterns: list[str] = []
        armature_node: ast.AST | None = None
        for keyword in value.keywords:
            if keyword.arg == "joint_names_expr":
                patterns = _strings(keyword.value)
            elif keyword.arg == "armature":
                armature_node = keyword.value
        if armature_node is None:
            # The group configures no armature: IsaacLab then resolves the field
            # from the USD asset, which for this URDF (no <dynamics> at all) is
            # zero.  That is a real zero, so it is stated as one.
            entries.extend((pattern, 0.0) for pattern in patterns)
            continue
        group = _armature_entries(armature_node, constants)
        for pattern in patterns:
            for sub_pattern, armature in group:
                if sub_pattern == ".*":
                    entries.append((pattern, armature))
                else:
                    # A dict key replaces the group pattern with the sub-pattern.
                    entries.append((sub_pattern, armature))
    if not entries:
        raise TrainingDynamicsError(f"{TRAINING_ARTICULATION} declares no actuator armature")
    return entries, constants


def resolve_armature(
    joint_names: list[str], entries: list[tuple[str, float]]
) -> tuple[list[float], list[str]]:
    """Per-joint armature in ``joint_names`` order, plus the joints left unset.

    Patterns are full matches, the way IsaacLab resolves ``joint_names_expr``.
    A joint no group mentions keeps the evaluation's own value and is reported,
    so "the training config does not speak about this joint" stays visible
    instead of being silently written as a zero.
    """
    values: list[float] = []
    unset: list[str] = []
    compiled = [(re.compile(pattern), armature) for pattern, armature in entries]
    for name in joint_names:
        matches = [armature for regex, armature in compiled if regex.fullmatch(name)]
        if len(set(matches)) > 1:
            raise TrainingDynamicsError(
                f"training config gives conflicting armatures for {name}: {matches}"
            )
        if not matches:
            unset.append(name)
            values.append(0.0)
        else:
            values.append(matches[0])
    return values, unset
