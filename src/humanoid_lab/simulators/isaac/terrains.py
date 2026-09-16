"""Generated grounds the Isaac G1 simulator can run on.

A preset resolves to a world that is declared somewhere else in the stack
rather than re-declared here: ``instinct_parkour_rough`` is the InstinctLab
parkour training terrain, read from the pinned InstinctLab checkout that is
importable in the same interpreter.  Keeping one definition matters because the
point of the option is to drive the same robot on the same ground the instinct
policy was trained on; a copy in this package would drift from that world.

Both the terrain generator and the importer come from that definition, so the
sub-terrain mix, the tile materials and the edge-cylinder collision layer are
the training world's, not this module's invention.  What this module does own is
the interactive trim: the walls come off and the grid shrinks, exactly as the
playback script trims it, because a viewer run pays for every tile it builds.

Nothing here imports InstinctLab at module import time: a profile without a
terrain never touches it.
"""

from __future__ import annotations

import copy
import dataclasses
from typing import Any

from .contracts import TERRAIN_PRESETS, ContractError, TerrainSpec

#: Where the presets are declared.  The instinct playback task and this option
#: therefore resolve to the same config object.
PARKOUR_CONFIG_MODULE = "instinctlab.tasks.parkour.config.parkour_env_cfg"

#: The interactive grid: the training scene is a 10x20 field of 8 m tiles, of
#: which an interactive run needs neither rows nor columns.  Four rows preserve
#: the difficulty band the training curriculum walks along; ten columns keep
#: most of the sub-terrain mix (the training grid's twenty give every sub-terrain
#: a column, at ten the two rarest share their neighbours').
PLAY_ROWS = 4
PLAY_COLS = 10


def _preset_source(preset: str) -> Any:
    """Import the module that declares the preset, with a readable failure."""
    try:
        return __import__(PARKOUR_CONFIG_MODULE, fromlist=["SceneCfg", "ROUGH_TERRAINS_CFG"])
    except ImportError as exc:  # pragma: no cover - depends on the runtime image
        raise ContractError(
            f"terrain preset {preset!r} is declared in {PARKOUR_CONFIG_MODULE}, which this "
            f"interpreter cannot import: {exc}"
        ) from exc


def _declared_default(cls: type, name: str) -> Any:
    """Read a declared field default, factory included.

    ``@configclass`` rewrites every member of a config class into
    ``field(default_factory=...)``, and Python deletes the class attribute of a
    factory-backed field, so ``SceneCfg.terrain`` does not exist at run time.
    The field list is the only place the default is still declared.
    """
    try:
        declared = {member.name: member for member in dataclasses.fields(cls)}
    except TypeError as exc:
        raise ContractError(f"{cls!r} is not a dataclass, so {name!r} cannot be read") from exc
    member = declared.get(name)
    if member is None:
        raise ContractError(f"{cls.__name__} declares no field named {name!r}")
    if member.default is not dataclasses.MISSING:
        return member.default
    if member.default_factory is not dataclasses.MISSING:
        return member.default_factory()
    raise ContractError(f"{cls.__name__}.{name} is declared without a default to copy")


def build_terrain_importer(spec: TerrainSpec) -> tuple[Any, dict[str, Any]]:
    """Build the declared preset's terrain importer and its provenance.

    Returns the ``TerrainImporterCfg`` the scene entity carries, plus a summary
    of what was built.  The summary is printed at start-up and stored in the run
    manifest, so a saved run states which world it ran in.
    """
    if spec.preset not in TERRAIN_PRESETS:
        raise ContractError(f"unknown terrain preset {spec.preset!r}; expected one of {TERRAIN_PRESETS}")
    source = _preset_source(spec.preset)

    generator = copy.deepcopy(source.ROUGH_TERRAINS_CFG)
    # The training scene wraps most tiles in 5 m walls so a policy cannot walk
    # off its tile.  Nothing here expects them, and they hide the terrain from
    # the viewport, which is the whole point of an interactive run.
    walls_removed = []
    for name, sub_terrain in generator.sub_terrains.items():
        if getattr(sub_terrain, "wall_prob", None) is not None:
            sub_terrain.wall_prob = [0.0, 0.0, 0.0, 0.0]
            walls_removed.append(name)
    generator.num_rows = min(generator.num_rows, PLAY_ROWS)
    generator.num_cols = min(generator.num_cols, PLAY_COLS)

    importer = copy.deepcopy(_declared_default(source.SceneCfg, "terrain"))
    importer.prim_path = "/World/ground"
    importer.terrain_type = "generator"
    importer.terrain_generator = generator
    # The environments sit on the tiles the generator laid out.  The importer's
    # other option would put them on a grid around the world origin, which is
    # the corner of this terrain: the robot would fall off the edge of the
    # ground plane it is supposed to be standing on.
    importer.use_terrain_origins = True
    # The robot starts on a tile the terrain importer chooses.  With the level
    # band pinned to its lowest value the choice is the easiest tile in the
    # world, deterministically, which is the only tile a support band and a
    # freshly loaded controller can be expected to survive.
    importer.max_init_terrain_level = spec.max_init_terrain_level

    summary = {
        "preset": spec.preset,
        "source": f"{PARKOUR_CONFIG_MODULE}:SceneCfg.terrain",
        "grid": [generator.num_rows, generator.num_cols],
        "tile_size_m": [float(generator.size[0]), float(generator.size[1])],
        "sub_terrains": list(generator.sub_terrains),
        "walls_removed": walls_removed,
        "curriculum": bool(generator.curriculum),
        "max_init_terrain_level": spec.max_init_terrain_level,
        "virtual_obstacles": list(getattr(importer, "virtual_obstacles", {})),
        "static_friction": float(importer.physics_material.static_friction),
        "dynamic_friction": float(importer.physics_material.dynamic_friction),
    }
    return importer, summary
