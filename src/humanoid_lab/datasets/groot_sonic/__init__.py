"""GR00T N1.7 training pack for the Unitree G1 Dex3 / SONIC v1.1 corpus.

:mod:`.contract` is the frozen conversion contract, :mod:`.split` reads the
existing Psi0 episode split, :mod:`.convert` maps one episode onto the pack
timeline, :mod:`.modality` is the stock ``unitree_g1_sonic`` modality box,
:mod:`.writer` writes the LeRobot v2.1 split and :mod:`.validate` re-derives
every produced episode from the frozen sources.

The pack's field names, hand orders and slices are the loader-facing interop
contract with the pre-registered ``UNITREE_G1_SONIC`` embodiment; they live in
:mod:`.contract` and are re-checked on every produced split.
"""

from .contract import (
    ACTION_DIM,
    ACTION_HAND_ORDER,
    CANONICAL_ACTION_HAND_NAMES,
    CANONICAL_STATE_NAMES,
    DEFAULT_CONFIG_PATH,
    FPS,
    GRAVITY_FIELD,
    HORIZON,
    INSTRUCTION_KEY,
    LEFT_HAND_FIELD,
    MODEL_STATE_DIM,
    MOTION_TOKEN_DIM,
    MOTION_TOKEN_FIELD,
    RIGHT_HAND_FIELD,
    STATE_DIM,
    STATE_FIELD,
    STATE_HAND_ORDER,
    TAIL_ROWS,
    ConversionConfig,
    load_config,
)
from .convert import ConversionError, ConvertedEpisode, convert_episode, read_raw_episode
from .modality import assert_modality_payload, modality_payload
from .split import SplitError, assert_manifest_matches_config, load_split_manifest, selection

__all__ = [
    "ACTION_DIM",
    "ACTION_HAND_ORDER",
    "CANONICAL_ACTION_HAND_NAMES",
    "CANONICAL_STATE_NAMES",
    "DEFAULT_CONFIG_PATH",
    "FPS",
    "GRAVITY_FIELD",
    "HORIZON",
    "INSTRUCTION_KEY",
    "LEFT_HAND_FIELD",
    "MODEL_STATE_DIM",
    "MOTION_TOKEN_DIM",
    "MOTION_TOKEN_FIELD",
    "RIGHT_HAND_FIELD",
    "STATE_DIM",
    "STATE_FIELD",
    "STATE_HAND_ORDER",
    "TAIL_ROWS",
    "ConversionConfig",
    "ConversionError",
    "ConvertedEpisode",
    "SplitError",
    "assert_manifest_matches_config",
    "assert_modality_payload",
    "convert_episode",
    "load_config",
    "load_split_manifest",
    "modality_payload",
    "read_raw_episode",
    "selection",
]
