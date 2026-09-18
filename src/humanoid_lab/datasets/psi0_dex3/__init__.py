"""Gate 0 and Gate 1 data pipeline for the Psi0 Unitree Dex3 / SONIC v1 pack.

:mod:`humanoid_lab.datasets.psi0_dex3.contract` is the frozen conversion
contract, :mod:`.split` the deterministic evaluation split, :mod:`.convert` the
per-episode 50->30 Hz conversion, :mod:`.writer` the LeRobot v2.1 output and
:mod:`.validate` the fail-closed validator.

The field names and widths of the produced pack are the loader-facing interop
contract and live once, in ``humanoid_lab.datasets.psi0.contract``; this package
imports them rather than declaring them again.
"""

from .contract import (
    ANCHOR_MASK_KEY,
    CANONICAL_HAND_NAMES,
    CANONICAL_STATE_NAMES,
    MASK_KEY,
    ConversionConfig,
    load_config,
)

__all__ = [
    "ANCHOR_MASK_KEY",
    "CANONICAL_HAND_NAMES",
    "CANONICAL_STATE_NAMES",
    "MASK_KEY",
    "ConversionConfig",
    "load_config",
]
