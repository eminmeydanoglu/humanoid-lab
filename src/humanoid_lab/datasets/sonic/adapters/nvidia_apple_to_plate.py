"""AppleToPlate body resolver; hands deliberately remain blocked."""

from __future__ import annotations

APPLE_BLOCK_ORDER = ("left_leg", "right_leg", "waist", "left_arm", "right_arm", "left_hand", "right_hand")
HAND_SCHEMA_STATUS = "unresolved"


def require_hand_schema() -> None:
    raise RuntimeError("AppleToPlate hand channel order/unit is unresolved; full 78D conversion is blocked")
