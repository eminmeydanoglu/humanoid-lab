"""Fail-closed validation of the served Psi0 checkpoint's ``/info`` contract.

The bridge never guesses a wire layout.  Before it will connect, the server's
``/info`` response must describe exactly the contract this bridge was built for:

* exactly one image key, named by the served run itself (this repo's own packs
  call it ``observation.images.egocentric``; a released checkpoint may call the
  same single camera ``observation.images.head``).  The bridge sends its one
  frame under the declared key, so the name is the server's to choose while the
  count is not: a checkpoint needing a second camera cannot be served;
* a single-frame (history = 1) raw state vector, 43D, unnormalized;
* an action width of 78 or 80 (64 token + 14 Dex3 + optional 2 neck);
* chunk metadata (``action_chunk_size`` and ``action_exec_horizon``);
* a ``resize`` transform plus a ``center_crop`` of the same size, both reported
  by the server.  The bridge never resizes: the served transform is the server's
  own, and the launcher checks that it equals the selected run's ``run_config``
  before anything is marked ready (see ``wait_for_info``'s identity checks).

The 43D number is the raw training layout from
``configs/datasets/psi0/unitree_dex3_sonic_v1.yaml`` (``state.dim: 43``); the
same config pads the model-side width to 45, so a server that reports 45 is
accepted and pads the 43D vector itself (``pad_state_dim``), while the bridge
still sends exactly 43 raw values.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping

#: The image key this repo's own Unitree Dex3 SONIC packs are built with.  It is
#: the expected key for the fine-tuned and base entries; a served run that
#: declares a different single key is still drivable, and the bridge sends the
#: frame under whatever ``/info`` declares.
IMAGE_KEY = "observation.images.egocentric"
STATE_KEY = "states"

BODY_DIM = 29
HAND_DIM = 7
RAW_STATE_DIM = BODY_DIM + 2 * HAND_DIM  # 43
TOKEN_DIM = 64
HAND14_DIM = 2 * HAND_DIM  # 14
NECK_DIM = 2

ACTION_DIMS = (78, 80)

# Raw wire width accepted from the server description, plus the model-side width
# the 43D vector is padded to before the checkpoint sees it.
SERVER_STATE_DIMS = (RAW_STATE_DIM, 45)

#: The transform this repo's own packs are built with (height, width).  The
#: served value is read from ``/info`` and checked against the selected run's
#: ``run_config``; this constant documents the repo's own contract and bounds
#: what a served transform may be.
RESIZE_SIZE = (240, 320)
#: Bounds a declared transform must satisfy; the model sees 3D image tokens, so a
#: non-positive or absurd size is a malformed run config rather than a variant.
MIN_TRANSFORM_SIDE = 32
MAX_TRANSFORM_SIDE = 4096

# The 80D neck block has no field in Protocol v4's ``pose`` message.  A
# near-zero neck therefore means "no neck command this frame"; anything above
# this tolerance is an instruction the bridge cannot forward, which is an error
# rather than a silent drop.
NECK_NOOP_TOLERANCE = 0.05

_STATE_DESC_RE = re.compile(r"^\s*(\d+)\s*x\s*(\d+)\b")


class ContractError(ValueError):
    """The server (or a caller) violates the declared bridge contract."""


@dataclass(frozen=True)
class ServerInfo:
    """The subset of ``/info`` the bridge depends on, after validation."""

    run_dir: str
    action_dim: int
    action_chunk_size: int
    action_exec_horizon: int
    state_dim: int
    history_length: int
    image_key: str
    resize_size: tuple[int, int]
    center_crop_size: tuple[int, int]
    normalize_state: bool
    rtc_enabled: bool
    ckpt_step: int | None
    dataset_name: str

    @property
    def online_state_dim(self) -> int:
        """The raw width the bridge sends; padding to ``state_dim`` is server-side."""
        return RAW_STATE_DIM


def _size_tuple(value: Any, what: str) -> tuple[int, int]:
    if isinstance(value, int) and not isinstance(value, bool):
        return (value, value)
    if isinstance(value, (list, tuple)) and len(value) == 2:
        try:
            return (int(value[0]), int(value[1]))
        except (TypeError, ValueError) as exc:
            raise ContractError(f"{what} size is not integral: {value!r}") from exc
    raise ContractError(f"{what} size is not an int or a 2-vector: {value!r}")


def _transform_size(transforms: Any, name: str) -> tuple[int, int] | None:
    if not isinstance(transforms, list):
        return None
    for entry in transforms:
        if isinstance(entry, Mapping) and entry.get("name") == name:
            if "size" not in entry:
                raise ContractError(f"transform {name!r} has no size")
            return _size_tuple(entry["size"], f"transform {name!r}")
    return None


def _require_sane_transform(size: tuple[int, int], what: str) -> None:
    """Reject a declared transform no image could satisfy."""
    height, width = size
    for side in size:
        if not MIN_TRANSFORM_SIDE <= side <= MAX_TRANSFORM_SIDE:
            raise ContractError(
                f"{what} size {size} is outside "
                f"[{MIN_TRANSFORM_SIDE}, {MAX_TRANSFORM_SIDE}]"
            )
    if height <= 0 or width <= 0:  # pragma: no cover - covered by the bounds above
        raise ContractError(f"{what} size {size} is not positive")


def _require_int(value: Any, what: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ContractError(f"{what} must be an int, got {value!r}")
    return value


def validate_info(raw: Any) -> ServerInfo:
    """Validate a decoded ``/info`` body; raise :class:`ContractError` on any mismatch."""
    if not isinstance(raw, Mapping):
        raise ContractError(f"/info must be a JSON object, got {type(raw).__name__}")

    expected = raw.get("expected_keys")
    if not isinstance(expected, Mapping):
        raise ContractError("/info is missing expected_keys")

    images = expected.get("image")
    if not isinstance(images, Mapping):
        raise ContractError("/info expected_keys.image is not an object")
    declared_keys = sorted(str(key) for key in images)
    if len(declared_keys) != 1:
        raise ContractError(
            "/info must declare exactly one image key: this bridge sends one camera frame, so a "
            f"checkpoint needing any other number of views cannot be served. Declared: {declared_keys}"
        )
    image_key = declared_keys[0]

    state_expected = expected.get("state")
    if not isinstance(state_expected, Mapping):
        raise ContractError("/info expected_keys.state is not an object")
    description = state_expected.get(STATE_KEY)
    if not isinstance(description, str):
        raise ContractError(f"/info does not declare a {STATE_KEY!r} state vector")
    match = _STATE_DESC_RE.match(description)
    if match is None:
        raise ContractError(f"unparseable state description: {description!r}")
    history_length, described_dim = int(match.group(1)), int(match.group(2))
    if history_length != 1:
        raise ContractError(
            f"state history must be 1 frame, /info describes {history_length}"
        )
    if described_dim not in SERVER_STATE_DIMS:
        raise ContractError(
            f"state dim {described_dim} is not one of {SERVER_STATE_DIMS}"
        )

    observation = raw.get("observation")
    if not isinstance(observation, Mapping):
        raise ContractError("/info is missing observation")
    state_dim = _require_int(observation.get("state_dim"), "observation.state_dim")
    if state_dim != described_dim:
        raise ContractError(
            f"observation.state_dim {state_dim} disagrees with the declared "
            f"{described_dim}D state vector"
        )
    if state_dim not in SERVER_STATE_DIMS:
        raise ContractError(
            f"observation.state_dim {state_dim} is not one of {SERVER_STATE_DIMS}"
        )
    if observation.get("normalize_state") is not True:
        raise ContractError("the served checkpoint must normalize states server-side")

    action = raw.get("action")
    if not isinstance(action, Mapping):
        raise ContractError("/info is missing action")
    action_dim = _require_int(action.get("action_dim"), "action.action_dim")
    if action_dim not in ACTION_DIMS:
        raise ContractError(
            f"action dim {action_dim} is neither 78 nor 80; refusing to guess a layout"
        )
    chunk = _require_int(action.get("action_chunk_size"), "action.action_chunk_size")
    if chunk <= 0:
        raise ContractError(f"action_chunk_size must be positive, got {chunk}")
    horizon = _require_int(action.get("action_exec_horizon"), "action.action_exec_horizon")
    if horizon <= 0 or horizon > chunk:
        raise ContractError(
            f"action_exec_horizon {horizon} is not in (0, {chunk}]"
        )

    resize = _transform_size(raw.get("transforms"), "resize")
    if resize is None:
        raise ContractError("/info declares no resize transform")
    _require_sane_transform(resize, "resize")
    crop = _transform_size(raw.get("transforms"), "center_crop")
    if crop is None:
        raise ContractError("/info declares no center_crop transform")
    _require_sane_transform(crop, "center_crop")
    if crop != resize:
        raise ContractError(
            f"/info center_crop {crop} is not the served resize {resize}: this bridge feeds the "
            "whole resized frame and cannot reproduce a narrower view"
        )

    # The server can run with PSI_RTC_INIT_PREV=1, which makes the *client* seed
    # the first chunk by sending state.init_prev_action.  This bridge only sends
    # the raw 43D state vector, so such a server cannot be driven correctly.
    rtc = raw.get("rtc")
    if isinstance(rtc, Mapping) and rtc.get("init_prev_enabled"):
        raise ContractError(
            "/info reports rtc.init_prev_enabled=true: the server expects the client to send "
            "state.init_prev_action, which this bridge does not produce"
        )

    # Run identity: the served checkpoint must be the run the launcher selected.
    # ``run_dir`` and ``ckpt_step`` are compared with the CLI directory (canonical
    # resolved paths) and the requested step; ``dataset_name`` is the request's
    # dataset field, so a server that omits it cannot be driven correctly.
    run_dir = raw.get("run_dir")
    if not isinstance(run_dir, str) or not run_dir.strip():
        raise ContractError(f"/info run_dir must be a non-empty string, got {run_dir!r}")
    ckpt_step = raw.get("ckpt_step")
    dataset_name = raw.get("dataset_name")
    if not isinstance(dataset_name, str) or not dataset_name.strip():
        raise ContractError(
            f"/info dataset_name must be a non-empty string, got {dataset_name!r}"
        )
    return ServerInfo(
        run_dir=run_dir,
        action_dim=action_dim,
        action_chunk_size=chunk,
        action_exec_horizon=horizon,
        state_dim=state_dim,
        history_length=history_length,
        image_key=image_key,
        resize_size=resize,
        center_crop_size=crop,
        normalize_state=True,
        rtc_enabled=bool(raw.get("rtc_enabled", False)),
        ckpt_step=ckpt_step if isinstance(ckpt_step, int) else None,
        dataset_name=dataset_name,
    )
