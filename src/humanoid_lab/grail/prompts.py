"""Deterministic English instructions for GRAIL ``pickup_table`` trajectories.

A GRAIL source trajectory is identified by its file stem
``<task_family>__<object_asset_id>__<motion_variant>`` (for example
``pickup_table__apple_17__003``).  Future camera variations are derived
trajectories: they replay one source motion from another viewpoint and reuse its
stem as their ``source_motion_id``.  Every instruction in this module is
therefore a pure function of that stem, so all camera variations of one source
trajectory resolve to exactly the same English prompt.

The layer is deliberately closed: it reads no GRAIL pickles, edits no release
file and calls no model.  Renaming a category, changing a template, adding a
template or changing the selection algorithm is a prompt-policy change and must
bump :data:`PROMPT_POLICY_VERSION`.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

MANIFEST_SCHEMA_VERSION = 1
PROMPT_POLICY_VERSION = "grail_pickup_table_prompt_v1"

# Selection is a fixed, documented hash: no ``hash()``, no per-run or per-machine
# entropy.  The key is the source motion id, so a stem always picks the same
# index on every machine and in every run.
SELECTION_ALGORITHM = "sha256-first8be-mod-v1"

# Semantic family: every template only asks to take an object off the table.
# Nothing here asks the robot to bring, place, hand over or fetch anything, and
# no template names a hand: the dataset is right-handed, but contrasting hands
# would be language supervision the trajectories do not carry.
TEMPLATE_FAMILY_ID = "table_pick_and_lift_v1"

FAMILY_SEPARATOR = "__"


class PromptLayerError(ValueError):
    """A trajectory stem or a prompt-layer input violates the declared contract."""


@dataclass(frozen=True)
class PromptTemplate:
    """One instruction template of :data:`TEMPLATE_FAMILY_ID`."""

    template_id: str
    instruction_format: str


# Order is policy: ``select_template`` indexes this tuple by hash.  Appending a
# template re-maps existing trajectories, so it requires a policy version bump.
TEMPLATES: tuple[PromptTemplate, ...] = (
    PromptTemplate("pick_from_table_v1", "pick up the {object} from the table"),
    PromptTemplate("grab_from_table_v1", "grab the {object} from the table"),
    PromptTemplate("lift_off_table_v1", "lift the {object} off the table"),
    PromptTemplate("pick_v1", "pick up the {object}"),
    PromptTemplate("grab_v1", "grab the {object}"),
    PromptTemplate("take_from_table_v1", "take the {object} from the table"),
)

# Object labels are the raw category with underscores replaced by spaces, except
# for the reviewed categories below.  Identity entries are explicit: they were
# reviewed for natural phrasing and keep the default.  The raw category stays in
# the manifest, so nothing is lost by normalizing the label.
OBJECT_NAME_OVERRIDES: dict[str, str] = {
    "alcohol": "alcohol bottle",
    "bagged_food": "bag of food",
    "bar": "snack bar",  # RoboCasa ``bar`` is a snack bar; ``bar_soap`` is separate
    "bottled_drink": "bottled drink",
    "boxed_drink": "boxed drink",
    "boxed_food": "boxed food",
    "canned_food": "canned food",
    "coffee_cup": "coffee cup",
    "condiment": "condiment",
    "spray": "spray bottle",
    "water_bottle": "water bottle",
}

_FIELD_RE = re.compile(r"[a-z][a-z0-9_]*")
_ASSET_INDEX_RE = re.compile(r"(?P<category>[a-z][a-z0-9_]*)_(?P<index>[0-9]+)")


@dataclass(frozen=True)
class MotionStem:
    """Structural identity of one GRAIL source trajectory."""

    source_motion_id: str
    task_family: str
    object_asset_id: str
    object_category_raw: str
    motion_variant: str


@dataclass(frozen=True)
class PromptRecord:
    """One manifest row: a source trajectory and the single prompt it carries."""

    source_motion_id: str
    task_family: str
    object_asset_id: str
    object_category_raw: str
    object_name: str
    motion_variant: str
    prompt_template_id: str
    instruction: str

    def to_manifest_row(self) -> dict[str, object]:
        """Return the JSON-serializable manifest row, in the manifest's field order."""
        return {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "prompt_policy_version": PROMPT_POLICY_VERSION,
            "source_motion_id": self.source_motion_id,
            "task_family": self.task_family,
            "object_asset_id": self.object_asset_id,
            "object_category_raw": self.object_category_raw,
            "object_name": self.object_name,
            "motion_variant": self.motion_variant,
            "prompt_template_id": self.prompt_template_id,
            "instruction": self.instruction,
        }


def _require_field(value: str, name: str, stem: str) -> str:
    if not _FIELD_RE.fullmatch(value):
        raise PromptLayerError(
            f"{stem!r}: {name} must be lower-case alphanumeric with underscores "
            f"(got {value!r})"
        )
    return value


def parse_motion_stem(stem: str) -> MotionStem:
    """Split a trajectory stem into its identity fields.

    The object category is derived by dropping exactly the trailing ``_<digits>``
    asset index from ``object_asset_id`` and nothing else, so category names that
    carry their own underscores (``bagged_food_14``, ``sweet_potato_1``) survive
    intact.  A stem that does not fit the schema raises :class:`PromptLayerError`.
    """
    parts = stem.split(FAMILY_SEPARATOR)
    if len(parts) != 3:
        raise PromptLayerError(
            f"{stem!r}: expected <task_family>__<object_asset_id>__<motion_variant>"
        )
    task_family, object_asset_id, motion_variant = parts
    _require_field(task_family, "task_family", stem)
    if task_family != "pickup_table":
        raise PromptLayerError(
            f"{stem!r}: task_family must be 'pickup_table' for this prompt policy "
            f"(got {task_family!r})"
        )
    _require_field(object_asset_id, "object_asset_id", stem)
    if not motion_variant.isdigit():
        raise PromptLayerError(
            f"{stem!r}: motion_variant must be digits (got {motion_variant!r})"
        )
    match = _ASSET_INDEX_RE.fullmatch(object_asset_id)
    if match is None:
        raise PromptLayerError(
            f"{stem!r}: object_asset_id must end in '_<digits>' "
            f"(got {object_asset_id!r})"
        )
    category = match.group("category")
    if category.endswith("_"):
        raise PromptLayerError(
            f"{stem!r}: object category must not be empty (got {category!r})"
        )
    return MotionStem(
        source_motion_id=stem,
        task_family=task_family,
        object_asset_id=object_asset_id,
        object_category_raw=category,
        motion_variant=motion_variant,
    )


def object_name_for_category(category: str) -> str:
    """Turn a raw object category into the noun phrase used in an instruction."""
    if not category:
        raise PromptLayerError("object category is empty")
    override = OBJECT_NAME_OVERRIDES.get(category)
    if override is not None:
        return override
    return category.replace("_", " ")


def select_template(source_motion_id: str) -> PromptTemplate:
    """Pick the template for a source motion, deterministically.

    The index is the first eight bytes (big-endian) of
    ``sha256(SELECTION_ALGORITHM + NUL + source_motion_id)`` modulo the number of
    templates.  The hash is SHA-256 rather than :func:`hash`, whose truncation
    and salting differ per process and per machine.
    """
    payload = f"{SELECTION_ALGORITHM}\x00{source_motion_id}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return TEMPLATES[int.from_bytes(digest[:8], "big") % len(TEMPLATES)]


def build_prompt_record(stem: str) -> PromptRecord:
    """Build the single prompt record for one trajectory stem."""
    parsed = parse_motion_stem(stem)
    template = select_template(parsed.source_motion_id)
    object_name = object_name_for_category(parsed.object_category_raw)
    return PromptRecord(
        source_motion_id=parsed.source_motion_id,
        task_family=parsed.task_family,
        object_asset_id=parsed.object_asset_id,
        object_category_raw=parsed.object_category_raw,
        object_name=object_name,
        motion_variant=parsed.motion_variant,
        prompt_template_id=template.template_id,
        instruction=template.instruction_format.format(object=object_name),
    )
