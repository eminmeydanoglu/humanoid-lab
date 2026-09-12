"""Tests for the GRAIL pickup_table prompt layer.

These run without numpy, joblib or the GRAIL release: the prompt layer is a pure
function of a trajectory stem plus its fixed tables.  The manifest tests use a
throwaway fixture directory of empty ``*.pkl`` files.
"""

from __future__ import annotations

import hashlib
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from humanoid_lab.grail.cli import build_manifest, main, write_manifest  # noqa: E402
from humanoid_lab.grail.prompts import (  # noqa: E402
    MANIFEST_SCHEMA_VERSION,
    OBJECT_NAME_OVERRIDES,
    PROMPT_POLICY_VERSION,
    SELECTION_ALGORITHM,
    TEMPLATES,
    PromptLayerError,
    build_prompt_record,
    object_name_for_category,
    parse_motion_stem,
    select_template,
)

# Stems copied from the real release: every one of these files exists under
# data/datasets/grail/data/pickup_table/robot/.
REAL_STEMS = (
    "pickup_table__alcohol_0__000",
    "pickup_table__apple_17__003",
    "pickup_table__bagged_food_14__002",
    "pickup_table__bar_2__001",
    "pickup_table__bar_soap_0__000",
    "pickup_table__coffee_cup_15__001",
    "pickup_table__condiment_0__000",
    "pickup_table__spray_0__000",
    "pickup_table__sweet_potato_1__000",
    "pickup_table__water_bottle_23__000",
)

# Pins the selection algorithm across machines and runs.  A change here is a
# prompt-policy change, not a refactor.
GOLDEN_TEMPLATES = {
    "pickup_table__alcohol_0__000": ("lift_off_table_v1", "lift the alcohol bottle off the table"),
    "pickup_table__apple_17__003": ("grab_v1", "grab the apple"),
    "pickup_table__bagged_food_14__002": (
        "grab_from_table_v1",
        "grab the bag of food from the table",
    ),
    "pickup_table__bar_2__001": ("pick_from_table_v1", "pick up the snack bar from the table"),
    "pickup_table__coffee_cup_15__001": (
        "lift_off_table_v1",
        "lift the coffee cup off the table",
    ),
    "pickup_table__condiment_0__000": (
        "pick_from_table_v1",
        "pick up the condiment from the table",
    ),
    "pickup_table__spray_0__000": (
        "grab_from_table_v1",
        "grab the spray bottle from the table",
    ),
    "pickup_table__water_bottle_23__000": (
        "take_from_table_v1",
        "take the water bottle from the table",
    ),
}


class ParseMotionStemTests(unittest.TestCase):
    def test_parses_a_plain_stem(self) -> None:
        stem = parse_motion_stem("pickup_table__apple_17__003")
        self.assertEqual(stem.source_motion_id, "pickup_table__apple_17__003")
        self.assertEqual(stem.task_family, "pickup_table")
        self.assertEqual(stem.object_asset_id, "apple_17")
        self.assertEqual(stem.object_category_raw, "apple")
        self.assertEqual(stem.motion_variant, "003")

    def test_keeps_underscored_categories_intact(self) -> None:
        cases = {
            "pickup_table__bagged_food_14__002": "bagged_food",
            "pickup_table__sweet_potato_1__000": "sweet_potato",
            "pickup_table__coffee_cup_15__001": "coffee_cup",
            "pickup_table__bar_soap_0__000": "bar_soap",
        }
        for stem, category in cases.items():
            with self.subTest(stem=stem):
                self.assertEqual(parse_motion_stem(stem).object_category_raw, category)

    def test_rejects_stems_without_three_fields(self) -> None:
        for stem in (
            "",
            "pickup_table",
            "pickup_table__apple_17",
            "pickup_table__apple_17__003__004",
            "__apple_17__003",
            "pickup_table____003",
        ):
            with self.subTest(stem=stem), self.assertRaises(PromptLayerError):
                parse_motion_stem(stem)

    def test_rejects_other_task_families(self) -> None:
        for stem in (
            "pickup_ground__apple_17__003",
            "sitting__chair_1__000",
        ):
            with self.subTest(stem=stem), self.assertRaises(PromptLayerError):
                parse_motion_stem(stem)

    def test_rejects_an_asset_without_a_numeric_index_suffix(self) -> None:
        for stem in (
            "pickup_table__apple__003",
            "pickup_table__apple_last__003",
        ):
            with self.subTest(stem=stem), self.assertRaises(PromptLayerError):
                parse_motion_stem(stem)

    def test_rejects_malformed_fields(self) -> None:
        for stem in (
            "pickup_table__Apple_17__003",
            "pickup_table__apple_17__00x",
            "pickup_table__apple_17__",
            "pickup_table__apple-17__003",
            "pickup_table__17__003",
            "pickup_table__apple_17__003 ",
        ):
            with self.subTest(stem=stem), self.assertRaises(PromptLayerError):
                parse_motion_stem(stem)


class ObjectNameTests(unittest.TestCase):
    def test_default_is_underscore_to_space(self) -> None:
        self.assertEqual(object_name_for_category("bell_pepper"), "bell pepper")
        self.assertEqual(object_name_for_category("cutting_board"), "cutting board")
        self.assertEqual(object_name_for_category("apple"), "apple")
        self.assertEqual(object_name_for_category("bar_soap"), "bar soap")

    def test_reviewed_categories_are_all_covered(self) -> None:
        reviewed = {
            "alcohol",
            "bagged_food",
            "bar",
            "bottled_drink",
            "boxed_drink",
            "boxed_food",
            "canned_food",
            "coffee_cup",
            "condiment",
            "spray",
            "water_bottle",
        }
        self.assertTrue(reviewed.issubset(OBJECT_NAME_OVERRIDES))

    def test_overrides_replace_the_underscore_default(self) -> None:
        self.assertEqual(object_name_for_category("alcohol"), "alcohol bottle")
        self.assertEqual(object_name_for_category("bagged_food"), "bag of food")
        self.assertEqual(object_name_for_category("bar"), "snack bar")
        self.assertEqual(object_name_for_category("spray"), "spray bottle")
        for category, label in OBJECT_NAME_OVERRIDES.items():
            with self.subTest(category=category):
                self.assertEqual(object_name_for_category(category), label)
                self.assertNotIn("_", label)

    def test_empty_category_is_an_error(self) -> None:
        with self.assertRaises(PromptLayerError):
            object_name_for_category("")


class TemplateSelectionTests(unittest.TestCase):
    def test_selection_follows_the_documented_hash(self) -> None:
        for stem in REAL_STEMS:
            payload = f"{SELECTION_ALGORITHM}\x00{stem}".encode("utf-8")
            expected = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % len(TEMPLATES)
            with self.subTest(stem=stem):
                self.assertEqual(select_template(stem).template_id, TEMPLATES[expected].template_id)

    def test_selection_is_stable_across_calls(self) -> None:
        for stem in REAL_STEMS:
            first = build_prompt_record(stem)
            self.assertEqual([build_prompt_record(stem) for _ in range(16)], [first] * 16)

    def test_golden_prompts_are_pinned(self) -> None:
        for stem, (template_id, instruction) in GOLDEN_TEMPLATES.items():
            with self.subTest(stem=stem):
                record = build_prompt_record(stem)
                self.assertEqual(record.prompt_template_id, template_id)
                self.assertEqual(record.instruction, instruction)

    def test_one_source_motion_id_gets_exactly_one_prompt(self) -> None:
        # A camera variation is a new file that reuses the source stem; lookup is
        # by source_motion_id, so it must resolve to the same template and text.
        stem = "pickup_table__apple_17__003"
        for _ in range(8):
            record = build_prompt_record(stem)
            self.assertEqual(record, build_prompt_record(record.source_motion_id))
            self.assertEqual(
                select_template(record.source_motion_id).template_id,
                record.prompt_template_id,
            )

    def test_all_templates_are_reachable(self) -> None:
        seen = {select_template(stem).template_id for stem in REAL_STEMS}
        self.assertEqual(seen, {template.template_id for template in TEMPLATES})


class TemplateFamilyTests(unittest.TestCase):
    def test_template_set_is_exactly_the_agreed_six(self) -> None:
        self.assertEqual(
            tuple((t.template_id, t.instruction_format) for t in TEMPLATES),
            (
                ("pick_from_table_v1", "pick up the {object} from the table"),
                ("grab_from_table_v1", "grab the {object} from the table"),
                ("lift_off_table_v1", "lift the {object} off the table"),
                ("pick_v1", "pick up the {object}"),
                ("grab_v1", "grab the {object}"),
                ("take_from_table_v1", "take the {object} from the table"),
            ),
        )

    def test_no_template_asks_for_anything_beyond_picking_off_the_table(self) -> None:
        forbidden = ("bring", "place", "put", "hand", "fetch", "give", "deliver", "move to", "left")
        for template in TEMPLATES:
            with self.subTest(template=template.template_id):
                text = template.instruction_format.lower()
                for word in forbidden:
                    self.assertNotIn(word, text)
                self.assertIn("the {object}", text)


class BuildPromptRecordTests(unittest.TestCase):
    def test_record_matches_the_manifest_contract(self) -> None:
        record = build_prompt_record("pickup_table__coffee_cup_15__001")
        self.assertEqual(
            record.to_manifest_row(),
            {
                "schema_version": MANIFEST_SCHEMA_VERSION,
                "prompt_policy_version": PROMPT_POLICY_VERSION,
                "source_motion_id": "pickup_table__coffee_cup_15__001",
                "task_family": "pickup_table",
                "object_asset_id": "coffee_cup_15",
                "object_category_raw": "coffee_cup",
                "object_name": "coffee cup",
                "motion_variant": "001",
                "prompt_template_id": "lift_off_table_v1",
                "instruction": "lift the coffee cup off the table",
            },
        )

    def test_record_is_json_serializable(self) -> None:
        row = build_prompt_record(REAL_STEMS[0]).to_manifest_row()
        self.assertEqual(json.loads(json.dumps(row)), row)


class ManifestTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.robot_dir = self.root / "robot"
        self.robot_dir.mkdir()

    def _write_stems(self, *stems: str) -> None:
        for stem in stems:
            (self.robot_dir / f"{stem}.pkl").touch()

    def _run(self, *extra: str) -> int:
        """Run the CLI, keeping its summary out of the test output."""
        with redirect_stdout(io.StringIO()):
            return main(["--robot-dir", str(self.robot_dir), "--output", *extra])

    def test_manifest_is_sorted_and_has_one_row_per_stem(self) -> None:
        self._write_stems(
            "pickup_table__coffee_cup_15__001",
            "pickup_table__apple_17__003",
            "pickup_table__apple_17__002",
        )
        output = self.root / "manifest.jsonl"
        self.assertEqual(self._run(str(output)), 0)
        rows = [json.loads(line) for line in output.read_text().splitlines()]
        self.assertEqual(
            [row["source_motion_id"] for row in rows],
            [
                "pickup_table__apple_17__002",
                "pickup_table__apple_17__003",
                "pickup_table__coffee_cup_15__001",
            ],
        )
        self.assertEqual(len(rows), len({row["source_motion_id"] for row in rows}))
        self.assertEqual({row["schema_version"] for row in rows}, {MANIFEST_SCHEMA_VERSION})
        self.assertEqual({row["prompt_policy_version"] for row in rows}, {PROMPT_POLICY_VERSION})
        self.assertTrue(all(row["instruction"] for row in rows))
        self.assertEqual(len(build_manifest(self.robot_dir)), 3)

    def test_rewriting_is_deterministic_and_leaves_no_temporary_file(self) -> None:
        self._write_stems("pickup_table__apple_17__003", "pickup_table__bar_2__001")
        output = self.root / "manifest.jsonl"
        self.assertEqual(self._run(str(output)), 0)
        first = output.read_bytes()
        self.assertEqual(self._run(str(output)), 0)
        self.assertEqual(output.read_bytes(), first)
        self.assertEqual(sorted(p.name for p in self.root.iterdir()), ["manifest.jsonl", "robot"])

    def test_malformed_stem_fails_loudly_and_writes_nothing(self) -> None:
        self._write_stems("pickup_table__apple_17__003", "pickup_table__apple__003")
        output = self.root / "manifest.jsonl"
        stderr = io.StringIO()
        with redirect_stderr(stderr):
            code = self._run(str(output))
        self.assertEqual(code, 2)
        self.assertIn("pickup_table__apple__003", stderr.getvalue())
        self.assertFalse(output.exists())

    def test_missing_robot_dir_is_an_error(self) -> None:
        output = self.robot_dir / "manifest.jsonl"
        stderr = io.StringIO()
        with redirect_stdout(io.StringIO()), redirect_stderr(stderr):
            code = main(["--robot-dir", str(self.root / "absent"), "--output", str(output)])
        self.assertEqual(code, 2)
        self.assertIn("does not exist", stderr.getvalue())
        self.assertFalse(output.exists())

    def test_empty_robot_dir_is_an_error(self) -> None:
        output = self.root / "manifest.jsonl"
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            code = self._run(str(output))
        self.assertEqual(code, 2)
        self.assertFalse(output.exists())

    def test_output_directory_is_rejected(self) -> None:
        self._write_stems("pickup_table__apple_17__003")
        stderr = io.StringIO()
        with redirect_stdout(io.StringIO()), redirect_stderr(stderr):
            code = self._run(str(self.root))
        self.assertEqual(code, 2)
        self.assertIn("is a directory", stderr.getvalue())

    def test_write_manifest_creates_missing_parent_directories(self) -> None:
        self._write_stems("pickup_table__apple_17__003")
        output = self.root / "nested" / "out" / "manifest.jsonl"
        write_manifest(build_manifest(self.robot_dir), output)
        self.assertTrue(output.is_file())

    def test_invalid_stem_raises_from_build_manifest(self) -> None:
        self._write_stems("pickup_table__spray_0__notanumber")
        with self.assertRaises(PromptLayerError):
            build_manifest(self.robot_dir)


if __name__ == "__main__":
    unittest.main()
