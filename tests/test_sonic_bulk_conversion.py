"""Bulk Unitree sweeps: episode selection and the declared corpus exclusions."""

from __future__ import annotations

import importlib.util
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/datasets/sonic/pilots.json"
CONVERTER = ROOT / "scripts/convert-unitree-sonic-production.py"


def load_converter():
    spec = importlib.util.spec_from_file_location("convert_unitree_sonic_production", CONVERTER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class EpisodeSelectionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.converter = load_converter()

    def test_excluded_episode_leaves_the_full_sweep(self) -> None:
        selection, excluded = self.converter.select_episodes(10, episodes=None, all_episodes=True, exclude=[0])
        self.assertEqual(selection, list(range(1, 10)))
        self.assertEqual(excluded, [0])

    def test_exclusion_is_recorded_only_when_it_is_selected(self) -> None:
        selection, excluded = self.converter.select_episodes(10, episodes=(5, 8), all_episodes=False, exclude=[0, 6])
        self.assertEqual(selection, [5, 7])
        self.assertEqual(excluded, [6])

    def test_exclusion_outside_the_collection_is_refused(self) -> None:
        with self.assertRaisesRegex(SystemExit, "outside 0..9"):
            self.converter.select_episodes(10, episodes=None, all_episodes=True, exclude=[10])

    def test_dropping_every_selected_episode_is_visible_to_the_caller(self) -> None:
        selection, excluded = self.converter.select_episodes(1, episodes=None, all_episodes=True, exclude=[0])
        self.assertEqual((selection, excluded), ([], [0]))


class DeclaredExclusionsTest(unittest.TestCase):
    def test_declared_exclusions_are_well_formed(self) -> None:
        entry = json.loads(CONFIG.read_text())["unitree"]
        declared = set(entry["bulk_datasets"])
        excluded = entry.get("bulk_excluded_episodes", {})
        for name, episodes in excluded.items():
            with self.subTest(dataset=name):
                self.assertIn(name, declared)
                self.assertTrue(episodes, "an empty exclusion list is noise; drop the key instead")
                self.assertTrue(all(isinstance(index, int) and index >= 0 for index in episodes))
        if excluded:
            self.assertTrue(str(entry.get("bulk_exclusion_reason", "")).strip())
