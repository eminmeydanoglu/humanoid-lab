"""Experiment 07's own decisions, as code rather than prose.

The experiment's result may only be A (causal), B (state fixed, target not),
C (inconsistent) or D (no valid warm start), and it may only be one of those
after the pre-declared validity gates have been applied.  Those two rules are
what these tests pin down: an invalid run can never contribute a target effect,
and the decision can never be softer than the gates and the acceptance allow.
"""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "groot-token-warmstart-ab.py"


def load_module():
    name = "groot_token_warmstart_ab"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def effect(label: str, drop: float, relative: float = -0.5, *, state_fixed: bool = True,
           baseline_in_p95: bool = False) -> dict:
    return {
        "pair": f"A{label[-1]}/{label}",
        "baseline": f"A{label[-1]}",
        "treatment": label,
        "first_5s_target_palm_z_delta_m": drop,
        "first_5s_target_palm_cube_min_distance_right_relative_delta": relative,
        "first_5s_target_palm_z_baseline_m": 0.5,
        "first_5s_target_palm_z_treatment_m": 0.5 + drop,
        "start_arms_in_p95_baseline": baseline_in_p95,
        "start_arms_in_p95_treatment": state_fixed,
    }


def gate(label: str, integrity: bool, state_fixed: bool | None = None) -> dict:
    state_fixed = integrity if state_fixed is None else state_fixed
    return {"label": label, "cell": label[0], "rollout": int(label[1:]),
            "gates": {"integrity_passed": integrity, "treatment_state_fixed": state_fixed,
                      "valid": bool(integrity and (label[0] != "W" or state_fixed))}}


class DecideTest(unittest.TestCase):
    def setUp(self) -> None:
        self.module = load_module()

    def test_a_lower_target_in_every_valid_pair_is_a(self) -> None:
        decision = self.module.decide(
            [], [effect("W1", -0.12), effect("W2", -0.30), effect("W3", -0.08)],
            [gate("A1", True), gate("A2", True), gate("A3", True),
             gate("W1", True), gate("W2", True), gate("W3", True)],
        )
        self.assertEqual(decision["decision"], "A")
        self.assertEqual(decision["paired_repeats"], 3)

    def test_a_much_shorter_target_to_cube_distance_alone_is_a(self) -> None:
        decision = self.module.decide(
            [], [effect("W1", -0.01, -0.55), effect("W2", -0.02, -0.40)],
            [gate("A1", True), gate("A2", True), gate("W1", True), gate("W2", True)],
        )
        self.assertEqual(decision["decision"], "A")

    def test_a_fixed_state_with_no_change_of_the_pre_declared_size_is_b(self) -> None:
        decision = self.module.decide(
            [], [effect("W1", -0.01, -0.02), effect("W2", -0.02, -0.01), effect("W3", -0.005, -0.05)],
            [gate("A1", True, False), gate("A2", True, False), gate("A3", True, False),
             gate("W1", True), gate("W2", True), gate("W3", True)],
        )
        self.assertEqual(decision["decision"], "B")
        self.assertEqual(decision["paired_repeats"], 3)

    def test_a_detectable_but_sign_flipping_effect_is_c(self) -> None:
        decision = self.module.decide(
            [], [effect("W1", -0.30), effect("W2", 0.25)],
            [gate("A1", True, False), gate("A2", True, False), gate("W1", True), gate("W2", True)],
        )
        self.assertEqual(decision["decision"], "C")

    def test_pairs_that_disagree_in_direction_are_c_even_at_noise_level(self) -> None:
        decision = self.module.decide(
            [], [effect("W1", 0.03), effect("W2", -0.062), effect("W3", -0.02)],
            [gate("A1", True, False), gate("A2", True, False), gate("A3", True, False),
             gate("W1", True), gate("W2", True), gate("W3", True)],
        )
        self.assertEqual(decision["decision"], "C")
        self.assertIn("do not agree", decision["reason"])

    def test_an_invalid_treatment_never_counts_as_evidence(self) -> None:
        decision = self.module.decide(
            [], [effect("W1", -0.42), effect("W2", -0.44)],
            [gate("A1", True, False), gate("A2", True, False),
             gate("W1", False, False), gate("W2", False, False)],
        )
        self.assertEqual(decision["decision"], "D")
        self.assertEqual(decision["paired_repeats"], 0)
        self.assertIn("not causal evidence", decision["reason"])

    def test_an_integrity_failure_invalidates_a_run_even_if_the_state_is_fixed(self) -> None:
        decision = self.module.decide(
            [], [effect("W1", -0.30), effect("W2", -0.31)],
            [gate("A1", True, False), gate("A2", True, False),
             gate("W1", False, True), gate("W2", False, True)],
        )
        self.assertEqual(decision["decision"], "D")

    def test_an_invalid_baseline_pair_is_dropped_not_used(self) -> None:
        decision = self.module.decide(
            [], [effect("W1", -0.42), effect("W2", -0.44)],
            [gate("A1", False), gate("A2", True, False), gate("W1", True), gate("W2", True)],
        )
        self.assertEqual(decision["decision"], "A")
        self.assertEqual(decision["paired_repeats"], 1)
        self.assertEqual(decision["first_5s_target_palm_z_delta_m"], [-0.44])


class GateLimitTest(unittest.TestCase):
    def test_the_gate_limits_are_pre_declared_numbers(self) -> None:
        module = load_module()
        limits = module.GATE_LIMITS
        self.assertEqual(limits["start_arms_mahalanobis_p95"], 5.39)
        self.assertEqual(limits["cube_displacement_max_m"], 0.01)
        self.assertEqual(module.ACCEPTANCE["target_palm_z_drop_m"], 0.05)
        self.assertEqual(module.ACCEPTANCE["target_palm_cube_min_distance_reduction"], 0.30)
        self.assertFalse(module.ACCEPTANCE["measured_droop_alone_counts"])


class SceneClearanceTest(unittest.TestCase):
    def test_the_chosen_start_state_is_above_the_scene(self) -> None:
        """The selected demonstration start pose clears the worktop and the cubes
        in this profile's own base-frame terms."""
        module = load_module()
        selection = module.read_json(module.DEFAULT_OUT / "tables" / "segment_selection.json")
        chosen = [row for row in selection["candidates"]
                  if row["episode_index"] == selection["chosen_episode_index"]][0]
        self.assertGreater(chosen["palm_left_above_cube_top_m"], 0.0)
        self.assertGreater(chosen["palm_right_above_cube_top_m"], 0.0)
        self.assertTrue(chosen["arms_inside_p95"])
        self.assertTrue(chosen["validated_by_exp01"])

    def test_the_token_stream_the_bridge_consumes_is_valid(self) -> None:
        module = load_module()
        path = module.DEFAULT_OUT / "tokens.json"
        if not path.is_file():
            self.skipTest("the select stage has not run yet")
        stream = module.exp01_module().fsq_quantize(np.asarray(
            module.read_json(path)["tokens"], dtype=float))
        payload = module.read_json(path)
        self.assertEqual(len(payload["tokens"]), len(payload["left_hand_joints"]))
        self.assertEqual(len(payload["tokens"][0]), 64)
        self.assertEqual(len(payload["left_hand_joints"][0]), 7)
        self.assertTrue(np.allclose(np.asarray(payload["tokens"], dtype=float), stream, atol=1e-9))


if __name__ == "__main__":
    unittest.main()
