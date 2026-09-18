"""The benchmark harness reads the paths' own log lines; these pin the formats."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _load_harness():
    spec = importlib.util.spec_from_file_location("benchmark_sim", ROOT / "scripts" / "benchmark-sim.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    # dataclasses resolve their own module through sys.modules while the class
    # body is executed, so the module has to be registered before exec_module.
    sys.modules["benchmark_sim"] = module
    spec.loader.exec_module(module)
    return module


bench = _load_harness()

SONIC_LINE = (
    "[isaac-g1] physics=212.4 Hz  render=26.8 FPS  RTF=1.06x  step=3.62 ms  "
    "render_call=8.39 ms  device=cpu"
)
SONIC_LINE_DETAIL = (
    "[isaac-g1] physics=212.4 Hz  render=26.8 FPS  RTF=1.06x  step=3.62 ms  "
    "render_call=8.39 ms  device=cpu  render_interval=16  "
    "[ms/tick: apply=0.31 physics=3.23 sensors=0.05 publish=0.00]"
)
INSTINCT_LINE = (
    "[perf] t=    3.5s sim |   16.8 loops/s |    16.8 env-steps/s | RTF  0.34x | cmd "
    "[+0.00, +0.00, +0.00] | pos ( -11.96, -35.87,+0.83) | vx -0.00 | "
    "ms/loop env_step=57.04 policy=0.52 depth_window=0.67"
)


class SonicLineTests(unittest.TestCase):
    def test_a_sonic_sample_line_parses(self) -> None:
        match = bench.SONIC_RE.search(SONIC_LINE)
        self.assertIsNotNone(match, "the per-second line format changed")
        values = {key: float(value) for key, value in match.groupdict().items() if key != "device"}
        self.assertAlmostEqual(values["physics_hz"], 212.4)
        self.assertAlmostEqual(values["render_fps"], 26.8)
        self.assertAlmostEqual(values["rtf"], 1.06)
        self.assertAlmostEqual(values["step_ms"], 3.62)
        self.assertAlmostEqual(values["render_call_ms"], 8.39)

    def test_a_detailed_line_still_parses_as_a_sample(self) -> None:
        match = bench.SONIC_RE.search(SONIC_LINE_DETAIL)
        self.assertIsNotNone(match)
        self.assertAlmostEqual(float(match.group("step_ms")), 3.62)
        interval = bench.SONIC_INTERVAL_RE.search(SONIC_LINE_DETAIL)
        self.assertIsNotNone(interval, "--render-interval must remain readable from the line")
        self.assertEqual(interval.group("render_interval"), "16")

    def test_an_instinct_line_does_not_match_the_sonic_format(self) -> None:
        self.assertIsNone(bench.SONIC_RE.search(INSTINCT_LINE))


class InstinctLineTests(unittest.TestCase):
    def test_an_instinct_sample_line_parses(self) -> None:
        match = bench.INSTINCT_RE.search(INSTINCT_LINE)
        self.assertIsNotNone(match, "the playback's [perf] line format changed")
        self.assertAlmostEqual(float(match.group("sim_s")), 3.5)
        self.assertAlmostEqual(float(match.group("loops_per_s")), 16.8)
        self.assertAlmostEqual(float(match.group("rtf")), 0.34)

    def test_a_final_total_line_parses(self) -> None:
        line = (
            "[perf] TOTAL 123 policy steps | 61.0s wall | 2.0 loops/s | 0.40x RTF | 2.500s sim"
        )
        match = bench.INSTINCT_TOTAL_RE.search(line)
        self.assertIsNotNone(match)
        self.assertEqual(int(match.group("steps")), 123)
        self.assertAlmostEqual(float(match.group("rtf")), 0.40)

    def test_playback_prints_precise_sim_time_for_windowed_rtf(self) -> None:
        source = (ROOT / "scripts/play-instinct-parkour.py").read_text()
        self.assertIn("sim_time:8.3f", source)
        self.assertIn("sim_time:.3f}s sim", source)


class CaseDefinitionTests(unittest.TestCase):
    def test_each_path_gets_its_own_flag_spelling(self) -> None:
        """The runner uses hyphens, the playback underscores; mixing them exits 2."""
        source = (ROOT / "src/humanoid_lab/simulators/isaac/cli.py").read_text()
        playback = (ROOT / "scripts/play-instinct-parkour.py").read_text()
        self.assertIn('"--perf-detail"', source)
        self.assertNotIn('"--perf_detail"', source)
        self.assertIn('"--perf_detail"', playback)

    def test_the_rough_case_uses_the_rough_launcher(self) -> None:
        cases = {case.name: case for case in bench.default_cases()}
        self.assertIn("isaac-g1-sonic-rough", cases["sonic-rough-paced"].argv)
        self.assertIn("isaac-g1-sonic", cases["sonic-flat-paced"].argv)
        self.assertNotIn("isaac-g1-sonic-rough", cases["sonic-flat-paced"].argv)

    def test_every_case_selects_both_ui_and_pacing_explicitly(self) -> None:
        for case in bench.default_cases():
            self.assertIsNotNone(case.kind)
            self.assertTrue(case.notes)
            self.assertGreater(case.timeout_s, case.warmup_s + case.window_s)


class AggregationTests(unittest.TestCase):
    def _result(self, values: list[float], kind: str = "sonic") -> object:
        case = bench.Case(
            name="unit", kind=kind, argv=[], warmup_s=2.0, window_s=10.0, timeout_s=60.0
        )
        samples = [
            bench.Sample(float(index), {"rtf": value, "physics_hz": 200.0, "step_ms": 1.0})
            for index, value in enumerate(values)
        ]
        return bench.CaseResult(case=case, command="unit", log_path="unit.log", samples=samples)

    def test_warmup_samples_are_excluded(self) -> None:
        result = self._result([9.0, 9.0, 1.0, 1.0])
        steady = bench.steady_samples(result)
        self.assertEqual([sample.values["rtf"] for sample in steady], [1.0, 1.0])

    def test_metrics_are_the_steady_window(self) -> None:
        # Sample 0 is time 0, sample 1 is time 1: warm-up ends at 2.0 s.
        aggregates = bench.aggregate(self._result([9.0, 9.0, 1.0, 2.0]))
        self.assertEqual(aggregates["metrics"]["rtf"]["median"], 1.5)
        self.assertEqual(aggregates["steady_sample_count"], 2)
        self.assertEqual(aggregates["sample_count"], 4)

    def test_a_sonic_summary_is_reduced_to_the_compared_fields(self) -> None:
        # Walls 0, 1, 2 with a 2 s warm-up: one steady sample, which is enough
        # for the summary fields that do not come from the sample stream.
        result = self._result([1.0, 1.0, 1.0])
        result.summary = {"result": "COMPLETED", "real_time_factor": 1.05, "pacing_overruns": 3, "junk": 1}
        aggregates = bench.aggregate(result)
        self.assertEqual(aggregates["run_summary"]["pacing_overruns"], 3)
        self.assertNotIn("junk", aggregates["run_summary"])
        self.assertEqual(aggregates["metrics"]["deadline_misses"], 3)

    def test_the_window_is_bounded_at_both_ends(self) -> None:
        """Samples past warmup+window are not part of the measured window.

        The Instinct playback bounds *simulated* seconds, so at RTF below 1 it
        keeps producing samples long after the wall window the harness asked
        for; those must not enter the aggregate.
        """
        result = self._result([9.0, 9.0, 1.0, 1.0, 1.0, 7.0, 7.0])
        steady = [sample.values["rtf"] for sample in bench.steady_samples(result)]
        # warmup 2 s ends at wall 2; window 10 s would end at wall 12, past the
        # samples that exist here, so everything from wall 2 on is kept.
        self.assertEqual(steady, [1.0, 1.0, 1.0, 7.0, 7.0])
        # A 2 s window ends at wall 4, which drops the trailing 7.0 samples.
        tight = bench.Case(name="unit", kind="sonic", argv=[], warmup_s=2.0, window_s=2.0, timeout_s=60.0)
        result.case = tight
        self.assertEqual(
            [sample.values["rtf"] for sample in bench.steady_samples(result)], [1.0, 1.0, 1.0]
        )

    def test_the_recorded_command_is_the_executed_one(self) -> None:
        """The artifact must describe the argv that ran, including harness flags."""
        case = bench.Case(name="unit", kind="sonic", argv=["./dev.sh", "isaac-g1-sonic"], warmup_s=1.0,
                          window_s=1.0, timeout_s=60.0)
        executed = [*case.argv, "--duration", "2", "--metrics-output", "x.json"]
        result = bench.CaseResult(
            case=case, command=" ".join(executed), log_path="unit.log", executed_argv=executed
        )
        aggregates = bench.aggregate(result)
        self.assertEqual(aggregates["argv"], executed)
        self.assertIn("--duration", aggregates["command"])

    def test_windowed_rtf_comes_from_consecutive_instinct_samples(self) -> None:
        case = bench.Case(name="unit", kind="instinct", argv=[], warmup_s=0.0, window_s=10.0, timeout_s=60.0)
        samples = [
            bench.Sample(0.0, {"sim_s": 0.0, "loops_per_s": 0.0, "env_steps_per_s": 0.0, "rtf": 0.0}),
            bench.Sample(1.0, {"sim_s": 0.5, "loops_per_s": 0.0, "env_steps_per_s": 0.0, "rtf": 0.0}),
            bench.Sample(2.0, {"sim_s": 0.9, "loops_per_s": 0.0, "env_steps_per_s": 0.0, "rtf": 0.0}),
        ]
        result = bench.CaseResult(case=case, command="unit", log_path="unit.log", samples=samples)
        aggregates = bench.aggregate(result)
        # Deltas are 0.5 and 0.4 simulated seconds per wall second; the reported
        # statistic is the median of those.
        self.assertEqual(aggregates["metrics"]["windowed_rtf"]["median"], 0.45)


class SummaryRecoveryTests(unittest.TestCase):
    def test_the_printed_summary_is_recovered_from_a_log_with_trailers(self) -> None:
        summary = {
            "result": "COMPLETED",
            "real_time_factor": 1.03,
            "physics_ticks": 2719,
            "step_ms_by_phase": {"physics": 3.2},
        }
        text = (
            "[isaac-g1] physics=1.0 Hz\n"
            + json.dumps(summary, indent=2)
            + '\n{"event": "isaac_g1_app_close", "closed": false}\n'
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run.log"
            path.write_text(text)
            recovered = bench._summary_from_log(path)
        self.assertIsNotNone(recovered)
        self.assertEqual(recovered["real_time_factor"], 1.03)
        self.assertEqual(recovered["step_ms_by_phase"], {"physics": 3.2})

    def test_a_log_without_a_summary_recovers_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run.log"
            path.write_text("[isaac-g1] physics=1.0 Hz\n")
            self.assertIsNone(bench._summary_from_log(path))


if __name__ == "__main__":
    unittest.main()
