#!/usr/bin/env python3
"""Unit tests for scripts/groot-finetune-launcher.py.

The shim exists because the pinned FinetuneConfig exposes no optimizer flag and
the pinned launcher hard-codes ``config.training.optim = "adamw_torch"``.  It
executes the pinned launcher source verbatim and intercepts only the final
``run(config)`` call, which is exactly what these tests pin: the allowlist, the
fail-closed source guard, the pass-through default, and that the launcher's own
``from gr00t.experiment.experiment import run`` binds the patched function.

No gr00t, torch or GPU is needed: the fake package below is injected into
sys.modules, so this stays an offline unit test.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import os
import pathlib
import sys
import tempfile
import types
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SHIM = ROOT / "scripts" / "groot-finetune-launcher.py"

# Mirrors the pinned launcher's import line and the shape of the config it hands
# to run(); the real launcher builds this config from FinetuneConfig.
FAKE_LAUNCHER = """\
from gr00t.experiment.experiment import run

import sys

class Training:
    optim = "adamw_torch"

class Config:
    training = Training()

print("ARGV: " + " ".join(sys.argv[1:]))
run(Config())
print("launcher finished")
"""

# A launcher that no longer binds run through the expected import: the override
# would be dropped silently, so the shim must refuse it.
FAKE_LAUNCHER_NO_IMPORT = """\
import sys

print("ARGV: " + " ".join(sys.argv[1:]))
"""


def load_shim():
    spec = importlib.util.spec_from_file_location("groot_finetune_launcher", SHIM)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class LauncherShimTests(unittest.TestCase):
    def setUp(self) -> None:
        self.shim = load_shim()
        self._env = {
            key: os.environ.get(key) for key in ("GROOT_OPTIM", "GROOT_LAUNCHER")
        }
        self._modules = dict(sys.modules)
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.launcher = pathlib.Path(self._tmp.name) / "launch_finetune.py"
        self.launcher.write_text(FAKE_LAUNCHER, encoding="utf-8")
        os.environ["GROOT_LAUNCHER"] = str(self.launcher)
        os.environ.pop("GROOT_OPTIM", None)

    def tearDown(self) -> None:
        for key, value in self._env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        sys.modules.clear()
        sys.modules.update(self._modules)

    def install_fake_gr00t(self) -> list[str]:
        calls: list[str] = []
        experiment = types.ModuleType("gr00t.experiment.experiment")

        def original_run(config):
            calls.append(config.training.optim)
            return "original"

        experiment.run = original_run
        for name in ("gr00t", "gr00t.experiment"):
            sys.modules[name] = types.ModuleType(name)
        sys.modules["gr00t.experiment.experiment"] = experiment
        return calls

    def call_main(self, argv: list[str]) -> tuple[int, str, str]:
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            try:
                rc = self.shim.main(argv)
            except SystemExit as exc:
                rc = exc.code
        return rc, out.getvalue(), err.getvalue()

    def test_override_reaches_the_launcher_config(self) -> None:
        calls = self.install_fake_gr00t()
        os.environ["GROOT_OPTIM"] = "adafactor"
        rc, out, err = self.call_main(["shim", "--base-model-path", "/model"])
        self.assertEqual(rc, 0, err)
        self.assertEqual(calls, ["adafactor"])
        self.assertIn(
            "optimizer override: training.optim 'adamw_torch' -> 'adafactor'", out
        )
        self.assertIn("launcher finished", out)
        self.assertIn("ARGV: --base-model-path /model", out)

    def test_without_optimizer_the_shim_is_a_pass_through(self) -> None:
        calls = self.install_fake_gr00t()
        rc, out, err = self.call_main(["shim", "--max-steps", "2"])
        self.assertEqual(rc, 0, err)
        self.assertEqual(calls, ["adamw_torch"])
        self.assertNotIn("optimizer override", out)

    def test_unknown_optimizer_is_refused(self) -> None:
        self.install_fake_gr00t()
        os.environ["GROOT_OPTIM"] = "paged_adamw_8bit"
        rc, _out, err = self.call_main(["shim"])
        self.assertEqual(rc, 2)
        self.assertIn("not one of", err)

    def test_missing_launcher_is_refused(self) -> None:
        os.environ["GROOT_LAUNCHER"] = str(pathlib.Path(self._tmp.name) / "absent.py")
        os.environ["GROOT_OPTIM"] = "adafactor"
        rc, _out, err = self.call_main(["shim"])
        self.assertEqual(rc, 2)
        self.assertIn("pinned launcher is missing", err)

    def test_launcher_without_the_run_import_is_refused(self) -> None:
        calls = self.install_fake_gr00t()
        self.launcher.write_text(FAKE_LAUNCHER_NO_IMPORT, encoding="utf-8")
        os.environ["GROOT_OPTIM"] = "adafactor"
        rc, _out, err = self.call_main(["shim"])
        self.assertEqual(rc, 2)
        self.assertIn("silently dropped", err)
        self.assertEqual(calls, [])


if __name__ == "__main__":
    unittest.main()
