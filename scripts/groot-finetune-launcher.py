#!/usr/bin/env python3
"""Run the pinned GR00T fine-tune launcher with a training-optimizer override.

Why this shim exists
--------------------
At the pinned commit 1a1837f20538b7d7e21f977a11a5aee14f99803c:

  * ``gr00t/configs/finetune_config.py`` (FinetuneConfig) exposes no optimizer
    flag, so tyro cannot select one from the command line;
  * ``gr00t/experiment/launch_finetune.py`` hard-codes
    ``config.training.optim = "adamw_torch"`` before calling
    ``gr00t.experiment.experiment.run``.

On the 32607 MiB RTX 5090 the 32 GB UNITREE_G1_SONIC profile (projector frozen,
diffusion head trainable: 1,293,159,424 parameters, all fp32 because
``from_pretrained`` is called without a dtype and transformers then keeps
``torch.get_default_dtype()``) needs, on top of 5,173 MiB of weights and
5,173 MiB of gradients:

  * AdamW: exp_avg + exp_avg_sq = 9,866 MiB, which OOMed inside the first
    ``optimizer.step`` next to a 6,443 MiB neighbour;
  * Adafactor: factored row/col second moment and no first moment
    (``beta1=None``) = 7 MiB for the same parameters.

This shim executes the pinned launcher source verbatim and intercepts only the
final ``run(config)`` call, setting ``config.training.optim`` from the
``GROOT_OPTIM`` environment variable.  Nothing else about the launch changes:
the CLI is still the pinned tyro schema and every other config field still comes
from the pinned launcher's own mapping.

The override is fail-closed.  If the pinned launcher stops importing ``run``
from ``gr00t.experiment.experiment``, the patch would be silently dropped and
the run would quietly go back to AdamW, so the shim refuses to start instead.

``GROOT_OPTIM`` must be one of ALLOWED_OPTIMIZERS; unset means "no override"
(identical to running the pinned launcher directly).  The list is deliberately
small: the groot-n17 venv has no bitsandbytes, torchao, deepspeed, schedulefree,
lomo or galore_torch, so the 8-bit, paged, low-rank and ZeRO names in
``transformers.training_args.OptimizerNames`` are not selectable here.  Keep it
in sync with the OPTIM validation in scripts/groot-unitree-dex3-sonic.sh.

Exit codes: 0 = launcher finished, 2 = this shim refused to run, otherwise the
launcher's own exit code.
"""

from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path
from typing import NoReturn

DEFAULT_LAUNCHER = Path("/opt/src/isaac-groot/gr00t/experiment/launch_finetune.py")
# Optimizers transformers 4.57.3 builds in-process, i.e. without the absent
# bitsandbytes/torchao/schedulefree/lomo/galore backends.
ALLOWED_OPTIMIZERS = ("adafactor", "adamw_torch", "adamw_torch_fused")
# The pinned launcher must bind `run` through this exact import for the patch
# below to take effect.
RUN_IMPORT = "from gr00t.experiment.experiment import run"


def fail(message: str) -> NoReturn:
    print(f"FAIL: {message}", file=sys.stderr)
    raise SystemExit(2)


def make_run_override(original_run, optimizer: str, launcher: Path):
    """Wrap ``gr00t.experiment.experiment.run`` with the optimizer override.

    ``config`` is the fully-built Config the pinned launcher is about to hand to
    the upstream ``run``, so this is the last point at which the hard-coded
    optimizer can be replaced without touching the pinned source.
    """

    def run(config):
        previous = config.training.optim
        config.training.optim = optimizer
        if config.training.optim != optimizer:  # pragma: no cover - defensive
            fail(f"could not set training.optim={optimizer!r}")
        print(
            f"optimizer override: training.optim {previous!r} -> {optimizer!r} "
            f"({launcher.name})",
            flush=True,
        )
        return original_run(config)

    return run


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv if argv is None else argv)
    optimizer = os.environ.get("GROOT_OPTIM", "").strip()
    launcher = Path(os.environ.get("GROOT_LAUNCHER") or DEFAULT_LAUNCHER)

    if not launcher.is_file():
        fail(f"pinned launcher is missing: {launcher}")
    if optimizer and optimizer not in ALLOWED_OPTIMIZERS:
        fail(
            f"GROOT_OPTIM={optimizer!r} is not one of {list(ALLOWED_OPTIMIZERS)}; "
            "the groot-n17 venv has no bitsandbytes/torchao/deepspeed backend for the "
            "8-bit, paged or ZeRO optimizer names"
        )

    if optimizer:
        source = launcher.read_text(encoding="utf-8")
        if RUN_IMPORT not in source:
            fail(
                f"{launcher} no longer imports the trainer entry point via "
                f"{RUN_IMPORT!r}; the optimizer override would be silently dropped, "
                "so this shim refuses to launch it"
            )
        from gr00t.experiment import experiment

        experiment.run = make_run_override(experiment.run, optimizer, launcher)

    # Keep argv honest for anything inside the launcher that inspects it.
    sys.argv = [str(launcher), *argv[1:]]
    runpy.run_path(str(launcher), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
