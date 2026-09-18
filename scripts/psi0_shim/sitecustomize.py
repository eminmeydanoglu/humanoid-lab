"""Interpreter-startup shim for the psi0 training wrapper.

Python imports ``sitecustomize`` automatically when its directory is on
``PYTHONPATH``; ``scripts/psi0-unitree-dex3-sonic-v1.sh`` prepends this one, so
Psi0's own entry points (``scripts/train.py`` and the gate verifier) get the SDPA
fallback and CUDA-peak reporting without touching the pinned ``third_party/Psi0``
checkout.  Every failure is swallowed: a startup shim must never be able to break
an unrelated interpreter.
"""

try:
    from humanoid_lab.psi0_compat import (
        install_accumulation_guard,
        install_sdpa_fallback,
        report_cuda_peak_at_exit,
    )

    install_sdpa_fallback()
    install_accumulation_guard()
    report_cuda_peak_at_exit()
except Exception:
    pass
