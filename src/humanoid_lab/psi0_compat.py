"""Runtime compatibility shims for the pinned Psi0 training line.

``psi/trainers/finetune.py`` builds the VLM with a hardcoded
``attn_implementation="flash_attention_2"`` and raises before any checkpoint can
load when that kernel package is absent, even though ``psi/models/psi0.py``
already drops to ``sdpa`` in that case.  The locked psi0 environment ships no
flash-attn wheel (and none matches its torch 2.7 / cp311), so the same fallback is
installed here for every consumer of the training path: the real ``train.py``
launch picks it up through ``scripts/psi0_shim/sitecustomize.py``, and the gate
verifier calls it directly.
"""

from __future__ import annotations

import atexit

_installed = False
_accumulation_guard_installed = False


def install_sdpa_fallback() -> bool:
    """Load Qwen3-VL with SDPA when flash-attn is unavailable.

    Returns True when this call owns the patch, False when flash-attn is present
    (nothing to do) or the patch was already installed.
    """
    global _installed
    if _installed:
        return False
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLForConditionalGeneration
    from transformers.utils import is_flash_attn_2_available

    _installed = True
    if is_flash_attn_2_available():
        return False
    original = Qwen3VLForConditionalGeneration.from_pretrained.__func__

    def from_pretrained(cls, *args, **kwargs):
        if kwargs.get("attn_implementation") == "flash_attention_2":
            kwargs["attn_implementation"] = "sdpa"
        return original(cls, *args, **kwargs)

    Qwen3VLForConditionalGeneration.from_pretrained = classmethod(from_pretrained)
    return True


def install_accumulation_guard() -> bool:
    """Verify the first optimizer step consumes the configured microbatches."""
    global _accumulation_guard_installed
    if _accumulation_guard_installed:
        return False

    from psi.trainers.finetune import FinetuneTrainer

    original = FinetuneTrainer.training_step
    microbatches = 0
    verified = False

    def training_step(self, batch):
        nonlocal microbatches, verified
        microbatches += 1
        result = original(self, batch)
        if result[0] and not verified:
            expected = int(self.cfg.train.gradient_accumulation_steps)
            if microbatches != expected:
                raise RuntimeError(
                    f"gradient accumulation mismatch: optimizer step used {microbatches} "
                    f"microbatches, configured {expected}"
                )
            print(f"[psi0] gradient accumulation verified: {microbatches}/{expected} microbatches", flush=True)
            verified = True
        if result[0]:
            microbatches = 0
        return result

    FinetuneTrainer.training_step = training_step
    _accumulation_guard_installed = True
    return True


def report_cuda_peak_at_exit() -> None:
    """Print the CUDA allocator peaks at interpreter exit (plan §15).

    ``train.py`` logs neither figure, and an in-process reading is the only way to
    get the torch peak rather than an nvidia-smi sample, so the training wrapper
    reports it here.  A no-op unless CUDA was actually used.
    """

    def _report() -> None:
        try:
            import torch

            if not torch.cuda.is_initialized():
                return
            print(
                "[psi0] CUDA peak allocated "
                f"{torch.cuda.max_memory_allocated() / 2**30:.2f} GiB / reserved "
                f"{torch.cuda.max_memory_reserved() / 2**30:.2f} GiB",
                flush=True,
            )
        except Exception:
            pass

    atexit.register(_report)


def install_sliced_state_delta_fallback() -> bool:
    """Let ``SonicRepackTransform`` address state columns by slice.

    The pinned ``SonicRepackTransform.__call__`` parses ``observation.state``
    slice keys (through ``parse_modality_key``) but its ``delta_timestamps``
    hands the raw key strings to LeRobot as column names, so a sliced state key
    raises ``KeyError: Column observation.state[0:36] not in the dataset`` before
    the transform ever runs.  The right-hand mapping of the v1.1 entrypoint uses
    exactly such keys; this shim collapses every state key to its base field
    (duplicates collapse harmlessly) while leaving non-sliced keys and all other
    delta entries untouched.  Behaviour for the historical one-key layout is
    bit-identical, so it is safe to install unconditionally.

    Returns True when this call owns the patch, False when it was already
    installed.
    """
    from psi.config.transform_psi0_sonic import SonicRepackTransform, parse_modality_key

    original = SonicRepackTransform.delta_timestamps
    if getattr(original, "_humanoid_lab_sliced_state_keys", False):
        return False

    def delta_timestamps(self, fps):
        delta = original(self, fps)
        for state_key in self.state_keys:
            base_key, _ = parse_modality_key(state_key)
            if base_key == state_key:
                continue
            delta.pop(state_key, None)
            delta[base_key] = [
                t / fps for t in range(-(self.num_past_frames + self.state_temporal_jitter),
                                       self.state_temporal_jitter + 1)
            ]
        return delta

    delta_timestamps._humanoid_lab_sliced_state_keys = True
    SonicRepackTransform.delta_timestamps = delta_timestamps
    return True
