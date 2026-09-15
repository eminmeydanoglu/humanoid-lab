"""Offline runner for the pinned SONIC v1.1 encoder ONNX model.

The deployment runs the encoder through TensorRT; offline dataset production runs
the very same ``model_encoder.onnx`` through onnxruntime on the CPU.  This module
owns the contract that keeps the two interchangeable:

* the model file is checksummed, and an expected checksum fails closed on
  mismatch;
* the session input/output names, shapes and dtypes must match the pinned model
  (``obs_dict`` ``[1, 1751]`` float32 in, ``encoded_tokens`` ``[1, 64]`` float32
  out) or the run stops;
* inference is always batch 1, because that is the exported model's batch
  dimension — ``encode_frames`` loops instead of feeding a wider tensor;
* inputs and outputs are checked for NaN/Inf before they reach a manifest.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .encoder_observation import ENCODER_INPUT_DIM, ENCODER_TOKEN_DIM
from .provenance import sha256_file
from .schema import compose_action

ENCODER_INPUT_NAME = "obs_dict"
ENCODER_OUTPUT_NAME = "encoded_tokens"
ENCODER_INPUT_SHAPE = (1, ENCODER_INPUT_DIM)
ENCODER_OUTPUT_SHAPE = (1, ENCODER_TOKEN_DIM)
ENCODER_INPUT_TYPE = "tensor(float)"
ENCODER_OUTPUT_TYPE = "tensor(float)"
DEFAULT_PROVIDERS: tuple[str, ...] = ("CPUExecutionProvider",)

ONNXRUNTIME_HINT = (
    "onnxruntime is not importable in this interpreter; the offline encoder runs in the "
    "sonic-sim environment (onnxruntime is pinned there): use './dev.sh sonic-encode ...' "
    "or '/opt/venvs/sonic-sim/bin/python'"
)


class EncoderContractError(RuntimeError):
    """The model or the data handed to it violates the pinned encoder contract."""


def require_onnxruntime() -> Any:
    """Import onnxruntime or explain which environment provides it."""
    try:
        import onnxruntime  # noqa: PLC0415 - optional dependency, imported on demand
    except ImportError as error:  # pragma: no cover - exercised by the CLI path
        raise RuntimeError(ONNXRUNTIME_HINT) from error
    return onnxruntime


@dataclass(frozen=True)
class EncoderModelInfo:
    """Everything a manifest needs to describe the encoder that produced tokens."""

    path: str
    sha256: str
    input_name: str
    input_shape: tuple[int, ...]
    input_type: str
    output_name: str
    output_shape: tuple[int, ...]
    output_type: str
    providers: tuple[str, ...]
    onnxruntime_version: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "sha256": self.sha256,
            "input_name": self.input_name,
            "input_shape": list(self.input_shape),
            "input_type": self.input_type,
            "output_name": self.output_name,
            "output_shape": list(self.output_shape),
            "output_type": self.output_type,
            "providers": list(self.providers),
            "onnxruntime_version": self.onnxruntime_version,
        }


def _as_shape(value: Sequence[Any]) -> tuple[int, ...]:
    shape: list[int] = []
    for item in value:
        if not isinstance(item, int):
            raise EncoderContractError(f"encoder model has a non-static input/output dimension: {value}")
        shape.append(int(item))
    return tuple(shape)


class G1EncoderRunner:
    """Batch-1 encoder inference with contract checks on both sides."""

    def __init__(
        self,
        model_path: Path,
        *,
        providers: Sequence[str] = DEFAULT_PROVIDERS,
        expected_sha256: str | None = None,
    ) -> None:
        self.model_path = Path(model_path)
        if not self.model_path.is_file():
            raise FileNotFoundError(f"encoder model not found: {self.model_path}")
        digest = sha256_file(self.model_path)
        if expected_sha256 is not None and digest.lower() != expected_sha256.lower():
            raise EncoderContractError(
                f"encoder checksum mismatch for {self.model_path}: {digest} != {expected_sha256}"
            )
        onnxruntime = require_onnxruntime()
        self._session = onnxruntime.InferenceSession(str(self.model_path), providers=list(providers))
        inputs = self._session.get_inputs()
        outputs = self._session.get_outputs()
        if len(inputs) != 1 or len(outputs) != 1:
            raise EncoderContractError(
                f"encoder must have exactly one input and one output, got {len(inputs)}/{len(outputs)}"
            )
        self._input = inputs[0]
        self._output = outputs[0]
        for label, actual, expected in (
            ("input name", self._input.name, ENCODER_INPUT_NAME),
            ("output name", self._output.name, ENCODER_OUTPUT_NAME),
            ("input type", self._input.type, ENCODER_INPUT_TYPE),
            ("output type", self._output.type, ENCODER_OUTPUT_TYPE),
            ("input shape", _as_shape(self._input.shape), ENCODER_INPUT_SHAPE),
            ("output shape", _as_shape(self._output.shape), ENCODER_OUTPUT_SHAPE),
        ):
            if actual != expected:
                raise EncoderContractError(f"encoder {label} is {actual!r}, expected {expected!r}")
        self.info = EncoderModelInfo(
            path=str(self.model_path),
            sha256=digest,
            input_name=self._input.name,
            input_shape=_as_shape(self._input.shape),
            input_type=self._input.type,
            output_name=self._output.name,
            output_shape=_as_shape(self._output.shape),
            output_type=self._output.type,
            providers=tuple(self._session.get_providers()),
            onnxruntime_version=str(onnxruntime.__version__),
        )

    def encode(self, observation: np.ndarray) -> np.ndarray:
        """Encode one 1751D observation into a finite ``[64]`` float32 token."""
        batch = self._prepare(observation)
        raw = self._session.run([self._output.name], {self._input.name: batch})[0]
        tokens = np.asarray(raw, dtype=np.float32).reshape(-1)
        if tokens.shape != (ENCODER_TOKEN_DIM,):
            raise EncoderContractError(f"encoder returned shape {tokens.shape}, expected ({ENCODER_TOKEN_DIM},)")
        if not np.isfinite(tokens).all():
            raise EncoderContractError("encoder returned NaN or Inf")
        return tokens

    def encode_frames(self, observations: np.ndarray) -> np.ndarray:
        """Encode ``[frames, 1751]`` one frame at a time and stack the tokens."""
        frames = np.asarray(observations)
        if frames.ndim != 2 or frames.shape[1] != ENCODER_INPUT_DIM:
            raise EncoderContractError(
                f"encoder input must be [frames, {ENCODER_INPUT_DIM}], got {frames.shape}"
            )
        if frames.shape[0] == 0:
            raise EncoderContractError("cannot encode an empty frame batch")
        return np.stack([self.encode(row) for row in frames], axis=0)

    def _prepare(self, observation: np.ndarray) -> np.ndarray:
        value = np.asarray(observation, dtype=np.float32)
        if value.shape == (ENCODER_INPUT_DIM,):
            value = value[None, :]
        if value.shape != ENCODER_INPUT_SHAPE:
            raise EncoderContractError(f"encoder input must be {ENCODER_INPUT_SHAPE}, got {value.shape}")
        if not np.isfinite(value).all():
            raise EncoderContractError("encoder input contains NaN or Inf")
        return np.ascontiguousarray(value)


def compose_pilot_action(
    motion_token: np.ndarray,
    left_hand_joints: np.ndarray | None,
    right_hand_joints: np.ndarray | None,
) -> np.ndarray | None:
    """The 78D GR00T action, or ``None`` while a hand schema is unresolved."""
    if left_hand_joints is None or right_hand_joints is None:
        return None
    return compose_action(motion_token, left_hand_joints, right_hand_joints)
