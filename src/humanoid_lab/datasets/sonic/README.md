# SONIC Dataset Pipeline

This package converts source robot datasets into the canonical whole-body format
expected by the pinned SONIC v1.1 motion encoder.

The production design has one explicit boundary:

```text
raw dataset
  -> dataset-specific adapter and joint mapping
  -> CanonicalEpisodeBuild
  -> CanonicalEpisode at 50 Hz
  -> production.build_encoder_input()
  -> SONIC observation [frames, 1751]
  -> production.encode_prepared_episode()
  -> motion_token [frames, 64]
  -> training_valid_mask (unclamped future windows only)
  -> optional Dex3 hands [frames, 7 + 7]
  -> action [frames, 78]
```

Dataset-specific behavior stops at `CanonicalEpisodeBuild`. Every dataset uses
the same observation builder, ONNX encoder, artifact writer, and validation after
that point.

## Package layout

| Path | Responsibility |
| --- | --- |
| `adapters/unitree_dex3.py` | Maps Unitree Dex3 arm/hand actions and completes the missing body with a static standing pose. |
| `adapters/nvidia_apple_to_plate.py` | Maps AppleToPlate body blocks and preserves unresolved normalized hand commands outside the canonical action. |
| `adapters/nvidia_fruits.py` | Maps named NVIDIA Fruits body and hand channels into canonical order. |
| `pilot.py` | Selects the source adapter, writes canonical references, runs quality checks, and records provenance. |
| `production.py` | Sole production path from a canonical 50 Hz episode to encoder input, motion tokens, and optional 78D actions. |
| `schema.py` | Canonical episode and final action contracts. |
| `encoder_observation.py` | Implements the pinned 1751D SONIC encoder observation layout. Diagnostic orientation variants remain available for tests. |
| `encoder_runner.py` | Validates and executes the pinned batch-1 ONNX encoder. |
| `reference.py` | Standing-pose definitions and reference-composition utilities. |
| `timeline.py` | Timestamp resampling and velocity derivation. |
| `joints.py` | Canonical body and Dex3 joint-order utilities. |
| `quality.py` | Joint limits, source-echo checks, derivative checks, and conversion thresholds. |
| `state_reference.py` | Source-rate state/action references used only for visual review. |
| `protocol_v4.py` | Packs persisted motion tokens for SONIC latent replay. |
| `tracking.py` | Simulation tracking and stability metrics. |

## Canonical episode contract

Every adapter must return a `CanonicalEpisodeBuild`. Its `episode` contains:

- `timestamps`: uniform 50 Hz timeline.
- `joint_pos`: 29 body joints in `SONIC_REFERENCE_JOINT_ORDER`.
- `joint_vel`: 29 velocities derived consistently at 50 Hz.
- `body_pos`: root position.
- `body_quat_wxyz`: root quaternion in `wxyz` order.
- `left_hand_joints`: seven canonical Dex3 motor positions.
- `right_hand_joints`: seven canonical Dex3 motor positions.

The common production boundary rejects a non-uniform or non-50 Hz episode.
Resampling and source semantics belong in the dataset adapter, not in the
encoder.

## Single production encoder path

`production.py` owns the only supported conversion after source mapping.

### `build_encoder_input(episode)`

This function:

1. Validates the canonical episode and its 50 Hz timeline.
2. Enforces the production orientation policy:
   `reference_root_current_frame`.
3. Builds the official `[frames, 1751]` observation.
4. Returns the observation, timestamps, and future-window clamp fractions.

Other orientation policies exist for diagnostics and tests only. Production
commands do not select between multiple policies.

### `encode_prepared_episode(pilot_dir, model_dir=...)`

This function:

1. Loads the prepared canonical reference and encoder observation.
2. Verifies `observation_config.yaml` describes the 1751D G1 layout.
3. Verifies the pinned `model_encoder.onnx` contract and SHA-256 checksum.
4. Runs batch-1 inference frame by frame: `[1, 1751] -> [1, 64]`.
5. Re-encodes probe frames and requires bitwise repeatability.
6. Writes `action_tokens.npz` and `encoder_manifest.json`.
7. Adds the two seven-joint Dex3 targets only when the adapter has verified the
   hand schema.

The supervised VLA contract is `observation_t -> z_t`, where `z_t` encodes the
reference window beginning at `t`. The final 45 rows are retained so a SONIC
replay can execute the complete episode, but their future windows contain
upstream last-frame clamping and `training_valid_mask` excludes them from VLA
losses.

The pinned encoder checksum is exported as `PINNED_ENCODER_SHA256`.

No other production module writes `action_tokens.npz` or
`encoder_manifest.json`.

## Dataset behavior

### Unitree G1 Dex3

The source collection records arms and Dex3 hands but does not contain a complete
body trajectory.

The adapter therefore:

1. Maps arms and hands by their source names.
2. Starts from one validated static standing frame.
3. Replaces the standing arms and hands with the recorded values.
4. Keeps legs, waist, root position, and root orientation static.
5. Resamples the completed trajectory to 50 Hz and derives velocities again.

This policy is intended for stationary manipulation. It must not be interpreted
as a recorded walking or pelvis trajectory.

The action hand schema is verified, so the output contains both:

- 64D SONIC motion tokens.
- 78D actions: `64D token + 7D left hand + 7D right hand`.

## Unitree production corpus

The lean production path is deliberately separate from pilot/replay evidence:

```bash
scripts/convert-all-unitree-sonic.sh
```

It is blocked until `unitree-sonic-v1.1-78d/qc/unitree-production-qc.json`
records a five-episode `PASS`. It writes only one `action.npz` and one
`manifest.json` per episode. No videos, CSV/parquet mirrors, plots, HTML,
Isaac rollout, or SONIC physics replay are part of bulk conversion.

`action[:, :64]` is the SONIC v1.1 body latent, `action[:, 64:71]` the verified
Unitree left Dex3 action, and `action[:, 71:78]` the verified right action.
`SonicTrainingEpisode.load()` refuses legacy files without
`training_valid_mask`. Frame loss uses `valid_actions()`. Chunked training uses
`valid_action_chunks(horizon)`, which admits an anchor only when every target in
the chunk is valid; no clamped-tail target can enter indirectly.

The sweep honours the `unitree.bulk_excluded_episodes` map in
`configs/datasets/sonic/pilots.json`: those episodes are dropped from the
selection and recorded per collection as `excluded_episodes`. A collection that
still fails no longer aborts the sweep — every collection is attempted, the
aggregate summary is written, and the run exits non-zero at the end.

The converter is sequential by collection to keep ONNX/GPU memory deterministic.
Within each collection it reuses one encoder session, caches LeRobot episode
metadata and source-file checksums, atomically replaces incomplete episode
directories, and skips only outputs whose source identity, model/config hashes,
schema and output checksum still match.

### NVIDIA AppleToPlate

Body motion comes from same-row `action_t`. The normalized hand command has no
verified conversion to Dex3 radians. Measured `observation.state_t` hands are
kept for diagnostics, but must not be mixed into a 78D action label: equal
timestamps do not make measured state and desired action the same semantic
target. AppleToPlate bulk conversion therefore remains excluded until the hand
command mapping and action-to-state latency are established; its safe output is
the 64D body latent only.

The adapter maps the 29 body action channels and resamples the 30 Hz source to
50 Hz before it reaches `production.py`.

The action hand blocks are normalized open/close commands, not proven Dex3
actuator radians. Consequently:

- The 64D body motion token is available.
- The final 78D action remains blocked.
- Raw normalized commands are retained in `raw_hand_command.npz`.
- Recorded hand radians from `observation.state` are used only for source-state
  visualization and are not substituted into the action silently.

### NVIDIA Fruits / `g1-pick-apple`

The source provides named whole-body and hand channels. The adapter maps them by
name and resamples the 20 Hz trajectory to 50 Hz.

This source is currently marked `excluded` in
`configs/datasets/sonic/pilots.json` following human visual review. It remains
archived but bulk conversion refuses to process it. Changing a
`human_review.json` file does not override this policy.

## Usage

Commands run through `dev.sh`, which selects the environment containing the
required dependencies.

### Build a complete pilot

```bash
./dev.sh sonic-pilot --pilot unitree
```

This one command performs:

1. Dataset loading and joint mapping.
2. Dex3 standing completion when applicable.
3. Canonical 50 Hz reference creation.
4. Quality and provenance checks.
5. 1751D observation construction.
6. Pinned ONNX encoding.
7. 64D token and optional 78D action writing.

Other configured sources use the same command:

```bash
./dev.sh sonic-pilot --pilot apple \
  --assume-unitree-mujoco-body-order

./dev.sh sonic-pilot --pilot fruits
```

Use `--prepare-only` to stop after canonical reference, quality checks, and
encoder-observation creation:

```bash
./dev.sh sonic-pilot --pilot unitree --prepare-only
```

### Re-encode an existing prepared episode

```bash
./dev.sh sonic-encode <pilot_dir>
```

This is a maintenance entry point, not a second encoder implementation. It calls
`production.encode_prepared_episode()`.

### Replay the canonical reference exactly

```bash
./dev.sh isaac-g1-kinematic dex3 --headless \
  --kinematic-reference <pilot_dir>/reference.npz \
  --kinematic-label completed_reference_50hz \
  --record-video <pilot_dir>/completed_reference_kinematic.mp4 \
  --tracking-output <pilot_dir>/completed_reference_kinematic_tracking.parquet \
  --metrics-output <pilot_dir>/completed_reference_kinematic_metrics.json
```

This bypasses SONIC, PD control, and physics tracking. It writes the canonical
body, hands, root position, and root quaternion directly into simulation. Use it
to inspect what the adapter actually produced.

### Replay persisted latent tokens through SONIC

```bash
scripts/run-sonic-pilot-sim.sh <pilot_dir> --duration 60
```

This publishes `action_tokens.npz` at 50 Hz of Isaac simulation time through
SONIC Protocol v4 and records the closed-loop robot motion. The replay writes
the publisher timeline and a 50 Hz trace of SONIC joint targets and measured
robot joints. `sonic_fidelity.json` compares A (canonical reference), B (SONIC
joint-position command), and C (robot response). Only A-to-C execution fidelity
and complete delivery determine PASS/FAIL. B is a controller diagnostic, not an
encoder reconstruction. Pelvis-local wrist position/orientation is primary;
world-frame wrist motion is diagnostic when A has a synthetic root. The trace also retains B's velocity target,
gains, feed-forward torque, and applied actuator torque for diagnosis.
Missing publisher, SONIC receiver, or robot-trace frames fail the completeness
gate; no review clip is made from a truncated replay. The fixed minimum wall-time
budget includes model startup.

### Build and serve the review page

```bash
./dev.sh sonic-review
./dev.sh sonic-review-serve --port 8767
```

The review separates:

- Source camera footage.
- Frame-exact canonical reference replay.
- Offline latent through the SONIC controller.
- Controller-only debugging runs.

### Bulk conversion

```bash
./dev.sh sonic-convert \
  --pilot unitree \
  --episodes START:STOP
```

Bulk conversion uses the same adapter and `production.py` path as a pilot. It
runs only when:

1. The source is marked `eligible` in `configs/datasets/sonic/pilots.json`.
2. The latest pilot review status is `accepted` or `accepted_with_notes`.

There is no command-line override for either gate.

## Main output artifacts

A completed pilot directory contains:

| Artifact | Meaning |
| --- | --- |
| `reference.npz` | Canonical 50 Hz body/root/hand episode. |
| `reference.parquet` | Inspectable tabular form of the canonical reference. |
| `encoder_observation.npz` | Common `[frames, 1751]` encoder input. |
| `action_tokens.npz` | 64D tokens, frame timestamps, `training_valid_mask`, and, when action semantics are verified, 7+7 hand targets and 78D actions. |
| `run_manifest.json` | Source mapping, assumptions, quality results, and provenance. |
| `encoder_manifest.json` | Model checksum, ONNX contract, repeatability, dimensions, and token statistics. |
| `human_review.json` | Explicit human review gate used by bulk conversion. |
| `source.mp4` | Source camera clip when requested. |
| `reference_state.npz` | Measured source state for visualization, when available. |
| `reference_action_source.npz` | Source-rate action visualization, when available. |
| `raw_hand_command.npz` | Preserved unresolved normalized hand commands, when applicable. |

## Adding a dataset

To add another source:

1. Create an adapter under `adapters/`.
2. Map channels by verified names or an explicitly documented schema.
3. Convert the source timeline to a uniform 50 Hz timeline.
4. Supply all canonical body, root, velocity, and hand arrays.
5. Record every synthetic field and assumption in provenance.
6. Return `CanonicalEpisodeBuild`.
7. Register the adapter in `pilot.map_source_episode()` and add pilot config.
8. Add source-echo, mapping, and fail-closed schema tests.

Do not add dataset-specific branches to `production.py`. Do not write a second
observation builder or token writer. Unresolved channel semantics must remain
blocked rather than being guessed.

## Tests

The focused production tests are:

```bash
./dev.sh sonic-tests
```

Relevant files include:

- `tests/test_sonic_production.py`
- `tests/test_sonic_encoder_observation.py`
- `tests/test_sonic_encoder_runner.py`
- `tests/test_sonic_pilot_loaders.py`
- `tests/test_sonic_protocol_v4.py`
- `tests/test_sonic_state_reference.py`
