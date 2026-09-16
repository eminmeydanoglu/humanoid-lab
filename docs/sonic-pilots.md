# SONIC v1.1 pilot workflow

Everything below works on **single episodes** until a human accepts the pilots.
The raw collection (`data/datasets/first_tur_ham`) is never written to: every
artifact lands in `data/datasets/first_tur_processed/sonic_v1_1/` under a new
UTC timestamp, and the review page lists each run from its own directory.

## Pipeline

```text
immutable raw episode
  → source adapter (name-based joint remap, synthetic fields declared)
  → canonical 50 Hz reference: joint_pos[29] + joint_vel[29] + root pose + Dex3 hands
  → exact 1751D G1 encoder observation (pinned observation_config.yaml layout)
  → model_encoder.onnx (batch 1)
  → motion_token[64]  (+ 7+7 Dex3 hand targets → 78D action)
  → frame-exact kinematic replay of the composed reference
  → free SONIC simulation (PD replay remains controller debugging only)
  → review page → human gate + source eligibility → (only then) bulk conversion
```

Conversion and QC run in the `isaac-sonic` environment (pyarrow); the encoder runs
in `sonic-sim` (onnxruntime). The `./dev.sh` wrappers below pick the right one.

There is one production boundary and one encoder path:

* `adapters/unitree_dex3.py`, `nvidia_fruits.py`, and `nvidia_apple_to_plate.py`
  only map source fields into `CanonicalEpisodeBuild`. Dex3 standing completion
  belongs to its adapter and does not leak into the encoder.
* `production.build_encoder_input()` is the only production conversion from a
  canonical 50 Hz whole-body episode to the 1751D SONIC observation.
* `production.encode_prepared_episode()` is the only writer of `action_tokens.npz`
  and `encoder_manifest.json`. Pilot, re-encode, and bulk commands all use it.

Diagnostic orientation variants remain test-only; production always uses
`reference_root_current_frame`.

## Commands

```bash
# 1. Build one complete pilot (dataset mapping → canonical 50 Hz → QC → latent)
./dev.sh sonic-pilot --pilot unitree
./dev.sh sonic-pilot --pilot fruits
./dev.sh sonic-pilot --pilot apple --assume-unitree-mujoco-body-order

# Optional maintenance command: re-encode an already prepared canonical episode.
# It calls the same production.encode_prepared_episode() implementation.
./dev.sh sonic-encode <pilot_dir>

# 2. Frame-exact composed-reference replay (no controller, PD, or tracking physics)
./dev.sh isaac-g1-kinematic dex3 --headless \
  --kinematic-reference <pilot_dir>/reference.npz \
  --kinematic-label completed_reference_50hz \
  --record-video <pilot_dir>/completed_reference_kinematic.mp4 \
  --tracking-output <pilot_dir>/completed_reference_kinematic_tracking.parquet \
  --metrics-output <pilot_dir>/completed_reference_kinematic_metrics.json

# 3. Offline latent → free SONIC: official deployment + Protocol v4 + Isaac G1
#    scripts/run-sonic-pilot-sim.sh first produces the root-inclusive frame-exact
#    replay, then optionally keeps the old fixed-base PD run as controller
#    debugging. The simulator comes up first, the deployment attaches, and only
#    then replay-sonic-latent.py publishes action_tokens.npz at 50 Hz of Isaac
#    simulation time. This path
#    tests the persisted offline token; the older reference→live-C++-encoder run
#    remains a separate comparison artifact.
scripts/run-sonic-pilot-sim.sh <pilot_dir>

# Primary replay acceptance: A=reference.npz, B=SONIC body/hand target,
# C=measured robot. The simulator clock and publisher timeline align samples.
# Incomplete publisher, SONIC receiver, or robot-trace coverage fails even if
# the observed prefix has low error.
# The left wrist is compared in pelvis-local and world-frame metres using the
# pinned G1 model and measured root pose.
# This command exits nonzero when the A/B/C gate fails.
docker exec humanoid-lab-dev bash -lc 'source /opt/humanoid-lab/entrypoint.sh && use-isaac-sonic && cd /workspace/humanoid-lab && PYTHONPATH=src python3 scripts/evaluate-sonic-latent-replay.py /data/outputs/sonic-abc-20260916/unitree_ep003_full_action --require-pass'

# Outputs: sonic_fidelity.json, sonic_left_wrist_world_abc.png, sonic_left_wrist_abc.png,
# sonic_left_arm_abc.png, sonic_left_hand_abc.png, and a verified complete
# sonic_latent_motion.mp4. A/B and A/C are primary fidelity values; B/C checks
# how well the simulated actuators followed the SONIC command. A visually
# wrong replay does not pass just because B/C is small.

# 4. Review page (videos side by side, per-joint plots, verdicts, assumptions)
./dev.sh sonic-review [--output <path>]

# 4b. Minimal three-video page for one episode: recorded head camera, the
#      whole-body trajectory written straight into the simulator, and the free
#      SONIC run driven by the produced latent.
python3 scripts/write-sonic-pilot-page.py --pilot-dir <pilot_dir> --output <pilot_dir>/review.html

# 4c. Serve it with byte ranges so the shared timeline can seek.  Plain
#     `python3 -m http.server` sends no Accept-Ranges/206, which leaves
#     HTMLMediaElement.seekable empty and freezes every video timeline.
./dev.sh sonic-review-serve [--port 8765]

# 5. Tests (unit suite + real ONNX encoder tests)
./dev.sh sonic-tests

# 6. Independent arithmetic check of a converted Unitree episode.  Re-derives the
#    canonical reference from the raw parquet WITHOUT importing the pipeline and
#    compares it against the stored artifacts, so a bug shared by the producer and
#    the checker cannot hide: joint mapping and cross-joint swaps, the 50 Hz
#    timeline, resampling, velocities (gain, lag, integrated displacement), the
#    pinned joint limits, hand order and motion, the full 1751D observation
#    rebuilt block by block, and the 78D action composition.
./dev.sh sonic-verify 3 74
```

## Bulk conversion

Bulk conversion is gated, resumable and per-dataset:

```bash
# One dataset, every episode it declares
./dev.sh sonic-convert --pilot unitree \
  --dataset unitree-g1-dex3/G1_Dex3_PickDoll_Dataset --all-episodes --no-video

# An explicit episode range (applied to every selected dataset)
./dev.sh sonic-convert --pilot unitree --pilot apple --episodes 0:50 --no-video
```

* `--all-episodes` uses each dataset's own declared `total_episodes`, so datasets
  of different sizes can be converted without hand-computing ranges.
* `--dataset` restricts a kind to one of its declared `bulk_datasets`; without it
  every declared dataset of the kind is converted.  The output directory follows
  the selected dataset (`<kind>_<dataset>_epNNN`), never the pilot's own dataset.
* Episodes that already have **both** `action_tokens.npz` and
  `encoder_manifest.json` are skipped, which makes a long batch resumable and
  prevents duplicate timestamped copies of the same episode.  `--force` re-runs
  them.
* `--no-video` skips cutting the source camera clip.  That clip is what makes a
  conversion ~15× slower, and it is only needed for pilots that get a review
  page, so bulk conversion normally runs without it.

## Encoder observation semantics

The 1751D input is the pinned layout of `observation_config.yaml`: `[0:4]` mode,
`[4:294]` 10 × joint positions at step 5, `[294:584]` 10 × joint velocities,
`[584:644]` 10 × heading-relative anchor orientation, and `[644:1751]` zero for
G1 mode. Two upstream details matter and are easy to get wrong:

* the mode slot is the **mode id followed by zeros** (`[0, 0, 0, 0]` for G1), not
  a one-hot vector;
* the orientation slots carry `motion_anchor_orientation_heading`, the reference
  root rotation made relative to the *robot's* heading — not the raw world
  quaternion.

"The robot's heading" is where live and offline differ:

* **live C++ semantics** use the measured base quaternion of the control cycle
  plus the operator's `apply_delta_heading`;
* **offline dataset production** has no recorded base, so it uses the
  deterministic assumption `reference_root_current_frame`: the reference root
  heading at the current frame stands in for the measured heading. With an
  identity `apply_delta_heading` this is algebraically the upstream `refheading`
  variant, which is why it needs no robot state.

The policy is recorded in every manifest (`encoder_orientation_policy`), and
`require_conversion_policy()` refuses the legacy raw-world orientation and the
measured-base policy for conversion — raw world yaw is never encoded into a
dataset.

Note on the current pilots: all three use a synthetic *upright* root, so the
heading-relative block and the raw world block coincide and the policy choice is
invisible in their tokens. The distinction becomes observable as soon as a source
carries a recorded (non-identity) root orientation; the earlier measurement on a
captured IDLE reference, whose root orientation did vary, is where the
raw-world encoding was off by 0.292 while the heading-relative encoding came in
at 0.125 against the same recorded deployment tokens.

Parity note: because the deployment's tokens depend on the live measured base,
offline tokens cannot be bit-compared with a recorded deployment run. Instead the
tests pin the model contract (names, shapes, dtypes, checksum), the official
packing layout, run-to-run repeatability, and golden tokens for a fixture whose
orientation block depends on the policy. The free SONIC run also records the
deployment's own tokens (`cpp_recorded_tokens.npz`) next to the pilot; comparing
them with the offline tokens is informative (0.12 mean absolute difference on the
Unitree pilot, dominated by the robot's measured tilt during the run) and is not
a parity gate.

## Source policies

* **Unitree Dex3** — arm-only collection. Lower body, waist and root are frozen
  at one validated standing frame (`static_stable_frame`, from the deployment's
  own standing pose), arms and hands are the recorded absolute desired positions,
  and velocities are re-derived from the composed 50 Hz positions. Deterministic
  for any episode length; marked `stationary_manipulation_only` because the feet
  never move.
* **Fruits 1K / `g1-pick-apple`** — 43D whole-body action (20 Hz) with channel
  names, so body and hands are remapped by name into the reference/IsaacLab order
  and the canonical Dex3 order; resampled to 50 Hz. Root pose is synthetic
  (declared). Human visual review found the recorded trajectory crouched and the
  free SONIC run walking backward, so this source is marked `excluded` for bulk
  processed conversion. Raw data and pilot evidence remain archived.
* **AppleToPlate** — modality boundaries name the body blocks but not individual
  channels. The body block order is verified as Unitree SDK/MuJoCo order using
  joint-limit fit, flat-footed recorded state, and source video. Its 30 Hz body
  action is interpolated by timestamp onto the uniform 50 Hz SONIC grid, then
  velocities are re-derived at 50 Hz; the terminal velocity copies the previous
  finite difference to match SONIC's own resampler. `observation.state` is kept
  separately as the measured 30 Hz replay and is not fed to the encoder.
  `action[29:36]` and `action[36:43]` are normalized open/close commands rather
  than actuator radians, so they are never written into the canonical hand slots.
  The hands instead come from `observation.state`, which carries the same channels
  as **measured radians in canonical Dex3 order**; that path is opt-in via
  `hands_from_state` in the pilot config (on for `apple`) and the manifest records
  which field supplied the hands. The measured left hand moves in 58/58 sampled
  episodes and stays inside the pinned limits; the right hand rests near its
  modelled zero and barely moves, so its channels carry little motion even though
  they are real measurements. The raw normalized commands are preserved as a pilot
  artifact either way.

## QC

`run_manifest.json` carries the numbers and the threshold decision for each:
`finite.non_finite_count`, `range.violation_count` (limits from the pinned MuJoCo
model), `velocity/acceleration/jerk.abs_p99`, `source_echo.max_abs_error_rad`
(stored canonical channels vs a direct interpolation of the source),
`synthetic.max_abs_error_rad` (synthetic channels vs the frame they claim), and
`future_clamp.mean_fraction`. Joint order is checked by name round-trip over a
marker vector, not by shape. The pilot verdict is `FAIL` if any threshold fails,
`UNVERIFIED` while a hand schema blocks the 78D action, otherwise `PASS`.

**QC is a gate, not a report.** Bulk conversion refuses an episode whose QC
verdict is `FAIL` (`require_episode_qc`), naming the failing metrics, so no tokens
are written for it. Only the 29 body joints are covered by
`range.violation_count`; the 14 Dex3 hand channels get their own
`hand_range.violating_channels` decision described below.

Simulation runs add per-joint MAE/p95/max, root height and tilt, and a fall flag
for the free window in `sonic_pilot_review.json`.

### Dex3 hand range policy

The Unitree Dex3 collections record hand values that leave the pinned model's
range on the index/middle distal joints, in **both** the commanded `action` and
the measured `observation.state`. The pinned `g1_29dof_with_hand` model, all three
shipped URDFs and the deployment's own `MAX_LIMITS_*`/`MIN_LIMITS_*` tables in
`dex3_hands.hpp` all stop at 1.7453 rad (100°); the data reaches 2.0944 rad (120°),
which is the documented stroke of the **earlier** Dex3 revision.

That is a hardware-revision difference, not a conversion defect: an exhaustive
search over all 5040 column permutations per hand ranks the canonical order first,
and the measured state violates in the same channels as the commanded action.

Measured over a 551-episode sample across all 13 collections, the excess is
bimodal and never in between — exactly 0.3491 rad (= 2.0944 − 1.7453) or ~0.0010
rad (a resting pose a hair past modelled zero). `HAND_RANGE_POLICY` in
`datasets/sonic/pilot.py` therefore keys on the **tolerance** rather than the
channel count: every hand channel may deviate, but by no more than the documented
stroke difference. Values are emitted as recorded — no clipping, no rescaling,
matching how the LeRobot converter stores raw `qpos`. Anything beyond the
tolerance is treated as a mapping error and fails the episode.

AppleToPlate's hands come from measured `observation.state` (see below); its right
hand rests about 0.053 rad below modelled zero, a real offset.

## Gate

Bulk conversion (`./dev.sh sonic-convert ...`) refuses to run unless **every**
requested source kind is marked `eligible` in the pilot configuration and has a
pilot whose `human_review.json` status is `accepted` or `accepted_with_notes`.
There is no flag that overrides either condition. A human review file cannot
re-enable a source marked `excluded`; Fruits / `g1-pick-apple` is currently
excluded and remains archive-only.

The gate resolves the reviewed run itself: `latest_pilot_dir` only considers runs
that actually wrote a `human_review.json`, so an aborted later run cannot shadow
the pilot a human accepted and turn the gate into a confusing "status is
'missing'" refusal.

Per episode the command runs the full pipeline, not only the reference composer:
`prepare_pilot()` writes the canonical reference plus the 1751D encoder
observation, and then `scripts/encode-sonic-episode.py` runs the pinned encoder
in the onnxruntime environment (`--encoder-python`, default
`/opt/venvs/sonic-sim/bin/python`) over that observation, inheriting `--model-dir`
and `--expected-sha256`. An episode is counted as converted only when both
`action_tokens.npz` and `encoder_manifest.json` exist in the output directory
taken from the returned manifest (`processed_dir`); any other outcome is reported
as that episode's failure and the batch continues. A `FAIL` QC verdict is fatal
for the episode and is reported as its failure. Sources with an unresolved hand
schema still produce their 64D body-latent token — only the 78D action stays
blocked, as recorded in the encoder manifest.
