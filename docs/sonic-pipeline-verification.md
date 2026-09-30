# SONIC pipeline verification — findings and fixes

Date: 2026-09-16 · checkout `33b2be4` (pre-fix) → fixes applied in this pass

Everything below was executed on this machine. Numbers come from the commands and
artifacts named next to them, not from reading the source alone.

## 1. What was verified and found correct

| Check | Evidence | Result |
|---|---|---|
| Encoder input layout (1751D) vs upstream | independent YAML parse + offsets scraped from the pinned C++ registry | identical, 0 discrepancy |
| ONNX IO contract | raw `onnxruntime` 1.22.0, no repo code | `obs_dict [1,1751] f32` → `encoded_tokens [1,64] f32` |
| `PINNED_ENCODER_SHA256` | sha256 of the actual model file | matches (`fb97de22…`) |
| Runner vs raw ORT | same array through both paths | bitwise equal (max diff `0.0`) |
| Encoder determinism | same frame × 5 | bitwise identical |
| Look-ahead packing | `value=frame*1000+joint` probe | `obs[10,4:294].reshape(10,29) == joint_pos[10+5k]` exactly |
| Mode block semantics | 822-frame real observation | `[0,0,0,0]` every row, not one-hot; other 8 blocks exactly 0.0 |
| Joint order (arm 14) | independent name-based remap from raw parquet | max error `5.9e-8` rad, no cross-joint swap |
| Joint order (hands 14) | independent remap | max error `1.2e-7` rad |
| Hand permutation | exhaustive search over all 5040 column permutations | canonical order ranks **1st** |
| Timeline | dt over 822 frames | exactly 20.000000 ms (50 Hz) |
| Velocities | recomputed finite difference | max diff `1.2e-7` rad/s; integrated displacement matches source |
| Body joint limits | pinned MuJoCo model | 0 violations, all 29 joints |
| Encoder observation rebuild | block-by-block from `reference.npz` | max abs diff **0.0** over `(822,1751)` |
| Token/action composition | `action_tokens.npz` vs reference | `action = [motion_token|left|right]`, max diff `0.0` |
| Frame-exact replay | Isaac kinematic metrics | body/hand write error **0.0 rad**, root `6e-8 m` |
| Latent replay fidelity | deployment debug stream | `first_token_echo_max_abs` **0.0** |

Independent verification script: `scripts/verify-sonic-unitree-conversion.py`,
runnable as `./dev.sh sonic-verify [episode ...]` (does not import the pipeline;
re-derives everything from raw parquet). Encoder audit:
`data/diagnostics/encoder-contract-audit.md`.

Current state: `./dev.sh sonic-verify 3 74` → **48/48 checks pass**;
`./dev.sh sonic-tests` → **118 passed, 8 skipped**.

## 2. Bugs found and fixed

### 2.1 Bulk conversion ignored a failing QC verdict — **fixed**

`prepare_pilot` computed per-episode QC and stored `qc.episode_result`, but
`scripts/convert-sonic-dataset.py` never read it: an episode whose QC verdict was
`FAIL` still got its tokens written and was counted in `"converted"`. A
non-finite value, a joint outside its modelled range or a garbage derivative would
have entered the processed corpus while the docs advertised QC as a gate.

Fix: `require_episode_qc()` in `scripts/convert-sonic-dataset.py` raises
`QcRejected` (naming the failing metrics) before the encoder ever runs.
Verified: `PASS`/`UNVERIFIED` accepted, `FAIL` rejected with
`range.violation_count=12.0 > 0.0`.

### 2.2 Selecting a bulk dataset filed episodes under the wrong dataset — **fixed**

`--dataset` narrowed which raw collection to read but replaced only
`PilotSpec.dataset`; `pilot_name` (and therefore the output directory) was derived
from the *pilot's* dataset and left untouched. Converting PickDoll wrote its
episodes into `unitree_G1_Dex3_ObjectPlacement_Dataset_epNNN/` directories — 779
mislabeled runs before it was caught — and those runs then shadowed the real
`ObjectPlacement` pilot, so the review gate reported `pending_human_review` for a
pilot that had been accepted.

Fix: the replacement recomputes `pilot_name` from the selected dataset.
Verified: `G1_Dex3_PickDoll_Dataset` now writes to
`unitree_G1_Dex3_PickDoll_Dataset_ep006/…`, and `BulkSourceNamingTest` pins the
naming for all 13 declared datasets. The 779 mislabeled runs were deleted, each
identified by reading its own `run_manifest.json` so genuine runs were kept.

### 2.3 The review gate could be shadowed by an aborted run — **fixed**

`latest_pilot_dir` returned the newest directory regardless of content, so any
later run that died before writing `human_review.json` became "the latest" and
turned the gate into a confusing `status is 'missing'` refusal. It now only
considers runs that wrote a review file. Verified by
`LatestPilotSelectionTest` and by a real conversion the gate had been blocking.

### 2.4 The Dex3 hand channels were never range-checked — **fixed**

`range_report` checks the 29 body joints only. All 14 hand channels could sit
arbitrarily far outside the modelled Dex3 range while the episode still reported
`range.violation_count = 0`.

Fix: `hand_range_report()` + `hand_range_decision()` in `quality.py`, wired into
`evaluate_episode_qc` as the `hand_range.violating_channels` decision.

### 2.5 A stale source-invariant test was failing — **fixed**

`test_loop_decouples_physics_from_render` asserted
`source.count("self._sim.step(render=False)") == 1`, but the frame-exact kinematic
stepper (commit `9a2a2c8`) added two more correct calls.
`106 passed, 1 failed` → the test now checks the invariant **per stepper** rather
than by a module-wide count.

### 2.6 AppleToPlate could only ever emit a 64D body latent — **changed**

The adapter blocked the 78D action because the *action* hand block is a quantized
normalized open/close command. That reasoning is correct for the action block, but
`observation.state` carries the same channels as **measured radians** in canonical
Dex3 order, and it is clean: over 58 sampled episodes the left hand moves in 58/58
and stays inside the pinned limits (worst deviation 0.054 rad).

Fix: `hands_from_state` option (config-driven, off by default) fills the canonical
hand slots from the measured state and marks the schema verified with the source
recorded in provenance. Never silently: the manifest states which field supplied
the hands, and the normalized action block is still preserved separately.

### 2.7 Bulk conversion could not address all datasets or resume — **added**

`--episodes START:STOP` was mandatory and its range applied to every dataset, so a
199-episode collection could not be converted alongside a 418-episode one, and a
long batch could not resume without piling up duplicate timestamped copies. Added
`--all-episodes` (each dataset uses its own declared `total_episodes`),
`--dataset`, `--no-video` (the camera clip is what makes a conversion ~15× slower)
and `--force`, plus skip-if-already-converted logic keyed on both encoder
artifacts. Verified: 203/203 PickDoll episodes with 0 failures, and a re-run
converting 0 and skipping 2.

The full corpus conversion was **not** run in this pass.

## 3. The Dex3 hand range question — investigated, not a data bug

Hand values leave the pinned model's range on both the index/middle distal joints
(2.0944 vs 1.7453 rad) and the proximal ones.

* The pinned `g1_29dof_with_hand` MuJoCo model, all three shipped URDFs, and the
  pinning deployment's own `MAX_LIMITS_*`/`MIN_LIMITS_*` tables in `dex3_hands.hpp`
  all agree on 1.7453 rad.
* The measured `observation.state` violates in the *same* channels and by the
  *same* amount as the commanded action, so the trajectory really was recorded
  there — it is not a packing error.
* An exhaustive search over all 5040 permutations per hand ranks the canonical
  order **first**, so no relabelling explains it.
* 2.0944 rad = 120° is the documented stroke of the **earlier** Dex3 revision;
  the current one is 100°.

Measured over a 551-episode sample covering all 13 collections, the excess is
bimodal and never in between: exactly 0.3491 rad (= 2.0944 − 1.7453, the revision
stroke difference) or ~0.0010 rad (a resting pose a hair past modelled zero).
Channel counts ranged 0–12 of 14, so the channel count is a weak signal and the
tolerance is the discriminator.

Conclusion: a hardware-revision difference, not a conversion defect. Values are
emitted as recorded (no clipping, no rescaling, matching how the LeRobot
converter stores raw qpos). The QC policy encodes this as a bounded,
source-specific rule — within the revision stroke on any number of channels
passes, anything worse fails — instead of the previous silent acceptance, which
caught 18 real out-of-policy episodes in a 242-episode probe before the policy was
widened to match the measured distribution.

## 4. Other observations (not fixed; no action taken)

* **`min_sole_z_m` is a misnomer.** The kinematic metrics report `0.0709` m for a
  grounded standing pose. `service.py` takes the minimum z of the `ankle_roll`
  *body origin*, not of the foot mesh. Ground truth from MuJoCo with the same
  standing pose and pelvis at 0.792563 m: lowest `ankle_roll` mesh vertex
  z = 0.0351 m, lowest foot vertex overall 0.0351 m — so the robot is grounded and
  the metric is simply named for something it does not measure.
* **Sentinel values in the raw Unitree state.** `observation.state` contains 2
  outliers across the data file — `kRightHandMiddle0` at −3363.483 (episode 23)
  and −119.361 (episode 96). The pipeline encodes from `action`, not
  `observation.state`, so they do not reach any token; they are a raw-data defect
  worth knowing about if the state field is ever used. The GR00T pack, which does
  use the measured state, found these are part of a larger right-hand dropout
  class — see section 6.
* **`right_hip_roll` +1.58 rad in the free run.** The joint ranges above the
  model limit seen in the SONIC tracking trace belong to the *policy's own output*
  during the free run, not to the dataset reference (all 29 reference joints are
  inside the limits).
* **Encoder throughput is not the bottleneck.** Measured 2961 fps through the repo
  runner on CPU; the full 2.76 M-frame corpus is ~14 min of inference. TensorRT
  reaches 13 162 fps but saves only ~11 min in total.
* **CUDA/TensorRT silently fall back to CPU in this container**
  (`libcublasLt.so.12` / `libnvrtc.so.12` not on the loader path). Production
  defaults to CPU so it is unaffected, but any future GPU run must assert
  `providers_active` rather than trusting the requested provider list.
* **Tokens are quantized.** The ONNX graph ends in a quantizer, so every output
  component is an exact multiple of 1/16 with |t| ≤ 0.9697. Losses and metrics
  should not treat the latent as unbounded/continuous.

## 5. Artifacts produced in this pass

Three videos per dataset episode, plus a minimal page each:

* `unitree_G1_Dex3_ObjectPlacement_Dataset_ep003/20260915T211315Z/` (822 frames)
* `apple_nvidia-gr00t-n1.7-apple-to-plate_ep179/20260915T213121Z/` (414 frames)
* the reviewed `unitree_G1_Dex3_ObjectPlacement_Dataset_ep074/20260915T152807Z/`
  keeps its earlier `completed_reference_kinematic.mp4` and
  `sonic_latent_free.mp4`

Each directory holds `source.mp4` (recorded head camera),
`completed_reference_kinematic.mp4` (canonical 50 Hz trajectory written straight
into Isaac) and `sonic_latent_free.mp4` (free SONIC run driven by the persisted
offline tokens), described by `review.html`. Generate with
`scripts/write-sonic-pilot-page.py --pilot-dir DIR --output DIR/review.html`, serve
with `scripts/serve-sonic-review.py` (byte-range support is required or the video
timeline cannot seek).

Both free runs completed with `hand_binding` bound on both sides and tracking MAE
0.1138 rad (Unitree, 822 frames) / 0.1011 rad (Apple, 414 frames). The kinematic
replays write every frame with body and hand write error `0.0` rad and root write
error `6e-8` m.

## 6. The measured-state hand dropouts — corpus audit and the 51-frame bound

Date: 2026-09-19. The first full GR00T pack conversion stopped at
`G1_Dex3_PickDoll_Dataset` episode 77: `right_hand_index_0_joint` carried an
invalid run of 18 samples, above the 15-frame repair bound declared by
`configs/datasets/groot/unitree_dex3_sonic_v1.yaml`. A read-only scan of every
raw episode of all 13 collections (predicate `~isfinite | abs(q) > 3.0 rad` on
the 14 measured hand columns of `observation.state` and the 14 desired hand
columns of `action`) places that failure in context:

* **104 runs / 314 samples**, all in `observation.state` and all on the
  **right hand**: `right_hand_index_0` 133, `right_hand_middle_0` 122,
  `right_hand_index_1` 31, `right_hand_middle_1` 28. The left hand, both arms
  and the whole desired action block are clean: 0 invalid samples and 0
  non-finite values anywhere in the 28 columns.
* **Stuck-value bursts, not scaled or noisy readings.** A run holds one constant
  for its length and may switch constant mid-run (episode 77,
  `kRightHandMiddle0`: −29.441 × 11 then −719.823 × 14). Magnitudes reach
  1921.6 rad against a ±2.09 rad mechanical range.
* **Run-length histogram** 72×1, 9×2, 5×3, 4×4, 4×5, 2×8, 2×9, 1×13, 1×15, and
  four runs above the old bound: 17, 18, 25 and 51 frames. No run touches an
  episode boundary.
* **Only two episodes fail the old policy**, both in the train split:
  PickDoll/77 (longest run 25 frames, worst channel 72/2227 = 3.23 %) and
  PickDoll/153 (longest run 51 frames, worst channel 53/1457 = 3.64 %). Every
  other affected episode stays inside 15 frames and 0.8 %.
* **Interpolation is defensible for all four long runs.** The valid samples on
  either side differ by ≤ 0.014 rad (0.0015, 1.5e−5, 0.014, 0.0022) while those
  channels' p95 per-frame motion is 0.008–0.013 rad, and the clean desired
  action moves ≤ 0.022 rad across the same spans — the hand is parked, so the
  interpolated bridge is a near-hold, never an invented motion.

The two episodes cannot be excluded: the split is the frozen Psi0
`split_manifest.json` and `assert_manifest_matches_config` requires the
contract's exclusions to equal the manifest's (PickDoll has none), so an
exclusion would fail closed or fork the shared split. The narrowest safe change
is therefore one number — `repair.max_gap_source_frames` 15 → 51 (the audited
corpus maximum) — with the 3.0 rad threshold, the 5 % per-channel fraction cap,
the boundary refusal, the all-invalid refusal and the action-clean check left
unchanged. With that bound both episodes convert end-to-end; their repair
ledgers (174 and 72 samples) match the audit per channel and the rebuilt-vs-
corpus hand cross-check stays at 6e-8 rad.
