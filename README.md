# humanoid-lab


## Setup from scratch

```bash
git clone git@github.com:eminmeydanoglu/humanoid-lab.git
cd humanoid-lab

docker login nvcr.io        # the Isaac Sim base image is EULA-gated on NGC
./setup.sh --verify-digests # resolves the base image digest from NGC into versions.lock.yaml
./setup.sh                  # preflight, .env, image build, container start, smoke test
```

`setup.sh --verify-digests` needs NGC access. Without a verified digest in the lock, `setup.sh`
refuses to build

Models are a separate, revision-pinned step:

```bash
./dev.sh hf-login      # once, if you need gated Hugging Face repositories; the token stays in the host HF cache
./dev.sh fetch-models  # pinned revisions into $HUMANOID_DATA_ROOT/models, plus MODEL_PROVENANCE.json with sha256 per file
```

`nvidia/Cosmos-Reason2-2B`, the GR00T N1.7 backbone, is gated: accept its license on Hugging Face
before `fetch-models`, otherwise the GR00T checks stay `BLOCKED`.


## G1 on Isaac Sim

```bash
./dev.sh isaac-g1 dex3          # Dex3 hands
./dev.sh isaac-g1 inspire-ftp   # Inspire FTP hands
./dev.sh isaac-g1 no_hands      # 29-DoF body only
```

Flags:

| Flag | Effect |
| --- | --- |
| `--headless` | no GUI and no X11 requirement |
| `--head-camera-window` | opens the 640x480 head camera in a second GPU viewport; costs render throughput |
| `--duration SECONDS` | bounded run |
| `--device {cpu,cuda}` | overrides the profile device; profiles default to CPU |
| `--controller none` | disables the controller a profile declares, falling back to passive |
| `--test {passive-fall\|controlled-hold\|controller-hold}` | one of the acceptance modes below |

**Head camera.** A profile declares the head camera by the horizontal angle of the stream it stands
for (`horizontal_fov_deg`) and the aperture the focal length is expressed against; the focal length
and the vertical aperture follow from them. Isaac Sim projects square pixels: an authored vertical
aperture has no effect on the image, and a 640x360 render is a plain centre crop of the 640x480 one
(probe renders, `tests/test_head_camera_projection.sh`). The rendered pair is therefore the declared
horizontal angle plus the vertical angle the image size implies: 54.90 x 42.57 degrees for the G1's
RealSense D435i color stream at 640x480. The D435i is quoted at 54.9 x 42.5 (the profiles record that
pair as `nominal_fov_deg`); it is 0.07 deg narrower vertically than any square-pixel 4:3 projection of
54.9 can be, so the render follows the projection and the manifest records the nominal pair, the
rendered pair and the prim's own values.


## Isaac and SONIC in two terminals

```bash
# terminal 1 — Isaac Sim: scene, physics loop, robot state on DDS, incoming joint commands applied
./dev.sh isaac-g1-sonic dex3

# terminal 2 — the official SONIC deployment: planner and keyboard on this terminal
./dev.sh sonic-controller
```

Keys in the SONIC terminal: `]` starts control, `T` plays the reference motion, `N` and `P` change
motion, `R` resets, `O` is the emergency stop. `Enter` to switch to planner to teleop.


## If you need


| Command | What it does |
| --- | --- |
| `./dev.sh` | interactive shell in the container, no environment pre-selected |
| `./dev.sh isaac-g1-test-controller dex3` | G1 driven by the deterministic scripted test controller |
| `./dev.sh isaac-demo <demo.py>` | an Isaac Lab demo from `/opt/src/isaaclab/scripts/demos` |
| `./dev.sh doctor` | full host + container report, written to `$HUMANOID_DATA_ROOT/diagnostics/` |
| `./dev.sh smoke` | in-container environment, asset and model checks |
| `./dev.sh sync` | re-provisions the environment whose lock or pinned source changed |
| `./dev.sh fetch-models` | downloads the pinned model revisions | |
| `./dev.sh groot-finetune-smoke [pipeline\|pretrained\|eval]` | optional GR00T N1.7 fine-tuning smoke (two optimizer steps, then open-loop eval) |
| `./dev.sh hf-login` | Hugging Face login for gated repositories |
| `./dev.sh stop` | stops the container, keeps the data root |
| `./dev.sh rebuild` | rebuilds the image with the same lock, recreates the container |
| `./dev.sh isaac` | shell with the `isaac-sonic` environment active |
| `./dev.sh sonic-sim` | shell with the `sonic-sim` environment (MuJoCo + G1) |
| `./dev.sh groot` | shell with the `groot-n17` environment |




## Repository layout

| Path | Contents |
| --- | --- |
| `setup.sh`, `dev.sh`, `doctor.sh` | host entry points: provisioning, container lifecycle and runs, diagnostics |
| `compose.yaml`, `Dockerfile`, `containers/` | image and container definition, entrypoint, environment selectors, venv provisioning |
| `configs/profiles/` | G1 run profiles: robot asset, hands, camera, controller, support band |
| `src/humanoid_lab/` | simulator service, profile and command contracts, controller bridge, GRAIL prompt layer |
| `scripts/` | Isaac G1 runner, GRAIL prompt manifest, model fetching, lock rendering, checks |
| `locks/` | frozen `uv` locks for the three Python environments |
| `tests/` | `unittest` modules and shell checks |
| `docs/` | SONIC-Isaac link notes with the measured acceptance results |
| `versions.lock.yaml` | single source of every version pin |



## GRAIL kinematic replay

Open any valid motion of the local GRAIL `pickup_table` release in Isaac and play the recorded
robot and object trajectory. No controller, DDS or SONIC is involved: gravity is off, nothing
drives the robot, and only the recorded state is written. Each 25 Hz source interval gets one
interpolated midpoint, so the original keyframes are preserved while hands render at 50 Hz.

```bash
./dev.sh isaac-g1 grail-replay pickup_table__apple_0__000            # GUI: external + head camera viewports
./dev.sh isaac-g1 grail-replay pickup_table__apple_0__000 --headless # writes external.mp4, head.mp4, manifest.json
```

The sequence key alone selects the motion, its object USD and textures, and the metadata table.
Headless output goes to `/outputs/grail-replay/<sequence-key>/`; `--replay-data-root` and
`--replay-output-dir` override the defaults. Missing or malformed dataset files fail the run.

### Target visibility in the head camera

`--visibility` adds a target-object visibility analysis around the grasp. The grasp itself is the
first negative-to-positive transition of the source robot pkl's discrete `hand_action_right`
command (`-1` open, `+1` closed); it is rejected if that series is absent, non-discrete,
non-finite, already closed or never closes.

```bash
./dev.sh isaac-g1 grail-replay pickup_table__apple_0__000 --headless \
  --visibility --visibility-threshold 0.5 --visibility-before 1.0 --visibility-after 1.0
```

The run writes `visibility.json` next to the videos and a `visibility` summary into
`manifest.json`. Every mesh leaf of the target gets a semantic label, and each render frame
carries `visible_fraction`, the primary metric:

```text
visible_fraction = target pixels visible in the actual 640x480 head-camera image
                   --------------------------------------------------------------------
                   target pixels the unoccluded object projects onto the head-camera
                   image plane, including pixels outside the image
```

Occlusion by the robot, hands or table and clipping at the image boundary both count as *not
visible*: 1.0 is a fully visible, unclipped object, about 0.5 is half outside the image, and 0.0
is an object that is fully occluded or outside the view. The denominator is measured by one extra
analysis camera: the head camera's pose, optics and pixel scale with a field of view twice as wide,
whose central 640x480 crop is the head image (checked per frame; a misaligned pair becomes an
unmeasured frame, never a silent zero). Frames whose denominator cannot be measured exactly — for
example a leaf fully covered while another is visible — keep their fraction but are marked
`exact: false` with a reason, and unmeasurable frames are `valid: false` and excluded from
summaries. A run in which the annotators never see the target at all fails instead of reporting a
clip of zeros.

The report also carries the grasp source frame/time, the configured window and threshold, the
contiguous below-threshold intervals, minimum/mean/grasp-time fractions, and the selection
booleans `object_below_threshold_for_whole_window` and `object_out_of_frame_for_whole_window`
(strict: every window frame measured). Window frames are the render frames whose timestamps lie in
the closed interval `[grasp - before, grasp + after]`, first frame `ceil(start * fps)` and last
`floor(end * fps)`, clamped to the clip.

Batch selection over finished reports:

```bash
python3 scripts/collect-grail-visibility.py --format jsonl --output data/outputs/grail/visibility_candidates.jsonl
# --below-threshold-only, --out-of-frame-only, --format csv, --reports-root PATH
```

Acceptance: `tests/test_grail_replay_clip.sh [sequence-key]` (renders with visibility enabled).
Unit tests: `PYTHONPATH=src python3 -m unittest tests.simulator.test_grail_replay tests.simulator.test_visibility`.


## GRAIL pickup_table prompt layer

Every GRAIL source trajectory gets exactly one English instruction, derived from its file stem
(`<task_family>__<object_asset_id>__<motion_variant>`, e.g. `pickup_table__apple_17__003`). The
stems under `data/datasets/grail/data/pickup_table/robot/` are the only source of truth; nothing
in the GRAIL release is read or modified.

```bash
python3 scripts/generate-grail-prompt-manifest.py \
  --output data/outputs/grail/pickup_table_prompt_manifest.jsonl
# --robot-dir defaults to data/datasets/grail/data/pickup_table/robot
```

The manifest is JSONL, sorted by `source_motion_id`, and each row carries `schema_version`,
`prompt_policy_version`, `source_motion_id`, `task_family`, `object_asset_id`,
`object_category_raw`, `object_name`, `motion_variant`, `prompt_template_id` and `instruction`.
A malformed stem, a missing robot directory or an empty directory fails the run with exit code 2
and leaves no output; a successful run replaces the output atomically.

**Determinism.** The template is chosen by
`sha256("sha256-first8be-mod-v1" + NUL + source_motion_id)`, first eight bytes big-endian, modulo
the template count. Python's `hash()` is never used, so the same input yields the same row on
every machine and every run. `PROMPT_POLICY_VERSION` in
[`src/humanoid_lab/grail/prompts.py`](src/humanoid_lab/grail/prompts.py) identifies the template
set, the selection algorithm and the label policy.

**Camera variations.** A camera variation is a derived trajectory that replays one source motion;
it keeps that motion's stem as its `source_motion_id`. Consumers therefore look the instruction up
by id instead of re-deriving it, and every variation of one source trajectory shares one prompt:

```python
import json
lookup = {row["source_motion_id"]: row["instruction"]
          for row in map(json.loads, open(manifest_path))}
instruction = lookup[source_motion_id]
```

**Templates and labels.** The first template family is only pick-and-lift off a table: six
templates that use `pick up`, `grab`, `lift ... off the table` and `take ... from the table`, with
no bring, place, hand-over, fetch or left/right hand wording. Each trajectory gets one template,
so variations are spread across trajectories rather than duplicated per trajectory. Object labels
are the raw category with underscores turned into spaces, except for reviewed categories in
`OBJECT_NAME_OVERRIDES` (`alcohol` → "alcohol bottle", `bagged_food` → "bag of food", `bar` →
"snack bar", `spray` → "spray bottle", and similar); the raw category is kept in the manifest.

Tests: `PYTHONPATH=src python3 -m unittest tests.test_grail_prompts`.



## Data root, mounts and environments

`HUMANOID_DATA_ROOT` in `.env` defaults to `./data` and holds everything persistent:
`datasets`, `checkpoints`, `models`, `hf-cache`, `uv-cache`, `venvs`, `isaac-cache`, `outputs`,
`diagnostics`, `rosbags`, `runtime`. It is bind-mounted into the container as `/data`, `/cache`,
`/outputs` and `/opt/venvs`. The repository itself is mounted at `/workspace/humanoid-lab`.


## Requirements

| Item | Value |
| --- | --- |
| OS / arch | Ubuntu 24.04, x86_64 |
| GPU | NVIDIA GPU, driver >= 580.65.06 (newer is kept, older fails the preflight) |
| Docker | Docker Engine with the Compose v2 plugin, plus the NVIDIA Container Toolkit |
| Disk | more than 100 GiB free where the repo lives |
| Host tools | `git`, `curl`, `python3` with PyYAML |
| NGC | account with the Isaac Sim EULA accepted, and `docker login nvcr.io` performed |
| GUI (optional) | a working X11 display; headless runs need none |
| Hugging Face (optional) | token, only for the gated GR00T backbone |

