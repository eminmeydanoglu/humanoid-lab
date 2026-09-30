# FLUX video evaluation

The evaluation application runs the pinned FLUX KinoVAE on the local RTX 5090.
It performs frozen inference and stores each run independently. The default
source is episode 711: `Unifolm/G1_Dex3_PickApple_Dataset/episode_0`, from the
local UniFoLM-to-SONIC LeRobot export, egocentric camera. The video has 676
frames at 30 FPS (22.533 seconds).

## Start

```bash
uv pip install --python data/venvs/flux-model/bin/python \
  -r scripts/flux-evaluation-requirements.txt
./scripts/flux-evaluation.sh
```

The launcher binds to the machine's Tailscale IPv4 address on port 7860.
On Raider the URL is **http://100.97.125.26:7860**. Access requires connectivity
to this machine through Tailscale. Tailnet permissions control who can access
it; the application has no separate login. Cross-origin browser writes are
rejected. No public listener or Tailscale Funnel is configured.

The installed persistent user service can be managed with:

```bash
systemctl --user status flux-evaluation
systemctl --user restart flux-evaluation
journalctl --user -u flux-evaluation -f
```

Overrides:

| Environment variable | Purpose |
| --- | --- |
| `FLUX_EVAL_HOST` / `FLUX_EVAL_PORT` | Listener, defaults to Tailscale IP / 7860 |
| `HUMANOID_DATA_ROOT` | Persistent data root |
| `FLUX_MODEL_PYTHON` | Existing FLUX environment interpreter |
| `FLUX_EVAL_WEIGHTS` | Local KinoVAE safetensors file |
| `FLUX_EVAL_DATASET` | UniFoLM LeRobot export root |
| `FLUX_EVAL_OUTPUT` | Run, upload, and preview storage root |

Default weights are reused from
`/home/aksoy-lab/code/flux-3-action/flux-action/outputs/weights/video_vae.safetensors`.
The model is loaded once, lazily, and remains on the GPU. Restart the service
to release GPU memory or change weights. The GPU worker serializes inference;
the queue accepts at most four outstanding jobs. Preview preparation uses a
separate lock and runs outside the GPU worker. The UI supports up to 24 local
apple episodes and user video uploads (512 MiB maximum).

## Evaluation contract

Resolution is **height × width**: 192×256 produces a video 256 pixels wide and
192 pixels high. Dimensions must be multiples of 32 and at least 160 pixels,
because decoder neighborhood attention requires five latent positions per axis. Resize uses stretch;
source frame rate is preserved. Start time and a frame limit are optional;
frame limit 0 evaluates the full remaining clip (maximum 3600 frames).

The encoder expects `(B,3,T,H,W)` RGB in `[-1,1]`, with `T=4k+1`. The application
repeats the last frame to meet this constraint and a minimum of 17 input frames
(five latent frames), records the padding, and removes extra output frames. KinoVAE uses its upstream 45-frame chunked encoder with
one-frame overlap, and the full-sequence decoder with an 8-frame activation
window (`decoder_max_t=8`). Latents are normalized KinoVAE latents,
`(B,96,1+(T-1)/4,H/32,W/32)`. Precision is bfloat16; compilation is disabled.

PSNR is computed from global mean squared error over all RGB pixels and frames.
MAE uses the 0–255 scale. Both metrics compare the resized source to rounded,
clamped uint8 decoded RGB **before** H.264 compression. Timings synchronize
CUDA and cover encode/decode calls. They exclude model loading, video I/O,
metrics, and artifact export. Peak GPU allocation is recorded for inference.
Per-frame errors are available in `metrics.json`.

Runs are under `data/outputs/flux-evaluation/runs/<run-id>/`:

- `run.json`: configuration, lifecycle state, timestamped logs, URLs, metrics.
- `metrics.json`: aggregate and per-frame reconstruction metrics.
- `latents.safetensors`: normalized latent tensor plus original/padded frame counts.
- `reconstruction.mp4`: H.264 video at the source frame rate.

Resized inputs are cached under `previews/`; uploads persist under `uploads/`.
Completed and failed jobs survive service restarts. An interrupted queued or
running job is marked failed after restart and can be rerun. Artifacts accumulate
on disk; retention is manual.

## Extending the platform

Modules are in `src/humanoid_lab/evaluation/`:

- `video.py`: source decoding, preparation, output video writing.
- `backends.py`: inference adapter protocol and registry; lazy model loading.
- `app.py`: catalog, upload validation, durable job state, queue, logs, metrics/API.
- `static/`: responsive UI, polling, history, synchronized video comparison.

To add trained-checkpoint reconstruction or a different VAE, implement a
backend with `label` and `reconstruct(frames, directory, log)`. Return a uint8
RGB array matching the prepared source shape and a JSON-compatible metrics
mapping. Register it in `registry()`; the backend selector reads the registry
automatically. Backends control their model loading and auxiliary artifacts.
For checkpoint-generated video without a reconstruction reference, extend the
job request and result contract with a separate task mode and optional reference
metrics. The application supports `vae_roundtrip`, `policy_window`, and `policy_dream` on the same serialized GPU worker.

API documentation is available at `/docs`. The main endpoints are `/api/catalog`,
`/api/preview`, `/api/runs`, `/api/runs/{id}`, and `/api/uploads`.

## Verification

```bash
uv pip install --python data/venvs/flux-model/bin/python pytest==8.3.5 httpx==0.28.1
PYTHONPATH=src:third_party/flux/flux-training/src \
  data/venvs/flux-model/bin/python -m pytest tests/test_flux_evaluation.py -q
```

Both full-video presets were exercised through the real browser with the
pretrained VAE. Initial measured results (676 source frames, 677 padded):

| H×W | Latent shape | PSNR (dB) | MAE | Peak allocated GPU (GiB) |
| --- | --- | --- | --- | --- |
| 192×256 | 1×96×170×6×8 | 37.531 | 1.995 | 4.888 |
| 288×384 | 1×96×170×9×12 | 38.690 | 1.760 | 8.029 |

CPU tests cover resizing/ranges, temporal padding metadata, cache reuse,
HTTP video byte ranges, validation, uploads, and persisted backend failures.

## G1 policy evaluation

The **Tek chunk** and **Uzun rüya** tabs use the validated original Dex3 index,
independently of the SONIC videos in the VAE tab. The default selection is apple
validation episode **181**, frame **150** (combined index ID `893`). Original
apple training episode 0 is combined ID `712`; its 1179-frame timeline is
independent of the SONIC preview's episode 711.

Select an episode, seek in its episode-relative source video, and click
**Videodaki kareyi kullan**. Frame, seconds, and ±1 controls resolve an explicit
zero-based frame index. The preview shows original RGB, the resized policy input,
28 named measured joints in physical and normalized units, and the exact edited
prompt. Empty prompts and invalid windows are rejected. The dataset caption is a
default; custom captions are passed through the checkpoint processor and CPU text
encoder, with cache misses logged. Changes to a draft visibly separate it from
an existing run's immutable snapshot.

- **Tek chunk:** base, raw fine-tune, or both produce 32 future RGB frames and
  32×28 absolute actions from the same input and seed. RGB reference is
  `t+1..t+32`; reference actions are `t..t+31`.
- **Uzun rüya:** defaults to 10 chunks, configurable from 1 to 20. Each model
  starts from the selected real input and then uses its own last uint8 decoded
  frame and last postprocessed physical action as the next RGB/state. The prompt
  stays fixed and every chunk uses seed 0. This is an idealized state=command
  assumption. Rollouts describe model behavior and do not measure physical robot
  success. References stop at episode/valid-range boundaries; generation can
  continue beyond the recording.

### Models and inference contract

Server-owned paths are fixed:

- Base: `/home/aksoy-lab/code/flux-training/flux-action/outputs/dex3/g1-base-policy`
- Raw FT: `data/models/flux-dex3/checkpoint-2500`

The base is the pretrained FLUX trunk with newly initialized G1 heads. The raw
LoRA adapter is loaded once; disabling it selects the original saved heads.
Both variants are verified against checkpoint tensors. The text encoder stays
on CPU, with a bounded 32-caption GPU-context cache. The VAE tab releases the
policy before loading its standalone VAE; policy jobs release that VAE first.

The strict deploy preset uses BF16, torch attention, no compilation, history
packing with one observation, absolute 28D actions, action scale 2, 192×256,
Cosmos UniPC with 4 steps, shift 5, video CFG 4, action CFG 1, seed 0, action
30 FPS and learned video positions at 24 FPS. Resolution overrides are absent
from the first policy release. The checkpoint-owned quantiles are reused.
Physical predicted actions are preserved without motor-range clamping; clipping
and quantile-range diagnostics are recorded separately. Normalized comparison
metrics apply the saved quantile transform and clipping to both trajectories;
the unclipped sampler values are also preserved.

The joint sampler retains video and action from one solver call. Decoding uses
one observed latent plus eight future latents, producing 33 frames; it removes
the observation frame to expose exactly 32 future RGB frames. References never
enter policy conditioning. GPU regression checks compare actions with the direct
deploy call and compare actual training/deploy snapshot conditioning tokens.
Single-frame versus full-clip encoding differences are reported separately.

### Jobs, artifacts and API

Policy jobs share the four-outstanding-job queue. **Koşuyu iptal et** requests
cooperative cancellation at a chunk boundary; completed chunks remain available.
Cancel intent and old VAE manifests survive restart. Interrupted active runs are
marked failed, while runs with persisted cancel intent become cancelled.

Each run stores `input_snapshot.json`, `input.npz`, original/resized PNGs, exact
config/provenance, reference video, and per-model prediction video, action JSON,
CSV/NPZ and metrics. Per-chunk NPZ/JSON store states, raw/normalized actions,
observed/future normalized latents, input/output RGB, effective sampler settings,
position IDs, token shapes, VAE implementation/weights, seeds, clipping and
timings. Dataset file size/mtime fingerprints guard a queued job against a changed
comparison basis. Outputs reside under `data/outputs/flux-evaluation/runs/`.

Additional endpoints:

- `GET /api/policy/catalog`
- `GET /api/policy/episodes/{episode_id}/video`
- `POST /api/policy/preview`
- `POST /api/policy/runs`
- `POST /api/runs/{run_id}/cancel`

A policy payload contains `task_mode`, `episode_id`, `start_frame`, `prompt`,
`models` (`base`/`ft`) and `chunks`. The existing run/history/artifact endpoints
serve all task modes. Policy inputs use paired dataset observations; video uploads
remain available in the VAE tab.

Dataset overrides: `FLUX_EVAL_POLICY_INDEX` and `FLUX_EVAL_POLICY_DATASET`.

```bash
PYTHONPATH=src:third_party/flux/flux-training/src \
  data/venvs/flux-model/bin/python -m pytest \
  tests/test_flux_evaluation.py tests/test_flux_policy*.py -q

# Requires exclusive FLUX GPU ownership; unload the web policy/VAE first.
FLUX_GPU_PARITY=1 PYTHONPATH=src:third_party/flux/flux-training/src \
  data/venvs/flux-model/bin/python -m pytest \
  tests/test_flux_policy.py -k actual_training_deploy -s
```

Real GPU probe evidence is stored in
`data/outputs/flux-evaluation/verification/joint-gpu-parity.json`.
