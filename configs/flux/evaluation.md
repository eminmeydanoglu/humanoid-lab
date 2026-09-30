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
metrics. The current implemented mode is source-video VAE reconstruction.

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
