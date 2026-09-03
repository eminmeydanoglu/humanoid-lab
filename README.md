# humanoid-lab

Portable single-container dev environment for SONIC whole-body control on
Isaac Sim 5.1.0 + Isaac Lab 2.3.2 + GR00T (N1.7), on one NVIDIA GPU host.

## What is inside the image

One immutable Isaac Sim 5.1.0 base (digest-pinned) + three deliberately
isolated Python environments, built from committed frozen locks:

| env (`use-...`)          | Python | Contents                                          |
|--------------------------|--------|---------------------------------------------------|
| `/opt/venvs/isaac-sonic` | 3.11   | Torch 2.7.0+cu128, Isaac Lab 2.3.2, SONIC         |
| `/opt/venvs/sonic-sim`   | 3.11   | Torch 2.7.0+cu128, MuJoCo 3.12.0, SONIC `[sim]`, CycloneDDS + Unitree SDK2 transport (G1 sim loop) |
| `/opt/venvs/groot-n17`   | 3.12   | Torch 2.9.0+cu128, flash-attn 2.8.3, Isaac-GR00T N1.7 |

The pinned SONIC source ships its G1 MuJoCo meshes as Git LFS pointers with
some ASCII STLs; the image pulls the pinned meshes and converts them to binary
STL in-place, so `use-sonic-sim` loads the real G1 asset without any host
mounts. Vulkan loader (`libvulkan1`) + host NVIDIA ICD via compose `devices`
keep Isaac Sim headless EGL/Vulkan render working.

## Quick start

```bash
git clone <url> humanoid-lab && cd humanoid-lab
./setup.sh        # host preflight, lock render, image build, smoke tests
./dev.sh          # enter the main dev container
./dev.sh sonic-sim   # example: select the MuJoCo/G1 sim env
./scripts/smoke-test.sh   # inside the container: hard smoke tests per env
```

Smoke-test statuses: `PASS` / `WARN` / `BLOCKED` / `FAIL`.
Missing gated-model artifacts (HF license decision gate) report `BLOCKED`,
never a silent pass; anything mandatory that is broken fails the run.

## Layout

- `Dockerfile`, `compose.yaml` — the single dev container (host network/ipc, GPU reservation)
- `versions.lock.yaml` — single version source; rendered via `scripts/render-lock-env.py`
- `setup.sh`, `dev.sh`, `doctor.sh` — bootstrap / container entry / validation
- `containers/` — env selectors (`use-isaac-sonic`, `use-sonic-sim`, `use-groot`), G1 asset prep, viewer
- `locks/` — frozen uv locks per env (committed; see `locks/README.md`)
- `scripts/` — render, fetch-models, verify-pins, smoke-test, ci-lint
- `ros/` — direct-DDS test plan + optional Foxy profile decision gate (see `ros/README.md`)

## Models (separate step)

```bash
./dev.sh hf-login       # host-persistent HF cache; token never enters the image
./dev.sh fetch-models   # pinned revisions + MODEL_PROVENANCE.json (sha256 per file)
```

GR00T N1.7 loads its Qwen3-VL backbone from `nvidia/Cosmos-Reason2-2B`
(gated:auto) — accept that license on Hugging Face before `hf-login`, or the
N1.7 model-load smoke test reports `BLOCKED`.

## Notes

- Unverified pins (`required:false` in versions.lock.yaml) block setup until verified.
- Tokens never enter `.env` or the image.
- Container: no privileged, no Docker socket, all data host bind-mounted
  (datasets/checkpoints/models/hf-cache/uv-cache/isaac-cache/rosbags/runtime/outputs).
