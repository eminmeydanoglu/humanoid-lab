"""Local/Tailscale video evaluation API with durable runs and one GPU worker."""
import hashlib
import json
import logging
import os
import threading
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, model_validator
from typing import Literal

from .backends import registry
from .video import padded_frames, prepare, write_video

ROOT = Path(__file__).resolve().parents[3]
DATA = Path(os.environ.get("HUMANOID_DATA_ROOT", ROOT / "data"))
STORE = Path(os.environ.get("FLUX_EVAL_OUTPUT", DATA / "outputs/flux-evaluation"))
DATASET = Path(os.environ.get("FLUX_EVAL_DATASET", DATA / "datasets/sonic/unifolm_sonic_lerobot_train"))
WEIGHTS = Path(os.environ.get("FLUX_EVAL_WEIGHTS", "/home/aksoy-lab/code/flux-training/flux-action/outputs/weights/video_vae.safetensors"))
for name in ("runs", "previews", "uploads"):
    (STORE / name).mkdir(parents=True, exist_ok=True)
BACKENDS = registry(WEIGHTS)
WORKER = ThreadPoolExecutor(max_workers=1, thread_name_prefix="vae-eval")
LOCK = threading.RLock()
PREVIEW_LOCK = threading.Lock()
RUNS = {}
VIDEOS = {}
POLICY_MANAGER = None
POLICY_DATASET = None
CANCELLED = set()


def discover():
    videos = {}
    meta = DATASET / "meta/episodes.jsonl"
    if meta.is_file():
        for line in meta.read_text().splitlines():
            episode = json.loads(line)
            if "PickApple" not in episode.get("episode_unique_id", ""):
                continue
            index = episode["episode_index"]
            path = DATASET / f"videos/chunk-{index // 1000:03d}/observation.images.egocentric/episode_{index:06d}.mp4"
            if path.is_file():
                key = f"unifolm-{index}"
                videos[key] = {"id": key, "label": f"UniFoLM G1 · elma alma · episode {index}", "path": path}
            if len(videos) >= 24:
                break
    for path in sorted((STORE / "uploads").glob("*.mp4")):
        label_file = path.with_suffix(".json")
        label = json.loads(label_file.read_text())["label"] if label_file.is_file() else f"Yüklenen · {path.stem}"
        videos[path.stem] = {"id": path.stem, "label": label, "path": path}
    VIDEOS.clear()
    VIDEOS.update(videos)


def save_run(run):
    path = STORE / "runs" / run["id"] / "run.json"
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(run, indent=2, ensure_ascii=False, allow_nan=False))
    tmp.replace(path)


discover()
for manifest in (STORE / "runs").glob("*/run.json"):
    run = json.loads(manifest.read_text())
    if run["status"] in ("queued", "running"):
        if run.get("cancel_requested"):
            run.update(status="cancelled", stage="cancelled", error=None)
        else:
            run.update(status="failed", error="Sunucu yeniden başlatıldı; koşuyu tekrar başlatın.")
        save_run(run)
    RUNS[run["id"]] = run

app = FastAPI(title="FLUX Video Evaluation", version="1.0")
app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")


@app.middleware("http")
async def same_origin_writes(request, call_next):
    origin = request.headers.get("origin")
    if request.method in ("POST", "PUT", "PATCH", "DELETE") and origin:
        from urllib.parse import urlsplit
        if urlsplit(origin).netloc != request.url.netloc:
            return JSONResponse({"detail": "İstek uygulamanın kendi adresinden gelmelidir."}, status_code=403)
    return await call_next(request)


class Configuration(BaseModel):
    task_mode: Literal["vae_roundtrip"] = "vae_roundtrip"
    video_id: str
    height: int = Field(default=192, ge=160, le=768)
    width: int = Field(default=256, ge=160, le=1024)
    start_seconds: float = Field(default=0, ge=0, le=3600, allow_inf_nan=False)
    max_frames: int = Field(default=0, ge=0, le=3600)
    backend: str = "flux_vae"

    @model_validator(mode="after")
    def check(self):
        if self.height % 32 or self.width % 32:
            raise ValueError("Yükseklik ve genişlik 32'nin katı olmalıdır.")
        return self


class PolicyConfiguration(BaseModel):
    task_mode: Literal["policy_window", "policy_dream"] = "policy_window"
    episode_id: str
    start_frame: int = Field(default=150, ge=0)
    prompt: str = Field(min_length=1, max_length=8000)
    models: list[Literal["base", "ft"]] = Field(default_factory=lambda: ["base", "ft"], min_length=1, max_length=2)
    chunks: int = Field(default=10, ge=1, le=20)

    @model_validator(mode="after")
    def validate_policy(self):
        if not self.prompt.strip():
            raise ValueError("Prompt boş olamaz.")
        if len(set(self.models)) != len(self.models):
            raise ValueError("Model seçimi tekrarlanamaz.")
        return self


def source(config):
    if config.video_id not in VIDEOS:
        raise HTTPException(404, "Video bulunamadı.")
    if config.backend not in BACKENDS:
        raise HTTPException(422, "Backend bulunamadı.")
    return VIDEOS[config.video_id]["path"]


def preview(config):
    with PREVIEW_LOCK:
        return _preview(config)


def _preview(config):
    path = source(config)
    key = hashlib.sha256(("v2-min17:" + json.dumps(config.model_dump(), sort_keys=True) + str(path.stat().st_mtime_ns)).encode()).hexdigest()[:24]
    movie = STORE / "previews" / f"{key}.mp4"
    metadata = movie.with_suffix(".json")
    if metadata.is_file() and movie.is_file():
        return json.loads(metadata.read_text())
    frames, fps = prepare(path, config.height, config.width, config.start_seconds, config.max_frames)
    temp = movie.with_name(f"{key}-{uuid.uuid4().hex}.mp4")
    write_video(temp, frames, fps)
    temp.replace(movie)
    info = {"url": f"/artifacts/previews/{key}.mp4", "frames": len(frames), "fps": fps,
            "height": config.height, "width": config.width, "padded_frames": padded_frames(len(frames))}
    metadata.write_text(json.dumps(info))
    return info


class RunHandler(logging.Handler):
    def __init__(self, callback):
        super().__init__()
        self.callback = callback

    def emit(self, record):
        self.callback("inference", self.format(record), record.levelname)


def execute(run_id):
    global POLICY_MANAGER
    run = RUNS[run_id]
    config = Configuration(**run["config"])
    directory = STORE / "runs" / run_id

    def log(stage, message, level="INFO"):
        with LOCK:
            run["stage"] = stage
            run["logs"].append({"time": datetime.now(timezone.utc).isoformat(), "level": level, "message": message})
            save_run(run)

    handler = RunHandler(log)
    logger = logging.getLogger("flux_action.models.video_vae")
    previous_level = logger.level
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    try:
        with LOCK:
            run["status"] = "running"
            save_run(run)
        if POLICY_MANAGER is not None:
            POLICY_MANAGER.unload()
            POLICY_MANAGER = None
        log("prepare", "Koşu başladı; resize=stretch, RGB, kaynak FPS korunur; eğitim/gradient kapalı.")
        info = preview(config)
        run["preview_url"] = info["url"]
        frames, fps = prepare(source(config), config.height, config.width, config.start_seconds, config.max_frames)
        log("prepare", f"Hazırlandı: {len(frames)} kare, {fps:.3f} FPS, H×W={config.height}×{config.width}")
        result, metrics = BACKENDS[config.backend].reconstruct(frames, directory, log)
        log("metrics", "Metrikler MP4 sıkıştırmasından önce, uint8 RGB rekonstrüksiyon üzerinde hesaplanıyor.")
        square_error = absolute_error = 0.0
        per_frame = []
        for index, (original, reconstructed) in enumerate(zip(frames, result, strict=True)):
            delta = original.astype(np.float32) - reconstructed.astype(np.float32)
            mse = float(np.square(delta).mean())
            mae = float(np.abs(delta).mean())
            square_error += mse
            absolute_error += mae
            per_frame.append({"frame": index, "mse": mse, "mae": mae, "psnr_db": float(10 * np.log10(255**2 / mse)) if mse else None})
        mse = square_error / len(frames)
        metrics.update(psnr_db=float(10 * np.log10(255**2 / mse)) if mse else None,
                       mae=absolute_error / len(frames), mse=mse, frames=len(frames), fps=fps,
                       height=config.height, width=config.width, source=str(source(config)),
                       resize_mode="stretch", metric_domain="uint8 RGB before MP4 compression")
        (directory / "metrics.json").write_text(json.dumps({**metrics, "per_frame": per_frame}, indent=2))
        log("export", "Rekonstrüksiyon H.264 MP4 olarak kaydediliyor.")
        write_video(directory / "reconstruction.mp4", result, fps)
        with LOCK:
            run.update(status="completed", metrics=metrics, output_url=f"/artifacts/runs/{run_id}/reconstruction.mp4",
                       artifacts={name: f"/artifacts/runs/{run_id}/{name}" for name in ("reconstruction.mp4", "latents.safetensors", "metrics.json", "run.json")})
        psnr = f"{metrics['psnr_db']:.3f}" if metrics["psnr_db"] is not None else "∞"
        log("completed", f"Tamamlandı; PSNR={psnr} dB; MAE={metrics['mae']:.3f}")
    except Exception as exc:
        with LOCK:
            run.update(status="failed", error=str(exc))
        log("failed", traceback.format_exc(), "ERROR")
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass
    finally:
        logger.removeHandler(handler)
        logger.setLevel(previous_level)


@app.get("/")
def index():
    return FileResponse(Path(__file__).parent / "static/index.html")


@app.get("/api/catalog")
def catalog():
    with LOCK:
        entries = [{"id": item["id"], "label": item["label"], "url": f"/api/videos/{item['id']}"} for item in VIDEOS.values()]
    return {"videos": entries, "default_video": entries[0]["id"] if entries else None,
            "backends": [{"id": key, "label": value.label} for key, value in BACKENDS.items()],
            "model": {"weights": str(WEIGHTS), "device": "RTX 5090 / CUDA · bfloat16"}, "resolution_order": "height × width"}


@app.get("/api/videos/{video_id}")
def video(video_id: str):
    if video_id not in VIDEOS:
        raise HTTPException(404, "Video bulunamadı.")
    return FileResponse(VIDEOS[video_id]["path"], media_type="video/mp4")


@app.post("/api/preview")
def create_preview(config: Configuration):
    try:
        return preview(config)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


@app.get("/api/runs")
def list_runs():
    with LOCK:
        return [json.loads(json.dumps(run)) for run in sorted(RUNS.values(), key=lambda r: r["created_at"], reverse=True)]


@app.get("/api/runs/{run_id}")
def get_run(run_id: str):
    with LOCK:
        if run_id not in RUNS:
            raise HTTPException(404, "Koşu bulunamadı.")
        return json.loads(json.dumps(RUNS[run_id]))


@app.post("/api/runs", status_code=202)
def create_run(config: Configuration):
    source(config)
    with LOCK:
        if sum(r["status"] in ("queued", "running") for r in RUNS.values()) >= 4:
            raise HTTPException(429, "GPU kuyruğu dolu; mevcut koşuların tamamlanmasını bekleyin.")
        run_id = uuid.uuid4().hex[:12]
        (STORE / "runs" / run_id).mkdir()
        run = {"id": run_id, "manifest_version": 2, "task_mode": "vae_roundtrip", "status": "queued", "stage": "queued", "config": config.model_dump(),
               "created_at": datetime.now(timezone.utc).isoformat(), "logs": [], "metrics": {}, "artifacts": {},
               "source_url": f"/api/videos/{config.video_id}", "preview_url": None, "output_url": None, "error": None}
        RUNS[run_id] = run
        save_run(run)
        response = json.loads(json.dumps(run))
    WORKER.submit(execute, run_id)
    return response


@app.get("/artifacts/{name:path}")
def artifact(name: str):
    path = (STORE / name).resolve()
    if not path.is_relative_to(STORE.resolve()) or not path.is_file() or path.suffix not in (".mp4", ".json", ".safetensors", ".png", ".csv", ".npz"):
        raise HTTPException(404, "Dosya bulunamadı.")
    return FileResponse(path)


@app.post("/api/uploads")
async def upload(file: UploadFile = File(...)):
    key = f"upload-{uuid.uuid4().hex[:12]}"
    path = STORE / "uploads" / f"{key}.mp4"
    size = 0
    try:
        with path.open("wb") as out:
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                if size > 512 * 1024**2:
                    raise HTTPException(413, "Video sınırı 512 MiB.")
                out.write(chunk)
        import av
        with av.open(str(path)) as container:
            if not container.streams.video:
                raise ValueError("Video akışı yok.")
            next(container.decode(video=0))
        label = f"Yüklenen · {file.filename}"
        path.with_suffix(".json").write_text(json.dumps({"label": label}, ensure_ascii=False))
        with LOCK:
            VIDEOS[key] = {"id": key, "label": label, "path": path}
        return {"id": key, "label": VIDEOS[key]["label"], "url": f"/api/videos/{key}"}
    except HTTPException:
        path.unlink(missing_ok=True)
        raise
    except Exception as exc:
        path.unlink(missing_ok=True)
        raise HTTPException(422, f"Video okunamadı: {exc}") from exc
    finally:
        await file.close()


def policy_dataset():
    global POLICY_DATASET
    if POLICY_DATASET is None:
        from .datasets import Dex3Dataset, INDEX, DATASET as ORIGINAL_DATASET
        POLICY_DATASET = Dex3Dataset(os.environ.get('FLUX_EVAL_POLICY_INDEX', INDEX),
                                     os.environ.get('FLUX_EVAL_POLICY_DATASET', ORIGINAL_DATASET))
    return POLICY_DATASET


def policy_url(directory, name):
    return f'/artifacts/{directory.relative_to(STORE).as_posix()}/{name}'


def policy_snapshot(config, directory):
    from PIL import Image
    from .normalization import normalize
    from .policy import DEPLOY, FT_CHECKPOINT
    dataset = policy_dataset()
    observation = dataset.observation(config.episode_id, config.start_frame)
    normalized, clipping = normalize(observation['state'], 'state', FT_CHECKPOINT)
    Image.fromarray(observation['image']).save(directory / 'original.png')
    import av
    resized = av.VideoFrame.from_ndarray(observation['image'], format='rgb24').to_ndarray(format='rgb24', width=256, height=192)
    Image.fromarray(resized).save(directory / 'input.png')
    available = dataset.reference_length(config.episode_id, config.start_frame, 32) == 32
    episode = dataset.describe(config.episode_id)
    info = {'episode': episode, 'frame': config.start_frame, 'timestamp': config.start_frame / dataset.fps,
            'prompt': config.prompt, 'dataset_prompt': episode['prompt'],
            'state': observation['state'].tolist(), 'normalized_state': normalized.tolist(),
            'joint_names': dataset.joint_names, 'original_url': policy_url(directory, 'original.png'),
            'image_url': policy_url(directory, 'input.png'), 'reference_available': available,
            'validation_error': '' if available or config.task_mode == 'policy_dream' else 'Tek chunk için 32 gelecek kare gerekir.',
            'effective_config': DEPLOY, 'clipping': clipping,
            'provenance': {'manifest': str(dataset.index / 'manifest.json'), 'rows': str(dataset.index / 'rows.f32.npy'),
                           'row_index': observation['row_index'], 'video_frame': observation['video_frame'],
                           'camera': episode['camera'], 'dataset_fingerprint': dataset.fingerprint(config.episode_id), 'timeline': 'input t; future RGB t+1..t+32; actions t..t+31',
                           'conditioning': 'Only selected RGB and measured state; future data is reference only'}}
    (directory / 'input_snapshot.json').write_text(json.dumps(info, ensure_ascii=False, indent=2, allow_nan=False))
    np.savez_compressed(directory / 'input.npz', image=resized, state=observation['state'], normalized_state=normalized)
    return info


@app.get('/api/policy/catalog')
def policy_catalog():
    from .policy import BASE_POLICY, FT_CHECKPOINT, DEPLOY
    try:
        dataset = policy_dataset()
        episodes = dataset.catalog()
        if not episodes:
            raise ValueError('Policy episode kataloğu boş.')
        default = next((e['id'] for e in episodes if 'PickApple' in e['dataset'] and e['source_episode'] == 181), episodes[0]['id'])
        return {'episodes': episodes, 'default_episode': default, 'default_frame': 150,
                'joint_names': dataset.joint_names, 'effective_config': DEPLOY,
                'models': [{'id': 'base', 'label': 'Base · pretrained trunk + zero G1 heads', 'path': str(BASE_POLICY)},
                           {'id': 'ft', 'label': 'Fine-tune · raw LoRA checkpoint-2500', 'path': str(FT_CHECKPOINT)}]}
    except (ValueError, OSError) as exc:
        raise HTTPException(503, f'Policy dataset hazır değil: {exc}') from exc


@app.get('/api/policy/episodes/{episode_id}/video')
def policy_source_video(episode_id: str):
    try:
        dataset = policy_dataset()
        episode = dataset.episode(episode_id)
        key = hashlib.sha256((str(dataset.index) + json.dumps(episode, sort_keys=True)).encode()).hexdigest()[:24]
        movie = STORE / 'previews' / f'episode-{key}.mp4'
        with PREVIEW_LOCK:
            if not movie.is_file():
                frames = dataset.read_frames(episode_id, 0, episode['n_frames'])
                temp = movie.with_name(f'{movie.stem}-{uuid.uuid4().hex}.mp4')
                write_video(temp, frames, dataset.fps)
                temp.replace(movie)
        return FileResponse(movie, media_type='video/mp4')
    except (ValueError, OSError) as exc:
        raise HTTPException(422, str(exc)) from exc


@app.post('/api/policy/preview')
def policy_preview(config: PolicyConfiguration):
    try:
        with PREVIEW_LOCK:
            dataset = policy_dataset()
            dataset.validate_frame(config.episode_id, config.start_frame)
            key = hashlib.sha256(json.dumps({'config': config.model_dump(), 'dataset': dataset.fingerprint(config.episode_id)}, sort_keys=True).encode()).hexdigest()[:24]
            directory = STORE / 'previews' / f'policy-{key}'
            directory.mkdir(exist_ok=True)
            return policy_snapshot(config, directory)
    except (ValueError, OSError) as exc:
        raise HTTPException(422, str(exc)) from exc


@app.post('/api/policy/runs', status_code=202)
def create_policy_run(config: PolicyConfiguration):
    try:
        dataset = policy_dataset()
        dataset.validate_frame(config.episode_id, config.start_frame)
        if config.task_mode == 'policy_window':
            dataset.validate_window(config.episode_id, config.start_frame)
        with LOCK:
            if sum(r['status'] in ('queued', 'running') for r in RUNS.values()) >= 4:
                raise HTTPException(429, 'GPU kuyruğu dolu.')
        run_id = uuid.uuid4().hex[:12]
        directory = STORE / 'runs' / run_id
        directory.mkdir()
        try:
            snapshot = policy_snapshot(config, directory)
        except Exception:
            import shutil
            shutil.rmtree(directory)
            raise
        with LOCK:
            if sum(r['status'] in ('queued', 'running') for r in RUNS.values()) >= 4:
                import shutil
                shutil.rmtree(directory)
                raise HTTPException(429, 'GPU kuyruğu dolu.')
            run = {'id': run_id, 'manifest_version': 2, 'task_mode': config.task_mode,
                   'status': 'queued', 'stage': 'queued', 'config': config.model_dump(),
                   'created_at': datetime.now(timezone.utc).isoformat(), 'logs': [], 'metrics': {},
                   'artifacts': {}, 'results': {}, 'input_snapshot': snapshot, 'error': None,
                   'progress': {'model': None, 'chunk': 0, 'total': 1 if config.task_mode == 'policy_window' else config.chunks},
                   'reference_url': None}
            RUNS[run_id] = run
            save_run(run)
            response = json.loads(json.dumps(run))
        WORKER.submit(execute_policy, run_id)
        return response
    except (ValueError, OSError) as exc:
        raise HTTPException(422, str(exc)) from exc


@app.post('/api/runs/{run_id}/cancel')
def cancel_run(run_id: str):
    with LOCK:
        if run_id not in RUNS:
            raise HTTPException(404, 'Koşu bulunamadı.')
        run = RUNS[run_id]
        if run.get('task_mode', 'vae_roundtrip') == 'vae_roundtrip':
            raise HTTPException(422, 'İptal policy koşuları için kullanılabilir.')
        if run['status'] in ('queued', 'running'):
            CANCELLED.add(run_id)
            run['cancel_requested'] = True
            save_run(run)
        return json.loads(json.dumps(run))


def execute_policy(run_id):
    global POLICY_MANAGER
    from .policy import FluxJointPolicy, FT_CHECKPOINT
    from .normalization import normalize
    from .policy_metrics import action_metrics, video_metrics, feedback_state
    import csv
    import gc
    import torch
    from PIL import Image
    run = RUNS[run_id]
    config = PolicyConfiguration(**run['config'])
    directory = STORE / 'runs' / run_id

    def log(stage, message, level='INFO'):
        with LOCK:
            run['stage'] = stage
            run['logs'].append({'time': datetime.now(timezone.utc).isoformat(), 'level': level, 'message': message})
            save_run(run)

    def cancelled():
        with LOCK:
            return run_id in CANCELLED

    try:
        if cancelled():
            with LOCK:
                run['status'] = 'cancelled'
            log('cancelled', 'Kuyruktaki koşu iptal edildi.')
            return
        with LOCK:
            run['status'] = 'running'
            save_run(run)
        for backend in BACKENDS.values():
            if getattr(backend, 'model', None) is not None:
                backend.model = None
        gc.collect()
        torch.cuda.empty_cache()
        log('model', 'Tek GPU worker; base ve raw FT aynı backbone üzerinde sırayla çalışır.')
        dataset = policy_dataset()
        if dataset.fingerprint(config.episode_id) != run['input_snapshot']['provenance']['dataset_fingerprint']:
            raise ValueError('Dataset files changed after submission; create a fresh input snapshot.')
        if POLICY_MANAGER is None:
            POLICY_MANAGER = FluxJointPolicy.load(log=log)
        POLICY_MANAGER.log = log
        total = 1 if config.task_mode == 'policy_window' else config.chunks
        reference = dataset.reference(config.episode_id, config.start_frame, total * 32)
        normalized_reference, reference_clipping = normalize(reference['actions'], 'action', FT_CHECKPOINT)
        if len(reference['frames']):
            write_video(directory / 'reference.mp4', reference['frames'], 30)
            with LOCK:
                run['reference_url'] = policy_url(directory, 'reference.mp4')
        log('reference', f"Kayıtlı referans: {reference['steps']} steps; dataset prompt={run['input_snapshot']['dataset_prompt']!r}; run prompt={config.prompt!r}")
        if config.task_mode == 'policy_dream':
            log('dream', 'Son decode RGB + son fiziksel-birim absolute action → sonraki input. State=command idealized varsayımı; prompt sabit, her chunk seed=0.')
        initial = np.load(directory / 'input.npz')
        for variant in config.models:
            if cancelled():
                break
            POLICY_MANAGER.reset()
            image, state = initial['image'].copy(), initial['state'].copy()
            frames, actions, normalized_actions, chunks = [], [], [], []
            model_dir = directory / variant
            model_dir.mkdir(exist_ok=True)
            torch.cuda.reset_peak_memory_stats()
            for chunk in range(total):
                if cancelled():
                    break
                with LOCK:
                    run['progress'] = {'model': variant, 'chunk': chunk + 1, 'total': total}
                log('sampling', f'{variant} · chunk {chunk + 1}/{total} · seed=0')
                result = POLICY_MANAGER.predict(image, state, config.prompt, variant=variant, seed=0)
                rgb, predicted = result['frames'], result['actions']
                if rgb.shape != (32,192,256,3) or rgb.dtype != np.uint8 or not np.isfinite(predicted).all():
                    raise ValueError('Policy returned invalid RGB/action output')
                chunk_dir = model_dir / f'chunk-{chunk:03d}'
                chunk_dir.mkdir()
                Image.fromarray(image).save(chunk_dir / 'input.png')
                Image.fromarray(rgb[-1]).save(chunk_dir / 'last.png')
                next_state = feedback_state(predicted)
                np.savez_compressed(chunk_dir / 'chunk.npz', input_state=state,
                                    normalized_state=result['normalized_state'], actions=predicted,
                                    normalized_actions=result['normalized_actions'], latents=result['latents'],
                                    observed_latent=result['latents'][:, :, :1], future_latents=result['latents'][:, :, 1:],
                                    output_state=next_state)
                _, input_clipping = normalize(state, 'state', FT_CHECKPOINT)
                entry = {'index': chunk, 'input_frame_url': policy_url(chunk_dir, 'input.png'),
                         'output_frame_url': policy_url(chunk_dir, 'last.png'),
                         'input_state': state.tolist(), 'output_state': next_state.tolist(),
                         'seed': 0, 'timings': result['timings'], 'clipping': {**result['clipping'], 'input': input_clipping},
                         'metadata': result['metadata'], 'artifact_url': policy_url(chunk_dir, 'chunk.npz')}
                (chunk_dir / 'chunk.json').write_text(json.dumps(entry, ensure_ascii=False, indent=2, allow_nan=False))
                chunks.append(entry)
                frames.append(rgb)
                actions.append(predicted)
                normalized_actions.append(result['normalized_actions'])
                image, state = rgb[-1].copy(), next_state
                log('chunk', f"{variant} chunk {chunk + 1}: {result['timings']['total_seconds']:.2f}s; normalization={input_clipping}; action clamping=false")
                del result
            if not frames:
                continue
            movie_frames, predicted, sampler_normalized = np.concatenate(frames), np.concatenate(actions), np.concatenate(normalized_actions)
            normalized, predicted_clipping = normalize(predicted, 'action', FT_CHECKPOINT)
            metrics = {'actions': action_metrics(predicted, reference['actions'], normalized, normalized_reference),
                       'video': video_metrics(movie_frames, reference['frames']),
                       'frames': len(movie_frames), 'chunks': len(chunks),
                       'peak_vram_gb': torch.cuda.max_memory_allocated() / 2**30,
                       'total_seconds': sum(c['timings']['total_seconds'] for c in chunks),
                       'reference_clipping': reference_clipping, 'predicted_clipping': predicted_clipping,
                       'normalized_metric_domain': 'Both trajectories use checkpoint quantile normalization and its saved clip; physical actions remain unclamped',
                       'reference_available_steps': reference['steps']}
            write_video(model_dir / 'prediction.mp4', movie_frames, 30)
            action_data = {'joint_names': dataset.joint_names, 'fps': 30, 'predicted': predicted.tolist(),
                           'reference': reference['actions'].tolist(), 'normalized_predicted': normalized.tolist(),
                           'normalized_reference': normalized_reference.tolist(), 'sampler_normalized_predicted': sampler_normalized.tolist(), 'chunks': chunks}
            (model_dir / 'actions.json').write_text(json.dumps(action_data, allow_nan=False))
            (model_dir / 'metrics.json').write_text(json.dumps(metrics, indent=2, allow_nan=False))
            np.savez_compressed(model_dir / 'actions.npz', predicted=predicted, reference=reference['actions'],
                                normalized_predicted=normalized, normalized_reference=normalized_reference,
                                sampler_normalized_predicted=sampler_normalized)
            with (model_dir / 'actions.csv').open('w', newline='') as output:
                writer = csv.writer(output)
                writer.writerow(['step', 'seconds', *dataset.joint_names])
                for step, row in enumerate(predicted):
                    writer.writerow([step, step / 30, *row.tolist()])
            with LOCK:
                run['results'][variant] = {'video_url': policy_url(model_dir, 'prediction.mp4'),
                                          'actions_url': policy_url(model_dir, 'actions.json'), 'metrics': metrics, 'chunks': chunks}
                run['metrics'][variant] = metrics
                for path in model_dir.rglob('*'):
                    if path.is_file():
                        name = path.relative_to(directory).as_posix()
                        run['artifacts'][name] = policy_url(path.parent, path.name)
                save_run(run)
        with LOCK:
            run['status'] = 'cancelled' if cancelled() else 'completed'
            for name in ('run.json', 'input_snapshot.json', 'input.npz', 'input.png', 'original.png', 'reference.mp4'):
                if (directory / name).is_file():
                    run['artifacts'][name] = policy_url(directory, name)
        log(run['status'], 'Koşu chunk sınırında iptal edildi; tamamlanan çıktılar saklandı.' if cancelled() else 'Policy değerlendirmesi tamamlandı.')
    except Exception as exc:
        with LOCK:
            run.update(status='failed', error=str(exc))
        log('failed', traceback.format_exc(), 'ERROR')
        if POLICY_MANAGER is not None:
            POLICY_MANAGER.unload()
            POLICY_MANAGER = None
        torch.cuda.empty_cache()
    finally:
        with LOCK:
            CANCELLED.discard(run_id)
