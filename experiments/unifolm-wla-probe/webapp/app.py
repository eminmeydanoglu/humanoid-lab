#!/usr/bin/env python3
"""Small web UI for the UnifoLM ER checkpoints.

    venvs/unifolm-wla/bin/python webapp/app.py --port 8321

Serves a single page: pick one of the extracted G1 head-camera frames, type a
prompt, run the model, read the answer. In point mode the coordinates the model
prints are drawn back onto the frame, because "the model says (816, 512)" is not
something anyone can check by eye.

Inference is serialised through one worker thread: the GPU holds one 4B model and
one request at a time, so a browser refresh cannot corrupt a run.
"""

import argparse
import glob
import json
import mimetypes
import os
import pathlib
import queue
import threading
import time
import uuid

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from PIL import Image

import points as points_mod
from engine import DEFAULT_MAX_NEW_TOKENS, ModelRunner

ROOT = pathlib.Path(__file__).resolve().parent
STATIC_DIR = ROOT / "static"

# The bare instruction "Point to the apple in the image." was tested and never yields a
# coordinate -- the model answers "Counting the apple shows a total of 1." All four
# phrasings here returned a coordinate pair in both test frames.
POINT_TEMPLATES = {
    "point_center": "Give the center point of the {target} in the image as [(x, y)].",
    "bbox": "Locate the {target} in the image and output its bounding box.",
    "grasp": "Where should the robot grasp the {target}? Give the grasp point.",
    "point_where": "Where is the {target}? Answer with the pixel coordinate.",
}

# Short labels for the dropdown; the full sentence being sent is shown under it, so the
# select does not have to carry the whole phrasing on a narrow screen.
POINT_TEMPLATE_LABELS = {
    "point_center": "Merkez noktası",
    "bbox": "Bounding box",
    "grasp": "Tutma noktası (grasp)",
    "point_where": "Where is? (koordinat)",
}


MANIFEST_NAMES = ("manifest.jsonl", "manifest_clean4.jsonl")


def load_frames(frames_dir, manifest_dir):
    """Index every PNG in frames_dir, enriched with whatever the manifests know."""
    meta = {}
    for name in MANIFEST_NAMES:
        path = manifest_dir / name
        if not path.exists():
            continue
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            meta.setdefault(pathlib.Path(row["image"]).name, row)

    frames = []
    for path in sorted(frames_dir.glob("*.png")):
        row = meta.get(path.name, {})
        with Image.open(path) as img:
            width, height = img.size
        frames.append(
            {
                "id": path.stem,
                "file": path.name,
                "url": f"/frames/{path.name}",
                "width": width,
                "height": height,
                "fruit": row.get("fruit", ""),
                "instruction": row.get("instruction", ""),
                "episode": row.get("episode"),
            }
        )
    return frames


class App:
    def __init__(self, frames, model_dirs, default_model):
        self.frames = {f["id"]: f for f in frames}
        self.frame_list = frames
        self.models = model_dirs
        self.default_model = default_model
        self.runner = ModelRunner(model_dirs)
        self.jobs = {}
        self.jobs_lock = threading.Lock()
        self.tasks = queue.Queue()
        self.worker = threading.Thread(target=self._work, daemon=True)
        self.worker.start()

    # -- job plumbing ---------------------------------------------------------
    def submit(self, kind, request):
        job_id = uuid.uuid4().hex[:12]
        job = {
            "id": job_id,
            "kind": kind,
            "status": "queued",
            "phase": "sırada",
            "request": request,
            "result": None,
            "error": None,
            "created": time.time(),
        }
        with self.jobs_lock:
            self.jobs[job_id] = job
        self.tasks.put(job_id)
        return job

    def job(self, job_id):
        with self.jobs_lock:
            return self.jobs.get(job_id)

    def _update(self, job_id, **fields):
        with self.jobs_lock:
            job = self.jobs.get(job_id)
            if job is not None:
                job.update(fields)

    def _work(self):
        while True:
            job_id = self.tasks.get()
            job = self.job(job_id)
            if job is None:
                continue
            try:
                self._update(job_id, status="running", phase="model yükleniyor", started=time.time())
                if job["kind"] == "warm":
                    self.runner.ensure(job["request"]["model"])
                    self._update(
                        job_id,
                        status="done",
                        phase="hazır",
                        finished=time.time(),
                        result={"warm": True, **self.runner.status()},
                    )
                else:
                    result = self._generate(job["request"], job_id)
                    self._update(job_id, status="done", phase="bitti", finished=time.time(), result=result)
            except Exception as exc:  # surfaced to the browser instead of dying silently
                self._update(
                    job_id,
                    status="error",
                    phase="hata",
                    finished=time.time(),
                    error=f"{type(exc).__name__}: {exc}",
                )

    def _generate(self, request, job_id):
        frame = self.frames[request["frame_id"]]
        prompt = request["prompt"]
        image = Image.open(frame_path(frame)).convert("RGB")

        if self.runner.resident != request["model"]:
            self.runner.ensure(request["model"])
        self._update(job_id, phase="üretim")
        out = self.runner.generate(
            request["model"],
            image,
            prompt,
            max_new_tokens=request["max_new_tokens"],
            assistant_prefix=request.get("assistant_prefix", ""),
        )

        markers = []
        if request["mode"] == "point":
            parsed = points_mod.extract(out["text"], frame["width"], frame["height"], request["scale"])
            markers = parsed["markers"]
            out["scale_used"] = parsed["mode"]
            out["warnings"] = list(parsed["warnings"])
            out["n_coord_groups"] = parsed["n_raw"]
            if not markers:
                out["warnings"].append(
                    "cevapta koordinat yok: model bu soruya sayı döndürmedi, çizilecek bir şey çıkmadı."
                )
        else:
            out["scale_used"] = None
            out["warnings"] = []
            out["n_coord_groups"] = 0

        out.pop("token_ids", None)
        return {
            **out,
            "model": request["model"],
            "mode": request["mode"],
            "frame_id": frame["id"],
            "frame_url": frame["url"],
            "frame_width": frame["width"],
            "frame_height": frame["height"],
            "markers": markers,
            "finished_at": time.time(),
        }


FRAME_DIR = None  # set in main(), used by frame_path()


def frame_path(frame):
    return FRAME_DIR / frame["file"]


class Handler(BaseHTTPRequestHandler):
    server_version = "UnifoLMPlayer/1.0"
    protocol_version = "HTTP/1.1"
    app = None

    def log_message(self, fmt, *args):
        if self.path.startswith("/api/job") and self.command == "GET":
            return
        print(f"{self.address_string()} {self.command} {self.path}", flush=True)

    # -- helpers --------------------------------------------------------------
    def _send(self, status, body, content_type="application/json; charset=utf-8", extra=None):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for key, value in (extra or {}).items():
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(body)

    def _json(self, status, payload):
        self._send(status, json.dumps(payload, ensure_ascii=False).encode())

    def _file(self, path, content_type=None):
        if not path.exists() or not path.is_file():
            self._json(404, {"error": f"not found: {path.name}"})
            return
        data = path.read_bytes()
        ctype = content_type or mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        if ctype.startswith("text/") or ctype.endswith(("javascript", "json")):
            ctype += "; charset=utf-8"
        self._send(200, data, ctype)

    def _read_json(self):
        length = int(self.headers.get("Content-Length") or 0)
        if not length:
            return {}
        return json.loads(self.rfile.read(length).decode())

    # -- routes ---------------------------------------------------------------
    def do_GET(self):
        path = self.path.split("?", 1)[0]
        app = self.app
        if path in ("/", "/index.html"):
            self._file(STATIC_DIR / "index.html")
        elif path.startswith("/static/"):
            rel = path[len("/static/") :]
            target = (STATIC_DIR / rel).resolve()
            if STATIC_DIR.resolve() not in target.parents and target != STATIC_DIR.resolve():
                self._json(403, {"error": "forbidden"})
                return
            self._file(target)
        elif path.startswith("/frames/"):
            name = pathlib.Path(path[len("/frames/") :]).name
            self._file(FRAME_DIR / name, "image/png")
        elif path == "/api/config":
            self._json(
                200,
                {
                    "frames": app.frame_list,
                    "models": [
                        {"id": mid, "label": label} for mid, label in app.model_labels.items()
                    ],
                    "default_model": app.default_model,
                    "point_templates": POINT_TEMPLATES,
                    "point_template_labels": POINT_TEMPLATE_LABELS,
                    "scale_modes": list(points_mod.SCALE_MODES),
                    "default_max_new_tokens": DEFAULT_MAX_NEW_TOKENS,
                },
            )
        elif path == "/api/status":
            with app.jobs_lock:
                active = [j for j in app.jobs.values() if j["status"] in ("queued", "running")]
            self._json(
                200,
                {
                    **app.runner.status(),
                    "queued": len(active),
                    "queue_size": app.tasks.qsize(),
                },
            )
        elif path.startswith("/api/job/"):
            job = app.job(path[len("/api/job/") :])
            if job is None:
                self._json(404, {"error": "unknown job"})
            else:
                self._json(200, job)
        else:
            self._json(404, {"error": f"no route for {path}"})

    def do_POST(self):
        path = self.path.split("?", 1)[0]
        app = self.app
        try:
            body = self._read_json()
        except Exception as exc:
            self._json(400, {"error": f"bad json: {exc}"})
            return

        if path == "/api/warm":
            model = body.get("model") or app.default_model
            if model not in app.models:
                self._json(400, {"error": f"unknown model {model!r}"})
                return
            job = app.submit("warm", {"model": model})
            self._json(200, {"job_id": job["id"]})
            return

        if path == "/api/generate":
            try:
                error = validate(app, body)
                if error:
                    self._json(400, {"error": error})
                    return
                request = build_request(app, body)
                job = app.submit("generate", request)
            except Exception as exc:
                # a dropped connection hides the traceback from the browser
                self._json(500, {"error": f"{type(exc).__name__}: {exc}"})
                return
            self._json(200, {"job_id": job["id"], "request": request})
            return

        self._json(404, {"error": f"no route for {path}"})


def validate(app, body):
    frame_id = body.get("frame_id")
    if frame_id not in app.frames:
        return "bilinmeyen kare"
    if body.get("model") not in app.models:
        return "bilinmeyen model"
    if body.get("mode") not in ("free", "point"):
        return "kip 'free' ya da 'point' olmalı"
    if body.get("mode") == "point":
        if body.get("template") not in POINT_TEMPLATES:
            return "bilinmeyen point şablonu"
        if body.get("scale") not in points_mod.SCALE_MODES:
            return "bilinmeyen ölçek kipi"
        if not (body.get("target") or "").strip():
            return "hedef nesne boş"
    else:
        prompt = (body.get("prompt") or "").strip()
        if not prompt:
            return "prompt boş"
        if len(prompt) > 2000:
            return "prompt çok uzun (2000 karakter sınırı)"
    tokens = body.get("max_new_tokens", DEFAULT_MAX_NEW_TOKENS)
    if not isinstance(tokens, int) or not 8 <= tokens <= 1024:
        return "max_new_tokens 8..1024 arasında olmalı"
    return None


def build_request(app, body):
    mode = body["mode"]
    if mode == "point":
        template = POINT_TEMPLATES[body["template"]]
        prompt = template.format(target=(body.get("target") or "apple").strip())
    else:
        prompt = body["prompt"].strip()
    return {
        "frame_id": body["frame_id"],
        "model": body["model"],
        "mode": mode,
        "prompt": prompt,
        "user_prompt": (body.get("prompt") or "").strip(),
        "template": body.get("template"),
        "target": (body.get("target") or "").strip(),
        "scale": body.get("scale", "norm1000"),
        "max_new_tokens": body.get("max_new_tokens", DEFAULT_MAX_NEW_TOKENS),
        "assistant_prefix": body.get("assistant_prefix", ""),
    }


def find_model_root(explicit):
    """Locate the directory holding the two checkpoints without hardcoding one layout.

    The probe data lives in /home/aksoy-msi/code/humanoid-lab-main/data while this
    work sits in a different worktree, so the repo-relative guess is only the second
    option: an explicit flag wins, then HUMANOID_DATA_ROOT, then a sibling checkout.
    """
    candidates = []
    if explicit:
        candidates.append(pathlib.Path(explicit))
    if os.environ.get("HUMANOID_DATA_ROOT"):
        candidates.append(pathlib.Path(os.environ["HUMANOID_DATA_ROOT"]) / "models" / "unifolm-wla-1.0")
    repo_root = ROOT.parents[2]
    candidates.append(repo_root / "data" / "models" / "unifolm-wla-1.0")
    candidates.extend(
        pathlib.Path(p) for p in sorted(glob.glob(str(pathlib.Path.home() / "code" / "*" / "data" / "models" / "unifolm-wla-1.0")))
    )
    for candidate in candidates:
        if (candidate / "UnifoLM-ER-1").exists() or (candidate / "UnifoLM-ER-Flow").exists():
            return candidate.resolve()
    raise SystemExit(
        "checkpoint directory not found; pass --model-root. Tried:\n  "
        + "\n  ".join(str(c) for c in candidates)
    )


def main():
    global FRAME_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames-dir", default=None, help="directory of PNG frames")
    ap.add_argument("--model-root", default=None, help="directory holding UnifoLM-ER-1 and UnifoLM-ER-Flow")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8321)
    ap.add_argument("--default-model", default="er-1")
    ap.add_argument("--warm", action="store_true", help="load the default model before serving")
    args = ap.parse_args()

    data_root = find_model_root(args.model_root)
    frames_dir = pathlib.Path(
        args.frames_dir
        or (data_root.parents[1] / "outputs" / "unifolm-wla-probe" / "frames")
    ).resolve()
    FRAME_DIR = frames_dir

    model_dirs = {
        "er-1": data_root / "UnifoLM-ER-1",
        "er-flow": data_root / "UnifoLM-ER-Flow",
    }
    model_dirs = {k: v for k, v in model_dirs.items() if v.exists()}
    if not model_dirs:
        raise SystemExit(f"no checkpoints under {data_root}")
    default_model = args.default_model if args.default_model in model_dirs else next(iter(model_dirs))

    frames = load_frames(frames_dir, frames_dir)
    app = App(frames, model_dirs, default_model)
    app.model_labels = {"er-1": "UnifoLM-ER-1", "er-flow": "UnifoLM-ER-Flow"}
    Handler.app = app

    if args.warm:
        app.submit("warm", {"model": default_model})

    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"{len(frames)} frames from {frames_dir}", flush=True)
    print(f"models: {', '.join(f'{k} -> {v}' for k, v in model_dirs.items())}", flush=True)
    print(f"listening on http://{args.host}:{args.port}/", flush=True)
    httpd.serve_forever()


if __name__ == "__main__":
    main()
