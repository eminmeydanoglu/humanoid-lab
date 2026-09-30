"""CPU tests for evaluation preparation and API boundaries."""
import importlib
import json
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient

from humanoid_lab.evaluation.video import prepare, write_video


@pytest.fixture()
def evaluation(tmp_path, monkeypatch):
    dataset = tmp_path / "dataset"
    movie = dataset / "videos/chunk-000/observation.images.egocentric/episode_000000.mp4"
    movie.parent.mkdir(parents=True)
    frames = np.full((8, 64, 96, 3), 100, dtype=np.uint8)
    write_video(movie, frames, 10)
    (dataset / "meta").mkdir()
    (dataset / "meta/episodes.jsonl").write_text(json.dumps({"episode_index": 0, "episode_unique_id": "Unifolm/G1_Dex3_PickApple_Dataset/episode_0"}))
    monkeypatch.setenv("FLUX_EVAL_DATASET", str(dataset))
    monkeypatch.setenv("FLUX_EVAL_OUTPUT", str(tmp_path / "outputs"))
    import humanoid_lab.evaluation.app as module
    module = importlib.reload(module)
    with TestClient(module.app) as client:
        yield module, client, movie
    module.WORKER.shutdown(wait=True)


def test_video_resize_and_range(evaluation):
    _, _, movie = evaluation
    frames, fps = prepare(movie, 96, 128, 0.2, 3)
    assert frames.shape == (3, 96, 128, 3)
    assert fps == 10
    with pytest.raises(ValueError, match="karesi yok"):
        prepare(movie, 64, 96, 100, 0)


def test_preview_and_validation(evaluation):
    _, client, _ = evaluation
    payload = {"video_id": "unifolm-0", "height": 160, "width": 192}
    preview = client.post("/api/preview", json=payload)
    assert preview.status_code == 200
    info = preview.json()
    assert info["frames"] == 8 and info["padded_frames"] == 17
    assert client.get(info["url"], headers={"Range": "bytes=0-63"}).status_code == 206
    assert client.post("/api/preview", json=payload).json() == info
    assert client.post("/api/preview", json={**payload, "width": 100}).status_code == 422
    assert client.post("/api/runs", json={**payload, "height": 128}).status_code == 422
    assert client.post("/api/runs", json=payload, headers={"Origin": "https://untrusted.example"}).status_code == 403
    assert client.post("/api/runs", json={**payload, "backend": "unknown"}).status_code == 422
    assert client.post("/api/runs", json={**payload, "video_id": "unknown"}).status_code == 404
    assert client.get("/artifacts/../missing").status_code == 404


def test_upload(evaluation):
    _, client, movie = evaluation
    bad = client.post("/api/uploads", files={"file": ("bad.mp4", b"invalid", "video/mp4")})
    assert bad.status_code == 422
    response = client.post("/api/uploads", files={"file": ("clip.mp4", movie.read_bytes(), "video/mp4")})
    assert response.status_code == 200
    item = response.json()
    assert client.get(item["url"]).status_code == 200
    assert item["id"] in {v["id"] for v in client.get("/api/catalog").json()["videos"]}


def test_backend_failure_is_persisted(evaluation):
    module, client, _ = evaluation

    class BrokenBackend:
        label = "test"

        def reconstruct(self, *args):
            raise RuntimeError("intentional test failure")

    module.BACKENDS["broken"] = BrokenBackend()
    response = client.post("/api/runs", json={"video_id": "unifolm-0", "backend": "broken"})
    assert response.status_code == 202
    run_id = response.json()["id"]
    for _ in range(100):
        run = client.get(f"/api/runs/{run_id}").json()
        if run["status"] == "failed":
            break
        time.sleep(0.02)
    assert run["status"] == "failed"
    assert "intentional test failure" in run["error"]
    stored = json.loads((module.STORE / "runs" / run_id / "run.json").read_text())
    assert stored["status"] == "failed"
    assert any(entry["level"] == "ERROR" for entry in stored["logs"])
