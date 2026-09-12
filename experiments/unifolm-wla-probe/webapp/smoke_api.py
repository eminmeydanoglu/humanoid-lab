#!/usr/bin/env python3
"""Smoke test for the running web app: drives the same API the browser uses."""

BASE = "http://localhost:8321"

GROUND_TRUTH = {
    "apple_ep000000_f0": {"fruit": "apple", "center": (165, 198)},
    "pear_ep000074_f0": {"fruit": "pear", "center": (535, 242)},
    "grapes_ep000000_f0": {"fruit": "grapes", "center": (497, 210)},
    "starfruit_ep000111_f0": {"fruit": "starfruit", "center": (195, 400)},
}


def post(path, payload):
    req = urllib.request.Request(
        BASE + path, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}
    )
    return json.load(urllib.request.urlopen(req, timeout=30))


def get(path):
    return json.load(urllib.request.urlopen(BASE + path, timeout=30))


def run(payload):
    job = post("/api/generate", payload)
    while True:
        state = get(f"/api/job/{job['job_id']}")
        if state["status"] in ("done", "error"):
            return state
        time.sleep(0.5)


def main():
    print("== point şablonları, 4 kare ==", flush=True)
    for frame_id, truth in GROUND_TRUTH.items():
        for template in ("bbox", "point", "grasp"):
            state = run(
                {
                    "frame_id": frame_id,
                    "model": "er-1",
                    "mode": "point",
                    "template": template,
                    "target": truth["fruit"],
                    "scale": "norm1000",
                    "max_new_tokens": 64,
                }
            )
            if state["status"] == "error":
                print(f"{frame_id:22s} {template:6s} HATA {state['error']}", flush=True)
                continue
            r = state["result"]
            markers = r["markers"]
            mark = ""
            if markers:
                px = markers[0]["px"]
                dx = round(px[0] - truth["center"][0])
                dy = round(px[1] - truth["center"][1])
                mark = f"px=({round(px[0])},{round(px[1])}) hedef=({truth['center'][0]},{truth['center'][1]}) sapma=({dx:+},{dy:+})"
            print(
                f"{frame_id:22s} {template:6s} n_marker={len(markers)} {mark} | {r['text_plain'].strip()[:70]!r}",
                flush=True,
            )

    print("\n== serbest kip ==", flush=True)
    state = run(
        {
            "frame_id": "apple_ep000000_f0",
            "model": "er-1",
            "mode": "free",
            "prompt": "Do not touch the apple.",
            "max_new_tokens": 64,
        }
    )
    print(json.dumps(state["result"]["text_plain"], ensure_ascii=False), flush=True)
    print(f"markers={len(state['result']['markers'])} süre={state['result']['seconds']}s", flush=True)

    print("\n== model değiştir: er-flow ==", flush=True)
    state = run(
        {
            "frame_id": "pear_ep000074_f0",
            "model": "er-flow",
            "mode": "point",
            "template": "bbox",
            "target": "pear",
            "scale": "norm1000",
            "max_new_tokens": 64,
        }
    )
    print(json.dumps(state["result"]["text_plain"], ensure_ascii=False), flush=True)
    print(f"markers={state['result']['markers']}", flush=True)
    print("status:", json.dumps(get("/api/status")), flush=True)

    print("\n== hata yolları ==", flush=True)
    for bad, why in [
        ({"frame_id": "yok", "model": "er-1", "mode": "free", "prompt": "x"}, "bilinmeyen kare"),
        ({"frame_id": "apple_ep000000_f0", "model": "er-1", "mode": "free", "prompt": "   "}, "boş prompt"),
        ({"frame_id": "apple_ep000000_f0", "model": "er-1", "mode": "point", "template": "yok", "target": "apple"}, "bilinmeyen şablon"),
        ({"frame_id": "apple_ep000000_f0", "model": "yok", "mode": "free", "prompt": "x"}, "bilinmeyen model"),
    ]:
        try:
            post("/api/generate", bad)
            print(f"{why}: HATA BEKLENİYORDU ama kabul edildi", flush=True)
        except urllib.error.HTTPError as exc:
            print(f"{why}: HTTP {exc.code} {exc.read().decode()}", flush=True)


if __name__ == "__main__":
    main()
