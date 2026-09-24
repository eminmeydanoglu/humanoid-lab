"""Small operator page for one real Unitree VLA session."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse, Response

from humanoid_lab.psi0_bridge.camera import encode_jpeg

from .controller import RobotController, RobotSessionError


def create_app(controller: RobotController) -> FastAPI:
    @asynccontextmanager
    async def lifespan(_app):
        controller.open()
        try:
            yield
        finally:
            controller.close()

    app = FastAPI(title="Unitree VLA", lifespan=lifespan)

    @app.get("/", response_class=HTMLResponse)
    def index():
        return HTMLResponse(PAGE)

    @app.get("/api/options")
    def options():
        return {
            "models": [{"id": m.id, "kind": m.kind} for m in controller.models.values()],
            "tasks": [{"id": k, "prompt": v} for k, v in controller.tasks.items()],
        }

    @app.get("/api/status")
    def status():
        return controller.status()

    @app.get("/api/camera.jpg")
    def camera():
        sample = controller.monitor.frame(max_age_s=0.5)
        if sample is None:
            return JSONResponse({"detail": "robot camera unavailable or stale"}, status_code=503)
        return Response(encode_jpeg(sample.frame), media_type="image/jpeg",
                        headers={"Cache-Control": "no-store"})

    @app.post("/api/select")
    def select(body: dict):
        try:
            if set(body) != {"model", "task"}:
                raise RobotSessionError("select requires model and canonical task IDs")
            return controller.select(body["model"], body["task"])
        except RobotSessionError as exc:
            return JSONResponse({"detail": str(exc)}, status_code=409)

    @app.post("/api/start")
    def start():
        try:
            return controller.start()
        except RobotSessionError as exc:
            return JSONResponse({"detail": str(exc)}, status_code=409)

    @app.post("/api/stop")
    def stop():
        return controller.stop()

    @app.post("/api/reset")
    def reset():
        return controller.reset()

    return app


PAGE = """<!doctype html>
<html lang="tr"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Unitree VLA</title>
<style>
body{margin:0;background:#121820;color:#e8ede9;font:16px system-ui,sans-serif}main{max-width:1120px;margin:auto;padding:24px}
h1{font-size:28px;margin:0 0 6px}p{color:#9faca7}section{background:#1c252d;border:1px solid #35413f;border-radius:12px;padding:18px;margin:18px 0}
.grid{display:grid;grid-template-columns:2fr 1fr;gap:18px}img{width:100%;aspect-ratio:4/3;object-fit:contain;background:#0b1015;border-radius:8px}
label{display:block;margin:12px 0 5px;color:#b9c9c1}select,button{font:inherit;border-radius:7px;padding:9px 12px}select{width:100%;background:#101a20;color:white;border:1px solid #65716d}
button{cursor:pointer;border:0;margin:10px 7px 0 0;background:#9ccab0;color:#102018}button.secondary{background:#c6d0cb}button.danger{background:#e6ae95}
dl{display:grid;grid-template-columns:1fr 1fr;gap:8px}dt{color:#98aaa1}dd{margin:0;text-align:right;overflow-wrap:anywhere}
#error{color:#f1b59e;min-height:24px}#prompt{font-size:14px;line-height:1.45;color:#d1ded4}
@media(max-width:760px){.grid{grid-template-columns:1fr}}
</style>
<main><h1>Unitree VLA</h1><p>Raider çıkarımı · robot SONIC durumu</p><div class="grid">
<section><img id="camera" alt="Robotun canlı renk kamerası"><p id="cameraNote">Kamera bekleniyor</p></section>
<section><label>Model</label><select id="model"></select><label>Görev</label><select id="task"></select>
<p id="prompt"></p><button id="start">Başlat</button><button class="secondary" id="stop">Durdur</button><button class="danger" id="reset">Sıfırla</button>
<p id="error"></p></section></div>
<section><dl><dt>Oturum</dt><dd id="session">—</dd><dt>Robotun bildirdiği mod</dt><dd id="mode">UNKNOWN</dd>
<dt>Mod raporu yaşı</dt><dd id="reportAge">—</dd><dt>Geçerli token yaşı</dt><dd id="tokenAge">—</dd>
<dt>Kamera yaşı</dt><dd id="cameraAge">—</dd><dt>Durum yaşı</dt><dd id="stateAge">—</dd>
<dt>Eylem yaşı</dt><dd id="actionAge">—</dd></dl></section></main>
<script>
const $=id=>document.getElementById(id),fmt=x=>x==null?'—':(x*1000).toFixed(0)+' ms';
let tasks={};
async function api(path,body){let r=await fetch('/api/'+path,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body||{})});let d=await r.json();if(!r.ok)throw Error(d.detail||r.status);return d}
async function options(){let d=await (await fetch('/api/options')).json();for(let m of d.models)$('model').add(new Option(m.id,m.id));for(let t of d.tasks){tasks[t.id]=t.prompt;$('task').add(new Option(t.id,t.id))}showPrompt()}
function showPrompt(){$('prompt').textContent=tasks[$('task').value]||''}
async function refresh(){try{let d=await(await fetch('/api/status',{cache:'no-store'})).json();$('session').textContent=d.session;
 $('mode').textContent=d.sonic.mode;$('reportAge').textContent=fmt(d.sonic.report_age_s);$('tokenAge').textContent=d.sonic.valid_token_age_ms==null?'—':d.sonic.valid_token_age_ms+' ms';
 $('cameraAge').textContent=fmt(d.camera.age_s);$('stateAge').textContent=fmt(d.state.age_s);$('actionAge').textContent=fmt(d.action.last_age_s);
 $('cameraNote').textContent=d.camera.alive?'Canlı RGB kare':'Kamera bayat veya erişilemiyor';$('error').textContent=d.error||d.action.error||'';
 $('model').disabled=$('task').disabled=!['IDLE','STOPPED'].includes(d.session);$('camera').src='/api/camera.jpg?t='+Date.now();
 }catch(e){$('error').textContent=String(e)}}
$('task').onchange=showPrompt;
$('model').onchange=$('task').onchange=async()=>{showPrompt();try{await api('select',{model:$('model').value,task:$('task').value});$('error').textContent=''}catch(e){$('error').textContent=String(e)}};
for(let id of ['start','stop','reset'])$(id).onclick=async()=>{try{await api(id);await refresh()}catch(e){$('error').textContent=String(e)}};
options().then(refresh);setInterval(refresh,500);
</script></html>"""
