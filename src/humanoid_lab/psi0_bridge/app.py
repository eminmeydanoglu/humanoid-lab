"""Minimal local web UI and control API for one Ψ₀–SONIC session.

The page has exactly what the operator needs: the task name, an editable prompt
box prefilled with the canonical instruction, ``Start``/``Stop``/``Reset``, the
policy checkpoint selector, the Ψ₀ / SONIC-state / camera status rows, the
timestamp of the last published action, the policy camera preview, and a link
plus instructions for the Isaac WebRTC viewer.  There is deliberately no
success/reward/task-stage display.

The checkpoint selector is an allowlist of the artifacts the launcher built
(fine-tuned + training-start base): ``POST /api/checkpoint`` accepts only an
``{"id": ...}`` body naming one of them -- never a path -- and a switch really
restarts the owned policy server and verifies the served ``/info`` identity
before the session is ready again.

The backend answers ``Start`` only for the exact canonical prompt; an edited or
paraphrased instruction is rejected before it can change the policy condition.
"""

from __future__ import annotations

import html
from typing import Any, Optional

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse, Response

from .camera import CameraError, encode_jpeg
from .checkpoints import (
    CheckpointController,
    CheckpointError,
    CheckpointUnavailable,
    SwitchInProgress,
    UnknownCheckpoint,
)
from .prompt import CANONICAL_PROMPT, PromptMismatch, require_canonical
from .session import Session, SessionError, info_summary

TASK_NAME = "BlockStacking"


def _index_html(prompt: str, task: str, subtitle: str, policy_label: str) -> str:
    return (
        _INDEX_HTML.replace("__TASK__", html.escape(task))
        .replace("__PROMPT__", html.escape(prompt))
        .replace("__SUBTITLE__", html.escape(subtitle))
        .replace("__POLICY_LABEL__", html.escape(policy_label))
    )


def _empty_checkpoint_status() -> dict[str, Any]:
    return {"options": [], "selected": None, "active": None, "serving_selected": False,
            "switching": False, "server": None}


def create_app(
    session: Session,
    *,
    webrtc: Optional[dict[str, Any]] = None,
    checkpoints: Optional[CheckpointController] = None,
    prompt: str = CANONICAL_PROMPT,
    task: str = TASK_NAME,
    subtitle: str = "Ψ₀ checkpoint → SONIC Protocol v4 → Isaac G1 + Dex3",
    policy_label: str = "Ψ₀ server",
    require_canonical_prompt: bool = True,
) -> FastAPI:
    """Build the shared evaluation UI around an already-constructed session."""

    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        session.monitor.start()
        try:
            yield
        finally:
            # The service owns the action socket for its whole lifetime: Stop
            # only closes the send gate, shutdown releases the socket.
            session.stop()
            session.monitor.stop()
            session.close()

    app = FastAPI(title=f"{task} evaluation", lifespan=lifespan)
    webrtc_info = dict(webrtc or {})

    @app.get("/", response_class=HTMLResponse)
    def index() -> HTMLResponse:
        return HTMLResponse(_index_html(prompt, task, subtitle, policy_label))

    @app.get("/api/meta")
    def meta() -> dict[str, Any]:
        return {
            "task": task,
            "prompt": prompt,
            "instruction_keys": ["instruction"],
            "webrtc": webrtc_info,
        }

    @app.get("/api/status")
    def status() -> dict[str, Any]:
        return session.status()

    @app.get("/api/checkpoints")
    def checkpoints_status() -> dict[str, Any]:
        if checkpoints is None:
            return _empty_checkpoint_status()
        if hasattr(checkpoints, "checkpoints_status"):
            return checkpoints.checkpoints_status()
        return checkpoints.status()

    @app.post("/api/checkpoint")
    def checkpoint_select(body: dict[str, Any]) -> Any:
        """Switch the served checkpoint; the body is exactly ``{"id": <allowlisted>}``."""
        if checkpoints is None:
            return JSONResponse({"detail": "checkpoint selection is not configured"}, status_code=503)
        if not isinstance(body, dict) or set(body) != {"id"} or not isinstance(body.get("id"), str):
            return JSONResponse(
                {"detail": 'body must be exactly {"id": "<allowlisted checkpoint id>"}; '
                           "arbitrary paths are not accepted"},
                status_code=400,
            )
        try:
            return checkpoints.switch(body["id"])
        except UnknownCheckpoint as exc:
            return JSONResponse({"detail": str(exc)}, status_code=400)
        except (SwitchInProgress, CheckpointUnavailable) as exc:
            return JSONResponse({"detail": str(exc)}, status_code=409)
        except CheckpointError as exc:
            return JSONResponse({"detail": str(exc)}, status_code=502)

    @app.post("/api/start")
    def start(body: dict[str, Any]) -> Any:
        instruction = body.get("instruction", "") if isinstance(body, dict) else ""
        if require_canonical_prompt:
            try:
                if prompt == CANONICAL_PROMPT:
                    require_canonical(str(instruction))
                elif str(instruction) != prompt:
                    raise PromptMismatch(f"instruction must equal the selected task prompt: {prompt}")
            except PromptMismatch as exc:
                return JSONResponse({"detail": str(exc), "canonical_prompt": prompt}, status_code=400)
        try:
            return session.start()
        except (SessionError, CheckpointError) as exc:
            return JSONResponse({"detail": str(exc)}, status_code=409)

    @app.post("/api/stop")
    def stop() -> dict[str, Any]:
        return session.stop()

    @app.post("/api/reset")
    def reset() -> dict[str, Any]:
        return session.reset()

    @app.get("/api/psi0/info")
    def psi0_info() -> Any:
        try:
            info = session.refresh_info()
        except Exception as exc:  # noqa: BLE001 - surfaced verbatim to the operator
            return JSONResponse({"detail": f"{type(exc).__name__}: {exc}"}, status_code=502)
        return {"info": info if isinstance(info, dict) else info_summary(info)}

    @app.get("/api/camera/frame")
    def camera_frame() -> Any:
        preview = session.preview_frame()
        if preview is None:
            return JSONResponse({"detail": "no camera frame yet"}, status_code=503)
        frame, _timestamp = preview
        try:
            jpeg = encode_jpeg(frame)
        except CameraError as exc:
            return JSONResponse({"detail": str(exc)}, status_code=503)
        return Response(content=jpeg, media_type="image/jpeg", headers={"Cache-Control": "no-store"})

    return app


_INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TASK__ · Psi0-SONIC</title>
<style>
  :root { color-scheme: dark; --bg:#0f1115; --panel:#171a21; --line:#262b36;
          --fg:#e8eaed; --dim:#9aa4b2; --accent:#4f9cf9; --ok:#3fb950; --bad:#f85149; }
  * { box-sizing: border-box; }
  body { margin:0; background:var(--bg); color:var(--fg);
         font:14px/1.5 ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, sans-serif; }
  header { padding:16px 20px; border-bottom:1px solid var(--line);
           display:flex; align-items:baseline; gap:12px; flex-wrap:wrap; }
  header h1 { margin:0; font-size:18px; font-weight:600; }
  header .sub { color:var(--dim); font-size:12px; }
  main { padding:16px 20px; display:grid; gap:16px; max-width:1200px; margin:0 auto;
         grid-template-columns: minmax(0,1fr); }
  .panel { background:var(--panel); border:1px solid var(--line); border-radius:10px; padding:14px; }
  .panel h2 { margin:0 0 10px; font-size:13px; text-transform:uppercase; letter-spacing:.06em; color:var(--dim); }
  textarea { width:100%; min-height:64px; resize:vertical; background:#0c0e12; color:var(--fg);
             border:1px solid var(--line); border-radius:8px; padding:10px; font:inherit; }
  .row { display:flex; gap:8px; flex-wrap:wrap; margin-top:10px; }
  button { background:var(--accent); color:#04101f; border:0; border-radius:8px; padding:10px 16px;
           font:inherit; font-weight:600; cursor:pointer; }
  button.secondary { background:#2b3242; color:var(--fg); }
  button.selected { background:#1f6f43; color:#eafff2; }
  button:disabled { opacity:.5; cursor:not-allowed; }
  #cp-detail { color:var(--dim); font-size:12px; margin-top:10px; word-break:break-all; }
  #cp-detail code { color:var(--fg); }
  #error { color:var(--bad); margin-top:8px; min-height:18px; font-size:13px; }
  dl { margin:0; display:grid; grid-template-columns:auto 1fr; gap:6px 14px; }
  dt { color:var(--dim); }
  dd { margin:0; text-align:right; font-variant-numeric:tabular-nums; }
  .dot { display:inline-block; width:9px; height:9px; border-radius:50%; background:var(--bad); margin-right:6px; }
  .dot.on { background:var(--ok); }
  img#preview { width:100%; height:auto; border-radius:8px; background:#000; display:block; }
  .grid2 { display:grid; gap:16px; }
  @media (min-width: 900px) { main { grid-template-columns: minmax(0,1fr) minmax(0,1fr); }
                              .span2 { grid-column: 1 / -1; } }
  code { background:#0c0e12; padding:1px 5px; border-radius:4px; }
  a { color:var(--accent); }
</style>
</head>
<body>
<header>
  <h1>__TASK__</h1>
  <span class="sub">__SUBTITLE__</span>
</header>
<main>
  <section class="panel">
    <h2>Instruction</h2>
    <textarea id="prompt" spellcheck="false">__PROMPT__</textarea>
    <div class="row">
      <button id="start">Start</button>
      <button id="stop" class="secondary">Stop</button>
      <button id="reset" class="secondary">Reset</button>
    </div>
    <div id="error"></div>
  </section>

  <section class="panel">
    <h2>Model</h2>
    <div class="row" id="cp-options"></div>
    <div id="cp-detail">—</div>
  </section>

  <section class="panel">
    <h2>Status</h2>
    <dl>
      <dt>Session</dt><dd id="session">—</dd>
      <dt>__POLICY_LABEL__</dt><dd id="psi0">—</dd>
      <dt>SONIC g1_debug</dt><dd id="sonic">—</dd>
      <dt>Camera</dt><dd id="camera">—</dd>
      <dt>Last action</dt><dd id="lastaction">—</dd>
    </dl>
  </section>

  <section class="panel span2">
    <h2>Policy camera (last fetched frame, 640×480)</h2>
    <img id="preview" alt="policy camera preview">
  </section>

  <section class="panel span2">
    <h2>Isaac WebRTC</h2>
    <div id="webrtc">—</div>
  </section>
</main>
<script>
const el = (id) => document.getElementById(id);
const setDot = (node, ok, text) => {
  node.innerHTML = '<span class="dot' + (ok ? ' on' : '') + '"></span>' + text;
};
async function post(path, body) {
  const options = { method: 'POST', headers: { 'Content-Type': 'application/json' } };
  if (body !== undefined) options.body = JSON.stringify(body);
  const response = await fetch(path, options);
  const data = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(data.detail || (path + ' failed: HTTP ' + response.status));
  return data;
}
async function tick() {
  try {
    const s = await (await fetch('/api/status')).json();
    const running = s.state === 'RUNNING';
    el('session').textContent = s.state + (s.starting ? ' (starting)' : '');
    const info = s.psi0 && s.psi0.info;
    setDot(el('psi0'), !!(s.psi0 && s.psi0.connected),
      (s.psi0 && s.psi0.connected ? 'connected' : 'disconnected') +
      (info ? ' · ' + info.action_dim + 'D · chunk ' + info.action_chunk_size : ''));
    const st = s.sonic_state || {};
    setDot(el('sonic'), !!st.alive, (st.endpoint || '') + (st.age_s != null ? ' · ' + st.age_s.toFixed(1) + 's' : ''));
    const cam = s.camera || {};
    setDot(el('camera'), !!cam.alive, (cam.endpoint || '') + (cam.shape ? ' · ' + cam.shape.join('×') : ''));
    el('lastaction').textContent = s.action.last_time
      ? s.action.last_time + ' (#' + s.action.last_index + ', ' + s.action.sent + ' sent)'
      : 'none';
    if (s.error) el('error').textContent = s.error;
  } catch (exc) {
    el('error').textContent = String(exc.message || exc);
  }
}
el('start').addEventListener('click', async () => {
  el('error').textContent = '';
  try { await post('/api/start', { instruction: el('prompt').value }); }
  catch (exc) { el('error').textContent = String(exc.message || exc); }
  tick();
});
el('stop').addEventListener('click', async () => {
  el('error').textContent = '';
  try { await post('/api/stop'); } catch (exc) { el('error').textContent = String(exc.message || exc); }
  tick();
});
el('reset').addEventListener('click', async () => {
  el('error').textContent = '';
  try { await post('/api/reset'); } catch (exc) { el('error').textContent = String(exc.message || exc); }
  tick();
});
// Checkpoint selector: only the two allowlisted options the launcher built.
let cpState = null;
function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, (c) => (
    { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}
function renderCheckpoints(cp) {
  const box = el('cp-options');
  cpState = cp;
  box.innerHTML = '';
  (cp.options || []).forEach((opt) => {
    const button = document.createElement('button');
    const isSelected = !!(cp.selected && cp.selected.id === opt.id);
    button.textContent = opt.label;
    button.className = isSelected ? 'selected' : 'secondary';
    button.disabled = !opt.available || !!cp.switching;
    if (!opt.available) button.title = 'unavailable: ' + (opt.reason || 'no artifact');
    button.addEventListener('click', () => selectCheckpoint(opt.id));
    box.appendChild(button);
  });
  if (!(cp.options || []).length) {
    el('cp-detail').textContent = 'no checkpoint list (bridge started without the selector)';
    return;
  }
  const lines = [];
  const selected = cp.selected;
  lines.push('selected: ' + (selected
    ? escapeHtml(selected.label) + ' · <code>' + escapeHtml(selected.run_dir) + '</code> · step ' + selected.step
    : '—'));
  const active = cp.active;
  lines.push('active: ' + (active
    ? escapeHtml(active.label || '') + ' · <code>' + escapeHtml(active.run_dir) + '</code>'
      + (active.step === null || active.step === undefined ? '' : ' · step ' + active.step)
      + (cp.serving_selected ? '' : ' (does not match the selection)')
    : 'no verified policy server'));
  (cp.options || []).forEach((opt) => {
    if (!opt.available) lines.push('unavailable: ' + escapeHtml(opt.label) + ' — ' + escapeHtml(opt.reason || ''));
  });
  if (cp.server) {
    lines.push('policy server: ' + (cp.server.alive ? 'pid ' + cp.server.pid : 'not running')
      + (cp.server.step != null ? ' · step ' + cp.server.step : ''));
  }
  if (cp.switching) lines.push('switching… the policy server restarts; this can take minutes');
  el('cp-detail').innerHTML = lines.join('<br>');
}
async function refreshCheckpoints() {
  try {
    renderCheckpoints(await (await fetch('/api/checkpoints')).json());
  } catch (exc) {
    el('cp-detail').textContent = String(exc.message || exc);
  }
}
async function selectCheckpoint(id) {
  el('error').textContent = '';
  const option = ((cpState && cpState.options) || []).find((candidate) => candidate.id === id);
  el('cp-detail').textContent = 'switching to ' + (option ? option.label : id)
    + ' — the session stops, the policy server restarts; this can take minutes…';
  try {
    await post('/api/checkpoint', { id: id });
  } catch (exc) {
    el('error').textContent = String(exc.message || exc);
  }
  await refreshCheckpoints();
  tick();
}
async function refreshPreview() {
  const image = el('preview');
  const next = new Image();
  next.onload = () => { image.src = next.src; };
  next.src = '/api/camera/frame?t=' + Date.now();
}
async function loadMeta() {
  try {
    const m = await (await fetch('/api/meta')).json();
    const w = m.webrtc || {};
    const parts = [];
    if (w.endpoint) parts.push('Connect the Isaac Sim WebRTC native client to <code>' + w.endpoint + '</code>.');
    if (w.client) parts.push('Client: <code>' + w.client + '</code> (launch it on this host; see <code>./dev.sh webrtc-client</code>).');
    if (w.url) parts.push('<a href="' + w.url + '" target="_blank" rel="noopener">Open WebRTC session</a>');
    el('webrtc').innerHTML = parts.length ? parts.join('<br>') : 'No WebRTC endpoint configured.';
  } catch (exc) { el('webrtc').textContent = String(exc.message || exc); }
}
loadMeta();
tick();
setInterval(tick, 1000);
setInterval(refreshCheckpoints, 1000);
setInterval(refreshPreview, 250);
</script>
</body>
</html>
"""
