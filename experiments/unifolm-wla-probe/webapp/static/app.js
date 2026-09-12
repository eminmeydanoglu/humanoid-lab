const state = {
  config: null,
  frameId: null,
  mode: "free",
  running: false,
};

const $ = (id) => document.getElementById(id);

async function api(path, options) {
  const res = await fetch(path, options);
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.error || `HTTP ${res.status}`);
  return data;
}

function init(config) {
  state.config = config;

  const modelSelect = $("model");
  modelSelect.innerHTML = "";
  config.models.forEach((m) => {
    const opt = document.createElement("option");
    opt.value = m.id;
    opt.textContent = m.label;
    modelSelect.appendChild(opt);
  });
  modelSelect.value = config.default_model;

  const templateSelect = $("template");
  templateSelect.innerHTML = "";
  const labels = config.point_template_labels || {};
  Object.keys(config.point_templates).forEach((key) => {
    const opt = document.createElement("option");
    opt.value = key;
    opt.textContent = labels[key] || config.point_templates[key];
    opt.title = config.point_templates[key];
    templateSelect.appendChild(opt);
  });

  $("maxTokens").value = config.default_max_new_tokens;

  const grid = $("frameGrid");
  grid.innerHTML = "";
  config.frames.forEach((frame) => {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "thumb";
    btn.dataset.frameId = frame.id;
    const img = document.createElement("img");
    img.src = frame.url;
    img.alt = frame.id;
    img.loading = "lazy";
    const label = document.createElement("span");
    label.textContent =
      frame.fruit && frame.episode !== null && frame.episode !== undefined
        ? `${frame.fruit} #${frame.episode}`
        : frame.fruit || frame.id;
    btn.title = frame.id;
    btn.append(img, label);
    btn.addEventListener("click", () => selectFrame(frame.id));
    grid.appendChild(btn);
  });

  const preferred = config.frames.find((f) => f.id === "apple_ep000000_f0") || config.frames[0];
  if (preferred) selectFrame(preferred.id);
}

function selectFrame(frameId) {
  state.frameId = frameId;
  document.querySelectorAll(".thumb").forEach((el) => {
    el.classList.toggle("active", el.dataset.frameId === frameId);
  });
  const frame = state.config.frames.find((f) => f.id === frameId);
  if (!frame) return;
  const fruit = frame.fruit || "apple";
  $("prompt").value = frame.instruction || `Pick up the ${fruit} and place it on the plate.`;
  $("target").value = fruit;
  updatePreview();
}

function currentMode() {
  return state.mode;
}

function updatePreview() {
  const template = state.config.point_templates[$("template").value];
  const target = $("target").value.trim() || "…";
  $("promptPreview").textContent = template.replace("{target}", target);
}

function setMode(mode) {
  state.mode = mode;
  document.querySelectorAll("#modeSeg button").forEach((el) => {
    el.classList.toggle("active", el.dataset.mode === mode);
  });
  $("freeBlock").hidden = mode !== "free";
  $("pointBlock").hidden = mode !== "point";
}

async function run() {
  if (state.running) return;
  if (!state.frameId) {
    showError("önce bir fotoğraf seç");
    return;
  }
  const body = {
    frame_id: state.frameId,
    model: $("model").value,
    mode: currentMode(),
    prompt: $("prompt").value,
    target: $("target").value,
    template: $("template").value,
    scale: $("scale").value,
    max_new_tokens: Number($("maxTokens").value) || 96,
    assistant_prefix: $("prefix").value,
  };

  setRunning(true);
  $("resultEmpty").hidden = true;
  $("resultBody").hidden = true;
  $("runHint").textContent = "";

  try {
    const { job_id: jobId } = await api("/api/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const result = await pollJob(jobId);
    render(result);
  } catch (err) {
    showError(err.message);
  } finally {
    setRunning(false);
  }
}

async function pollJob(jobId) {
  const started = Date.now();
  while (true) {
    const job = await api(`/api/job/${jobId}`);
    const seconds = ((Date.now() - started) / 1000).toFixed(0);
    if (job.status === "queued" || job.status === "running") {
      $("runHint").textContent = `${job.phase}… (${seconds}s)`;
      await new Promise((r) => setTimeout(r, 400));
      continue;
    }
    if (job.status === "error") throw new Error(job.error);
    return job.result;
  }
}

function setRunning(running) {
  state.running = running;
  $("run").disabled = running;
  $("run").textContent = running ? "Çalışıyor…" : "Çalıştır";
}

function showError(message) {
  $("resultEmpty").hidden = true;
  $("resultBody").hidden = false;
  $("canvasWrap").hidden = true;
  $("markerList").innerHTML = "";
  $("warnings").innerHTML = "";
  $("meta").innerHTML = "";
  $("answer").innerHTML = "";
  const box = document.createElement("div");
  box.className = "error";
  box.textContent = message;
  $("answer").appendChild(box);
}

function render(result) {
  $("resultBody").hidden = false;

  const meta = [
    ["model", result.model],
    ["süre", `${result.seconds} sn`],
    ["üretilen token", result.n_new_tokens],
    ["token/sn", result.tokens_per_second],
    ["prompt token", result.prompt_tokens],
    ["token sınırı", result.hit_max_new_tokens ? "sınıra dayandı" : "dolmadı"],
  ];
  if (result.scale_used) meta.push(["ölçek kipi", result.scale_used]);
  $("meta").innerHTML = meta
    .map(([k, v]) => `<span>${k}: <b>${escapeHtml(String(v))}</b></span>`)
    .join("");

  const wrap = $("canvasWrap");
  const img = $("resultImage");
  if (img.getAttribute("src") !== result.frame_url) img.src = result.frame_url;
  wrap.hidden = false;

  const answer = result.text_plain !== undefined && result.text_plain.trim()
    ? result.text_plain
    : result.text;
  $("answer").textContent = answer;

  renderPromptBand(result);
  drawMarkers(result);
  renderMarkerList(result.markers || []);

  $("warnings").innerHTML = (result.warnings || [])
    .map((w) => `<div>${escapeHtml(w)}</div>`)
    .join("");
  $("sentPromptPlain").textContent = naturalPrompt(result);
  $("sentPrompt").textContent = result.prompt_sent || "";
  $("runHint").textContent = "";
}

/* The prompt as natural language, without the chat-template scaffolding: on the frame we
   want the instruction a person would read, not `<|im_start|>user …`. */
function naturalPrompt(result) {
  if (result.prompt && result.prompt.trim()) return result.prompt.trim();
  return (result.prompt_sent || "")
    .replace(/<\|[a-z_]+\|>/g, "")
    .replace(/^(user|assistant|system)\s*/i, "")
    .trim();
}

function renderPromptBand(result) {
  const band = $("promptBand");
  const text = naturalPrompt(result);
  $("promptBandText").textContent = text;
  band.hidden = !text;
  band.classList.toggle("long", text.length > 120);
  band.classList.toggle("verylong", text.length > 300);
  band.classList.remove("truncated");
  // The band is capped at a share of the frame; flag it when the text does not fit in full.
  requestAnimationFrame(() => {
    band.classList.toggle("truncated", band.scrollHeight > band.clientHeight + 2);
  });
}

function drawMarkers(result) {
  const svg = $("overlay");
  const width = result.frame_width;
  const height = result.frame_height;
  svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
  svg.innerHTML = "";
  const ns = "http://www.w3.org/2000/svg";

  (result.markers || []).forEach((marker) => {
    const colour = marker.in_bounds ? "#e11d48" : "#f59e0b";
    const dash = marker.in_bounds ? "none" : "6 4";
    const clamp = (v, max) => Math.min(Math.max(v, 0), max);

    if (marker.kind === "box") {
      const [x1, y1, x2, y2] = marker.px;
      const rect = document.createElementNS(ns, "rect");
      rect.setAttribute("x", clamp(Math.min(x1, x2), width));
      rect.setAttribute("y", clamp(Math.min(y1, y2), height));
      rect.setAttribute("width", Math.abs(x2 - x1));
      rect.setAttribute("height", Math.abs(y2 - y1));
      rect.setAttribute("fill", "none");
      rect.setAttribute("stroke", colour);
      rect.setAttribute("stroke-width", 3);
      rect.setAttribute("stroke-dasharray", dash);
      svg.appendChild(rect);
      svg.appendChild(labelNode(ns, marker.label, rect.getAttribute("x"), rect.getAttribute("y"), colour));
      return;
    }

    const cx = clamp(marker.px[0], width);
    const cy = clamp(marker.px[1], height);
    const circle = document.createElementNS(ns, "circle");
    circle.setAttribute("cx", cx);
    circle.setAttribute("cy", cy);
    circle.setAttribute("r", 9);
    circle.setAttribute("fill", "none");
    circle.setAttribute("stroke", colour);
    circle.setAttribute("stroke-width", 3);
    circle.setAttribute("stroke-dasharray", dash);
    svg.appendChild(circle);

    const cross = document.createElementNS(ns, "path");
    cross.setAttribute("d", `M ${cx - 15} ${cy} H ${cx + 15} M ${cx} ${cy - 15} V ${cy + 15}`);
    cross.setAttribute("stroke", colour);
    cross.setAttribute("stroke-width", 2);
    cross.setAttribute("stroke-dasharray", dash);
    svg.appendChild(cross);

    svg.appendChild(labelNode(ns, marker.label, cx + 14, cy - 10, colour));
  });
}

function labelNode(ns, text, x, y, colour) {
  const node = document.createElementNS(ns, "text");
  node.setAttribute("x", x);
  node.setAttribute("y", y);
  node.setAttribute("fill", colour);
  node.setAttribute("font-size", 18);
  node.setAttribute("font-weight", "700");
  node.setAttribute("stroke", "#ffffff");
  node.setAttribute("stroke-width", 4);
  node.setAttribute("paint-order", "stroke");
  node.textContent = text;
  return node;
}

function renderMarkerList(markers) {
  const box = $("markerList");
  box.innerHTML = "";
  if (!markers.length) return;
  markers.forEach((marker) => {
    const pairs = [];
    for (let i = 0; i < marker.raw.length; i += 2) {
      pairs.push(`(${marker.raw[i]}, ${marker.raw[i + 1]})`);
    }
    const pxPairs = [];
    for (let i = 0; i < marker.px.length; i += 2) {
      pxPairs.push(`(${Math.round(marker.px[i])}, ${Math.round(marker.px[i + 1])})`);
    }
    const chip = document.createElement("span");
    chip.className = marker.in_bounds ? "chip" : "chip out";
    chip.textContent = `${marker.label} ${pairs.join(" ")} → px ${pxPairs.join(" ")}`;
    box.appendChild(chip);
  });
}

function escapeHtml(text) {
  return text.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

async function refreshStatus() {
  try {
    const status = await api("/api/status");
    const bits = [];
    bits.push(`<span class="dot ${status.resident ? "" : "busy"}">●</span>`);
    bits.push(status.resident ? `yüklü: <b>${status.resident}</b>` : "model yüklü değil");
    bits.push(`GPU: <b>${status.cuda_allocated_gb} GB</b>`);
    if (status.load_seconds) bits.push(`son yükleme: ${status.load_seconds} sn`);
    if (status.queue_size) bits.push(`sırada: ${status.queue_size}`);
    $("status").innerHTML = bits.join(" · ");
  } catch (err) {
    $("status").textContent = `durum alınamadı: ${err.message}`;
  }
}

document.addEventListener("DOMContentLoaded", async () => {
  document.querySelectorAll("#modeSeg button").forEach((btn) => {
    btn.addEventListener("click", () => setMode(btn.dataset.mode));
  });
  ["template", "target"].forEach((id) => $(id).addEventListener("input", updatePreview));
  $("run").addEventListener("click", run);
  $("prompt").addEventListener("keydown", (event) => {
    if (event.key === "Enter" && (event.metaKey || event.ctrlKey)) run();
  });
  try {
    init(await api("/api/config"));
  } catch (err) {
    showError(`ayarlar alınamadı: ${err.message}`);
  }
  refreshStatus();
  setInterval(refreshStatus, 4000);
});
