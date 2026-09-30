const $ = id => document.getElementById(id);
let catalog, activeId = null, selectedRun = null, generation = 0, busy = false;
const labels = {queued:'Kuyrukta',running:'Çalışıyor',completed:'Tamamlandı',failed:'Hata'};
async function api(path, options = {}) {
  const response = await fetch(path, options);
  const data = await response.json();
  if (!response.ok) {
    const message = Array.isArray(data.detail) ? data.detail.map(e => `${e.loc?.slice(1).join('.') || 'Ayarlar'}: ${e.msg.replace(/^Value error, /, '')}`).join('\n') : (data.detail || 'İstek başarısız.');
    throw new Error(message);
  }
  return data;
}
function notice(message = '') { $('notice').hidden = !message; $('notice').textContent = message; }
function config() {
  return {video_id:$('video').value, backend:$('backend').value, height:Number($('height').value), width:Number($('width').value), start_seconds:Number($('start').value), max_frames:Number($('frames').value)};
}
function player(id, url) {
  const element = $(id);
  if (element.getAttribute('src') === (url || null)) return;
  element.pause();
  if (url) element.src = url; else element.removeAttribute('src');
  element.load();
}
function updateEmpty() {
  $('preview-empty').hidden = !!$('preview-player').getAttribute('src');
  $('output-empty').hidden = !!$('output-player').getAttribute('src');
}
function resetResults() {
  generation++;
  activeId = null; selectedRun = null;
  player('preview-player', null); player('output-player', null);
  $('input-size').textContent = `${$('height').value} × ${$('width').value}`;
  $('preview-info').textContent = 'Ayarlar değişti. Yeni bir önizleme hazırlayın.';
  $('run-title').textContent = 'Yeni bir deney başlat';
  $('status').textContent = 'Hazır'; $('status').className = 'status';
  $('metrics').replaceChildren(...['PSNR','MAE','Encode','Decode'].map(label => metric(label, '—')));
  $('details').textContent = ''; $('artifacts').replaceChildren();
  $('stage').textContent = 'BEKLENİYOR';
  $('logs').textContent = 'Yeni ayarlar için encode / decode çalıştırmaya hazır.';
  notice(); updateEmpty(); refreshHistory();
}
function metric(label, value) {
  const div = document.createElement('div'), small = document.createElement('small'), b = document.createElement('b');
  small.textContent = label; b.textContent = value; div.append(small,b); return div;
}
function render(run) {
  selectedRun = run;
  $('run-title').textContent = `${run.config.height} × ${run.config.width} · ${run.id}`;
  $('input-size').textContent = `${run.config.height} × ${run.config.width}`;
  $('status').textContent = labels[run.status] || run.status;
  $('status').className = `status ${run.status}`;
  $('stage').textContent = run.stage.toUpperCase();
  player('source-player', run.source_url);
  player('preview-player', run.preview_url); player('output-player', run.output_url); updateEmpty();
  const m = run.metrics || {};
  const number = (n, suffix = '') => n == null ? '—' : `${n.toFixed(2)}${suffix}`;
  $('metrics').replaceChildren(metric('PSNR',m.mse === 0 ? '∞ dB' : number(m.psnr_db,' dB')),metric('MAE',number(m.mae)),metric('Encode',number(m.encode_seconds,' s')),metric('Decode',number(m.decode_seconds,' s')));
  $('details').textContent = m.frames ? `${m.frames} kare · ${number(m.fps)} FPS · padding sonrası ${m.padded_frames} kare · GPU peak ${number(m.peak_vram_gb)} GiB\nLatent: [${m.latent_shape.join(', ')}] · ${m.dtype} · decoder window=${m.decoder_max_t}` : '';
  $('preview-info').textContent = m.frames ? `${m.frames} kaynak karesi · ${number(m.fps)} FPS · ${run.config.height} × ${run.config.width}` : 'Koşu için VAE girdisi hazırlanıyor.';
  $('artifacts').replaceChildren();
  Object.entries(run.artifacts || {}).forEach(([name,url]) => {
    const a = document.createElement('a'); a.href = url; a.download = name; a.textContent = `${name} ↓`; $('artifacts').append(a);
  });
  const logs = $('logs'), atEnd = logs.scrollHeight - logs.scrollTop - logs.clientHeight < 40;
  logs.textContent = run.logs.length ? run.logs.map(line => `${line.time.slice(11,19)}  ${line.level.padEnd(7)} ${line.message}`).join('\n') : 'GPU kuyruğunda bekleniyor…';
  if (atEnd) logs.scrollTop = logs.scrollHeight;
  notice(run.error || '');
}
async function loadCatalog(selected) {
  catalog = await api('/api/catalog');
  $('video').replaceChildren(...catalog.videos.map(v => new Option(v.label,v.id)));
  $('video').value = selected || catalog.default_video || '';
  $('backend').replaceChildren(...catalog.backends.map(b => new Option(b.label,b.id)));
  $('model-info').textContent = `Model: ${catalog.model.weights}\nCihaz: ${catalog.model.device}\nEncode: 45-kare parçalar, 1-kare overlap; T=4k+1.\nDecode: 8-kare aktivasyon penceresi.\nRGB normalizasyonu: [-1,1]. Latent: 96 kanal, uzayda 32×, zamanda 4× sıkıştırma.\nResize: stretch. Latent ve metrikler kalıcı diskte saklanır.`;
  const video = catalog.videos.find(v => v.id === $('video').value);
  player('source-player',video?.url);
  $('run').disabled = !video; $('preview').disabled = !video;
}
async function refreshHistory() {
  try {
    const runs = (await api('/api/runs')).filter(run => (run.task_mode || 'vae_roundtrip') === 'vae_roundtrip');
    $('history').replaceChildren();
    if (!runs.length) { $('history').textContent = 'Henüz bir koşu yok.'; return; }
    runs.forEach(run => {
      const button = document.createElement('button'), small = document.createElement('small');
      button.className = run.id === activeId ? 'active' : '';
      button.append(document.createTextNode(`${run.config.height} × ${run.config.width} · ${labels[run.status]}`));
      small.textContent = `${run.id} · ${new Date(run.created_at).toLocaleTimeString('tr-TR')}`;
      button.append(small); button.onclick = () => selectRun(run.id); $('history').append(button);
    });
    return runs;
  } catch (error) { notice(`Koşu geçmişi okunamadı: ${error.message}`); }
}
async function selectRun(id) {
  try {
    generation++;
    const run = await api(`/api/runs/${id}`);
    activeId = id; localStorage.setItem('flux-eval-run',id);
    Object.entries({video:run.config.video_id,backend:run.config.backend,height:run.config.height,width:run.config.width,start:run.config.start_seconds,frames:run.config.max_frames}).forEach(([key,value]) => $(key).value = value);
    const preset = `${run.config.height},${run.config.width}`;
    $('preset').value = ['192,256','288,384'].includes(preset) ? preset : 'custom';
    render(run); await refreshHistory();
  } catch(error) { notice(error.message); }
}
function setBusy(value) {
  busy = value;
  ['preview','run','upload','video','preset','height','width','start','frames','backend'].forEach(id => $(id).disabled = value);
}
$('preset').onchange = () => {
  if ($('preset').value !== 'custom') [$('height').value,$('width').value] = $('preset').value.split(',');
  resetResults();
};
['video','backend','height','width','start','frames'].forEach(id => $(id).addEventListener('change',() => {
  if (['height','width'].includes(id)) $('preset').value = 'custom';
  resetResults();
  if (id === 'video') player('source-player',catalog.videos.find(v => v.id === $('video').value)?.url);
}));
$('preview').onclick = async () => {
  notice(); setBusy(true); const version = generation;
  $('preview-info').textContent = 'Video seçilen çözünürlükte hazırlanıyor…';
  try {
    const result = await api('/api/preview',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(config())});
    if (version === generation) {
      player('preview-player',result.url); updateEmpty();
      $('preview-info').textContent = `${result.frames} kare · ${result.fps.toFixed(2)} FPS · ${result.height} × ${result.width} · VAE padding: ${result.padded_frames-result.frames}`;
    }
  } catch(error) { notice(error.message); $('preview-info').textContent = 'Önizleme hazırlanamadı. Ayarları kontrol edin.'; }
  finally { setBusy(false); }
};
$('run').onclick = async () => {
  notice(); setBusy(true);
  try {
    const run = await api('/api/runs',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(config())});
    activeId = run.id; localStorage.setItem('flux-eval-run',run.id); render(run); await refreshHistory();
  } catch(error) { notice(error.message); }
  finally { setBusy(false); }
};
$('upload').onchange = async () => {
  const file = $('upload').files[0]; if (!file) return;
  notice(); setBusy(true);
  try {
    const form = new FormData(); form.append('file',file);
    const item = await api('/api/uploads',{method:'POST',body:form});
    await loadCatalog(item.id); resetResults();
  } catch(error) { notice(error.message); }
  finally { setBusy(false); $('upload').value = ''; }
};
$('sync').onclick = async () => {
  const players = ['source-player','preview-player','output-player'].map($).filter(v => v.getAttribute('src'));
  if (!players.length) return;
  if (players.some(v => !v.paused)) { players.forEach(v => v.pause()); return; }
  const target = $('output-player').getAttribute('src') ? $('output-player').currentTime : $('preview-player').currentTime;
  const start = selectedRun?.config.start_seconds ?? Number($('start').value);
  players.forEach(v => v.currentTime = target + (v.id === 'source-player' ? start : 0));
  const result = await Promise.allSettled(players.map(v => v.play()));
  if (result.some(r => r.status === 'rejected')) notice('Video henüz hazır değil; yüklenmesini bekleyip yeniden oynatın.');
};
async function poll() {
  try {
    const id = activeId;
    if (id && ['queued','running'].includes(selectedRun?.status)) {
      const run = await api(`/api/runs/${id}`);
      if (id === activeId) render(run);
    }
    await refreshHistory();
  } catch(error) { notice(`Sunucu bağlantısı: ${error.message}`); }
  finally { setTimeout(poll,2000); }
}
(async () => {
  try {
    await loadCatalog(); const runs = await refreshHistory();
    const saved = localStorage.getItem('flux-eval-run');
    if (runs?.some(r => r.id === saved)) await selectRun(saved);
    else {
      const ready = runs?.find(r => r.status === 'completed' && r.config.video_id === catalog.default_video && r.config.height === 192 && r.config.width === 256 && r.config.start_seconds === 0 && r.config.max_frames === 0);
      if (ready) await selectRun(ready.id);
    }
    updateEmpty();
  } catch(error) { notice(`Başlatma hatası: ${error.message}`); }
  poll();
})();
