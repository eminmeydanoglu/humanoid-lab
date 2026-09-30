(() => {
  'use strict';
  const el = id => document.getElementById(`policy-${id}`);
  const modes = ['policy_window', 'policy_dream'];
  const labels = {queued:'Kuyrukta', running:'Çalışıyor', completed:'Tamamlandı', failed:'Hata', cancelled:'İptal edildi', cancelling:'İptal ediliyor'};
  const live = run => ['queued', 'running', 'cancelling'].includes(run?.status);
  const json = value => JSON.stringify(value, null, 2);
  const numeric = value => typeof value === 'number' && Number.isFinite(value);
  const fmt = value => numeric(value) ? value.toFixed(5) : '—';
  let catalog = null, mode = 'policy_window', revision = 0, preview = null, previewRevision = -1;
  let activeRun = null, selection = 0, historyRequest = 0, catalogRequest = 0, previewRequest = 0;
  let initialized = false, submitting = false, previewing = false, cancelling = false, playing = false, secondsDirty = false;
  let actionData = {}, actionRequests = new Map(), actionErrors = {}, polling = false, playbackRevision = 0;
  const tabs = [...document.querySelectorAll('.evaluation-tabs [role=tab]')];
  const playerIds = ['reference', 'base-video', 'ft-video'];
  const storage = {
    get(key) { try { return localStorage.getItem(key); } catch { return null; } },
    set(key, value) { try { localStorage.setItem(key, value); } catch { /* Storage can be disabled. */ } }
  };
  async function api(path, body) {
    const response = await fetch(path, body === undefined ? {} : {method:'POST', headers:{'Content-Type':'application/json'}, body:json(body)});
    let data;
    try { data = await response.json(); } catch { throw new Error(`Sunucu yanıtı okunamadı (${response.status}).`); }
    if (!response.ok) {
      const detail = data.detail;
      throw new Error(Array.isArray(detail) ? detail.map(item => `${item.loc?.slice(1).join('.') || 'Ayarlar'}: ${item.msg}`).join('\n') : (typeof detail === 'string' ? detail : json(detail || data.error || `HTTP ${response.status}`)));
    }
    return data;
  }
  function notice(message = '') { el('notice').textContent = message; el('notice').hidden = !message; }
  function episode() { return catalog?.episodes?.find(item => String(item.id) === el('episode').value); }
  function config() {
    return {task_mode:mode, episode_id:el('episode').value, start_frame:Number(el('frame').value), prompt:el('prompt').value,
      models:['base', 'ft'].filter(name => el(name).checked), chunks:mode === 'policy_dream' ? Number(el('chunks').value) : 1};
  }
  function validate() {
    const c = config(), item = episode();
    if (secondsDirty) return 'Zaman alanından çıkın; kesin kare indeksi henüz çözümlenmedi.';
    const frameValid = !!el('frame').value && Number.isInteger(c.start_frame) && c.start_frame >= 0 && (!numeric(item?.length) || c.start_frame < item.length);
    el('frame').setAttribute('aria-invalid', String(!frameValid));
    el('prompt').setAttribute('aria-invalid', String(!c.prompt.trim()));
    if (!item) return 'Geçerli bir episode seçin. Kayıtlı episode katalogda yoksa tekrar çalıştırılamaz.';
    if (!frameValid) return `Kare indeksi 0–${numeric(item.length) ? item.length - 1 : 'episode sonu'} aralığında tam sayı olmalı. İndeks otomatik düzeltilmez.`;
    if (!c.prompt.trim()) return 'Komut boş bırakılamaz.';
    if (!c.models.length) return 'En az bir model seçin.';
    if (c.models.some(id => !catalog.models?.some(model => model.id === id))) return 'Seçilen model katalogda bulunmuyor.';
    if (mode === 'policy_dream' && (!el('chunks').value || !Number.isInteger(c.chunks) || c.chunks < 1 || c.chunks > 20)) return 'Rüya uzunluğu 1–20 arasında tam sayı olmalı.';
    return '';
  }
  function updateButtons() {
    const error = validate();
    el('preview').disabled = submitting || previewing || !!error;
    el('run').disabled = submitting || previewing || !!error || previewRevision !== revision || !preview || !!preview.validation_error;
    el('cancel').disabled = cancelling || !live(activeRun) || activeRun.status === 'cancelling';
    el('use-frame').disabled = !episode();
    if (error) el('preview-info').textContent = error;
  }
  function sameConfig(left, right) {
    return ['task_mode', 'episode_id', 'start_frame', 'prompt'].every(key => left[key] === right[key]) &&
      (left.task_mode !== 'policy_dream' || left.chunks === right.chunks) && json(left.models || []) === json(right.models || []);
  }
  function snapshotNote() {
    if (!activeRun) return;
    const c = {...activeRun.config, task_mode:activeRun.task_mode || activeRun.config?.task_mode};
    const stale = !sameConfig(config(), c);
    el('snapshot-note').dataset.stale = String(stale);
    el('snapshot-note').textContent = stale ? 'Ayarlar değişti. Görünen videolar, metrikler ve günlükler aşağıdaki eski koşu snapshot’ına aittir; yeni girdinin sonucu değildir.' : 'Görünen sonuçlar bu koşunun değişmez snapshot’ına aittir. Önizleme koşu çıktısını değiştirmez.';
  }
  function clearPreview() {
    preview = null; previewRevision = -1;
    el('rgb').hidden = true; el('rgb').removeAttribute('src'); el('rgb-empty').hidden = false;
    el('model-rgb').hidden = true; el('model-rgb').removeAttribute('src');
    el('original-link').hidden = true; el('original-link').removeAttribute('href');
    el('rgb-info').textContent = ''; el('state-body').replaceChildren();
    el('settings').textContent = json({effective_config:catalog?.effective_config || {}, models:catalog?.models || []});
    el('preview-badge').textContent = 'Yeniden önizle';
  }
  function changed() {
    revision++; clearPreview(); notice();
    el('preview-info').textContent = 'Girdi değişti. Kesin RGB + state için yeniden önizleyin.';
    updateFrameInfo(); updateButtons(); snapshotNote();
  }
  function setVideo(id, url) {
    const video = el(id);
    if (video.getAttribute('src') === (url || null)) return;
    video.pause();
    if (url) video.src = url; else video.removeAttribute('src');
    video.load();
  }
  function updateFrameInfo() {
    secondsDirty = false;
    el('seconds').setAttribute('aria-invalid', 'false');
    const item = episode(), frame = Number(el('frame').value);
    if (numeric(item?.fps) && item.fps > 0 && Number.isInteger(frame) && el('frame').value) {
      el('seconds').value = String(Number((frame / item.fps).toFixed(9)));
      el('frame-info').textContent = `Kare ${frame} = ${(frame / item.fps).toFixed(6)} sn · ${item.fps} FPS · 0 tabanlı indeks. Sınır dışı indeksler düzeltilmez.`;
    } else { el('seconds').value = ''; el('frame-info').textContent = 'Kare indeksi tam sayı olmalı; zaman için episode FPS bilgisi gerekir.'; }
  }
  function episodeInfo() {
    const item = episode();
    el('episode-info').textContent = item ? `${item.split || 'Split belirtilmedi'} · kaynak ${item.source_episode ?? item.id} · ${item.length ?? '—'} kare · ${item.fps ?? '—'} FPS\nKamera: ${item.camera || '—'}\nDataset task: ${item.prompt || '—'}\nGeçerli başlangıç aralıkları: ${json(item.valid_ranges ?? 'Backend doğrular')}` : 'Episode katalogda bulunmuyor. Kayıtlı sonuçlar snapshot üzerinden incelenebilir.';
  }
  function chooseEpisode(resetPrompt = true) {
    const item = episode(); episodeInfo();
    setVideo('source', item ? `/api/policy/episodes/${encodeURIComponent(item.id)}/video` : null);
    if (resetPrompt) el('prompt').value = item?.prompt || '';
    changed();
  }
  function activate(nextMode, focus = false) {
    const tab = tabs.find(button => button.dataset.mode === nextMode) || tabs[0];
    storage.set('flux-eval-tab', nextMode);
    tabs.forEach(button => { const selected = button === tab; button.setAttribute('aria-selected', String(selected)); button.tabIndex = selected ? 0 : -1; });
    document.getElementById('vae-panel').hidden = nextMode !== 'vae';
    el('panel').hidden = nextMode === 'vae';
    if (focus) tab.focus();
    pausePlayers();
    if (nextMode === 'vae') { selection++; revision++; clearPreview(); updateButtons(); return; }
    el('panel').setAttribute('aria-labelledby', tab.id);
    if (mode !== nextMode) { mode = nextMode; changed(); }
    const dream = mode === 'policy_dream';
    el('heading').innerHTML = dream ? 'Bir komutla,<br>uzun bir rüya.' : 'Bir kareden,<br>bir sonraki harekete.';
    el('description').textContent = dream ? 'Aynı komutla art arda 32 karelik video + action üret. Her model kendi son karesi ve action state’iyle devam eder; fizik simülasyonu yok.' : 'Aynı RGB, state ve komutla base / fine-tuned policy çıktısını karşılaştır. 32 karelik pencere; eğitim veya fizik simülasyonu yok.';
    el('chunks-field').hidden = !dream; el('dream-caveat').hidden = !dream && activeRun?.task_mode !== 'policy_dream';
    if (!initialized) { initialized = true; initialize(); }
    else { refreshHistory(); drawChart(); }
  }
  tabs.forEach((tab, index) => {
    tab.addEventListener('click', () => activate(tab.dataset.mode));
    tab.addEventListener('keydown', event => {
      let next;
      if (event.key === 'ArrowRight') next = (index + 1) % tabs.length;
      if (event.key === 'ArrowLeft') next = (index + tabs.length - 1) % tabs.length;
      if (event.key === 'Home') next = 0;
      if (event.key === 'End') next = tabs.length - 1;
      if (next !== undefined) { event.preventDefault(); activate(tabs[next].dataset.mode, true); }
    });
  });
  async function loadCatalog() {
    const request = ++catalogRequest, version = revision;
    const result = await api('/api/policy/catalog');
    if (request !== catalogRequest || version !== revision) {
      if (request === catalogRequest && !catalog) {
        el('episode-info').textContent = 'Ayarlar katalog yüklenirken değişti. Güncel katalog için Yenile’ye basın.';
      }
      return false;
    }
    catalog = result;
    el('episode').replaceChildren(...(catalog.episodes || []).map(item => new Option(item.label || String(item.id), String(item.id))));
    el('episode').disabled = !catalog.episodes?.length;
    const selected = catalog.default_episode;
    if (selected != null && catalog.episodes?.some(item => String(item.id) === String(selected))) el('episode').value = String(selected);
    el('frame').value = catalog.default_frame ?? 150;
    ['base', 'ft'].forEach(id => {
      const model = catalog.models?.find(item => item.id === id);
      el(id).disabled = !model; el(id).checked = !!model; el(`${id}-label`).textContent = model?.label || `${id} (kullanılamıyor)`;
    });
    chooseEpisode(); populateJoints();
    if (!catalog.episodes?.length) notice('Policy episode kataloğu boş. Dataset ve model ayarlarını kontrol edin.');
    return true;
  }
  function renderPreview(data, validForDraft = true) {
    if (validForDraft) { preview = data; previewRevision = revision; }
    const rgbUrl = data.original_url || data.image_url;
    el('rgb').hidden = !rgbUrl; el('rgb-empty').hidden = !!rgbUrl;
    if (rgbUrl) el('rgb').src = rgbUrl;
    el('model-rgb').hidden = !data.image_url;
    if (data.image_url) el('model-rgb').src = data.image_url;
    el('original-link').hidden = !data.original_url;
    if (data.original_url) el('original-link').href = data.original_url;
    const id = typeof data.episode === 'object' ? data.episode?.id : data.episode;
    el('rgb-info').textContent = `Episode ${id ?? el('episode').value} · kesin kare ${data.frame ?? '—'} · ${numeric(data.timestamp) ? data.timestamp.toFixed(6) : '—'} sn · referans ${data.reference_available ? 'mevcut' : 'yok'}`;
    const names = data.joint_names || catalog?.joint_names || Array.from({length:28}, (_, i) => `joint_${i}`);
    el('state-body').replaceChildren(...names.map((name, i) => {
      const row = document.createElement('tr');
      [name, fmt(data.state?.[i]), fmt(data.normalized_state?.[i])].forEach(text => { const cell = document.createElement('td'); cell.textContent = text; row.append(cell); });
      return row;
    }));
    el('settings').textContent = json({episode:data.episode, frame:data.frame, timestamp:data.timestamp, prompt:data.prompt, dataset_prompt:data.dataset_prompt,
      reference_available:data.reference_available, validation_error:data.validation_error, effective_config:data.effective_config || catalog?.effective_config,
      provenance:data.provenance, clipping:data.clipping, original_url:data.original_url, models:catalog?.models});
    el('preview-badge').textContent = validForDraft ? (data.validation_error ? 'Girdi geçersiz' : 'Kesin girdi hazır') : 'Koşu snapshot’ı';
    el('preview-info').textContent = data.validation_error ? (typeof data.validation_error === 'string' ? data.validation_error : json(data.validation_error)) : (validForDraft ? 'Kesin RGB ve state hazır. Bu ayarlarla koşu başlatabilirsiniz.' : 'Kayıtlı koşu girdisi gösteriliyor; yeniden çalıştırmak için önizleyin.');
    updateButtons();
  }
  async function makePreview() {
    const error = validate(); if (error) { notice(error); return; }
    const version = revision, request = ++previewRequest, c = config();
    previewing = true; notice(); updateButtons(); el('preview-info').textContent = 'Kesin kare, state ve normalizasyon hazırlanıyor…';
    try {
      const result = await api('/api/policy/preview', c);
      if (version === revision && request === previewRequest) {
        if (result.frame !== c.start_frame) throw new Error(`Backend kare ${result.frame} döndürdü; istenen kesin indeks ${c.start_frame}. Koşu engellendi.`);
        const returnedEpisode = typeof result.episode === 'object' ? result.episode?.id : result.episode;
        if (returnedEpisode != null && String(returnedEpisode) !== String(c.episode_id)) throw new Error('Önizleme farklı episode’a ait. Koşu engellendi.');
        if (result.prompt !== c.prompt) throw new Error('Önizleme komutu istenen komutla aynı değil. Koşu engellendi.');
        if (!result.validation_error && (!result.image_url || result.state?.length !== 28 || result.normalized_state?.length !== 28)) throw new Error('Kesin RGB veya 28 eklem state girdisi eksik. Koşu engellendi.');
        renderPreview(result);
      }
    } catch (error) {
      if (version === revision && request === previewRequest) { clearPreview(); notice(error.message); el('preview-info').textContent = 'Önizleme başarısız. Girdi ve geçerli aralıkları kontrol edin.'; }
    } finally { if (request === previewRequest) { previewing = false; updateButtons(); } }
  }
  async function refreshHistory() {
    const request = ++historyRequest, version = revision, selected = selection;
    try {
      const runs = await api('/api/runs');
      if (request !== historyRequest || version !== revision || selected !== selection) return null;
      const policyRuns = runs.filter(run => modes.includes(run.task_mode || run.config?.task_mode));
      el('history').replaceChildren();
      if (!policyRuns.length) el('history').textContent = 'Henüz bir policy koşusu yok.';
      policyRuns.forEach(run => {
        const button = document.createElement('button'), small = document.createElement('small'), c = run.config || {};
        button.className = run.id === activeRun?.id ? 'active' : '';
        button.append(document.createTextNode(`${(run.task_mode || c.task_mode) === 'policy_dream' ? 'Rüya' : 'Tek chunk'} · ${c.episode_id} / kare ${c.start_frame} · ${labels[run.status] || run.status}`));
        small.textContent = `${run.id} · ${run.created_at ? new Date(run.created_at).toLocaleString('tr-TR') : ''}`;
        button.append(small); button.addEventListener('click', () => selectRun(run.id)); el('history').append(button);
      });
      return policyRuns;
    } catch (error) {
      if (request === historyRequest && version === revision && selected === selection) {
        el('history').textContent = `Geçmiş yüklenemedi: ${error.message}. Yenile ile tekrar deneyin.`;
      }
      return null;
    }
  }
  async function selectRun(id) {
    const request = ++selection;
    revision++; clearPreview(); updateButtons(); notice();
    const version = revision;
    try {
      const run = await api(`/api/runs/${encodeURIComponent(id)}`);
      if (request !== selection || version !== revision) return;
      if (!modes.includes(run.task_mode || run.config?.task_mode)) throw new Error('Bu kayıt bir policy koşusu değil.');
      restoreConfig(run); showRun(run); storage.set('flux-policy-run', run.id); refreshHistory();
    } catch (error) { if (request === selection && version === revision) notice(`Koşu açılamadı: ${error.message}`); }
  }
  function restoreConfig(run) {
    const c = run.config || {}, taskMode = run.task_mode || c.task_mode;
    activate(taskMode);
    if (![...el('episode').options].some(option => option.value === String(c.episode_id))) el('episode').add(new Option(`${c.episode_id} · katalogda yok`, c.episode_id));
    el('episode').value = c.episode_id; el('frame').value = c.start_frame; el('prompt').value = c.prompt ?? '';
    el('chunks').value = c.chunks ?? 10;
    ['base', 'ft'].forEach(id => el(id).checked = c.models?.includes(id) || false);
    chooseEpisode(false);
    if (run.input_snapshot) renderPreview(run.input_snapshot, false);
  }
  async function startRun() {
    const error = validate();
    if (error || previewRevision !== revision || !preview || preview.validation_error) { notice(error || 'Önce güncel girdiyi önizleyin.'); return; }
    const c = config(), version = revision, request = ++selection;
    submitting = true; notice(); updateButtons();
    try {
      const run = await api('/api/policy/runs', c);
      if (request === selection) {
        storage.set('flux-policy-run', run.id);
        showRun(run);
        if (version !== revision) snapshotNote();
        refreshHistory();
      }
    } catch (error) { if (request === selection) notice(error.message); }
    finally { submitting = false; updateButtons(); }
  }
  async function cancelRun() {
    if (!live(activeRun) || cancelling) return;
    const id = activeRun.id, request = selection;
    cancelling = true; notice(); updateButtons();
    try {
      await api(`/api/runs/${encodeURIComponent(id)}/cancel`, {});
      const run = await api(`/api/runs/${encodeURIComponent(id)}`);
      if (activeRun?.id === id && request === selection) showRun(run);
      refreshHistory();
    } catch (error) { if (activeRun?.id === id && request === selection) notice(`İptal edilemedi: ${error.message}`); }
    finally { cancelling = false; updateButtons(); }
  }
  function showRun(run) {
    const newRun = activeRun?.id !== run.id;
    if (newRun) {
      pausePlayers(); actionData = {}; actionErrors = {}; actionRequests = new Map();
      el('timeline').value = 0; el('chunk').replaceChildren();
      playerIds.forEach(id => setVideo(id, null));
    }
    activeRun = run;
    const c = run.config || {};
    el('run-title').textContent = `${run.task_mode === 'policy_dream' ? 'Rüya' : 'Tek chunk'} · ${c.episode_id} · kare ${c.start_frame} · ${run.id}`;
    el('status').textContent = labels[run.status] || run.status;
    el('status').className = `status ${run.status}`;
    el('snapshot').textContent = json({id:run.id, task_mode:run.task_mode, config:c, input_snapshot:run.input_snapshot, effective_config:run.effective_config, provenance:run.provenance});
    el('dream-caveat').hidden = mode !== 'policy_dream' && run.task_mode !== 'policy_dream';
    const p = run.progress || {};
    el('progress').max = Number(p.total) > 0 ? Number(p.total) : 1;
    el('progress').value = Number(p.chunk) || 0;
    el('progress-text').textContent = `${p.model || '—'} · chunk ${p.chunk ?? 0} / ${p.total ?? '—'} · ${labels[run.status] || run.status}`;
    setVideo('reference', run.reference_url);
    ['base', 'ft'].forEach(id => setVideo(`${id}-video`, run.results?.[id]?.video_url));
    ['reference', 'base', 'ft'].forEach(id => {
      const vid = el(id === 'reference' ? id : `${id}-video`);
      el(`${id}-empty`).hidden = !!vid.getAttribute('src');
      if (id === 'reference') el('reference-empty').textContent = run.input_snapshot?.reference_available === false || run.status === 'completed' ? 'Bu koşu için referans yok.' : 'Referans hazırlanıyor.';
      else el(`${id}-empty`).textContent = c.models?.includes(id) ? `${id === 'base' ? 'Base' : 'FT'} sonucu bekleniyor.` : 'Bu model koşuda seçilmedi.';
    });
    const logs = el('logs'), atEnd = logs.scrollHeight - logs.scrollTop - logs.clientHeight < 40;
    logs.textContent = (run.logs || []).map(line => typeof line === 'string' ? line : `${line.time?.slice(11, 19) || ''}  ${line.level || 'INFO'}  ${line.message || ''}`).join('\n') || 'GPU kuyruğunda bekleniyor…';
    if (atEnd) logs.scrollTop = logs.scrollHeight;
    el('log-stage').textContent = run.stage || p.model || (labels[run.status] || run.status);
    el('artifacts').replaceChildren();
    const artifacts = {...run.artifacts};
    ['base', 'ft'].forEach(id => { if (run.results?.[id]?.actions_url) artifacts[`${id} actions.json`] = run.results[id].actions_url; });
    Object.entries(artifacts).forEach(([name, url]) => {
      if (typeof url !== 'string') return;
      const link = document.createElement('a'); link.href = url; link.textContent = `${name} ↓`; link.download = name; el('artifacts').append(link);
    });
    if (run.error) notice(typeof run.error === 'string' ? run.error : json(run.error));
    renderMetrics(); renderChunks(); updateTimeline(); snapshotNote(); updateButtons(); loadActions();
  }
  function flatten(value, prefix = '', entries = []) {
    Object.entries(value || {}).forEach(([key, item]) => {
      const name = prefix ? `${prefix} / ${key}` : key;
      if (item && typeof item === 'object' && !Array.isArray(item)) flatten(item, name, entries);
      else entries.push([name, numeric(item) ? item.toFixed(5) : item == null ? '—' : Array.isArray(item) ? `${item.length} değer · tam metriklerde` : String(item)]);
    });
    return entries;
  }
  function renderMetrics() {
    el('metrics').replaceChildren();
    ['base', 'ft'].forEach(id => {
      if (!activeRun?.config.models?.includes(id)) return;
      const section = document.createElement('section'), title = document.createElement('h3'), list = document.createElement('dl');
      title.textContent = `${id === 'base' ? 'Base' : 'Fine-tuned'} · raw + normalized metrikler`;
      const entries = flatten(activeRun.results?.[id]?.metrics);
      if (!entries.length) entries.push(['Metrikler', 'Henüz yok / referans bulunmuyor']);
      entries.forEach(([name, value]) => { const row = document.createElement('div'), dt = document.createElement('dt'), dd = document.createElement('dd'); dt.textContent = name; dd.textContent = value; row.append(dt, dd); list.append(row); });
      const details = document.createElement('details'), summary = document.createElement('summary'), full = document.createElement('pre');
      details.className = 'model-details'; summary.textContent = 'Tam metrikler / eklem bazında'; full.className = 'policy-json';
      full.textContent = json(activeRun.results?.[id]?.metrics || {}); details.append(summary, full);
      section.append(title, list, details); el('metrics').append(section);
    });
  }
  async function loadActions() {
    const id = activeRun?.id, requests = actionRequests;
    ['base', 'ft'].forEach(async model => {
      const url = activeRun?.results?.[model]?.actions_url;
      if (!url || actionRequests.get(model) === url) return;
      actionRequests.set(model, url);
      try {
        const data = await api(url);
        if (activeRun?.id !== id || actionRequests !== requests || activeRun?.results?.[model]?.actions_url !== url) return;
        actionData[model] = data; delete actionErrors[model]; populateJoints(); updateTimeline(); renderChunks(); drawChart();
      } catch (error) {
        if (activeRun?.id === id && actionRequests === requests) { actionErrors[model] = error.message; drawChart(); }
      }
    });
    drawChart();
  }
  function jointNames() { return actionData.ft?.joint_names || actionData.base?.joint_names || catalog?.joint_names || []; }
  function jointGroup(name) {
    const side = /left/i.test(name) ? 'left' : 'right';
    return `${side}_${/hand/i.test(name) ? 'hand' : 'arm'}`;
  }
  function populateJoints() {
    const previous = el('joint').value, group = el('joint-group').value;
    el('joint').replaceChildren(...jointNames().flatMap((name, i) => group === 'all' || jointGroup(String(name)) === group ? [new Option(name, String(i))] : []));
    if ([...el('joint').options].some(option => option.value === previous)) el('joint').value = previous;
    drawChart();
  }
  function frameCount() {
    return activeRun?.task_mode === 'policy_dream' ? (activeRun.config.chunks || 1) * 32 : 32;
  }
  function updateTimeline() {
    el('timeline').max = frameCount() - 1;
    el('timeline').disabled = !players().length && !Object.keys(actionData).length;
    el('play').disabled = !players().length;
    timelineLabel();
  }
  function timelineLabel() {
    const frame = Number(el('timeline').value);
    el('timeline-info').textContent = `${frame} / ${el('timeline').max} · chunk ${Math.floor(frame / 32) + 1} · iç kare ${frame % 32} / 31 · ${(frame / 30).toFixed(3)} sn`;
    const referenceSteps = actionData.ft?.reference?.length ?? actionData.base?.reference?.length ?? activeRun?.results?.ft?.metrics?.reference_available_steps ?? activeRun?.results?.base?.metrics?.reference_available_steps;
    if (numeric(referenceSteps) && el('reference').getAttribute('src')) {
      const unavailable = frame >= referenceSteps;
      el('reference-empty').hidden = !unavailable;
      el('reference-empty').textContent = unavailable ? `Bu anda referans yok. Kayıtlı gelecek ${referenceSteps} karede bitti; model devam ediyor.` : '';
      el('reference').style.opacity = unavailable ? '0.25' : '1';
    } else el('reference').style.opacity = '1';
    if (activeRun?.task_mode === 'policy_dream' && el('chunk').options.length) {
      const chunk = String(Math.floor(frame / 32));
      if (el('chunk').value !== chunk) { el('chunk').value = chunk; chunkInfo(); }
    }
  }
  function players() { return playerIds.map(el).filter(video => video.getAttribute('src')); }
  function pausePlayers() { playbackRevision++; players().forEach(video => video.pause()); playing = false; el('play').textContent = 'Birlikte oynat'; }
  function seek(frame) {
    const time = frame / 30;
    players().forEach(video => { if (video.readyState >= 1) video.currentTime = numeric(video.duration) ? Math.min(time, video.duration) : time; });
    el('timeline').value = frame; timelineLabel(); drawChart();
  }
  function masterPlayer() { return el('ft-video').getAttribute('src') ? el('ft-video') : el('base-video').getAttribute('src') ? el('base-video') : el('reference'); }
  async function togglePlayback() {
    if (playing) { pausePlayers(); return; }
    const videos = players(); if (!videos.length) return;
    const master = masterPlayer();
    if (master.ended) el('timeline').value = 0;
    seek(Number(el('timeline').value));
    const version = ++playbackRevision;
    playing = true; el('play').textContent = 'Birlikte duraklat';
    const results = await Promise.allSettled(videos.filter(video => !video.ended).map(video => video.play()));
    if (version !== playbackRevision) return;
    if (results.some(result => result.status === 'rejected')) { pausePlayers(); notice('Video oynatılamadı. Metadata yüklenmesini bekleyip yeniden deneyin.'); }
  }
  playerIds.forEach(id => {
    const video = el(id);
    video.addEventListener('loadedmetadata', () => { updateTimeline(); if (!playing) seek(Number(el('timeline').value)); });
    video.addEventListener('error', () => { if (video.getAttribute('src')) notice(`${id}: video yüklenemedi. Artefakt bağlantısını ve sunucu günlüklerini kontrol edin.`); });
    video.addEventListener('timeupdate', () => {
      if (!playing || video !== masterPlayer()) return;
      const frame = Math.min(Number(el('timeline').max), Math.floor(video.currentTime * 30));
      el('timeline').value = frame; timelineLabel(); drawChart();
      players().filter(other => other !== video && !other.ended).forEach(other => {
        if (Math.abs(other.currentTime - video.currentTime) > 0.08 && (!numeric(other.duration) || video.currentTime < other.duration)) other.currentTime = video.currentTime;
      });
    });
    video.addEventListener('ended', () => { if (video === masterPlayer()) pausePlayers(); });
  });
  function renderChunks() {
    const dream = activeRun?.task_mode === 'policy_dream';
    el('chunk-browser').hidden = !dream;
    if (!dream) return;
    const previous = el('chunk').value, total = activeRun.config.chunks || 1;
    el('chunk').replaceChildren(...Array.from({length:total}, (_, i) => new Option(`Chunk ${i + 1} · ${i * 32}–${i * 32 + 31}`, String(i))));
    if ([...el('chunk').options].some(option => option.value === previous)) el('chunk').value = previous;
    chunkInfo();
  }
  function chunkInfo() {
    const index = Number(el('chunk').value), info = {chunk:index + 1, output_frames:[index * 32, index * 32 + 31], prompt:activeRun?.config.prompt, state_semantics:'Last predicted command becomes next input state; not measured physics.'};
    ['base', 'ft'].forEach(id => {
      if (!activeRun?.config.models?.includes(id)) return;
      info[id] = {metadata:activeRun.results?.[id]?.chunks?.[index] ?? actionData[id]?.chunks?.[index] ?? 'Chunk henüz üretilmedi',
        last_predicted_command:actionData[id]?.predicted?.[index * 32 + 31] ?? null,
        normalized_last_command:actionData[id]?.normalized_predicted?.[index * 32 + 31] ?? null};
    });
    el('chunk-info').textContent = json(info);
  }
  function drawChart() {
    const canvas = el('chart'), width = canvas.clientWidth;
    if (!width) return;
    const height = 240, ratio = window.devicePixelRatio || 1;
    canvas.width = Math.round(width * ratio); canvas.height = height * ratio;
    const ctx = canvas.getContext('2d'); if (!ctx) return;
    ctx.scale(ratio, ratio); ctx.clearRect(0, 0, width, height);
    const joint = Number(el('joint').value), normalized = el('action-scale').value === 'normalized';
    const start = activeRun?.task_mode === 'policy_dream' ? Number(el('chunk').value) * 32 : 0;
    const series = [];
    const reference = actionData.ft || actionData.base;
    if (reference) series.push({label:'Referans', color:'#68716b', values:(normalized ? reference.normalized_reference : reference.reference)?.slice(start, start + 32)});
    ['base', 'ft'].forEach(id => { if (actionData[id]) series.push({label:id, color:id === 'base' ? '#a66c30' : '#285c43', values:(normalized ? actionData[id].normalized_predicted : actionData[id].predicted)?.slice(start, start + 32)}); });
    const values = series.flatMap(line => (line.values || []).map(row => row?.[joint]).filter(numeric));
    const errors = Object.entries(actionErrors).map(([id, message]) => `${id}: ${message}`).join(' · ');
    if (!values.length || !el('joint').options.length) {
      ctx.fillStyle = '#68716b'; ctx.font = '12px monospace'; ctx.fillText('Action verisi bekleniyor.', 20, 35);
      el('chart-info').textContent = errors || 'Bu ölçekte / eklem grubunda action verisi yok. Normalized veriler backend’den gelir; UI normalizasyon yapmaz.';
      return;
    }
    let min = Math.min(...values), max = Math.max(...values);
    const padding = (max - min || 1) * 0.12; min -= padding; max += padding;
    const left = Math.min(76, width * 0.23), right = width - 16, top = 18, bottom = height - 30;
    const x = i => left + i / 31 * (right - left), y = value => bottom - (value - min) / (max - min) * (bottom - top);
    ctx.font = '10px monospace';
    for (let i = 0; i <= 4; i++) {
      const value = min + (max - min) * i / 4, py = y(value);
      ctx.strokeStyle = '#d8ddd5'; ctx.beginPath(); ctx.moveTo(left, py); ctx.lineTo(right, py); ctx.stroke();
      ctx.fillStyle = '#68716b'; ctx.fillText(value.toFixed(3), 6, py + 3);
    }
    [0, 8, 16, 24, 31].forEach(i => { ctx.fillStyle = '#68716b'; ctx.fillText(String(start + i), x(i) - 6, height - 10); });
    series.forEach(line => {
      ctx.strokeStyle = line.color; ctx.lineWidth = 2; ctx.setLineDash(line.label === 'Referans' ? [4, 3] : []); ctx.beginPath();
      let connected = false;
      (line.values || []).forEach((row, i) => { const value = row?.[joint]; if (!numeric(value)) { connected = false; return; } if (connected) ctx.lineTo(x(i), y(value)); else ctx.moveTo(x(i), y(value)); connected = true; });
      ctx.stroke();
    });
    ctx.setLineDash([]);
    const current = Number(el('timeline').value) - start;
    if (current >= 0 && current < 32) { ctx.strokeStyle = '#242c28'; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(x(current), top); ctx.lineTo(x(current), bottom); ctx.stroke(); }
    const name = jointNames()[joint];
    el('chart-info').textContent = `${name} · ${normalized ? 'normalized' : 'raw'} · çıktı kare ${start}–${start + 31} · ${series.filter(line => line.values?.length).map(line => line.label).join(' / ')}${series.some(line => line.label === 'Referans' && line.values?.length) ? '' : ' · bu chunk’ta referans yok'}${errors ? ` · ${errors}` : ''}`;
    canvas.setAttribute('aria-label', el('chart-info').textContent);
  }
  async function poll() {
    if (polling) return;
    polling = true;
    try {
      if (live(activeRun)) {
        const id = activeRun.id, request = selection;
        const run = await api(`/api/runs/${encodeURIComponent(id)}`);
        if (activeRun?.id === id && request === selection) showRun(run);
      }
      if (!el('panel').hidden) await refreshHistory();
    } catch (error) { if (!el('panel').hidden) notice(`Sunucu bağlantısı: ${error.message}`); }
    finally { polling = false; window.setTimeout(poll, 2500); }
  }
  async function initialize() {
    const version = revision;
    try {
      if (await loadCatalog()) {
        const afterCatalog = revision, runs = await refreshHistory(), saved = storage.get('flux-policy-run');
        if (afterCatalog === revision && !el('panel').hidden && runs?.some(run => run.id === saved)) await selectRun(saved);
      }
    } catch (error) { if (revision === version) notice(`Policy başlatılamadı: ${error.message}. Yenile ile tekrar deneyin.`); }
    updateButtons(); poll();
  }
  el('episode').addEventListener('change', () => chooseEpisode());
  ['frame', 'prompt', 'chunks'].forEach(id => el(id).addEventListener('input', changed));
  ['base', 'ft'].forEach(id => el(id).addEventListener('change', changed));
  el('seconds').addEventListener('input', () => {
    secondsDirty = true;
    revision++; clearPreview(); notice(); snapshotNote();
    el('preview').disabled = true; el('run').disabled = true;
    el('preview-info').textContent = 'Zamanı tamamlayın; alandan çıkınca kesin kare indeksi çözümlenir.';
  });
  el('seconds').addEventListener('change', () => {
    secondsDirty = false;
    const fps = episode()?.fps, seconds = Number(el('seconds').value);
    if (!el('seconds').value || !numeric(fps) || fps <= 0 || !Number.isFinite(seconds) || seconds < 0) {
      revision++; clearPreview(); el('frame').value = ''; el('seconds').setAttribute('aria-invalid', 'true'); notice('Zaman için pozitif episode FPS ve geçerli saniye değeri gerekir.'); updateButtons(); snapshotNote(); return;
    }
    const frame = Math.round(seconds * fps);
    el('frame').value = frame; el('seconds').setAttribute('aria-invalid', 'false'); changed();
    el('frame-info').textContent += ` Girilen ${seconds} sn → en yakın kare ${frame}; çözümlenen indeks açıkça gösterilir.`;
  });
  el('use-frame').addEventListener('click', () => {
    const fps = episode()?.fps;
    if (!numeric(fps) || fps <= 0) { notice('Kaynak FPS bilinmiyor; videodan kesin indeks çözülemez.'); return; }
    const time = el('source').currentTime; el('frame').value = Math.floor(time * fps + 1e-6); changed();
    el('frame-info').textContent += ` Video zamanı ${time.toFixed(6)} sn → içinde bulunulan kare.`;
  });
  ['prev', 'next'].forEach(id => el(id).addEventListener('click', () => {
    const frame = Number(el('frame').value);
    if (!el('frame').value || !Number.isInteger(frame)) { notice('Önce tam sayı kare indeksi girin.'); return; }
    el('frame').value = frame + (id === 'prev' ? -1 : 1); changed();
  }));
  el('reset-prompt').addEventListener('click', () => { el('prompt').value = episode()?.prompt || ''; changed(); });
  el('preview').addEventListener('click', makePreview); el('run').addEventListener('click', startRun); el('cancel').addEventListener('click', cancelRun);
  el('refresh').addEventListener('click', async () => {
    notice();
    if (!catalog) { try { await loadCatalog(); } catch (error) { notice(error.message); } }
    actionRequests = new Map(); loadActions(); await refreshHistory();
  });
  el('play').addEventListener('click', togglePlayback);
  el('timeline').addEventListener('input', () => { pausePlayers(); seek(Number(el('timeline').value)); });
  el('joint-group').addEventListener('change', populateJoints);
  ['joint', 'action-scale'].forEach(id => el(id).addEventListener('change', drawChart));
  el('chunk').addEventListener('change', () => { pausePlayers(); seek(Number(el('chunk').value) * 32); chunkInfo(); drawChart(); });
  if (window.ResizeObserver) new ResizeObserver(drawChart).observe(el('chart')); else window.addEventListener('resize', drawChart);
  const savedTab = storage.get('flux-eval-tab');
  if (modes.includes(savedTab)) activate(savedTab);
})();
