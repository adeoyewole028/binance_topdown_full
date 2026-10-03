/* Same-origin observations update read-only panels, never operator drafts. */
(() => {
  'use strict';
  const status = document.getElementById('connection-status');
  const error = document.getElementById('request-error');
  const login = document.getElementById('sign-in-again');
  const toggle = document.getElementById('soundToggle');
  let latest = null, currentRead = null, paused = false, sound = false, audio = null;
  let reviewedVersion = null;
  const namespace = document.body.dataset.namespace;
  const seenKey = 'signal-candles:' + namespace;
  let seen = new Set();
  try {
    sound = localStorage.getItem('soundEnabled') === '1';
    const saved = JSON.parse(localStorage.getItem(seenKey) || '[]');
    if (Array.isArray(saved)) seen = new Set(saved.filter(value => typeof value === 'string').slice(-500));
  } catch (_) { /* Storage is optional. */ }
  function soundLabel() {
    toggle.textContent = 'Sound: ' + (sound ? 'On' : 'Off');
    toggle.setAttribute('aria-pressed', String(sound));
  }
  function beep() {
    if (!sound || !audio || audio.state !== 'running') return;
    const oscillator = audio.createOscillator(), gain = audio.createGain();
    oscillator.frequency.value = 880; gain.gain.value = 0.08;
    oscillator.connect(gain); gain.connect(audio.destination);
    oscillator.start(); oscillator.stop(audio.currentTime + 0.25);
  }
  soundLabel();
  async function armAudio() {
    if (!sound) return;
    try { audio=audio || new (window.AudioContext || window.webkitAudioContext)(); await audio.resume(); } catch (_) {}
  }
  window.addEventListener('pointerdown',armAudio,{once:true});
  window.addEventListener('keydown',armAudio,{once:true});
  toggle.addEventListener('click', async () => {
    sound = !sound;
    try { localStorage.setItem('soundEnabled', sound ? '1' : '0'); } catch (_) {}
    soundLabel();
    if (sound) {
      try { audio = audio || new (window.AudioContext || window.webkitAudioContext)(); await audio.resume(); beep(); } catch (_) {}
    }
  });
  function announceSignals(payload) {
    if (!payload.market_health.execution_ready) return;
    const fresh = [];
    for (const signal of payload.signals) {
      if (!signal.signal_id || seen.has(signal.signal_id)) continue;
      seen.add(signal.signal_id); fresh.push(signal.symbol);
    }
    seen = new Set([...seen].slice(-500));
    try { localStorage.setItem(seenKey, JSON.stringify([...seen])); } catch (_) {}
    if (fresh.length) {
      document.getElementById('toast-root').textContent = 'New completed-candle entry signal: ' + fresh.join(', ');
      beep();
    }
  }
  function observation(payload) {
    if (payload.namespace !== namespace) throw new Error('Account namespace changed. Review this account before continuing.');
    latest = payload;
    const incoming = new DOMParser().parseFromString(payload.dashboard_view, 'text/html');
    const csrf = incoming.querySelector('[name="csrf_token"]')?.value;
    if (csrf) for (const input of document.querySelectorAll('[name="csrf_token"]')) input.value = csrf;
    for (const panel of document.querySelectorAll('[data-status-panel]')) {
      // Dynamic action lists retain the active control until focus leaves it.
      if (panel.contains(document.activeElement)) continue;
      const next = incoming.querySelector('[data-status-panel="' + panel.dataset.statusPanel + '"]');
      if (next) panel.replaceChildren(...Array.from(next.childNodes));
    }
    const settings = document.querySelector('form[action="/update-config"]');
    const version = settings.elements.namedItem('expected_config_version');
    if (Number(version.value) !== payload.config_version) {
      document.getElementById('settings-conflict').textContent = 'Saved settings changed. Your draft is preserved. Show current saved settings before saving.';
    }
    status.textContent = payload.presentation.label + '. Status updated.';
    status.dataset.state = payload.presentation.state;
    login.hidden = true;
    announceSignals(payload);
  }
  function readStatus() {
    if (!currentRead) currentRead=performRead().finally(() => { currentRead=null; });
    return currentRead;
  }
  async function performRead() {
    status.setAttribute('aria-busy', 'true');
    if (!latest) { status.textContent='Loading current status; your inputs remain available.'; status.dataset.state='loading'; }
    const controller = new AbortController(), timeout = setTimeout(() => controller.abort(), 10000);
    try {
      const response = await fetch('/api/status?view=1', {credentials: 'same-origin', cache: 'no-store', signal: controller.signal});
      if (response.status === 401 || response.status === 403) {
        status.textContent = 'Permission limited. Sign in again; your current inputs are preserved.';
        status.dataset.state = 'permission_limited'; login.hidden = false; return false;
      }
      if (!response.ok) throw new Error('Status unavailable. Your current inputs are preserved. Try updating again.');
      observation(await response.json());
      return true;
    } catch (failure) {
      status.textContent = failure.message === 'Account namespace changed. Review this account before continuing.' ? failure.message :
        'Offline or status unavailable. Last displayed data may be stale; your current inputs are preserved.';
      status.dataset.state = 'offline';
      return false;
    } finally { clearTimeout(timeout); status.removeAttribute('aria-busy'); }
  }
  document.getElementById('refreshStatus').addEventListener('click', readStatus);
  document.getElementById('pollToggle').addEventListener('click', event => {
    paused = !paused;
    event.currentTarget.textContent = paused ? 'Resume automatic updates' : 'Pause automatic updates';
    event.currentTarget.setAttribute('aria-pressed', String(paused));
  });
  setInterval(() => { if (!paused && !document.hidden) readStatus(); }, 15000);
  readStatus();
  document.getElementById('reviewSettings').addEventListener('click', () => {
    if (!latest) { error.textContent = 'Update status before reviewing saved settings.'; return; }
    const root = document.getElementById('saved-settings'); root.replaceChildren();
    for (const [field, value] of Object.entries(latest.config)) {
      if (value !== null && typeof value === 'object') {
        const title=document.createElement('p'); title.textContent=field.replaceAll('_',' '); root.appendChild(title);
        for (const [name,recorded] of Object.entries(value)) {
          const row=document.createElement('p'); row.textContent=name.replaceAll('_',' ')+': '+recorded; root.appendChild(row);
        }
        continue;
      }
      const row = document.createElement('p');
      const input = document.querySelector('form[action="/update-config"] [name="' + field + '"]');
      row.textContent = (input?.labels?.[0]?.textContent.trim() || field.replaceAll('_', ' ')) + ': ' + value;
      root.appendChild(row);
    }
    reviewedVersion = latest.config_version;
    document.getElementById('acceptVersion').hidden = false;
  });
  document.getElementById('acceptVersion').addEventListener('click', event => {
    if (!latest || reviewedVersion !== latest.config_version) {
      error.textContent = 'Saved settings changed again. Review the current values.'; return;
    }
    for (const version of document.querySelectorAll('[name="expected_config_version"]')) version.value=reviewedVersion;
    document.getElementById('settings-conflict').textContent = 'Current version reviewed; your draft is preserved.';
    event.currentTarget.hidden = true;
  });
  for (const form of document.querySelectorAll('form[action="/update-config"],form[action="/reset-config"],form[action="/paper/setup"],form[action="/exchange/setup"]')) {
    form.addEventListener('submit', async event => {
      event.preventDefault();
      if (form.dataset.submitting === 'true') return;
      form.dataset.submitting = 'true'; form.setAttribute('aria-busy', 'true');
      for (const control of form.querySelectorAll('[aria-invalid]')) control.setAttribute('aria-invalid', 'false');
      for (const row of form.querySelectorAll('.field-error')) row.textContent = '';
      error.textContent = '';
      const controller=new AbortController(), timeout=setTimeout(() => controller.abort(),10000);
      try {
        const response = await fetch(form.action, {method:'POST', credentials:'same-origin',
          headers:{Accept:'application/json'}, body:new FormData(form),signal:controller.signal});
        if (!response.ok) {
          const payload = await response.json();
          error.textContent = payload.error || 'Request failed. Your inputs are preserved.';
          if (response.status===401 || response.status===403) login.hidden=false;
          for (const [field, message] of Object.entries(payload.fields || {})) {
            const input = form.elements.namedItem(field);
            if (input && input.type !== 'hidden') {
              input.setAttribute('aria-invalid', 'true');
              const target = document.getElementById((form.action.endsWith('/setup') ? 'setup-error-' : 'error-') + field);
              if (target) target.textContent = message;
            }
          }
          error.focus(); return;
        }
        if (response.redirected && new URL(response.url).pathname === '/login') {
          error.textContent = 'Sign in again. Your inputs are preserved.'; login.hidden = false; return;
        }
        const acknowledged=await response.json();
        if (currentRead) await currentRead;
        const confirmed=await readStatus();
        const version = form.elements.namedItem('expected_config_version');
        if (version) version.value=acknowledged.saved_config_version;
        if (!confirmed) { error.textContent='Settings accepted, but current status could not be verified. Update status before retrying.'; return; }
        if (form.action.endsWith('/reset-config') && latest.config_version===acknowledged.saved_config_version) {
          const settings=document.querySelector('form[action="/update-config"]');
          for (const [name,value] of Object.entries(latest.config)) {
            const control=settings.elements.namedItem(name);
            if (control && control.type!=='hidden' && typeof value!=='object') control.value=String(value);
          }
          settings.elements.namedItem('expected_config_version').value=acknowledged.saved_config_version;
        }
        document.getElementById('settings-conflict').textContent=latest.config_version===acknowledged.saved_config_version ?
          'Settings saved at version '+acknowledged.saved_config_version+'.' : 'A newer settings version exists. Review it before saving again; your draft is preserved.';
        status.textContent = acknowledged.message+' Review the current account status before execution.';
      } catch (_) { error.textContent = 'Request unavailable. Your inputs are preserved; check saved settings before retrying.'; }
      finally { clearTimeout(timeout); form.dataset.submitting = 'false'; form.removeAttribute('aria-busy'); }
    });
  }
  const risk = document.getElementById('riskSizeBtn');
  if (risk) risk.addEventListener('click', () => {
    document.querySelector('select[name="sizing"]').value = 'risk';
    document.getElementById('qtyInput').value = '';
  });
})();
