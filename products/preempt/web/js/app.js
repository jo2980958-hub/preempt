/* The shell: pick a room, watch it, show what came back. */

import { esc, bytes, has, stateOf } from './format.js';
import * as api from './api.js';
import { renderResults } from './render.js';

const $ = (sel) => document.querySelector(sel);
const FIRST = 'bed-exit-steady';
const FAILURE_CASE = 'curtain-drawn';

const ui = {
  status: $('#status'), results: $('#results'), empty: $('#empty'),
  picker: $('#picker'), samples: $('#samples'), progress: $('#progress'),
  fill: $('#progress-fill'), msg: $('#progress-msg'), pct: $('#progress-pct'),
  start: $('#start'),
  startLabel: $('#start-label'), startNote: $('#start-note'), file: $('#file'),
  title: $('#title'), subtitle: $('#subtitle'), crumb: $('#crumb'),
  crumbDot: $('#crumb-dot'), source: $('#tb-source'), privacy: $('#tb-privacy'),
};

let samples = [];
let chosen = FIRST;
let upload = null;
let busy = false;

/* ── theme ─────────────────────────────────────────────────────────── */
function setTheme(mode) {
  document.documentElement.dataset.theme = mode;
  try { localStorage.setItem('preempt-theme', mode); } catch { /* private mode */ }
  const dark = mode === 'dark';
  $('#theme-label').textContent = dark ? 'Bring the lights up' : 'Dim the screen';
  $('#theme').setAttribute('aria-pressed', String(dark));
}
function initTheme() {
  let saved = null;
  try { saved = localStorage.getItem('preempt-theme'); } catch { /* ignore */ }
  const dark = saved ? saved === 'dark'
    : window.matchMedia('(prefers-color-scheme: dark)').matches;
  setTheme(dark ? 'dark' : 'light');
  $('#theme').addEventListener('click', () =>
    setTheme(document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark'));
}

/* ── status ────────────────────────────────────────────────────────── */
function status(text, tone = '') {
  ui.status.textContent = text;
  ui.status.className = `status ${tone}`;
}

/* ── the sample picker ─────────────────────────────────────────────── */
function renderSamples() {
  const ordered = [...samples].sort((a, b) =>
    (a.name === FIRST ? -1 : b.name === FIRST ? 1 : 0));
  ui.samples.innerHTML = ordered.map((s) => {
    const tag = s.name === FIRST ? '<span class="tag">Start here</span>'
      : s.name === FAILURE_CASE ? '<span class="tag refuse">The failure case</span>' : '';
    return `<button class="sample" type="button" data-demo="sample-${esc(s.name)}"
        data-name="${esc(s.name)}" aria-pressed="${s.name === chosen}">
        <span class="sn">${esc(s.name)}${tag}</span>
        <span class="sd">${esc(s.description)}</span>
        <span class="sm">${(s.duration_s ?? 0).toFixed(1)} s of pose track</span>
      </button>`;
  }).join('');
  ui.samples.querySelectorAll('.sample').forEach((b) =>
    b.addEventListener('click', () => choose(b.dataset.name)));
}

function choose(name) {
  chosen = name;
  upload = null;
  ui.file.value = '';
  ui.samples.querySelectorAll('.sample').forEach((b) =>
    b.setAttribute('aria-pressed', String(b.dataset.name === name)));
  const s = samples.find((x) => x.name === name);
  ui.startLabel.textContent = 'Watch this room';
  ui.startNote.textContent = s ? `${s.name}, ${(s.duration_s ?? 0).toFixed(1)} s of pose track` : name;
  run();
}

/* ── running a job ─────────────────────────────────────────────────── */
async function run() {
  if (busy) return;
  busy = true;
  ui.start.disabled = true;
  ui.empty.hidden = true;
  ui.progress.hidden = false;
  ui.fill.style.width = '2%';
  ui.pct.textContent = '0%';
  ui.msg.textContent = 'Handing the recording to the service';
  status('Running', 'busy');
  ui.crumbDot.className = 'dot';

  const label = upload ? upload.name : chosen;
  ui.source.textContent = `Watching ${label}`;

  try {
    const job = upload ? await api.startUpload(upload) : await api.startSample(chosen);
    const done = await api.follow(job, {
      onProgress: (e) => {
        if (has(e.percent)) {
          ui.fill.style.width = `${e.percent}%`;
          ui.pct.textContent = `${Math.round(e.percent)}%`;
        }
        if (e.message) { ui.msg.textContent = e.message; status(`Running — ${e.message}`, 'busy'); }
      },
      onNote: (e) => { if (e.message) ui.msg.textContent = e.message; },
    });
    ui.fill.style.width = '100%';
    ui.pct.textContent = '100%';
    if (done.status === 'failed' || !done.result) throw new Error(done.error || 'the service returned no record');
    show(done.result, job.job_id);
    status('Finished', 'done');
  } catch (err) {
    fail(err);
  } finally {
    busy = false;
    ui.start.disabled = false;
    setTimeout(() => { ui.progress.hidden = true; }, 500);
  }
}

function fail(err) {
  status('Failed', 'failed');
  ui.results.hidden = false;
  ui.results.innerHTML = `<div class="error"><h2>The run did not finish</h2>
    <p>${esc(err.message || String(err))}</p>
    <p class="dim">This is a fault in the service, not a reading about the room.
      Nothing here says anything about whether anybody is standing up.</p></div>`;
}

/* ── showing a finished record ─────────────────────────────────────── */
function show(record, jobId) {
  ui.empty.hidden = true;
  ui.results.hidden = false;
  ui.picker.classList.add('compact');
  renderResults(ui.results, record, jobId);

  const m = record.metrics;
  const room = record.input?.room || 'the room';
  ui.title.textContent = room.replace(/^\w/, (c) => c.toUpperCase());
  ui.subtitle.textContent = record.input?.source
    ? `${record.input.source}, at ${record.input.fps ?? '—'} frames a second. This page is everything the device knows about it.`
    : 'This page is everything the device knows about the recording.';
  ui.crumb.textContent = room;
  ui.crumbDot.className = `dot ${m.highest_rung === 'urgent' ? 'hot' : 'live'}`;
  ui.source.textContent = `${record.input?.source || 'recording'} · ${(m.duration_s ?? 0).toFixed(1)} s`;

  const p = m.privacy || {};
  ui.privacy.className = p.camera_bytes_persisted === 0 ? 'chip ok' : 'chip hot';
  ui.privacy.textContent = p.camera_bytes_persisted === 0
    ? 'Nothing recognisable leaves this room'
    : `${bytes(p.camera_bytes_persisted)} of camera pixels were written`;

  counts({
    keypoints: `${(m.privacy?.frames_examined ?? 0).toLocaleString()}`,
    movement: String(stateRuns(record)),
    calls: String(m.calls ?? 0),
    hazards: String((record.results.find((r) => r.kind === 'hazards')?.hazards || []).length),
    bytes: `${p.camera_bytes_persisted ?? 0} B`,
    view: `${(((m.view?.usable_fraction) ?? 1) * 100).toFixed(0)}%`,
  });
  document.querySelector('.nav a[href="#sec-calls"]')
    ?.classList.toggle('alert', (m.calls ?? 0) > 0);
  $('#f-kept').textContent = bytes(p.keypoint_bytes_kept);
  $('#f-frames').textContent = (p.frames_examined ?? 0).toLocaleString();
}

/** How many times the room changed state — a live count, not a word. */
function stateRuns(record) {
  const samples = record.results.find((r) => r.kind === 'timeline')?.samples || [];
  let runs = 0, last = null;
  for (const s of samples) {
    const state = s.risk?.state;
    if (state !== last) { runs += 1; last = state; }
  }
  return runs;
}

function counts(map) {
  for (const [key, value] of Object.entries(map)) {
    const el = document.querySelector(`.nav [data-count="${key}"]`);
    if (el) el.textContent = value;
  }
}

/* ── nav highlight: whichever section the reader has reached ──────── */
function watchSections() {
  const links = [...document.querySelectorAll('.nav a[href^="#sec-"]')];
  let pending = false;
  const mark = () => {
    pending = false;
    // Two columns means DOM order is not reading order, so pick the section
    // whose top has most recently passed the reading line.
    const line = window.innerHeight * 0.28;
    let current = null, best = -Infinity;
    for (const section of document.querySelectorAll('[id^="sec-"]')) {
      const top = section.getBoundingClientRect().top;
      if (top <= line && top > best) { best = top; current = section.id; }
    }
    links.forEach((a) => a.classList.toggle('on', a.getAttribute('href') === `#${current}`));
  };
  const schedule = () => {
    if (pending) return;
    pending = true;
    requestAnimationFrame(mark);
  };
  window.addEventListener('scroll', schedule, { passive: true });
  window.addEventListener('resize', schedule);
  new MutationObserver(schedule).observe(ui.results, { childList: true });
}

/* ── boot ──────────────────────────────────────────────────────────── */
async function boot() {
  initTheme();
  watchSections();
  ui.start.addEventListener('click', run);
  ui.file.addEventListener('change', () => {
    upload = ui.file.files?.[0] || null;
    if (!upload) return;
    ui.samples.querySelectorAll('.sample').forEach((b) => b.setAttribute('aria-pressed', 'false'));
    ui.startLabel.textContent = 'Watch this video';
    ui.startNote.textContent = `${upload.name}, ${bytes(upload.size)}`;
    run();
  });

  try {
    const [version, list] = await Promise.all([api.getVersion(), api.getSamples()]);
    $('#f-opencv').textContent = version.opencv_version || '—';
    $('#f-engine').textContent = Object.values(version.models || {})[0]?.name || 'RTMPose-t';
    samples = Array.isArray(list) ? list : (list.samples || []);
    renderSamples();
    status('Ready');
  } catch (err) {
    ui.samples.innerHTML = `<p class="loading">The sample list did not load: ${esc(err.message)}.
      The upload below still works.</p>`;
    status('Service unreachable', 'failed');
  }
}

boot();
