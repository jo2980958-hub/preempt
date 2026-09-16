/* Assembles the results region and wires the two things that move:
   the timeline playhead and the ledger-to-figure highlight. */

import { esc, has, secs, stateOf } from './format.js';
import { drawFigure, highlight } from './figure.js';
import { ribbonHtml, readoutsHtml, currentWords } from './timeline.js';
import { riskPanel, callsPanel, hazardsPanel } from './panels.js';
import { ledgerPanel, artefactsPanel, viewPanel, evidencePanel } from './privacy.js';

const pick = (record, kind) => (record.results || []).find((r) => r.kind === kind);
const reduced = () => window.matchMedia('(prefers-reduced-motion: reduce)').matches;

export function kpis(record) {
  const m = record.metrics;
  const lead = m.lead_time_s;
  const usable = (m.view?.usable_fraction ?? 1) * 100;
  const p = m.privacy || {};
  const blind = usable < 100;
  const leadTile = has(lead)
    ? { cls: 'good', v: `${lead.toFixed(2)}<u>s</u>`, n: 'between the first call and standing upright' }
    : blind
      ? { cls: 'refuse', v: '<span class="na">Not measurable</span>',
          n: 'the view was blocked for part of the recording' }
      : { cls: 'mute', v: 'none<u>needed</u>', n: 'nobody left the bed or the chair' };
  const tiles = [
    { ...leadTile, k: 'Warning before upright' },
    { cls: m.highest_rung === 'urgent' ? 'hot' : (m.calls ? '' : 'mute'), k: 'Calls raised',
      v: `${m.calls ?? 0}`, n: `highest rung: ${esc(m.highest_rung || 'none')}` },
    { cls: usable < 100 ? 'refuse' : '', k: 'View usable',
      v: `${usable.toFixed(1)}<u>%</u>`, n: `${m.view?.frames ?? 0} frames graded` },
    { cls: 'good', k: 'Camera bytes kept',
      v: `${p.camera_bytes_persisted ?? 0}<u>B</u>`, n: `${p.frames_retained ?? 0} of ${p.frames_examined ?? 0} frames retained` },
    { cls: 'mute', k: 'Recording watched',
      v: `${(m.duration_s ?? 0).toFixed(1)}<u>s</u>`, n: `${m.samples ?? 0} pose samples` },
  ];
  return `<div class="kpis">${tiles.map((t) =>
    `<div class="kpi ${t.cls}"><div class="k">${t.k}</div><div class="v">${t.v}</div>
     <div class="n">${t.n}</div></div>`).join('')}</div>`;
}

function roomPanel(record) {
  const ledger = pick(record, 'keypoint-ledger');
  const timeline = pick(record, 'timeline');
  const calls = pick(record, 'calls')?.calls || [];
  const samples = timeline?.samples?.filter((s) => has(s.time_s)) || [];
  const frame = ledger?.frame;
  const state = stateOf(record.metrics.peak_state);

  const kept = (frame?.keypoints || []).filter((k) => k.seen).length;
  const total = (frame?.keypoints || []).length;
  const thin = total && kept < 11
    ? `<div class="field-plate"><b>${kept} of ${total} keypoints</b>
       <span>Too few to say anything about how anyone is standing.</span></div>` : '';

  return `<section class="panel" id="sec-room" aria-labelledby="room-h">
    <div class="ph"><h2 id="room-h">The room, as the device holds it</h2>
      <span class="chip ${state.tone}">${esc(state.word)}</span>
      <span class="sub">${samples.length} samples over ${secs(record.metrics.duration_s, 1)}</span></div>
    <div class="field">
      <div class="field-canvas">${drawFigure(frame)}${thin}</div>
      <p class="field-note"><b>No image is produced at any point.</b>
        ${kept} of ${total} coordinates cleared the threshold at ${secs(frame?.time_s)}, five of them
        the head — and a head that is five numbers is not a face. Scrub the ribbon and the figure does
        not move: coordinates were kept for that one instant and thrown away for every other.</p>
    </div>
    <div class="tl">
      <div class="tl-controls">
        <button class="tl-play" id="tl-play" type="button" aria-label="Play the recording">
          <svg width="15" height="15" viewBox="0 0 16 16" aria-hidden="true"><path d="M4.4 3.2l8 4.8-8 4.8z" fill="currentColor"/></svg>
        </button>
        <span class="tl-time mono"><span id="tl-now">00:00.000</span><span class="of"> / ${secs(record.metrics.duration_s, 2)}</span></span>
        <input type="range" id="tl-range" min="0" max="${samples.length - 1}" value="0" step="1"
          aria-label="Position in the recording">
        <span class="tl-state" id="tl-chip"></span>
      </div>
      ${ribbonHtml(samples, calls, record.metrics.duration_s)}
      <div class="tl-legend">${legend(samples)}</div>
      <dl class="readouts" id="tl-readouts">${readoutsHtml(samples[0])}</dl>
      <p class="tl-reasons" id="tl-why"></p>
    </div>
  </section>`;
}

function legend(samples) {
  const kinds = [...new Set(samples.map((s) => s.risk?.state).filter(Boolean))];
  return kinds.map((k) => {
    const s = stateOf(k);
    return `<span><i style="background-color:${s.band}"></i>${esc(s.word)}</span>`;
  }).join('');
}

export function renderResults(root, record, jobId) {
  const risk = pick(record, 'risk');
  const calls = pick(record, 'calls')?.calls || [];
  const privacy = record.metrics.privacy || {};
  // A refused view is a result, not an error, so when it happens it leads the page.
  const blind = (record.refusals || []).some((r) => r.code === 'VIEW_UNUSABLE')
    || (record.metrics.view?.usable_fraction ?? 1) < 1;
  const view = viewPanel(record, blind);
  root.innerHTML = `
    ${kpis(record)}
    ${errorsFor(record)}
    ${blind ? view : ''}
    <div class="cols">
      <div class="col">
        ${roomPanel(record)}
        ${riskPanel(record, risk)}
        ${callsPanel(calls)}
        ${evidencePanel(record, jobId)}
      </div>
      <div class="col">
        ${ledgerPanel(pick(record, 'keypoint-ledger')?.frame, privacy)}
        ${blind ? '' : view}
        ${hazardsPanel(pick(record, 'hazards'))}
        ${artefactsPanel(privacy)}
      </div>
    </div>`;
  wireLedger(root, pick(record, 'keypoint-ledger')?.frame);
  wireTimeline(root, record);
}

function errorsFor(record) {
  const others = (record.refusals || []).filter((r) => r.code !== 'VIEW_UNUSABLE');
  const warnings = record.warnings || [];
  if (!others.length && !warnings.length) return '';
  return `<section class="panel refusal"><div class="ph"><h3>Preempt held something back</h3></div>
    <div class="pb"><ul class="haz">
      ${others.map((r) => `<li><span class="m chip refuse">${esc(r.code)}</span><span>${esc(r.message)}</span></li>`).join('')}
      ${warnings.map((w) => `<li><span class="m chip warn">warning</span><span>${esc(w)}</span></li>`).join('')}
    </ul></div></section>`;
}

function wireLedger(root, frame) {
  if (!frame) return;
  const svg = root.querySelector('.field svg');
  const body = root.querySelector('#ledger-body');
  const rows = body ? [...body.querySelectorAll('tr')] : [];
  const light = (i) => {
    rows.forEach((r) => r.classList.toggle('lit', Number(r.dataset.kp) === i));
    highlight(svg, frame, i);
  };
  rows.forEach((r) => {
    r.addEventListener('mouseenter', () => light(Number(r.dataset.kp)));
    r.addEventListener('mouseleave', () => light(null));
  });
  svg?.querySelectorAll('.kp-hit').forEach((hit) => {
    const i = Number(hit.dataset.kp);
    ['mouseenter', 'focus'].forEach((ev) => hit.addEventListener(ev, () => light(i)));
    ['mouseleave', 'blur'].forEach((ev) => hit.addEventListener(ev, () => light(null)));
  });
}

function wireTimeline(root, record) {
  const samples = pick(record, 'timeline')?.samples?.filter((s) => has(s.time_s)) || [];
  if (!samples.length) return;
  const range = root.querySelector('#tl-range');
  const play = root.querySelector('#tl-play');
  const now = root.querySelector('#tl-now');
  const chip = root.querySelector('#tl-chip');
  const head = root.querySelector('#tl-head');
  const outs = root.querySelector('#tl-readouts');
  const why = root.querySelector('#tl-why');
  const span = record.metrics.duration_s || samples.at(-1).time_s || 1;

  let i = 0, timer = null;
  const show = (n) => {
    i = Math.max(0, Math.min(samples.length - 1, n));
    const s = samples[i];
    const st = stateOf(s.risk?.state);
    range.value = String(i);
    now.textContent = fmt(s.time_s);
    chip.innerHTML = `<span class="chip ${st.tone}">${esc(st.word)}</span>`;
    head.style.left = `${((s.time_s / span) * 100).toFixed(3)}%`;
    outs.innerHTML = readoutsHtml(s);
    why.innerHTML = currentWords(s);
  };
  const stop = () => {
    clearInterval(timer); timer = null;
    play.setAttribute('aria-label', 'Play the recording');
    play.innerHTML = '<svg width="15" height="15" viewBox="0 0 16 16" aria-hidden="true"><path d="M4.4 3.2l8 4.8-8 4.8z" fill="currentColor"/></svg>';
  };
  play.addEventListener('click', () => {
    if (timer) return stop();
    if (i >= samples.length - 1) show(0);
    play.setAttribute('aria-label', 'Pause the recording');
    play.innerHTML = '<svg width="15" height="15" viewBox="0 0 16 16" aria-hidden="true"><path d="M4.6 3h2.6v10H4.6zM8.8 3h2.6v10H8.8z" fill="currentColor"/></svg>';
    // With reduced motion the playhead does not sweep: it lands on each state
    // change in turn, so what renders is always an end state.
    if (reduced()) {
      const stops = samples.reduce((acc, s, n) => {
        if (n === 0 || s.risk?.state !== samples[n - 1].risk?.state) acc.push(n);
        return acc;
      }, []);
      let at = stops.findIndex((n) => n > i);
      timer = setInterval(() => {
        if (at < 0 || at >= stops.length) return stop();
        show(stops[at]);
        at += 1;
      }, 800);
      return;
    }
    const stepMs = 1000 / Math.max(samples.length / span, 1);
    timer = setInterval(() => (i >= samples.length - 1 ? stop() : show(i + 1)), Math.max(stepMs, 16));
  });
  range.addEventListener('input', () => { stop(); show(Number(range.value)); });

  // Start on the instant the ledger frame belongs to: the moment that mattered.
  const peak = pick(record, 'keypoint-ledger')?.frame?.time_s ?? 0;
  const nearest = samples.reduce((best, s, n) =>
    Math.abs(s.time_s - peak) < Math.abs(samples[best].time_s - peak) ? n : best, 0);
  show(nearest);
}

const fmt = (t) => {
  const m = Math.floor(t / 60), s = Math.floor(t % 60), ms = Math.round((t % 1) * 1000);
  return `${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}.${String(ms).padStart(3, '0')}`;
};
