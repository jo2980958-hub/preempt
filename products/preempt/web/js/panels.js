/* The reasoning panels: what state the room was in, who was called, and what
   was on the route. Each one prints the service's own words, not a paraphrase. */

import { esc, has, stamp, metres, stateOf, rungOf } from './format.js';

const listOf = (items, cls = '') =>
  items.map((r) => `<li class="${cls}">${esc(r)}</li>`).join('');

export function riskPanel(record, risk) {
  const state = stateOf(record.metrics.peak_state);
  const tone = { ok: 'calm', hot: 'hot', refuse: 'refuse', warn: '', plain: 'calm' }[state.tone] ?? '';
  const reasons = risk?.reasons?.length ? risk.reasons : ['no movement that precedes standing'];
  const hazards = risk?.hazards || [];
  const lead = record.metrics.lead_time_s;

  return `<section class="panel" id="sec-risk" aria-labelledby="risk-h">
    <div class="risk-head ${tone}">
      <h2 id="risk-h">${esc(record.metrics.peak_headline || state.word)}</h2>
      <div class="risk-meta">
        <span class="chip ${state.tone}">${esc(state.word)}</span>
        <span class="chip plain">state named at ${stamp(risk?.at_s ?? 0)}</span>
        ${has(lead) ? `<span class="chip accent">${lead.toFixed(2)} s of warning before upright</span>`
    : '<span class="chip plain">nobody stood up, so there is no lead time</span>'}
      </div>
      <ul class="reasons">
        ${listOf(reasons)}
        ${listOf(hazards, 'hazard')}
      </ul>
      <p class="certainty"><b>How sure this is</b>${esc(risk?.certainty || 'observed')}</p>
    </div>
  </section>`;
}

export function callsPanel(calls) {
  const order = ['nudge', 'station', 'urgent'];
  const raised = new Set(calls.map((c) => c.rung));
  const rungs = order.map((key) => {
    const r = rungOf(key);
    const mine = calls.filter((c) => c.rung === key);
    const on = mine.length > 0;
    return `<div class="rung ${on ? 'raised' : 'quiet'} ${key}">
      <span class="badge" aria-hidden="true">${on ? r.n : '·'}</span>
      <div>
        <p class="rt">${esc(r.name)}
          ${on ? `<span class="chip ${key === 'urgent' ? 'hot' : 'accent'}">raised ${mine.length}&times;, first at ${stamp(mine[0].time_s)}</span>`
    : '<span class="chip plain">not raised</span>'}</p>
        <p class="rd">${esc(r.what)}</p>
      </div>
    </div>`;
  }).join('');

  const maint = calls.filter((c) => c.rung === 'maintenance');
  const maintRung = maint.length ? `<div class="rung raised maintenance">
      <span class="badge" aria-hidden="true">!</span>
      <div><p class="rt">${esc(rungOf('maintenance').name)}
        <span class="chip refuse">raised at ${stamp(maint[0].time_s)}</span></p>
      <p class="rd">${esc(rungOf('maintenance').what)}</p></div></div>` : '';

  return `<section class="panel" id="sec-calls" aria-labelledby="calls-h">
    <div class="ph"><h2 id="calls-h">Who was called, and why</h2>
      <span class="sub">${calls.length} ${calls.length === 1 ? 'call' : 'calls'}${raised.size ? '' : ' — the ladder stayed quiet'}</span></div>
    <div class="ladder">${rungs}${maintRung}</div>
    ${callLog(calls)}
  </section>`;
}

function callLog(calls) {
  if (!calls.length) {
    return `<div class="pb"><p class="dim">Nothing crossed a threshold, so nobody was called.
      A camera that calls when nothing is happening is a camera a ward switches off.</p></div>`;
  }
  const rows = calls.map((c) => `<tr>
      <td>${stamp(c.time_s)}</td>
      <td><b>${esc(c.wording)}</b>
        <div class="why">${esc((c.reasons || []).join('; '))}</div></td>
      <td class="num">${c.raised_by_hazard
    ? '<span class="chip warn">hazard on the route</span>'
    : `<span class="chip plain">${esc(c.risk_state)}</span>`}</td>
    </tr>`).join('');
  return `<table class="calllog">
    <thead><tr><th>Time</th><th>What the handset said</th><th class="num">Raised by</th></tr></thead>
    <tbody>${rows}</tbody></table>`;
}

export function hazardsPanel(h) {
  if (!h) return '';
  const found = (h.hazards || []).map((x) => `<li>
      <span class="m chip warn">${metres(x.metres)}</span>
      <span><b>${esc(x.kind)}</b><br>${esc(x.description)}.
        <span class="faint small">Confidence ${(x.confidence ?? 0).toFixed(2)}${has(x.area_cm2) ? `, area ${x.area_cm2} cm²` : ''}, on the floor plan at ${x.floor_xy?.map((n) => n.toFixed(1)).join(' m, ')} m.</span></span></li>`).join('');

  return `<section class="panel" id="sec-hazards" aria-labelledby="haz-h">
    <div class="ph"><h2 id="haz-h">What was on the route</h2>
      <span class="sub">${(h.hazards || []).length} found, ${(h.checked || []).length} checked, ${(h.skipped || []).length} skipped</span></div>
    <div class="pb">
      <p class="haz-title">Found</p>
      <ul class="haz">${found || '<li><span class="tick">✓</span><span class="dim">Nothing on the walking route.</span></li>'}</ul>
      <p class="haz-title">Checked</p>
      <ul class="haz">${(h.checked || []).map((c) =>
    `<li><span class="tick">✓</span><span>${esc(c)}</span></li>`).join('')}</ul>
      <p class="haz-title">Skipped, and why</p>
      <ul class="haz">${(h.skipped || []).map((s) =>
      `<li><span class="cross">✕</span><span class="dim">${esc(s)}</span></li>`).join('')
    || '<li><span class="dim">Nothing was skipped.</span></li>'}</ul>
    </div>
  </section>`;
}

const SOURCE_WORDS = {
  measured: 'measured', assumed: 'assumed', synthetic: 'exact (synthetic camera)', unstated: 'not stated',
};
const ORIGIN_WORDS = {
  uploaded: 'a room setup sent with this video',
  default: 'the default room, from the synthetic ward',
  'pose track': "the pose track's own room",
  'command line': 'a room setup given on the command line',
};

/** Which room the numbers were measured against, and whether its camera was measured. */
export function setupPanel(record) {
  const setup = record.input?.room_setup;
  if (!setup) return '';
  const cal = setup.calibration || {};
  const synthetic = cal.camera_height === 'synthetic';
  const tone = cal.calibrated ? 'ok' : 'warn';
  const chip = synthetic ? 'Synthetic camera, exact' : cal.calibrated ? 'Camera measured' : 'Camera assumed, not measured';
  const caveats = [];
  if (!cal.calibrated) {
    caveats.push(`The camera height and focal length for this room were ${cal.camera_height === 'unstated' ? 'never stated, so Preempt treats them as assumed' : 'assumed rather than measured'}. Every height and distance on this page rests on them: read the metres as approximate.`);
  }
  if (setup.source === 'default' && record.params?.input_kind === 'video') {
    caveats.push('This video was measured against the synthetic ward\'s room. Unless it was filmed in that room, the zones and heights do not describe it.');
  }
  return `<section class="panel setup" id="sec-setup" data-demo="room-setup" aria-labelledby="setup-h">
    <div class="ph"><h2 id="setup-h">Measured against ${esc(setup.name)}</h2>
      <span class="chip ${tone}">${esc(chip)}</span>
      <span class="sub">${esc(ORIGIN_WORDS[setup.source] || setup.source)}</span></div>
    <div class="pb">
      <dl class="setup-dl">
        <div><dt>Camera height</dt><dd>${esc(SOURCE_WORDS[cal.camera_height] || cal.camera_height)}</dd></div>
        <div><dt>Focal length</dt><dd>${esc(SOURCE_WORDS[cal.focal_length] || cal.focal_length)}</dd></div>
        <div><dt>Zones</dt><dd>${(setup.zones || []).map(esc).join(', ') || 'none'}</dd></div>
      </dl>
      ${cal.note ? `<p class="setup-note">${esc(cal.note)}</p>` : ''}
      ${caveats.map((c) => `<p class="setup-caveat">${esc(c)}</p>`).join('')}
    </div>
  </section>`;
}
