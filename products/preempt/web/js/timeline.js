/* The scrubbable state ribbon.

   The samples carry a state per frame but no keypoints — only the ledger frame
   has those — so the ribbon animates the state and the measured angles, and the
   figure stays at the one instant whose coordinates were kept. */

import { esc, has, stateOf, stamp, unit } from './format.js';

export function ribbonHtml(samples, calls, duration) {
  const runs = contiguous(samples);
  const span = duration || samples.at(-1)?.time_s || 1;
  const pct = (t) => `${((t / span) * 100).toFixed(3)}%`;

  const bands = runs.map((r) => {
    const s = stateOf(r.state);
    return `<i style="width:${pct(r.end - r.start)};background-color:${s.band}" title="${esc(s.word)}"></i>`;
  }).join('');

  // Label the longest run of each state once, so the strip reads as words not stripes.
  const named = new Map();
  runs.forEach((r, i) => {
    const best = named.get(r.state);
    if (best === undefined || runs[best].end - runs[best].start < r.end - r.start) named.set(r.state, i);
  });
  const labels = runs.map((r, i) => {
    const s = stateOf(r.state);
    // Only label a band that is wide enough for the word at phone width (~340 px).
    const show = named.get(r.state) === i && ((r.end - r.start) / span) * 320 > s.word.length * 8.6;
    return `<i style="width:${pct(r.end - r.start)}">${show ? esc(s.word) : ''}</i>`;
  }).join('');

  const pins = calls.map((c, i) => {
    const hot = c.rung === 'urgent';
    return `<b class="pin${hot ? ' hot' : ''}"
      style="left:clamp(12px, ${pct(c.time_s)}, calc(100% - 12px))"
      title="${esc(c.wording)} at ${stamp(c.time_s)}"><span>${i + 1}</span></b>`;
  }).join('');

  return `<div class="ribbon-wrap">
      <div class="pins">${pins}</div>
      <div class="ribbon" role="img" aria-label="${esc(runs.map((r) => stateOf(r.state).word).join(', then '))}">
        ${bands}<span class="head" id="tl-head" style="left:0%"></span>
      </div>
      <div class="ribbon-labels">${labels}</div>
      ${traceHtml(samples, span)}
    </div>`;
}

/** Hip height, metres, across the whole recording. A real measured series. */
function traceHtml(samples, span) {
  const pts = samples
    .map((s) => [s.time_s, s.exit?.hip_height_m])
    .filter(([, h]) => has(h));
  if (pts.length < 3) return '';
  const hs = pts.map((p) => p[1]);
  const lo = Math.min(...hs), hi = Math.max(...hs);
  const range = Math.max(hi - lo, 0.05);
  // A gap in the series is a gap in the drawing; the line never bridges one.
  const stepMax = (samples.at(-1).time_s / Math.max(samples.length - 1, 1)) * 4;
  const d = pts.map(([t, h], i) => {
    const jump = i === 0 || t - pts[i - 1][0] > stepMax;
    return `${jump ? 'M' : 'L'}${((t / span) * 1000).toFixed(2)} ${(100 - ((h - lo) / range) * 88 - 6).toFixed(2)}`;
  }).join(' ');
  return `<div class="trace">
    <svg viewBox="0 0 1000 100" preserveAspectRatio="none" aria-hidden="true">
      <path d="${d}" fill="none" stroke="var(--accent)" stroke-width="1.6"
        vector-effect="non-scaling-stroke" stroke-linejoin="round"/>
    </svg>
    <span class="trace-key">hip height, ${lo.toFixed(2)} m to ${hi.toFixed(2)} m</span>
  </div>`;
}

function contiguous(samples) {
  const runs = [];
  for (const s of samples) {
    const state = s.risk?.state || 'watch';
    const last = runs.at(-1);
    if (last && last.state === state) last.end = s.time_s;
    else runs.push({ state, start: s.time_s, end: s.time_s });
  }
  const tail = runs.at(-1);
  if (tail && tail.end === tail.start) tail.end = tail.start + 0.05;
  return runs;
}

/** The readouts beside the ribbon, for whichever sample the playhead is on. */
export function readoutsHtml(sample) {
  const e = sample?.exit || {};
  const g = sample?.gait;
  const tiles = [
    ['Hip height', unit(e.hip_height_m, ' m', 2)],
    ['Knee angle', unit(e.knee_deg, '°', 0)],
    ['Trunk angle', unit(e.trunk_deg, '°', 0)],
    ['Lean over feet', unit(e.lean_offset, '', 2)],
    ['Steadiness', g?.scored ? unit(g.score, '', 2) : '<span class="faint">not scored</span>'],
  ];
  return tiles.map(([k, v]) => `<div><dt>${k}</dt><dd>${v}</dd></div>`).join('');
}

export function currentWords(sample) {
  const risk = sample?.risk || {};
  const view = sample?.view || {};
  const reasons = (risk.reasons || []).filter((r, i, a) => a.indexOf(r) === i);
  const bits = [];
  if (reasons.length) bits.push(`<b>Why:</b> ${esc(reasons.join('; '))}.`);
  if (view.state && view.state !== 'usable') {
    bits.push(`<b>View:</b> ${esc(view.state)} — ${esc(view.remedy || view.detail || 'no detail given')}.`);
  }
  if (sample?.gait?.insufficient) bits.push(`<b>Steadiness:</b> ${esc(sample.gait.insufficient)}.`);
  return bits.join(' ') || '<span class="faint">Nothing worth saying about this instant.</span>';
}
