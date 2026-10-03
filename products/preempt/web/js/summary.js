/* What a finished run says about itself outside the results region: the page
   heading and the one line in the topbar that names what was watched. */

import { ui } from './dom.js';

export function describeRun(record) {
  const m = record.metrics;
  const room = record.input?.room || 'the room';
  ui.title.textContent = room.replace(/^\w/, (c) => c.toUpperCase());
  ui.subtitle.textContent = record.input?.source
    ? `${record.input.source}, at ${record.input.fps ?? '—'} frames a second.`
    : 'Everything below was read from this recording.';
  ui.crumb.textContent = room;
  ui.crumbDot.className = `dot ${m.highest_rung === 'urgent' ? 'hot' : 'live'}`;
  ui.source.textContent = `${record.input?.source || 'recording'} · ${(m.duration_s ?? 0).toFixed(1)} s`;
  document.querySelector('.nav a[href="#sec-calls"]')
    ?.classList.toggle('alert', (m.calls ?? 0) > 0);
}
