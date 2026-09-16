/* What a finished run says about itself outside the results region:
   the page title, the topbar crumb and privacy chip, and the rail's counts. */

import { $, ui } from './dom.js';
import { bytes } from './format.js';

export function describeRun(record) {
  const m = record.metrics;
  const room = record.input?.room || 'the room';
  ui.title.textContent = room.replace(/^\w/, (c) => c.toUpperCase());
  ui.subtitle.textContent = record.input?.source
    ? `${record.input.source}, at ${record.input.fps ?? '—'} frames a second.`
      + ' This page is everything the device knows about it.'
    : 'This page is everything the device knows about the recording.';
  ui.crumb.textContent = room;
  ui.crumbDot.className = `dot ${m.highest_rung === 'urgent' ? 'hot' : 'live'}`;
  ui.source.textContent = `${record.input?.source || 'recording'} · ${(m.duration_s ?? 0).toFixed(1)} s`;

  const p = m.privacy || {};
  const clean = p.camera_bytes_persisted === 0;
  ui.privacy.className = clean ? 'chip ok' : 'chip hot';
  ui.privacy.textContent = clean
    ? 'Nothing recognisable leaves this room'
    : `${bytes(p.camera_bytes_persisted)} of camera pixels were written`;
}

export function railCounts(record) {
  const m = record.metrics;
  const p = m.privacy || {};
  const counts = {
    keypoints: (p.frames_examined ?? 0).toLocaleString(),
    movement: String(stateRuns(record)),
    calls: String(m.calls ?? 0),
    hazards: String((record.results.find((r) => r.kind === 'hazards')?.hazards || []).length),
    bytes: `${p.camera_bytes_persisted ?? 0} B`,
    view: `${((m.view?.usable_fraction ?? 1) * 100).toFixed(0)}%`,
  };
  for (const [key, value] of Object.entries(counts)) {
    const el = document.querySelector(`.nav [data-count="${key}"]`);
    if (el) el.textContent = value;
  }
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
