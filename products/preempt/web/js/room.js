/* The room panel: the drawn figure on a plain field, and the scrubbable
   ribbon under it.

   The timeline samples carry a state per frame but no keypoints — only the
   ledger frame has those — so the figure is the one retained instant and the
   panel says so, rather than animating coordinates it does not have. */

import { esc, has, secs, stateOf } from './format.js';
import { drawFigure } from './figure.js';
import { ribbonHtml, readoutsHtml } from './timeline.js';

export const pick = (record, kind) => (record.results || []).find((r) => r.kind === kind);
export const samplesOf = (record) =>
  pick(record, 'timeline')?.samples?.filter((s) => has(s.time_s)) || [];

export const PLAY_ICON =
  '<svg width="15" height="15" viewBox="0 0 16 16" aria-hidden="true"><path d="M4.4 3.2l8 4.8-8 4.8z" fill="currentColor"/></svg>';
export const PAUSE_ICON =
  '<svg width="15" height="15" viewBox="0 0 16 16" aria-hidden="true"><path d="M4.6 3h2.6v10H4.6zM8.8 3h2.6v10H8.8z" fill="currentColor"/></svg>';

export function roomPanel(record) {
  const frame = pick(record, 'keypoint-ledger')?.frame;
  const calls = pick(record, 'calls')?.calls || [];
  const samples = samplesOf(record);
  const state = stateOf(record.metrics.peak_state);

  const keypoints = frame?.keypoints || [];
  const kept = keypoints.filter((k) => k.seen).length;
  const total = keypoints.length;
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
        <button class="tl-play" id="tl-play" type="button" aria-label="Play the recording">${PLAY_ICON}</button>
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
