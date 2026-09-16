/* The signature element.

   The privacy claim is not a badge. It is the camera's entire output for one
   instant, printed as numbers, beside the figure those numbers produced, above
   a line saying how many bytes left the device and how many frames were kept. */

import { esc, has, bytes, secs, viewColour } from './format.js';
import { evidenceUrl } from './api.js';

export function ledgerPanel(frame, privacy) {
  const kps = frame?.keypoints || [];
  const seen = kps.filter((k) => k.seen).length;
  const rows = kps.map((k, i) => `<tr data-kp="${i}" class="${k.seen ? '' : 'low'}">
      <td class="name">${esc(k.name)}</td>
      <td class="xy">${k.x.toFixed(1)}</td>
      <td class="xy">${k.y.toFixed(1)}</td>
      <td class="sc">${k.score.toFixed(3)}</td>
    </tr>`).join('');

  return `<section class="panel" id="sec-privacy" aria-labelledby="led-h">
    <div class="ph"><h2 id="led-h">Camera output, this instant</h2>
      <span class="sub mono">frame ${frame?.index ?? '—'} · ${secs(frame?.time_s)}</span></div>
    <table class="ledger">
      <thead><tr><th>Keypoint</th><th class="num">x</th><th class="num">y</th><th class="num">score</th></tr></thead>
      <tbody id="ledger-body">${rows}</tbody>
    </table>
    <div class="pledge">
      <p class="l"><span>Keypoints above the threshold</span><b>${seen} of ${kps.length}</b></p>
      <p class="l"><span>Camera bytes read from the sensor</span><b>${bytes(privacy.camera_bytes_read)}</b></p>
      <p class="l"><span>Camera bytes written to disk</span><b class="is-ok">${bytes(privacy.camera_bytes_persisted)}</b></p>
      <p class="l"><span>Frames examined</span><b>${privacy.frames_examined?.toLocaleString() ?? '—'}</b></p>
      <p class="l"><span>Frames retained</span><b class="is-ok">${privacy.frames_retained ?? '—'}</b></p>
      <p class="l"><span>Keypoints kept, whole run</span><b>${bytes(privacy.keypoint_bytes_kept)}</b></p>
      <p class="say">That list is all of it. No frame outlived the pose that was read from
        it, so there is no photograph to leak, subpoena or lose.</p>
    </div>
  </section>`;
}

export function artefactsPanel(privacy) {
  const rows = (privacy.artefacts || []).map((a) => `<tr>
      <td><b>${esc(a.name)}</b><div class="faint">${esc(a.provenance)}, ${bytes(a.bytes)}</div></td>
      <td class="hash">${esc(a.sha256)}</td>
    </tr>`).join('');
  return `<section class="panel" aria-labelledby="art-h">
    <div class="ph"><h2 id="art-h">Everything this run wrote down</h2>
      <span class="sub">${privacy.clean ? 'ledger balances' : 'ledger does not balance'}</span></div>
    <table class="artefacts">
      <thead><tr><th>Artefact</th><th>SHA-256</th></tr></thead>
      <tbody>${rows || '<tr><td colspan="2" class="dim">Nothing was written.</td></tr>'}</tbody>
    </table>
    <div class="pledge">
      <p class="l"><span>Written from synthetic drawing</span><b>${bytes(privacy.synthetic_bytes_persisted)}</b></p>
      <p class="l"><span>Written from camera pixels</span><b class="is-ok">${bytes(privacy.camera_bytes_persisted)}</b></p>
      <p class="l"><span>Privacy mode</span><b>${esc(privacy.mode)}</b></p>
    </div>
  </section>`;
}

const sentence = (t) => (/[.!?]$/.test(String(t).trim()) ? t : `${t}.`);

/** What it could see, and — when it could not — what to do about that. */
export function viewPanel(record, lead = false) {
  const view = record.metrics.view || {};
  const refusal = (record.refusals || []).find((r) => r.code === 'VIEW_UNUSABLE');
  const degraded = refusal || (view.usable_fraction ?? 1) < 1;
  const total = view.frames || 1;
  const by = Object.entries(view.by_state || {});

  // Anything that is not a usable frame is also hatched, so the bar is never colour alone.
  const bar = by.map(([k, n]) =>
    `<i class="${k === 'usable' ? '' : 'hatched'}"
      style="width:${((n / total) * 100).toFixed(2)}%;background-color:${viewColour(k)}"></i>`).join('');
  const keys = by.map(([k, n]) =>
    `<span><i class="${k === 'usable' ? '' : 'hatched'}" style="background-color:${viewColour(k)}"></i>${esc(k)} — ${n} of ${total} frames</span>`).join('');

  const head = degraded
    ? `<h4>Preempt is not watching, and will not guess</h4>
       <p class="dim">${sentence(refusal?.message || 'The view was not usable for part of this recording')}
       For those frames there is no state, no steadiness score and no call about the patient —
       a gap Preempt reports rather than fills in. Withholding is the price of being allowed in the room at all.</p>`
    : `<h4>The view held for the whole recording</h4>
       <p class="dim">Every frame was gradeable, so every state on the ribbon is a reading rather than a gap.</p>`;

  const remedy = refusal ? `<p class="remedy"><b>What would fix it</b>
      ${esc(refusal.details?.remedy || refusal.message)}
      ${has(refusal.details?.frames) ? `<br><span class="faint">${refusal.details.frames} of ${total} frames were affected.</span>` : ''}</p>` : '';

  const meter = `<div><div class="viewbar" role="img"
      aria-label="${esc(by.map(([k, n]) => `${k}: ${n} frames`).join(', '))}">${bar}</div>
    <div class="viewkeys">${keys}</div>${remedy}</div>`;

  return `<section class="panel ${degraded ? 'refusal' : ''} ${lead ? 'lead' : ''}"
      id="sec-view" aria-labelledby="view-h">
    <div class="ph"><h3 id="view-h">What it could see</h3>
      ${degraded ? '<span class="chip refuse">watching withheld</span>' : '<span class="chip ok">view held</span>'}
      <span class="sub">${((view.usable_fraction ?? 1) * 100).toFixed(1)}% of frames usable</span></div>
    <div class="pb">${lead ? `<div>${head}</div>${meter}` : `${head}${meter}`}</div>
  </section>`;
}

export function evidencePanel(record, jobId) {
  const items = (record.evidence || []).map((ev) => `<figure>
      <div class="plate"><img src="${evidenceUrl(jobId, ev.uri)}" alt="${esc(ev.caption || ev.label)}"></div>
      <figcaption><b>${esc(ev.label)}${has(ev.timestamp_ms) ? ` · ${(ev.timestamp_ms / 1000).toFixed(1)} s` : ''}</b>
        ${esc(ev.caption || '')}</figcaption>
    </figure>`).join('');
  if (!items) return '';
  return `<section class="panel" id="sec-evidence" data-demo="evidence" aria-labelledby="ev-h">
    <div class="ph"><h2 id="ev-h">Drawn by the service, from keypoints only</h2>
      <span class="sub">${record.evidence.length} images, 0 of them photographs</span></div>
    <div class="evid">${items}</div>
  </section>`;
}
