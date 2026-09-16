/* Assembles the results region: the KPI band, then two columns of evidence. */

import { esc, has } from './format.js';
import { pick, roomPanel } from './room.js';
import { wireLedger, wireTimeline } from './scrub.js';
import { riskPanel, callsPanel, hazardsPanel, setupPanel } from './panels.js';
import { ledgerPanel, artefactsPanel, viewPanel, evidencePanel } from './privacy.js';

export function kpis(record) {
  const m = record.metrics;
  const lead = m.lead_time_s;
  const usable = (m.view?.usable_fraction ?? 1) * 100;
  const p = m.privacy || {};
  const tiles = [
    { ...leadTile(lead, usable < 100), k: 'Warning before upright' },
    { cls: m.highest_rung === 'urgent' ? 'hot' : (m.calls ? '' : 'mute'), k: 'Calls raised',
      v: `${m.calls ?? 0}`, n: `highest rung: ${esc(m.highest_rung || 'none')}` },
    { cls: usable < 100 ? 'refuse' : '', k: 'View usable',
      v: `${usable.toFixed(1)}<u>%</u>`, n: `${m.view?.frames ?? 0} frames graded` },
    { cls: 'good', k: 'Camera bytes kept', v: `${p.camera_bytes_persisted ?? 0}<u>B</u>`,
      n: `${p.frames_retained ?? 0} of ${p.frames_examined ?? 0} frames retained` },
    { cls: 'mute', k: 'Recording watched', v: `${(m.duration_s ?? 0).toFixed(1)}<u>s</u>`,
      n: `${m.samples ?? 0} pose samples` },
  ];
  return `<div class="kpis">${tiles.map((t) =>
    `<div class="kpi ${t.cls}"><div class="k">${t.k}</div><div class="v">${t.v}</div>
     <div class="n">${t.n}</div></div>`).join('')}</div>`;
}

/** The headline number, or an honest reason there is not one. */
function leadTile(lead, blind) {
  if (has(lead)) {
    return { cls: 'good', v: `${lead.toFixed(2)}<u>s</u>`,
      n: 'between the first call and standing upright' };
  }
  if (blind) {
    return { cls: 'refuse', v: '<span class="na">Not measurable</span>',
      n: 'the view was blocked for part of the recording' };
  }
  return { cls: 'mute', v: 'none<u>needed</u>', n: 'nobody left the bed or the chair' };
}

export function renderResults(root, record, jobId) {
  const privacy = record.metrics.privacy || {};
  const frame = pick(record, 'keypoint-ledger')?.frame;
  // A refused view is a result, not an error, so when it happens it leads the page.
  const blind = (record.refusals || []).some((r) => r.code === 'VIEW_UNUSABLE')
    || (record.metrics.view?.usable_fraction ?? 1) < 1;
  const view = viewPanel(record, blind);

  root.innerHTML = `
    ${kpis(record)}
    ${setupPanel(record)}
    ${heldBack(record)}
    ${blind ? view : ''}
    <div class="cols">
      <div class="col">
        ${roomPanel(record)}
        ${riskPanel(record, pick(record, 'risk'))}
        ${callsPanel(pick(record, 'calls')?.calls || [])}
        ${evidencePanel(record, jobId)}
      </div>
      <div class="col">
        ${ledgerPanel(frame, privacy)}
        ${blind ? '' : view}
        ${hazardsPanel(pick(record, 'hazards'))}
        ${artefactsPanel(privacy)}
      </div>
    </div>`;

  wireLedger(root, frame);
  wireTimeline(root, record);
}

/** Refusals other than the unusable view, and any warnings, said plainly. */
function heldBack(record) {
  const others = (record.refusals || []).filter((r) => r.code !== 'VIEW_UNUSABLE');
  const warnings = record.warnings || [];
  if (!others.length && !warnings.length) return '';
  return `<section class="panel refusal"><div class="ph"><h3>Preempt held something back</h3></div>
    <div class="pb"><ul class="haz">
      ${others.map((r) =>
    `<li><span class="m chip refuse">${esc(r.code)}</span><span>${esc(r.message)}</span></li>`).join('')}
      ${warnings.map((w) =>
      `<li><span class="m chip warn">warning</span><span>${esc(w)}</span></li>`).join('')}
    </ul></div></section>`;
}
