/* The abstracted figure.

   This is the whole of what the device keeps: seventeen coordinates. The field
   behind the figure is a plain weave with nothing in it, because there is
   nothing behind it — no frame was retained to put there. */

import { esc } from './format.js';
import { headNote, highlight as ringKeypoint } from './figure-notes.js';

const EDGES = [
  [5, 6], [5, 7], [7, 9], [6, 8], [8, 10], [5, 11], [6, 12],
  [11, 12], [11, 13], [13, 15], [12, 14], [14, 16], [0, 5], [0, 6],
];

const W = 720;
const H = 470;
/* The state chip sits above the field and the statement over the bottom of it,
   so the figure is fitted into what is left rather than into the whole panel. */
const BOX = { x: 130, y: 58, w: W - 260, h: H - 58 - 142 };

/** Fit the recorded pixel coordinates into the field without distorting them. */
function fit(points) {
  const live = points.filter((p) => p.seen);
  const use = live.length >= 2 ? live : points;
  const xs = use.map((p) => p.x);
  const ys = use.map((p) => p.y);
  const minX = Math.min(...xs), maxX = Math.max(...xs);
  const minY = Math.min(...ys), maxY = Math.max(...ys);
  const spanX = Math.max(maxX - minX, 1);
  const spanY = Math.max(maxY - minY, 1);
  const scale = Math.min(BOX.w / spanX, BOX.h / spanY);
  const offX = BOX.x + (BOX.w - spanX * scale) / 2 - minX * scale;
  const offY = BOX.y + (BOX.h - spanY * scale) / 2 - minY * scale;
  return (p) => ({ x: p.x * scale + offX, y: p.y * scale + offY });
}

/**
 * Draw the figure for one frame of the keypoint ledger.
 * @param {{index:number,time_s:number,keypoints:Array}} frame
 */
export function drawFigure(frame) {
  const pts = frame?.keypoints || [];
  if (!pts.length) return emptyField('No keypoints were kept for this run.');

  const project = fit(pts);
  const at = pts.map(project);
  const seen = pts.map((p) => p.seen);

  const edges = EDGES
    .filter(([a, b]) => seen[a] && seen[b] && at[a] && at[b])
    .map(([a, b]) => `<line class="kp-edge" x1="${at[a].x.toFixed(1)}" y1="${at[a].y.toFixed(1)}"
      x2="${at[b].x.toFixed(1)}" y2="${at[b].y.toFixed(1)}"/>`)
    .join('');

  const dots = pts.map((p, i) => {
    const r = p.seen ? 3.4 + p.score * 3.6 : 3.6;
    return `<circle class="kp-dot${p.seen ? '' : ' unseen'}" data-kp="${i}"
      cx="${at[i].x.toFixed(1)}" cy="${at[i].y.toFixed(1)}" r="${r.toFixed(1)}"/>`;
  }).join('');

  const hits = pts.map((p, i) =>
    `<circle class="kp-hit" data-kp="${i}" tabindex="0" role="img"
      aria-label="${esc(p.name)}, x ${p.x}, y ${p.y}, score ${p.score}"
      cx="${at[i].x.toFixed(1)}" cy="${at[i].y.toFixed(1)}" r="13"/>`).join('');

  return `<svg viewBox="0 0 ${W} ${H}" role="group"
      aria-label="The figure Preempt draws from seventeen coordinates. No photograph exists.">
    <rect width="${W}" height="${H}" fill="url(#pt-weave)"/>
    ${headNote(pts, at, W)}
    <g>${edges}</g>
    <g>${dots}</g>
    <g id="kp-lit"></g>
    <g>${hits}</g>
  </svg>`;
}

export function emptyField(message) {
  return `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${esc(message)}">
    <rect width="${W}" height="${H}" fill="url(#pt-weave)"/>
    <text x="${W / 2}" y="${H / 2}" text-anchor="middle"
      style="font:600 18px var(--font-ui)" fill="var(--fg-muted)">${esc(message)}</text>
  </svg>`;
}

export const highlight = (svg, frame, index) => ringKeypoint(svg, frame, index, W);
