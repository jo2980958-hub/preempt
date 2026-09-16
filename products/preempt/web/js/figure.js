/* The abstracted figure.

   This is the whole of what the device keeps: seventeen coordinates. The field
   behind the figure is a plain weave with nothing in it, because there is
   nothing behind it — no frame was retained to put there. */

import { esc } from './format.js';

const EDGES = [
  [5, 6], [5, 7], [7, 9], [6, 8], [8, 10], [5, 11], [6, 12],
  [11, 12], [11, 13], [13, 15], [12, 14], [14, 16], [0, 5], [0, 6],
];

const W = 720;
const H = 470;
/* The badges sit over the top of the field and the statement over the bottom,
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
    ${head(pts, at)}
    <g>${edges}</g>
    <g>${dots}</g>
    <g id="kp-lit"></g>
    <g>${hits}</g>
  </svg>`;
}

/** The head is five coordinates and no pixels; say so on the drawing. */
function head(pts, at) {
  const idx = [0, 1, 2, 3, 4].filter((i) => pts[i]?.seen);
  if (idx.length < 2) return '';
  const cx = idx.reduce((s, i) => s + at[i].x, 0) / idx.length;
  const cy = idx.reduce((s, i) => s + at[i].y, 0) / idx.length;
  const r = Math.max(24, Math.max(...idx.map((i) => Math.hypot(at[i].x - cx, at[i].y - cy))) + 13);
  const text = `${idx.length} head coordinates, no pixels`;
  const width = text.length * 9.2;
  const right = cx + r + 42 + width < W - 10;     // keep the note inside the field
  const dir = right ? 1 : -1;
  const tipX = Math.min(Math.max(cx + dir * (r + 34), width + 16), W - width - 16);
  const tipY = Math.max(cy - r - 20, 30);
  return `<g>
    <circle cx="${cx.toFixed(1)}" cy="${cy.toFixed(1)}" r="${r.toFixed(1)}"
      fill="none" stroke="var(--danger)" stroke-width="1.6" stroke-dasharray="5 5" opacity=".85"/>
    <line x1="${(cx + dir * r * 0.72).toFixed(1)}" y1="${(cy - r * 0.72).toFixed(1)}"
      x2="${tipX.toFixed(1)}" y2="${tipY.toFixed(1)}" stroke="var(--danger)" stroke-width="1.4"/>
    <text class="kp-label" text-anchor="${right ? 'start' : 'end'}"
      x="${(tipX + dir * 6).toFixed(1)}" y="${(tipY - 5).toFixed(1)}">${text}</text>
  </g>`;
}

export function emptyField(message) {
  return `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${esc(message)}">
    <rect width="${W}" height="${H}" fill="url(#pt-weave)"/>
    <text x="${W / 2}" y="${H / 2}" text-anchor="middle"
      style="font:600 18px var(--font-ui)" fill="var(--fg-muted)">${esc(message)}</text>
  </svg>`;
}

/** Ring and name the one keypoint the reader is pointing at. */
export function highlight(svg, frame, index) {
  const layer = svg?.querySelector('#kp-lit');
  if (!layer) return;
  const dot = svg.querySelector(`.kp-dot[data-kp="${index}"]`);
  if (index === null || !dot) { layer.innerHTML = ''; return; }
  const p = frame.keypoints[index];
  const cx = Number(dot.getAttribute('cx'));
  const cy = Number(dot.getAttribute('cy'));
  const label = `${p.name} · ${p.x}, ${p.y}`;
  const wide = label.length * 9.1;
  const right = cx < W - wide - 40;
  const bx = right ? cx + 16 : cx - 16 - wide;
  layer.innerHTML = `<circle class="kp-ring" cx="${cx}" cy="${cy}" r="11"/>
    <rect class="kp-plate" x="${bx.toFixed(1)}" y="${(cy - 30).toFixed(1)}"
      width="${wide.toFixed(1)}" height="26" rx="4"/>
    <text class="kp-label" x="${(bx + 8).toFixed(1)}" y="${(cy - 12).toFixed(1)}">${esc(label)}</text>`;
}
