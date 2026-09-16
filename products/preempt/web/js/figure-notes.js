/* What gets drawn over the figure: the note saying the head is coordinates
   rather than a face, and the ring that names one keypoint on demand. */

import { esc } from './format.js';

/** The head is a handful of numbers and no pixels; say so on the drawing. */
export function headNote(pts, at, width) {
  const idx = [0, 1, 2, 3, 4].filter((i) => pts[i]?.seen);
  if (idx.length < 2) return '';
  const cx = idx.reduce((s, i) => s + at[i].x, 0) / idx.length;
  const cy = idx.reduce((s, i) => s + at[i].y, 0) / idx.length;
  const r = Math.max(24, Math.max(...idx.map((i) => Math.hypot(at[i].x - cx, at[i].y - cy))) + 13);

  const text = `${idx.length} head coordinates, no pixels`;
  const runs = text.length * 9.2;
  const right = cx + r + 42 + runs < width - 10;   // keep the note inside the field
  const dir = right ? 1 : -1;
  const tipX = Math.min(Math.max(cx + dir * (r + 34), runs + 16), width - runs - 16);
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

/** Ring and name the one keypoint the reader is pointing at. */
export function highlight(svg, frame, index, width) {
  const layer = svg?.querySelector('#kp-lit');
  if (!layer) return;
  const dot = svg.querySelector(`.kp-dot[data-kp="${index}"]`);
  if (index === null || !dot) { layer.innerHTML = ''; return; }

  const p = frame.keypoints[index];
  const cx = Number(dot.getAttribute('cx'));
  const cy = Number(dot.getAttribute('cy'));
  const label = `${p.name} · ${p.x}, ${p.y}`;
  const runs = label.length * 9.1;
  const right = cx < width - runs - 40;
  const bx = right ? cx + 16 : cx - 16 - runs;

  layer.innerHTML = `<circle class="kp-ring" cx="${cx}" cy="${cy}" r="11"/>
    <rect class="kp-plate" x="${bx.toFixed(1)}" y="${(cy - 30).toFixed(1)}"
      width="${runs.toFixed(1)}" height="26" rx="4"/>
    <text class="kp-label" x="${(bx + 8).toFixed(1)}" y="${(cy - 12).toFixed(1)}">${esc(label)}</text>`;
}
