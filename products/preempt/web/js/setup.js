/* The room a video will be measured against, drawn over its first frame before
   anything runs.

   Real footage showed why this exists: a CDC clip run against the default room
   put the patient's feet inside the default bed, so every stand read as
   "supported" and all three were missed. A wrong room gives heights that look
   exactly like right ones. Drawn over the picture, it is obvious in a second.

   The frame is decoded by the browser from the local file and never leaves it. */

import { ui } from './dom.js';
import { esc } from './format.js';
import * as api from './api.js';

const PROCESS_MAX_SIDE = 1280;   // the service scales frames to this before posing

let video = null;        // the local file, as a decoded first frame
let roomFile = null;     // the File the person picked, sent as-is with the job
let room = null;         // { room, summary } as the service validated it
let problem = '';        // why the room cannot be used, if it cannot

export const currentRoomFile = () => roomFile;
export const setupReady = () => Boolean(room) && !problem;

export function mountSetup(onGo) {
  ui.roomFile.addEventListener('change', async () => {
    roomFile = ui.roomFile.files?.[0] || null;
    await loadRoom();
    draw();
  });
  ui.previewGo.addEventListener('click', onGo);
}

export async function previewVideo(file) {
  ui.preview.hidden = false;
  ui.previewGo.disabled = true;
  video = await firstFrame(file).catch((err) => ({ error: err.message }));
  if (!room || (!roomFile && room.summary.source !== 'default')) await loadRoom();
  draw();
}

async function loadRoom() {
  problem = '';
  room = null;
  try {
    room = roomFile ? await api.checkRoom(await roomFile.text()) : await api.getDefaultRoom();
  } catch (err) {
    problem = `This room setup cannot be used: ${esc(err.message)}`;
  }
}

function firstFrame(file) {
  return new Promise((resolve, reject) => {
    const el = document.createElement('video');
    const url = URL.createObjectURL(file);
    const done = (fn, arg) => { URL.revokeObjectURL(url); fn(arg); };
    el.muted = true;
    el.preload = 'auto';
    el.playsInline = true;
    el.addEventListener('loadeddata', () => { el.currentTime = 0; }, { once: true });
    el.addEventListener('seeked', () => {
      const canvas = document.createElement('canvas');
      canvas.width = el.videoWidth;
      canvas.height = el.videoHeight;
      canvas.getContext('2d').drawImage(el, 0, 0);
      done(resolve, { canvas, width: el.videoWidth, height: el.videoHeight });
    }, { once: true });
    el.addEventListener('error', () =>
      done(reject, new Error('this browser cannot decode the video, so the room cannot be previewed')), { once: true });
    el.src = url;
  });
}

const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const ZONE_TOKEN = { bed: '--accent', chair: '--accent', door: '--ok', wall: '--fg-faint', aid: '--warn', exclude: '--refuse' };

function draw() {
  const setup = room?.room;
  const canvas = ui.previewCanvas;
  const ctx = canvas.getContext('2d');
  const w = video?.width || 1280;
  const h = video?.height || 720;
  canvas.width = w;
  canvas.height = h;
  ctx.fillStyle = css('--inset');
  ctx.fillRect(0, 0, w, h);
  if (video?.canvas) ctx.drawImage(video.canvas, 0, 0);

  // Room coordinates are in the frame the service poses, which is scaled down
  // to 1280 px on the long side; the preview is at the file's own size.
  const scale = Math.max(w, h) > PROCESS_MAX_SIDE ? Math.max(w, h) / PROCESS_MAX_SIDE : 1;
  const px = ([x, y]) => [x * scale, y * scale];
  const line = Math.max(2, w / 400);
  let outside = 0;
  let total = 0;
  const count = (pts) => pts.forEach(([x, y]) => {
    total += 1;
    if (x < 0 || y < 0 || x > w / scale || y > h / scale) outside += 1;
  });

  if (setup?.floor) {
    const quad = setup.floor.image_points;
    count(quad);
    poly(ctx, quad.map(px), css('--fg'), null, line, [line * 4, line * 3]);
  }
  for (const zone of setup?.zones || []) {
    count(zone.points);
    const colour = css(ZONE_TOKEN[zone.kind] || '--accent');
    poly(ctx, zone.points.map(px), colour, colour, line, []);
    label(ctx, zone.points.map(px), `${zone.name} (${zone.kind})`, colour, w);
  }
  if (video?.error) {
    ctx.fillStyle = css('--fg-muted');
    ctx.font = `${Math.round(w / 40)}px sans-serif`;
    ctx.fillText(video.error, w * 0.04, h / 2);
  }

  facts(setup, outside, total);
  ui.previewGo.disabled = !setupReady();
}

function poly(ctx, pts, stroke, fill, width, dash) {
  if (!pts.length) return;
  ctx.beginPath();
  pts.forEach(([x, y], i) => (i ? ctx.lineTo(x, y) : ctx.moveTo(x, y)));
  ctx.closePath();
  if (fill) {
    ctx.globalAlpha = 0.22;
    ctx.fillStyle = fill;
    ctx.fill();
    ctx.globalAlpha = 1;
  }
  ctx.setLineDash(dash);
  ctx.lineWidth = width;
  ctx.strokeStyle = stroke;
  ctx.stroke();
  ctx.setLineDash([]);
}

function label(ctx, pts, text, colour, w) {
  const x = Math.min(...pts.map((p) => p[0]));
  const y = Math.min(...pts.map((p) => p[1]));
  const size = Math.max(14, Math.round(w / 40));
  ctx.font = `600 ${size}px sans-serif`;
  const width = ctx.measureText(text).width;
  ctx.fillStyle = css('--surface');
  ctx.fillRect(x, y - size - 6, width + 12, size + 8);
  ctx.fillStyle = colour;
  ctx.fillText(text, x + 6, y - 6);
}

const SOURCE = { measured: 'measured', assumed: 'assumed', synthetic: 'exact (synthetic camera)', unstated: 'not stated' };

function facts(setup, outside, total) {
  const summary = room?.summary;
  const warn = [];
  if (problem) warn.push(problem);
  if (summary) {
    const cal = summary.calibration;
    const measured = cal.calibrated;
    const borrowed = summary.source === 'default';
    ui.previewCal.className = `chip ${measured && !borrowed ? 'ok' : 'warn'}`;
    ui.previewCal.textContent = borrowed ? "The synthetic ward's camera, not this one"
      : measured ? 'Camera measured' : 'Camera assumed, not measured';
    ui.previewFacts.innerHTML = `
      <div><dt>Room</dt><dd>${esc(summary.name)}</dd></div>
      <div><dt>Where it came from</dt><dd>${summary.source === 'default' ? 'the default room, from the synthetic ward' : 'the file you picked'}</dd></div>
      <div><dt>Camera height</dt><dd>${esc(SOURCE[cal.camera_height] || cal.camera_height)}</dd></div>
      <div><dt>Focal length</dt><dd>${esc(SOURCE[cal.focal_length] || cal.focal_length)}</dd></div>
      <div><dt>Zones</dt><dd>${(setup?.zones || []).map((z) => esc(z.name)).join(', ') || 'none'}</dd></div>`;
    if (summary.source === 'default') {
      warn.push('No room file picked, so this video will be measured against the synthetic ward. Unless it was filmed there, the zones and heights will not match it.');
    }
    if (!measured && cal.camera_height !== 'synthetic') {
      warn.push(`Every height in the result rests on a camera height and focal length that were ${cal.camera_height === 'unstated' ? 'never stated, so they are treated as assumed' : 'assumed'}. ${esc(cal.note || '')}`);
    }
    if (outside) {
      warn.push(`${outside} of ${total} points in this room fall outside the video frame. It was set up for a different camera.`);
    }
  } else {
    ui.previewCal.className = 'chip hot';
    ui.previewCal.textContent = 'No usable room';
    ui.previewFacts.innerHTML = '';
  }
  ui.previewWarn.hidden = !warn.length;
  ui.previewWarn.innerHTML = warn.map((w) => `<span>${w}</span>`).join('');
}
