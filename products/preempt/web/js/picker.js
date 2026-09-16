/* The sample picker, and the file input beside it.

   A judge on a cold start should be one click from the whole loop, so the
   clean bed exit leads and the case where the camera cannot see closes. */

import { ui } from './dom.js';
import { esc, bytes } from './format.js';

export const FIRST = 'bed-exit-steady';
export const FAILURE_CASE = 'curtain-drawn';

let samples = [];
let chosen = FIRST;
let upload = null;
let onPick = () => {};

export const currentSample = () => chosen;
export const currentUpload = () => upload;

export function mountPicker(handler) {
  onPick = handler;
  ui.file.addEventListener('change', () => {
    upload = ui.file.files?.[0] || null;
    if (!upload) return;
    ui.samples.querySelectorAll('.sample').forEach((b) => b.setAttribute('aria-pressed', 'false'));
    ui.startLabel.textContent = 'Watch this video';
    ui.startNote.textContent = `${upload.name}, ${bytes(upload.size)}`;
    onPick();
  });
}

export function setSamples(list) {
  samples = list;
  // The clean bed exit leads, the failure case closes, everything else keeps
  // the order the service gave it.
  const rank = (s) => (s.name === FIRST ? 0 : s.name === FAILURE_CASE ? 2 : 1);
  const ordered = [...samples].sort((a, b) => rank(a) - rank(b));
  document.querySelector('#picker-h').textContent = `${samples.length} recorded rooms`;
  ui.samples.innerHTML = ordered.map(card).join('');
  ui.samples.querySelectorAll('.sample').forEach((b) =>
    b.addEventListener('click', () => choose(b.dataset.name)));
}

function card(s) {
  const tag = s.name === FIRST ? '<span class="tag">Start here</span>'
    : s.name === FAILURE_CASE ? '<span class="tag refuse">The failure case</span>' : '';
  return `<button class="sample" type="button" data-demo="sample-${esc(s.name)}"
      data-name="${esc(s.name)}" aria-pressed="${s.name === chosen}">
      <span class="sn">${esc(s.name)}${tag}</span>
      <span class="sd">${esc(s.description)}</span>
      <span class="sm">${(s.duration_s ?? 0).toFixed(1)} s of pose track</span>
    </button>`;
}

function choose(name) {
  chosen = name;
  upload = null;
  ui.file.value = '';
  ui.samples.querySelectorAll('.sample').forEach((b) =>
    b.setAttribute('aria-pressed', String(b.dataset.name === name)));
  const s = samples.find((x) => x.name === name);
  ui.startLabel.textContent = 'Watch this room';
  ui.startNote.textContent = s
    ? `${s.name}, ${(s.duration_s ?? 0).toFixed(1)} s of pose track`
    : name;
  onPick();
}

/** The list did not load. The upload still works, so say that rather than sulk. */
export function pickerUnavailable(message) {
  ui.samples.innerHTML = `<p class="loading">The sample list did not load: ${esc(message)}.
    The upload below still works.</p>`;
}

export const compactPicker = () => ui.picker.classList.add('compact');
