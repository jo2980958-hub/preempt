/* The two things on the page that move: the playhead along the ribbon, and
   the link between a printed keypoint and the dot it drew. */

import { esc, stateOf } from './format.js';
import { highlight } from './figure.js';
import { readoutsHtml, currentWords } from './timeline.js';
import { pick, samplesOf, PLAY_ICON, PAUSE_ICON } from './room.js';

const reduced = () => window.matchMedia('(prefers-reduced-motion: reduce)').matches;

const clock = (t) => {
  const m = Math.floor(t / 60), s = Math.floor(t % 60), ms = Math.round((t % 1) * 1000);
  return `${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}.${String(ms).padStart(3, '0')}`;
};

/** Pointing at a row in the ledger rings the dot it produced, and the reverse. */
export function wireLedger(root, frame) {
  if (!frame) return;
  const svg = root.querySelector('.field svg');
  const rows = [...(root.querySelector('#ledger-body')?.querySelectorAll('tr') || [])];
  const light = (i) => {
    rows.forEach((r) => r.classList.toggle('lit', Number(r.dataset.kp) === i));
    highlight(svg, frame, i);
  };
  rows.forEach((r) => {
    r.addEventListener('mouseenter', () => light(Number(r.dataset.kp)));
    r.addEventListener('mouseleave', () => light(null));
  });
  svg?.querySelectorAll('.kp-hit').forEach((hit) => {
    const i = Number(hit.dataset.kp);
    ['mouseenter', 'focus'].forEach((ev) => hit.addEventListener(ev, () => light(i)));
    ['mouseleave', 'blur'].forEach((ev) => hit.addEventListener(ev, () => light(null)));
  });
}

export function wireTimeline(root, record) {
  const samples = samplesOf(record);
  if (!samples.length) return;
  const el = {
    range: root.querySelector('#tl-range'),
    play: root.querySelector('#tl-play'),
    now: root.querySelector('#tl-now'),
    chip: root.querySelector('#tl-chip'),
    head: root.querySelector('#tl-head'),
    outs: root.querySelector('#tl-readouts'),
    why: root.querySelector('#tl-why'),
  };
  const span = record.metrics.duration_s || samples.at(-1).time_s || 1;

  let i = 0, timer = null;
  const show = (n) => {
    i = Math.max(0, Math.min(samples.length - 1, n));
    const s = samples[i];
    const st = stateOf(s.risk?.state);
    el.range.value = String(i);
    el.now.textContent = clock(s.time_s);
    el.chip.innerHTML = `<span class="chip ${st.tone}">${esc(st.word)}</span>`;
    el.head.style.left = `${((s.time_s / span) * 100).toFixed(3)}%`;
    el.outs.innerHTML = readoutsHtml(s);
    el.why.innerHTML = currentWords(s);
  };
  const stop = () => {
    clearInterval(timer);
    timer = null;
    el.play.setAttribute('aria-label', 'Play the recording');
    el.play.innerHTML = PLAY_ICON;
  };

  el.play.addEventListener('click', () => {
    if (timer) return stop();
    if (i >= samples.length - 1) show(0);
    el.play.setAttribute('aria-label', 'Pause the recording');
    el.play.innerHTML = PAUSE_ICON;
    timer = reduced() ? stepStates(samples, i, show, stop) : sweep(samples, span, () => i, show, stop);
  });
  el.range.addEventListener('input', () => { stop(); show(Number(el.range.value)); });

  // Open on the instant the ledger frame belongs to: the moment that mattered.
  const peak = pick(record, 'keypoint-ledger')?.frame?.time_s ?? 0;
  show(samples.reduce((best, s, n) =>
    Math.abs(s.time_s - peak) < Math.abs(samples[best].time_s - peak) ? n : best, 0));
}

/** Real time, one sample per frame interval. */
function sweep(samples, span, at, show, stop) {
  const stepMs = Math.max(1000 / Math.max(samples.length / span, 1), 16);
  return setInterval(() => (at() >= samples.length - 1 ? stop() : show(at() + 1)), stepMs);
}

/** With reduced motion the playhead does not sweep. It lands on each state
    change in turn, so what renders is always an end state. */
function stepStates(samples, from, show, stop) {
  const stops = samples.reduce((acc, s, n) => {
    if (n === 0 || s.risk?.state !== samples[n - 1].risk?.state) acc.push(n);
    return acc;
  }, []);
  let at = stops.findIndex((n) => n > from);
  return setInterval(() => {
    if (at < 0 || at >= stops.length) return stop();
    show(stops[at]);
    at += 1;
  }, 800);
}
