/* The chrome around the results: appearance, the status word, the progress
   card, and which section of the run the reader has reached. */

import { $, ui } from './dom.js';

/* ── appearance. A ward dims its screens at night. ─────────────────── */
function setTheme(mode) {
  document.documentElement.dataset.theme = mode;
  try { localStorage.setItem('preempt-theme', mode); } catch { /* private mode */ }
  const dark = mode === 'dark';
  $('#theme-label').textContent = dark ? 'Bring the lights up' : 'Dim the screen';
  $('#theme').setAttribute('aria-pressed', String(dark));
}

export function initTheme() {
  let saved = null;
  try { saved = localStorage.getItem('preempt-theme'); } catch { /* ignore */ }
  const dark = saved ? saved === 'dark'
    : window.matchMedia('(prefers-color-scheme: dark)').matches;
  setTheme(dark ? 'dark' : 'light');
  $('#theme').addEventListener('click', () =>
    setTheme(document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark'));
}

/* ── the one element that says whether a job is running or finished ── */
export function status(text, tone = '') {
  ui.status.textContent = text;
  ui.status.className = `status ${tone}`;
}

export function beginProgress(message) {
  ui.progress.hidden = false;
  ui.fill.style.width = '2%';
  ui.pct.textContent = '0%';
  ui.msg.textContent = message;
}

export function stepProgress(percent, message) {
  if (percent !== null && percent !== undefined) {
    ui.fill.style.width = `${percent}%`;
    ui.pct.textContent = `${Math.round(percent)}%`;
  }
  if (message) ui.msg.textContent = message;
}

export function endProgress() {
  ui.fill.style.width = '100%';
  ui.pct.textContent = '100%';
  setTimeout(() => { ui.progress.hidden = true; }, 500);
}

export function showEnvironment(version) {
  $('#f-opencv').textContent = version.opencv_version || '—';
  $('#f-engine').textContent = Object.values(version.models || {})[0]?.name || 'RTMPose-t';
}

/* ── which section the reader has reached ──────────────────────────── */
export function watchSections() {
  const links = [...document.querySelectorAll('.nav a[href^="#sec-"]')];
  let pending = false;
  const mark = () => {
    pending = false;
    // Two columns means DOM order is not reading order, so pick the section
    // whose top has most recently passed the reading line.
    const line = window.innerHeight * 0.28;
    let current = null, best = -Infinity;
    for (const section of document.querySelectorAll('[id^="sec-"]')) {
      const top = section.getBoundingClientRect().top;
      if (top <= line && top > best) { best = top; current = section.id; }
    }
    links.forEach((a) => a.classList.toggle('on', a.getAttribute('href') === `#${current}`));
  };
  const schedule = () => {
    if (pending) return;
    pending = true;
    requestAnimationFrame(mark);
  };
  window.addEventListener('scroll', schedule, { passive: true });
  window.addEventListener('resize', schedule);
  new MutationObserver(schedule).observe(ui.results, { childList: true });
}
