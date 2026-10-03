/* The chrome around the results: the status word, the progress card, and which
   section of the run the reader has reached. */

import { ui } from './dom.js';

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
