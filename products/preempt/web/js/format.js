/* Units, words and the small vocabulary the whole page shares.
   Every number rendered by this file carries its unit. */

export const esc = (s) =>
  String(s ?? '').replace(/[&<>"']/g, (c) =>
    ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c]);

export const has = (v) => v !== null && v !== undefined && v !== '';

/** A number with its unit, or an em dash that says why there is no number. */
export function unit(value, u, digits = 2, absent = '—') {
  if (!has(value) || (typeof value === 'number' && !Number.isFinite(value))) return absent;
  const n = typeof value === 'number' ? value.toFixed(digits) : value;
  return `${n}<u>${esc(u)}</u>`;
}

export const secs = (v, d = 2) => (has(v) ? `${Number(v).toFixed(d)} s` : '—');
export const metres = (v, d = 2) => (has(v) ? `${Number(v).toFixed(d)} m` : '—');
export const degrees = (v, d = 0) => (has(v) ? `${Number(v).toFixed(d)}°` : '—');

export function bytes(n) {
  if (!has(n)) return '—';
  if (n === 0) return '0 bytes';
  if (n < 1000) return `${n} bytes`;
  if (n < 1e6) return `${(n / 1000).toFixed(1)} kB`;
  return `${(n / 1e6).toFixed(2)} MB`;
}

/** mm:ss.mmm from the start of the recording — a timestamp in a log, so monospace. */
export function stamp(t) {
  if (!has(t)) return '—';
  const total = Number(t);
  const m = Math.floor(total / 60);
  const s = Math.floor(total % 60);
  const ms = Math.round((total - Math.floor(total)) * 1000);
  return `${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}.${String(ms).padStart(3, '0')}`;
}

/* The six states the product can be in. Each carries a word and a tone; the
   tone drives a chip whose mark is a dot, so colour is never alone. */
export const STATES = {
  'settled': { word: 'Settled', tone: 'ok', band: 'var(--ok-wash)', dark: false },
  'watch': { word: 'Moving', tone: 'plain', band: 'var(--inset)', dark: false },
  'rising soon': { word: 'About to get up', tone: 'warn', band: 'var(--warn-wash)', dark: false },
  'unsteady': { word: 'Unsteady on their feet', tone: 'warn', band: 'var(--warn)', dark: true },
  'on the floor': { word: 'On the floor', tone: 'hot', band: 'var(--danger)', dark: true },
  'view unusable': { word: 'Cannot see the room', tone: 'refuse', band: 'var(--refuse-wash)', dark: false },
};
export const stateOf = (s) =>
  STATES[s] || { word: s || 'unknown', tone: 'plain', band: 'var(--inset)', dark: false };

/* The three rungs Preempt escalates through, plus the one it rings for itself. */
export const RUNGS = {
  nudge: {
    n: 1, name: 'A quiet nudge in the room',
    what: 'A light and a recorded voice, in the room itself. Nobody is called and nothing is logged against the patient.',
  },
  station: {
    n: 2, name: 'A call to the nurses’ station',
    what: 'The bed, the reason and the two numbers go to the handsets. No picture travels with it, because no picture exists.',
  },
  urgent: {
    n: 3, name: 'An urgent call',
    what: 'Someone is on the floor. Every handset in the bay alerts, and the alert does not stop on its own.',
  },
  maintenance: {
    n: 0, name: 'A call to maintenance',
    what: 'Not about the patient. The camera cannot see, and somebody has to go and look at the lens.',
  },
};
export const rungOf = (r) => RUNGS[r] || { n: 0, name: r, what: '' };

export const VIEW_BANDS = {
  usable: 'var(--ok)',
  blocked: 'var(--refuse)',
  dark: 'var(--fg-faint)',
  blurred: 'var(--warn)',
  moved: 'var(--warn)',
  glare: 'var(--warn)',
};
export const viewColour = (s) => VIEW_BANDS[s] || 'var(--border-strong)';
