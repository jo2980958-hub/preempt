/* The elements the shell owns, looked up once. */

export const $ = (sel) => document.querySelector(sel);

export const ui = {
  status: $('#status'),
  results: $('#results'),
  empty: $('#empty'),
  picker: $('#picker'),
  samples: $('#samples'),
  progress: $('#progress'),
  fill: $('#progress-fill'),
  msg: $('#progress-msg'),
  pct: $('#progress-pct'),
  start: $('#start'),
  startLabel: $('#start-label'),
  startNote: $('#start-note'),
  file: $('#file'),
  roomFile: $('#room-file'),
  preview: $('#preview'),
  previewCanvas: $('#preview-canvas'),
  previewCal: $('#preview-cal'),
  previewFacts: $('#preview-facts'),
  previewWarn: $('#preview-warn'),
  previewGo: $('#preview-go'),
  title: $('#title'),
  subtitle: $('#subtitle'),
  crumb: $('#crumb'),
  crumbDot: $('#crumb-dot'),
  source: $('#tb-source'),
  privacy: $('#tb-privacy'),
};
