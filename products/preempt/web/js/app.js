/* Pick a room, watch it, show what came back. */

import { ui } from './dom.js';
import { esc } from './format.js';
import * as api from './api.js';
import { renderResults } from './render.js';
import { describeRun, railCounts } from './summary.js';
import {
  initTheme, status, watchSections, showEnvironment,
  beginProgress, stepProgress, endProgress,
} from './shell.js';
import {
  mountPicker, setSamples, currentSample, currentUpload,
  pickerUnavailable, compactPicker, isTrack,
} from './picker.js';
import { mountSetup, previewVideo, currentRoomFile, setupReady } from './setup.js';

let busy = false;

async function run() {
  if (busy) return;
  const video = currentUpload();
  if (video && !isTrack(video) && !setupReady()) {
    status('Pick a room setup the service accepts first', 'failed');
    return;
  }
  busy = true;
  ui.start.disabled = true;
  ui.empty.hidden = true;
  ui.crumbDot.className = 'dot';
  beginProgress('Handing the recording to the service');
  status('Running', 'busy');

  const upload = currentUpload();
  ui.source.textContent = `Watching ${upload ? upload.name : currentSample()}`;

  try {
    const room = upload && !isTrack(upload) ? currentRoomFile() : null;
    const job = upload ? await api.startUpload(upload, {}, room) : await api.startSample(currentSample());
    const done = await api.follow(job, {
      onProgress: (e) => {
        stepProgress(e.percent, e.message);
        if (e.message) status(`Running — ${e.message}`, 'busy');
      },
      onNote: (e) => stepProgress(null, e.message),
    });
    if (done.status === 'failed' || !done.result) {
      throw new Error(done.error || 'the service returned no record');
    }
    show(done.result, job.job_id);
    status('Finished', 'done');
  } catch (err) {
    fail(err);
  } finally {
    busy = false;
    ui.start.disabled = false;
    endProgress();
  }
}

/** A fault in the service is not a reading about the room, and must not read as one. */
function fail(err) {
  status('Failed', 'failed');
  ui.results.hidden = false;
  ui.results.innerHTML = `<div class="error"><h2>The run did not finish</h2>
    <p>${esc(err.message || String(err))}</p>
    <p class="dim">This is a fault in the service, not a reading about the room.
      Nothing here says anything about whether anybody is standing up.</p></div>`;
}

function show(record, jobId) {
  ui.empty.hidden = true;
  ui.results.hidden = false;
  compactPicker();
  renderResults(ui.results, record, jobId);
  describeRun(record);
  railCounts(record);
}

async function boot() {
  initTheme();
  watchSections();
  mountPicker(run, previewVideo);
  mountSetup(run);
  ui.start.addEventListener('click', run);

  try {
    const [version, list] = await Promise.all([api.getVersion(), api.getSamples()]);
    showEnvironment(version);
    setSamples(Array.isArray(list) ? list : (list.samples || []));
    status('Ready');
  } catch (err) {
    pickerUnavailable(err.message);
    status('Service unreachable', 'failed');
  }
}

boot();
