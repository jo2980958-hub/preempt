/* Everything that talks to the service. Nothing here knows about the DOM. */

async function json(url, init) {
  const res = await fetch(url, init);
  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`;
    try {
      const body = await res.json();
      detail = body.error?.message || body.message || JSON.stringify(body);
    } catch { /* the body was not JSON; the status line will have to do */ }
    throw new Error(detail);
  }
  return res.json();
}

export const getConfig = () => json('/api/config');
export const getVersion = () => json('/version');
export const getSamples = () => json('/api/samples');
export const getJob = (id) => json(`/api/jobs/${encodeURIComponent(id)}`);

export const startSample = (name) =>
  json(`/api/samples/${encodeURIComponent(name)}`, { method: 'POST' });

export function startUpload(file, params = {}, roomFile = null) {
  const form = new FormData();
  form.append('file', file);
  form.append('params', JSON.stringify(params));
  if (roomFile) form.append('room', roomFile);
  return json('/api/jobs', { method: 'POST', body: form });
}

export const getDefaultRoom = () => json('/api/rooms/default');

/** The service's own validator, so the preview and the run agree on what is valid. */
export const checkRoom = (text) =>
  json('/api/rooms/check', { method: 'POST', body: text, headers: { 'Content-Type': 'application/json' } });

/**
 * Follow a job to its end. Resolves with the finished job record.
 * Falls back to polling if the event stream cannot be opened.
 */
export function follow(job, { onStatus, onProgress, onNote } = {}) {
  return new Promise((resolve, reject) => {
    let settled = false;
    const finish = async () => {
      if (settled) return;
      settled = true;
      source.close();
      try {
        resolve(await getJob(job.job_id));
      } catch (err) {
        reject(err);
      }
    };

    const source = new EventSource(job.events_url || `/api/jobs/${job.job_id}/events`);
    source.addEventListener('status', (e) => onStatus?.(parse(e)));
    source.addEventListener('progress', (e) => onProgress?.(parse(e)));
    source.addEventListener('note', (e) => onNote?.(parse(e)));
    source.addEventListener('end', finish);
    source.onerror = async () => {
      if (settled) return;
      // The stream dropped. Poll once; if the job is finished this is not an error.
      try {
        const record = await getJob(job.job_id);
        if (record.status === 'done' || record.status === 'failed') {
          settled = true;
          source.close();
          resolve(record);
          return;
        }
      } catch { /* fall through to the poller below */ }
      poll();
    };

    let tries = 0;
    async function poll() {
      if (settled) return;
      try {
        const record = await getJob(job.job_id);
        if (record.status === 'done' || record.status === 'failed') {
          settled = true;
          source.close();
          resolve(record);
          return;
        }
      } catch (err) {
        if (++tries > 8) {
          settled = true;
          source.close();
          reject(err);
          return;
        }
      }
      setTimeout(poll, 700);
    }
  });
}

function parse(event) {
  try {
    return JSON.parse(event.data);
  } catch {
    return {};
  }
}

/** Evidence arrives as a server-side path; the bytes are served per job. */
export function evidenceUrl(jobId, uri) {
  if (!uri) return '';
  if (/^(https?:)?\/\//.test(uri) || uri.startsWith('/api/')) return uri;
  const name = uri.split(/[\\/]/).pop();
  return `/api/jobs/${encodeURIComponent(jobId)}/evidence/${encodeURIComponent(name)}`;
}
