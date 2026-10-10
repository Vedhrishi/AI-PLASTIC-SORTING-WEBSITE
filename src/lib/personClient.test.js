import { describe, expect, it } from 'vitest';
import { loadPersonDetector } from './personClient';
import { classifyTfLoadError } from './tfErrors';

/** @param {{ initError?: any, detections?: any[] }} [options] */
function fakeWorker({ initError, detections = [{ bbox: [1, 2, 3, 4], class: 'person', score: 0.9 }] } = {}) {
  return {
    terminated: false,
    posted: [],
    onmessage: null,
    onerror: null,
    terminate() {
      this.terminated = true;
    },
    postMessage(message) {
      this.posted.push(message);
      queueMicrotask(() => {
        const reply = (payload) => this.onmessage({ data: { id: message.id, ...payload } });
        if (message.type === 'init') {
          return initError ? reply({ type: 'error', error: initError }) : reply({ type: 'init-done', backend: 'cpu' });
        }
        if (message.type === 'detect') return reply({ type: 'detect-done', detections });
      });
    },
  };
}

const fakeCanvas = (width, height) => ({
  width,
  height,
  getContext: () => ({ getImageData: () => ({ data: new Uint8ClampedArray(width * height * 4) }) }),
});

describe('loadPersonDetector', () => {
  it('reports the backend the worker settled on and passes the model url through', async () => {
    const worker = fakeWorker();
    const detector = await loadPersonDetector({ modelUrl: '/m.json', preferredBackends: ['cpu'], workerFactory: () => worker });
    expect(detector.backend).toBe('cpu');
    expect(worker.posted[0]).toMatchObject({ type: 'init', modelUrl: '/m.json', preferredBackends: ['cpu'] });
  });

  it('sends the canvas pixels (transferred) and returns the detections', async () => {
    const worker = fakeWorker();
    const detector = await loadPersonDetector({ modelUrl: '/m.json', workerFactory: () => worker });
    const detections = await detector.detect(fakeCanvas(4, 2));
    expect(detections).toEqual([{ bbox: [1, 2, 3, 4], class: 'person', score: 0.9 }]);
    expect(worker.posted.at(-1)).toMatchObject({ type: 'detect', width: 4, height: 2 });
    expect(worker.posted.at(-1).data).toHaveLength(32);
  });

  it('keeps the error kind (so the UI can say offline / HTTP 404) and stops the worker', async () => {
    const worker = fakeWorker({ initError: { message: 'Request to /m.json failed with status code 404', kind: 'http', status: 404 } });
    const err = await loadPersonDetector({ modelUrl: '/m.json', workerFactory: () => worker }).catch((e) => e);
    expect(err).toMatchObject({ kind: 'http', status: 404 });
    expect(worker.terminated).toBe(true);
  });
});

describe('classifyTfLoadError', () => {
  it('extracts the HTTP status from TF.js messages', () => {
    expect(classifyTfLoadError(new Error('Request to /m.json failed with status code 404. Please verify this URL.'))).toEqual({ kind: 'http', status: 404 });
  });

  it('distinguishes offline from a dropped connection', () => {
    expect(classifyTfLoadError(new TypeError('Failed to fetch'), true)).toEqual({ kind: 'network' });
    expect(classifyTfLoadError(new TypeError('Failed to fetch'), false)).toEqual({ kind: 'offline' });
  });

  it('returns no kind for unrelated errors', () => {
    expect(classifyTfLoadError(new Error('shader compile failed'))).toEqual({});
  });
});
