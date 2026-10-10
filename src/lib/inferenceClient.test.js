import { describe, expect, it } from 'vitest';
import { loadInferenceSessions } from './inferenceClient';
import { ModelLoadError } from './modelLoader';
import { Tensor } from './tensor';

// An in-memory stand-in for the worker that speaks the real message protocol.
function fakeWorker(behaviour = {}) {
  const worker = {
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
          if (behaviour.initError) return reply({ type: 'error', error: behaviour.initError });
          if (behaviour.progress) behaviour.progress.forEach((fraction) => reply({ type: 'progress', fraction }));
          return reply({
            type: 'init-done',
            threads: 2,
            models: { plastic: { inputNames: ['images'], outputNames: ['output0'] } },
          });
        }
        if (message.type === 'run') {
          if (behaviour.runError) return reply({ type: 'error', error: behaviour.runError });
          return reply({ type: 'run-done', data: new Float32Array([1, 2, 3, 4]), dims: [1, 4] });
        }
      });
    },
  };
  return worker;
}

const options = (worker, extra = {}) => ({
  models: { plastic: { url: '/m.onnx', reportProgress: true } },
  workerFactory: () => worker,
  ...extra,
});

describe('loadInferenceSessions', () => {
  it('returns session-like objects and reports progress and thread count', async () => {
    const progress = [];
    const worker = fakeWorker({ progress: [0.25, 1] });
    const { sessions, threads } = await loadInferenceSessions(options(worker, { onProgress: (p) => progress.push(p) }));

    expect(threads).toBe(2);
    expect(progress).toEqual([0.25, 1]);
    expect(sessions.plastic.inputNames).toEqual(['images']);
    expect(sessions.plastic.outputNames).toEqual(['output0']);
  });

  it('runs a tensor through the worker (transferring the input buffer) and returns an output tensor', async () => {
    const worker = fakeWorker();
    const { sessions } = await loadInferenceSessions(options(worker));
    const input = new Tensor('float32', new Float32Array(8), [1, 8]);

    const out = await sessions.plastic.run({ images: input });
    expect(out.output0).toBeInstanceOf(Tensor);
    expect(Array.from(out.output0.data)).toEqual([1, 2, 3, 4]);
    expect(out.output0.dims).toEqual([1, 4]);

    const runMessage = worker.posted.find((m) => m.type === 'run');
    expect(runMessage).toMatchObject({ model: 'plastic', inputName: 'images', outputName: 'output0', dims: [1, 8] });
  });

  it('rebuilds model-load errors with their kind, status and url, and stops the worker', async () => {
    const worker = fakeWorker({ initError: { name: 'ModelLoadError', message: 'HTTP 404', kind: 'http', status: 404, url: '/m.onnx' } });
    const err = await loadInferenceSessions(options(worker)).catch((e) => e);
    expect(err).toBeInstanceOf(ModelLoadError);
    expect(err).toMatchObject({ kind: 'http', status: 404, url: '/m.onnx' });
    expect(worker.terminated).toBe(true);
  });

  it('surfaces inference errors thrown inside the worker', async () => {
    const worker = fakeWorker({ runError: { name: 'Error', message: 'Session already started' } });
    const { sessions } = await loadInferenceSessions(options(worker));
    const input = new Tensor('float32', new Float32Array(4), [1, 4]);
    await expect(sessions.plastic.run({ images: input })).rejects.toThrow('Session already started');
  });

  it('turns a crashed worker into a clear init error instead of hanging', async () => {
    const worker = fakeWorker();
    worker.postMessage = function () {
      queueMicrotask(() => this.onerror({ message: 'script failed to load' }));
    };
    const err = await loadInferenceSessions(options(worker)).catch((e) => e);
    expect(err).toMatchObject({ kind: 'init' });
    expect(err.message).toMatch(/script failed to load/);
  });
});
