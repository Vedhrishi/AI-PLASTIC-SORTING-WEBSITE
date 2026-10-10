import { ModelLoadError } from './modelLoader';

/**
 * Rebuilds an error that crossed the worker boundary so callers (and
 * describeModelError) still see its kind, status and url.
 * @param {{ name?: string, message: string, kind?: any, status?: number, url?: string }} info
 */
export function rebuildError(info) {
  if (info.kind) return new ModelLoadError(info.message, { kind: info.kind, url: info.url, status: info.status });
  const err = new Error(info.message);
  if (info.name) err.name = info.name;
  return err;
}

/**
 * Minimal request/response layer over a Worker. Every request gets an id; the
 * worker answers with { id, type: 'error' | anything-else }. Messages of type
 * 'progress' are routed to onProgress instead of resolving a request.
 *
 * @param {any} worker
 * @param {{ onProgress?: (fraction: number) => void }} [options]
 */
export function createRpc(worker, { onProgress } = {}) {
  /** @type {Map<number, { resolve: (value: any) => void, reject: (err: any) => void }>} */
  const pending = new Map();
  let nextId = 1;

  const failAll = (err) => {
    for (const { reject } of pending.values()) reject(err);
    pending.clear();
  };

  worker.onmessage = (event) => {
    const msg = event.data;
    if (msg.type === 'progress') {
      onProgress?.(msg.fraction);
      return;
    }
    const entry = pending.get(msg.id);
    if (!entry) return;
    pending.delete(msg.id);
    if (msg.type === 'error') entry.reject(rebuildError(msg.error));
    else entry.resolve(msg);
  };
  worker.onerror = (event) => {
    failAll(new ModelLoadError(`Inference worker crashed: ${event?.message ?? 'unknown error'}`, { kind: 'init' }));
  };

  return {
    /**
     * @param {Record<string, any>} message
     * @param {Transferable[]} [transfer]
     * @returns {Promise<any>}
     */
    request(message, transfer = []) {
      return new Promise((resolve, reject) => {
        const id = nextId++;
        pending.set(id, { resolve, reject });
        worker.postMessage({ id, ...message }, transfer);
      });
    },
    dispose() {
      worker.terminate();
    },
  };
}
