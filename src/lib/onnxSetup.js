// WASM-only build of onnxruntime-web: smaller than the default bundle (which
// also ships the WebGPU runtime) and it never touches a third-party CDN —
// the runtime files are served from this app's own /ort/ directory
// (see scripts/prepare-ort.mjs).
import * as ort from 'onnxruntime-web/wasm';
import { ModelLoadError, fetchModelBytes } from './modelLoader';

const MAX_THREADS = 4;

let configured = false;

/**
 * Applies runtime settings once. Multi-threading needs SharedArrayBuffer,
 * which browsers only enable on cross-origin-isolated pages (COOP/COEP
 * headers); without isolation a single thread is the only safe choice.
 * This module only ever runs inside src/workers/inference.worker.js, so the UI
 * thread never blocks on inference.
 * @param {{ baseUrl?: string, isolated?: boolean, cores?: number }} [options]
 * @returns {{ threads: number }}
 */
export function configureOrt({
  baseUrl = import.meta.env?.BASE_URL ?? '/',
  isolated = globalThis.crossOriginIsolated === true,
  cores = globalThis.navigator?.hardwareConcurrency ?? 1,
} = {}) {
  const threads = isolated ? Math.max(1, Math.min(MAX_THREADS, cores)) : 1;
  if (!configured) {
    ort.env.wasm.wasmPaths = `${baseUrl}ort/`;
    ort.env.wasm.numThreads = threads;
    // Already running inside our own inference worker (src/workers), so ORT's
    // built-in proxy worker is neither needed nor safe to spawn from a bundle.
    ort.env.wasm.proxy = false;
    configured = true;
  }
  return { threads: ort.env.wasm.numThreads ?? threads };
}

/** Test hook: forget that ORT was configured. */
export function resetOrtConfigForTests() {
  configured = false;
}

const sessionCache = new Map();

/**
 * Loads (and caches) an ONNX session. The cache holds the *promise*, so two
 * concurrent callers (React StrictMode, retry clicks) share one download and
 * one initialisation; a failed load is evicted so Retry really retries.
 * @param {string} modelPath
 * @param {{ executionProviders?: string[], onProgress?: (fraction: number) => void, fetchImpl?: import('./modelLoader').FetchLike }} [options]
 * @returns {Promise<import('onnxruntime-web').InferenceSession>}
 */
export function loadSession(modelPath, { executionProviders = ['wasm'], onProgress, fetchImpl } = {}) {
  const key = `${modelPath}|${executionProviders.join(',')}`;
  const cached = sessionCache.get(key);
  if (cached) return cached;

  const promise = (async () => {
    configureOrt();
    const bytes = await fetchModelBytes(modelPath, { onProgress, fetchImpl });
    try {
      return await ort.InferenceSession.create(bytes, { executionProviders });
    } catch (err) {
      if (executionProviders.length === 1 && executionProviders[0] === 'wasm') {
        throw new ModelLoadError(`Could not initialise ${modelPath}: ${err?.message ?? err}`, { kind: 'init', url: modelPath, cause: err });
      }
      console.warn(`${executionProviders.join('/')} unavailable for ${modelPath}, falling back to WASM`, err);
      try {
        return await ort.InferenceSession.create(bytes, { executionProviders: ['wasm'] });
      } catch (fallbackErr) {
        throw new ModelLoadError(`Could not initialise ${modelPath}: ${fallbackErr?.message ?? fallbackErr}`, {
          kind: 'init',
          url: modelPath,
          cause: fallbackErr,
        });
      }
    }
  })();

  sessionCache.set(key, promise);
  promise.catch(() => sessionCache.delete(key));
  return promise;
}

export function clearSessionCacheForTests() {
  sessionCache.clear();
}

export { ort };
