import { loadInferenceSessions } from './inferenceClient';
import { loadPersonDetector } from './personClient';

const BASE = import.meta.env?.BASE_URL ?? '/';

export const MODEL_PATHS = {
  plastic: `${BASE}models/best.onnx`,
  resin: `${BASE}models/resin-resnet18.onnx`,
  person: `${BASE}models/coco-ssd/model.json`,
};

let personPromise = null;
let inferencePromise = null;
const progressListeners = new Set();

// Escape hatch for broken GPU drivers: open the app with ?tfBackend=cpu to
// skip WebGL for the person detector.
function preferredTfBackends() {
  const requested = new URLSearchParams(globalThis.location?.search ?? '').get('tfBackend');
  return requested === 'cpu' ? ['cpu'] : undefined;
}

// COCO-SSD is served from this origin (public/models/coco-ssd) rather than
// the default storage.googleapis.com URL, so no unpinned third-party model is
// fetched at runtime. Loads are cached as promises (and evicted on failure so
// Retry works): React StrictMode and retry clicks share one download. Each
// model family lives in its own worker, so neither blocks the UI thread nor
// the other.
function loadPerson() {
  if (!personPromise) {
    personPromise = loadPersonDetector({
      modelUrl: MODEL_PATHS.person,
      preferredBackends: preferredTfBackends(),
      workerFactory: () => new Worker(new URL('../workers/person.worker.js', import.meta.url), { type: 'module' }),
    });
    personPromise.catch(() => {
      personPromise = null;
    });
  }
  return personPromise;
}

function loadInference() {
  if (!inferencePromise) {
    inferencePromise = loadInferenceSessions({
      models: {
        plastic: { url: MODEL_PATHS.plastic, reportProgress: true },
        // The small ResNet18 classifier stays on WASM: no GPU op-coverage or
        // precision surprises can silently corrupt its output.
        resin: { url: MODEL_PATHS.resin },
      },
      onProgress: (fraction) => progressListeners.forEach((listener) => listener(fraction)),
      workerFactory: () => new Worker(new URL('../workers/inference.worker.js', import.meta.url), { type: 'module' }),
    });
    inferencePromise.catch(() => {
      inferencePromise = null;
    });
  }
  return inferencePromise;
}

/**
 * Loads all models in parallel and reports the 45 MB detector's download progress.
 * @param {{ onProgress?: (fraction: number) => void }} [options]
 */
export async function loadAllModels({ onProgress } = {}) {
  if (onProgress) progressListeners.add(onProgress);
  try {
    const [person, inference] = await Promise.all([loadPerson(), loadInference()]);
    const { threads } = inference;
    return {
      personModel: person,
      plasticSession: inference.sessions.plastic,
      resinSession: inference.sessions.resin,
      engine: `ONNX Runtime WASM (${threads} thread${threads === 1 ? '' : 's'}) · COCO-SSD ${person.backend.toUpperCase()} · both in workers`,
    };
  } finally {
    if (onProgress) progressListeners.delete(onProgress);
  }
}
