import * as ort from 'onnxruntime-web';

ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/';
ort.env.wasm.numThreads = 1;

const sessionCache = new Map();

export async function loadSession(modelPath) {
  if (sessionCache.has(modelPath)) return sessionCache.get(modelPath);
  const session = await ort.InferenceSession.create(modelPath, {
    executionProviders: ['wasm'],
  });
  sessionCache.set(modelPath, session);
  return session;
}

export { ort };
