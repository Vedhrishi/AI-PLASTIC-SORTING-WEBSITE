import * as ort from 'onnxruntime-web';

ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/';
ort.env.wasm.numThreads = 1;

const sessionCache = new Map();

export async function loadSession(modelPath) {
  if (sessionCache.has(modelPath)) return sessionCache.get(modelPath);

  let session;
  try {
    // Prefer the mobile/desktop GPU (WebGL) for the conv-heavy YOLO/ResNet
    // graphs; unsupported ops fall back to WASM within the same session.
    session = await ort.InferenceSession.create(modelPath, {
      executionProviders: ['webgl', 'wasm'],
    });
  } catch (err) {
    console.warn(`WebGL EP unavailable for ${modelPath}, falling back to WASM only`, err);
    session = await ort.InferenceSession.create(modelPath, {
      executionProviders: ['wasm'],
    });
  }

  sessionCache.set(modelPath, session);
  return session;
}

export { ort };
