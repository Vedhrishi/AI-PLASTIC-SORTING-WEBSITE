// Runs every ONNX model off the main thread, so downloading, initialising and
// running the 45 MB detector can never freeze the interface. Protocol:
//   -> { id, type: 'init', models: { [key]: { url, providers?, reportProgress? } } }
//   <- { id, type: 'progress', fraction }            (zero or more)
//   <- { id, type: 'init-done', threads, models: { [key]: { inputNames, outputNames } } }
//   -> { id, type: 'run', model, inputName, outputName, data, dims }
//   <- { id, type: 'run-done', data, dims }
//   <- { id, type: 'error', error: { name, message, kind?, status?, url? } }
import { configureOrt, loadSession, ort } from '../lib/onnxSetup';

// The DOM lib types `self` as a Window; inside a worker it is the worker scope.
const scope = /** @type {any} */ (self);

const sessions = {};
// A session must not be entered concurrently (ORT throws "Session already
// started"), so every request is queued behind the previous one.
let queue = Promise.resolve();

function serialiseError(err) {
  return { name: err?.name, message: err?.message ?? String(err), kind: err?.kind, status: err?.status, url: err?.url };
}

async function handle({ id, type, ...msg }) {
  if (type === 'init') {
    const { threads } = configureOrt();
    const described = {};
    await Promise.all(
      Object.entries(msg.models).map(async ([key, spec]) => {
        const session = await loadSession(spec.url, {
          executionProviders: spec.providers ?? ['wasm'],
          onProgress: spec.reportProgress ? (fraction) => scope.postMessage({ id, type: 'progress', fraction }) : undefined,
        });
        sessions[key] = session;
        described[key] = { inputNames: [...session.inputNames], outputNames: [...session.outputNames] };
      })
    );
    scope.postMessage({ id, type: 'init-done', threads, models: described });
    return;
  }

  if (type === 'run') {
    const session = sessions[msg.model];
    if (!session) throw new Error(`Model "${msg.model}" is not loaded in the inference worker`);
    const input = new ort.Tensor('float32', msg.data, msg.dims);
    const outputs = await session.run({ [msg.inputName]: input });
    const output = outputs[msg.outputName];
    const data = output.data.slice(); // detach from WASM memory before transferring
    const dims = [...output.dims];
    input.dispose?.();
    for (const tensor of Object.values(outputs)) tensor.dispose?.();
    scope.postMessage({ id, type: 'run-done', data, dims }, [data.buffer]);
    return;
  }

  throw new Error(`Unknown inference worker message "${type}"`);
}

scope.onmessage = (event) => {
  const { id } = event.data;
  queue = queue
    .then(() => handle(event.data))
    .catch((err) => scope.postMessage({ id, type: 'error', error: serialiseError(err) }));
};
