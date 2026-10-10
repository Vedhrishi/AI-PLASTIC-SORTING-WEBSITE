// Hosts COCO-SSD (TF.js) off the main thread. WebGL runs on an OffscreenCanvas
// inside the worker; if that is unavailable TF.js falls back to its CPU
// backend. Protocol:
//   -> { id, type: 'init', modelUrl, preferredBackends? }   <- { id, type: 'init-done', backend }
//   -> { id, type: 'detect', data, width, height }          <- { id, type: 'detect-done', detections }
//   <- { id, type: 'error', error: { name, message, kind?, status? } }
import * as tf from '@tensorflow/tfjs';
import * as cocoSsd from '@tensorflow-models/coco-ssd';
import { selectTfBackend } from '../lib/tfBackend';
import { classifyTfLoadError } from '../lib/tfErrors';

// The DOM lib types `self` as a Window; inside a worker it is the worker scope.
const scope = /** @type {any} */ (self);

let model = null;
let queue = Promise.resolve();

function serialiseError(err, phase) {
  const classified = phase === 'init' ? classifyTfLoadError(err, scope.navigator?.onLine !== false) : {};
  return { name: err?.name, message: err?.message ?? String(err), ...classified };
}

async function handle({ id, type, ...msg }) {
  if (type === 'init') {
    const backend = await selectTfBackend(tf, msg.preferredBackends);
    model = await cocoSsd.load({ base: 'lite_mobilenet_v2', modelUrl: msg.modelUrl });
    scope.postMessage({ id, type: 'init-done', backend });
    return;
  }

  if (type === 'detect') {
    if (!model) throw new Error('The person detector is not loaded in its worker');
    const image = new ImageData(new Uint8ClampedArray(msg.data), msg.width, msg.height);
    const found = await model.detect(image);
    scope.postMessage({
      id,
      type: 'detect-done',
      detections: found.map(({ bbox, class: label, score }) => ({ bbox, class: label, score })),
    });
    return;
  }

  throw new Error(`Unknown person worker message "${type}"`);
}

scope.onmessage = (event) => {
  const { id, type } = event.data;
  queue = queue
    .then(() => handle(event.data))
    .catch((err) => scope.postMessage({ id, type: 'error', error: serialiseError(err, type) }));
};
