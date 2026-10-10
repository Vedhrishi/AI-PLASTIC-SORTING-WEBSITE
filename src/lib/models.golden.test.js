// Runs the REAL shipped model files with onnxruntime-node (same ONNX graphs,
// same operators as the browser's WASM runtime) to prove they are intact and
// still behave the way the app's decoding and preprocessing code assumes.
import { createHash } from 'node:crypto';
import { readFileSync, statSync } from 'node:fs';
import { join } from 'node:path';
import { describe, expect, it } from 'vitest';

const ortNode = await import('onnxruntime-node');
const ort = ortNode.default ?? ortNode;
const { classifyResin, CLASSIFIER_INPUT_SIZE } = await import('./classify');
const { decodeYoloOutput } = await import('./yolo');

// Wraps an onnxruntime-node session in the same session-like shape the app's
// worker client exposes, so the REAL classifyResin / detectPlastic run on it.
function adapt(session) {
  return {
    inputNames: session.inputNames,
    outputNames: session.outputNames,
    async run(feeds) {
      const [name] = Object.keys(feeds);
      const tensor = feeds[name];
      const outputs = await session.run({ [name]: new ort.Tensor(tensor.type, tensor.data, tensor.dims) });
      return Object.fromEntries(Object.entries(outputs).map(([key, value]) => [key, { data: value.data, dims: value.dims }]));
    },
  };
}

const modelsDir = join(process.cwd(), 'public', 'models');
const manifest = JSON.parse(readFileSync(join(modelsDir, 'MANIFEST.json'), 'utf8'));

// Deterministic pseudo-noise so the reference outputs below are reproducible.
function noise(length) {
  const data = new Float32Array(length);
  for (let i = 0; i < length; i++) data[i] = ((i * 2654435761) % 1000) / 1000;
  return data;
}

function solidCanvas(r, g, b) {
  const size = CLASSIFIER_INPUT_SIZE;
  const rgba = new Uint8ClampedArray(size * size * 4);
  for (let i = 0; i < size * size; i++) rgba.set([r, g, b, 255], i * 4);
  return { width: size, height: size, getContext: () => ({ getImageData: () => ({ data: rgba }) }) };
}

describe('model integrity', () => {
  it('every shipped model file matches the size and SHA-256 recorded in MANIFEST.json', () => {
    const names = Object.keys(manifest);
    expect(names).toEqual(expect.arrayContaining(['best.onnx', 'resin-resnet18.onnx', 'coco-ssd/model.json']));
    for (const name of names) {
      const path = join(modelsDir, name);
      expect(statSync(path).size, `${name} size`).toBe(manifest[name].bytes);
      const hash = createHash('sha256').update(readFileSync(path)).digest('hex');
      expect(hash, `${name} sha256`).toBe(manifest[name].sha256);
    }
  });
});

describe('YOLOv8 plastic detector (public/models/best.onnx)', () => {
  it('has the I/O contract decodeYoloOutput relies on and reproduces its reference output', async () => {
    const session = await ort.InferenceSession.create(join(modelsDir, 'best.onnx'));
    expect(session.inputNames).toEqual(['images']);
    expect(session.outputNames).toEqual(['output0']);

    const out = (await session.run({ images: new ort.Tensor('float32', noise(3 * 640 * 640), [1, 3, 640, 640]) })).output0;
    expect(out.dims).toEqual([1, 5, 8400]); // 4 box coords + 1 "plastic" class, 8400 anchors

    // Reference values recorded from the shipped model for the noise input above.
    const expected = [5.5247, 8.91567, 14.03811, 26.80103, 37.75773, 43.38403, 51.25631, 59.35045];
    expected.forEach((value, i) => expect(out.data[i]).toBeCloseTo(value, 1));

    // And the decoder turns real model output into well-formed boxes.
    const detections = decodeYoloOutput(out, 640, 640, { scale: 1, offsetX: 0, offsetY: 0 }, { confThreshold: 0.001 });
    for (const { box, score } of detections) {
      expect(box[2]).toBeGreaterThanOrEqual(box[0]);
      expect(box[3]).toBeGreaterThanOrEqual(box[1]);
      expect(score).toBeGreaterThanOrEqual(0.001);
      expect(score).toBeLessThanOrEqual(1);
    }
  }, 120000);
});

describe('ResNet18 resin classifier (public/models/resin-resnet18.onnx)', () => {
  it('has the I/O contract the classifier code relies on and reproduces its reference logits', async () => {
    const session = await ort.InferenceSession.create(join(modelsDir, 'resin-resnet18.onnx'));
    expect(session.inputNames).toEqual(['input']);
    expect(session.outputNames).toEqual(['output']);

    const out = (await session.run({ input: new ort.Tensor('float32', noise(3 * 224 * 224), [1, 3, 224, 224]) })).output;
    expect(out.dims).toEqual([1, 4]); // PET, HDPE, PP, PS
    [4.75419, -4.91608, 0.25235, -1.28452].forEach((value, i) => expect(out.data[i]).toBeCloseTo(value, 1));
  }, 120000);

  it('classifyResin (real preprocessing + real model + real softmax) is stable on reference inputs', async () => {
    const session = await ort.InferenceSession.create(join(modelsDir, 'resin-resnet18.onnx'));
    const gray = await classifyResin(adapt(session), solidCanvas(128, 128, 128));
    const red = await classifyResin(adapt(session), solidCanvas(255, 0, 0));
    // Reference values recorded from the shipped model (regression guard: a
    // swapped model or a change to preprocessing/softmax shows up here).
    expect(gray).toMatchObject({ label: 'PS', code: 6 });
    expect(gray.confidence).toBeCloseTo(0.9515, 2);
    expect(red).toMatchObject({ label: 'HDPE', code: 2 });
    expect(red.confidence).toBeCloseTo(0.6615, 2);
  }, 120000);
});
