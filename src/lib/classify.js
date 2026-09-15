import { ort } from './onnxSetup';

export const CLASSIFIER_INPUT_SIZE = 224;

// Standard ImageNet normalization, used by both ResNet18 heads.
const MEAN = [0.485, 0.456, 0.406];
const STD = [0.229, 0.224, 0.225];

export const RESIN_CODES = {
  0: { code: 1, label: 'PET' },
  1: { code: 2, label: 'HDPE' },
  2: { code: 5, label: 'PP' },
  3: { code: 6, label: 'PS' },
};

export const CONTAMINATION_LABELS = {
  0: 'Clean/Light Soiling',
  1: 'Moderate Contamination',
  2: 'Heavy Contamination',
};

// Crops [x1,y1,x2,y2] from `source` and resizes to 224x224, returns the canvas.
export function cropAndResize(source, box) {
  const [x1, y1, x2, y2] = box;
  const w = Math.max(1, x2 - x1);
  const h = Math.max(1, y2 - y1);

  const canvas = document.createElement('canvas');
  canvas.width = CLASSIFIER_INPUT_SIZE;
  canvas.height = CLASSIFIER_INPUT_SIZE;
  const ctx = canvas.getContext('2d');
  ctx.drawImage(source, x1, y1, w, h, 0, 0, CLASSIFIER_INPUT_SIZE, CLASSIFIER_INPUT_SIZE);
  return canvas;
}

// Converts a 224x224 canvas into a PyTorch-style ImageNet input tensor:
// planar (channels-first) layout [1, 3, 224, 224] — all R values, then all
// G values, then all B values — normalized to 0-1 and then to ImageNet
// mean/std. NOT interleaved/HWC [1, 224, 224, 3], which is what a naive
// pixel-order copy of ImageData would produce and what a ResNet18 trained
// on torchvision-style preprocessing will silently misinterpret.
export function canvasToNormalizedTensor(canvas) {
  const width = canvas.width;
  const height = canvas.height;
  const ctx = canvas.getContext('2d');
  const { data: rgba } = ctx.getImageData(0, 0, width, height); // Uint8ClampedArray, RGBA interleaved

  const pixelCount = width * height; // 224 * 224 = 50,176
  const tensorData = new Float32Array(1 * 3 * height * width); // 150,528 elements

  const rOffset = 0;
  const gOffset = pixelCount;
  const bOffset = pixelCount * 2;

  for (let i = 0; i < pixelCount; i++) {
    const rgbaIndex = i * 4;

    const r = rgba[rgbaIndex] / 255;
    const g = rgba[rgbaIndex + 1] / 255;
    const b = rgba[rgbaIndex + 2] / 255;

    tensorData[rOffset + i] = (r - MEAN[0]) / STD[0];
    tensorData[gOffset + i] = (g - MEAN[1]) / STD[1];
    tensorData[bOffset + i] = (b - MEAN[2]) / STD[2];
  }

  return new ort.Tensor('float32', tensorData, [1, 3, height, width]);
}

function softmaxArgmax(data) {
  const max = Math.max(...data);
  const exps = Array.from(data, (v) => Math.exp(v - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  const probs = exps.map((e) => e / sum);
  let bestIdx = 0;
  let bestProb = -Infinity;
  for (let i = 0; i < probs.length; i++) {
    if (probs[i] > bestProb) {
      bestProb = probs[i];
      bestIdx = i;
    }
  }
  return { index: bestIdx, confidence: bestProb };
}

export async function classifyResin(session, croppedCanvas) {
  try {
    const tensor = canvasToNormalizedTensor(croppedCanvas);
    const inputName = session.inputNames[0];
    const outputName = session.outputNames[0];
    const results = await session.run({ [inputName]: tensor });
    const { index, confidence } = softmaxArgmax(results[outputName].data);
    const resin = RESIN_CODES[index] ?? { code: null, label: 'Unknown' };
    return { ...resin, confidence };
  } catch (err) {
    console.error('[classifyResin] resin-resnet18 inference failed', err);
    throw err;
  }
}

export async function classifyContamination(session, croppedCanvas) {
  try {
    const tensor = canvasToNormalizedTensor(croppedCanvas);
    const inputName = session.inputNames[0];
    const outputName = session.outputNames[0];
    const results = await session.run({ [inputName]: tensor });
    const { index, confidence } = softmaxArgmax(results[outputName].data);
    return { label: CONTAMINATION_LABELS[index] ?? 'Unknown', level: index, confidence };
  } catch (err) {
    console.error('[classifyContamination] contamination-resnet18 inference failed', err);
    throw err;
  }
}
