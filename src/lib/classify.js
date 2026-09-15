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

export function canvasToNormalizedTensor(canvas) {
  const ctx = canvas.getContext('2d');
  const { data } = ctx.getImageData(0, 0, canvas.width, canvas.height);
  const size = canvas.width * canvas.height;
  const float32 = new Float32Array(size * 3);

  for (let i = 0; i < size; i++) {
    const r = data[i * 4] / 255;
    const g = data[i * 4 + 1] / 255;
    const b = data[i * 4 + 2] / 255;
    float32[i] = (r - MEAN[0]) / STD[0];
    float32[size + i] = (g - MEAN[1]) / STD[1];
    float32[size * 2 + i] = (b - MEAN[2]) / STD[2];
  }

  return new ort.Tensor('float32', float32, [1, 3, canvas.height, canvas.width]);
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
  const tensor = canvasToNormalizedTensor(croppedCanvas);
  const inputName = session.inputNames[0];
  const outputName = session.outputNames[0];
  const results = await session.run({ [inputName]: tensor });
  const { index, confidence } = softmaxArgmax(results[outputName].data);
  const resin = RESIN_CODES[index] ?? { code: null, label: 'Unknown' };
  return { ...resin, confidence };
}

export async function classifyContamination(session, croppedCanvas) {
  const tensor = canvasToNormalizedTensor(croppedCanvas);
  const inputName = session.inputNames[0];
  const outputName = session.outputNames[0];
  const results = await session.run({ [inputName]: tensor });
  const { index, confidence } = softmaxArgmax(results[outputName].data);
  return { label: CONTAMINATION_LABELS[index] ?? 'Unknown', level: index, confidence };
}
