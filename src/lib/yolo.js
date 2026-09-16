import { ort } from './onnxSetup';
import { nms } from './geometry';

export const YOLO_INPUT_SIZE = 640;

// Letterbox-resizes `source` (video/canvas/image) into a square
// YOLO_INPUT_SIZE x YOLO_INPUT_SIZE canvas, preserving aspect ratio.
export function letterbox(source, srcW, srcH) {
  const canvas = document.createElement('canvas');
  canvas.width = YOLO_INPUT_SIZE;
  canvas.height = YOLO_INPUT_SIZE;
  const ctx = canvas.getContext('2d');
  ctx.fillStyle = 'rgb(114,114,114)';
  ctx.fillRect(0, 0, YOLO_INPUT_SIZE, YOLO_INPUT_SIZE);

  const scale = Math.min(YOLO_INPUT_SIZE / srcW, YOLO_INPUT_SIZE / srcH);
  const drawW = srcW * scale;
  const drawH = srcH * scale;
  const offsetX = (YOLO_INPUT_SIZE - drawW) / 2;
  const offsetY = (YOLO_INPUT_SIZE - drawH) / 2;

  ctx.drawImage(source, 0, 0, srcW, srcH, offsetX, offsetY, drawW, drawH);

  return { canvas, scale, offsetX, offsetY };
}

export function canvasToCHWTensor(canvas) {
  const ctx = canvas.getContext('2d');
  const { data } = ctx.getImageData(0, 0, canvas.width, canvas.height);
  const size = canvas.width * canvas.height;
  const float32 = new Float32Array(size * 3);

  for (let i = 0; i < size; i++) {
    const r = data[i * 4] / 255;
    const g = data[i * 4 + 1] / 255;
    const b = data[i * 4 + 2] / 255;
    float32[i] = r; // R plane
    float32[size + i] = g; // G plane
    float32[size * 2 + i] = b; // B plane
  }

  return new ort.Tensor('float32', float32, [1, 3, canvas.height, canvas.width]);
}

// Decodes a YOLOv8-style single-class detection head with output shape
// [1, 4 + numClasses, numBoxes] (cx, cy, w, h, class scores...).
// Returns boxes in the ORIGINAL (srcW x srcH) coordinate space.
export function decodeYoloOutput(outputTensor, srcW, srcH, letterboxInfo, opts = {}) {
  // 0.25 (not the previous 0.4): heavily crushed/deformed plastic items
  // produce weaker, less bottle-shaped activations, so the higher threshold
  // was silently dropping valid detections before they ever reached Stage 3.
  // NMS IoU stays at 0.45 to still collapse duplicate/overlapping boxes now
  // that more lower-confidence candidates survive the score gate.
  const { confThreshold = 0.25, iouThreshold = 0.45 } = opts;
  const dims = outputTensor.dims; // [1, C, N]
  const C = dims[1];
  const N = dims[2];
  const numClasses = C - 4;
  const data = outputTensor.data;

  const boxes = [];
  const scores = [];
  const classIds = [];

  for (let i = 0; i < N; i++) {
    let bestClass = 0;
    let bestScore = -Infinity;
    for (let c = 0; c < numClasses; c++) {
      const score = data[(4 + c) * N + i];
      if (score > bestScore) {
        bestScore = score;
        bestClass = c;
      }
    }
    if (bestScore < confThreshold) continue;

    const cx = data[0 * N + i];
    const cy = data[1 * N + i];
    const w = data[2 * N + i];
    const h = data[3 * N + i];

    // undo letterbox -> original image coords
    const x1 = (cx - w / 2 - letterboxInfo.offsetX) / letterboxInfo.scale;
    const y1 = (cy - h / 2 - letterboxInfo.offsetY) / letterboxInfo.scale;
    const x2 = (cx + w / 2 - letterboxInfo.offsetX) / letterboxInfo.scale;
    const y2 = (cy + h / 2 - letterboxInfo.offsetY) / letterboxInfo.scale;

    boxes.push([
      Math.max(0, x1),
      Math.max(0, y1),
      Math.min(srcW, x2),
      Math.min(srcH, y2),
    ]);
    scores.push(bestScore);
    classIds.push(bestClass);
  }

  const keep = nms(boxes, scores, iouThreshold);

  return keep.map((i) => ({
    box: boxes[i],
    score: scores[i],
    classId: classIds[i],
  }));
}

export async function detectPlastic(session, videoEl, srcW, srcH, opts) {
  const { canvas: lbCanvas, scale, offsetX, offsetY } = letterbox(videoEl, srcW, srcH);
  const inputTensor = canvasToCHWTensor(lbCanvas);
  const inputName = session.inputNames[0];
  const outputName = session.outputNames[0];
  const results = await session.run({ [inputName]: inputTensor });
  const output = results[outputName];
  return decodeYoloOutput(output, srcW, srcH, { scale, offsetX, offsetY }, opts);
}
