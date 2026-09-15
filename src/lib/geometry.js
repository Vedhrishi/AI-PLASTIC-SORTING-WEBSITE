// box: [x1, y1, x2, y2]
export function iou(boxA, boxB) {
  const xA = Math.max(boxA[0], boxB[0]);
  const yA = Math.max(boxA[1], boxB[1]);
  const xB = Math.min(boxA[2], boxB[2]);
  const yB = Math.min(boxA[3], boxB[3]);

  const interW = Math.max(0, xB - xA);
  const interH = Math.max(0, yB - yA);
  const interArea = interW * interH;
  if (interArea === 0) return 0;

  const areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]);
  const areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1]);

  return interArea / (areaA + areaB - interArea);
}

// Fraction of boxA (the plastic box) covered by boxB (the person box).
export function overlapFraction(boxA, boxB) {
  const xA = Math.max(boxA[0], boxB[0]);
  const yA = Math.max(boxA[1], boxB[1]);
  const xB = Math.min(boxA[2], boxB[2]);
  const yB = Math.min(boxA[3], boxB[3]);

  const interW = Math.max(0, xB - xA);
  const interH = Math.max(0, yB - yA);
  const interArea = interW * interH;
  if (interArea === 0) return 0;

  const areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]);
  if (areaA === 0) return 0;
  return interArea / areaA;
}

// Maps a box from the native source frame (e.g. video.videoWidth x
// video.videoHeight) into the coordinate space of a container that displays
// that frame with `object-fit: cover` (crops to fill, preserving aspect
// ratio). Needed because CSS `object-fit: cover` visually crops/scales the
// video independently of the pixel buffer the model actually saw, so a raw
// model-space box drawn without this correction drifts off the real object
// whenever the camera's native aspect ratio differs from the container's.
export function mapCoverBox(box, nativeW, nativeH, containerW, containerH) {
  const scale = Math.max(containerW / nativeW, containerH / nativeH);
  const visibleW = containerW / scale;
  const visibleH = containerH / scale;
  const cropX = (nativeW - visibleW) / 2;
  const cropY = (nativeH - visibleH) / 2;

  const [x1, y1, x2, y2] = box;
  return [(x1 - cropX) * scale, (y1 - cropY) * scale, (x2 - cropX) * scale, (y2 - cropY) * scale];
}

export function nms(boxes, scores, iouThreshold = 0.45) {
  const indices = scores
    .map((s, i) => i)
    .sort((a, b) => scores[b] - scores[a]);

  const keep = [];
  const suppressed = new Set();

  for (const i of indices) {
    if (suppressed.has(i)) continue;
    keep.push(i);
    for (const j of indices) {
      if (j === i || suppressed.has(j)) continue;
      if (iou(boxes[i], boxes[j]) > iouThreshold) {
        suppressed.add(j);
      }
    }
  }
  return keep;
}
