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
