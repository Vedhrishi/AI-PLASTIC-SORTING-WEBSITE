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

// Computes the scale/offset that CSS `object-fit: cover` applies when a
// videoW x videoH source is displayed inside a containerW x containerH box:
// the source is scaled up by the LARGER of the two axis ratios (so it fully
// covers the container with no gaps) and centered, cropping whichever axis
// overflows. offsetX/offsetY can come out negative — that's the crop, not a
// bug — a native-space point maps to screen space via `point * scale +
// offset`. This must be recomputed from the element's *live* clientWidth/
// clientHeight (not an assumed constant), since that's the only value that
// actually reflects the real on-screen box across phones, orientations, and
// dynamic viewport resizing.
export function computeCoverProjection(videoW, videoH, containerW, containerH) {
  const videoRatio = videoW / videoH;
  const containerRatio = containerW / containerH;
  let scale;
  let offsetX = 0;
  let offsetY = 0;

  if (containerRatio > videoRatio) {
    scale = containerW / videoW;
    offsetY = (containerH - videoH * scale) / 2;
  } else {
    scale = containerH / videoH;
    offsetX = (containerW - videoW * scale) / 2;
  }

  return { scale, offsetX, offsetY };
}

// Projects a box from native source-frame coordinates into the screen-space
// coordinates produced by `computeCoverProjection`.
export function projectBox(box, { scale, offsetX, offsetY }) {
  const [x1, y1, x2, y2] = box;
  return [x1 * scale + offsetX, y1 * scale + offsetY, x2 * scale + offsetX, y2 * scale + offsetY];
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
