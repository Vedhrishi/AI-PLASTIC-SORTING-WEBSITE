import { describe, expect, it, vi } from 'vitest';

import { canvasToCHWTensor, decodeYoloOutput, detectPlastic, YOLO_INPUT_SIZE } from './yolo';

// YOLOv8 head layout: [1, 4 + numClasses, numBoxes], channel-major.
function makeOutput(boxes) {
  const n = boxes.length;
  const data = new Float32Array(5 * n);
  boxes.forEach(([cx, cy, w, h, score], i) => {
    data[0 * n + i] = cx;
    data[1 * n + i] = cy;
    data[2 * n + i] = w;
    data[3 * n + i] = h;
    data[4 * n + i] = score;
  });
  return { dims: [1, 5, n], data };
}

const identity = { scale: 1, offsetX: 0, offsetY: 0 };

describe('decodeYoloOutput', () => {
  it('converts centre/size boxes to corner boxes in source coordinates', () => {
    const out = decodeYoloOutput(makeOutput([[100, 200, 40, 60, 0.9]]), 640, 640, identity);
    expect(out).toHaveLength(1);
    expect(out[0].box).toEqual([80, 170, 120, 230]);
    expect(out[0].score).toBeCloseTo(0.9, 5);
    expect(out[0].classId).toBe(0);
  });

  it('applies the default 0.25 confidence gate (crushed items still pass, noise does not)', () => {
    const out = decodeYoloOutput(
      makeOutput([
        [100, 100, 40, 40, 0.26],
        [300, 300, 40, 40, 0.24],
      ]),
      640,
      640,
      identity
    );
    expect(out.map((d) => d.score.toFixed(2))).toEqual(['0.26']);
  });

  it('honours a custom confidence threshold', () => {
    const out = decodeYoloOutput(makeOutput([[100, 100, 40, 40, 0.5]]), 640, 640, identity, { confThreshold: 0.6 });
    expect(out).toHaveLength(0);
  });

  it('suppresses duplicate overlapping detections with NMS', () => {
    const out = decodeYoloOutput(
      makeOutput([
        [100, 100, 50, 50, 0.9],
        [102, 101, 50, 50, 0.7],
        [400, 400, 50, 50, 0.8],
      ]),
      640,
      640,
      identity
    );
    expect(out.map((d) => d.score.toFixed(1))).toEqual(['0.9', '0.8']);
  });

  it('undoes the letterbox scale and offset and clamps to the image', () => {
    // 640x480 source letterboxed into 640x640: scale 1, vertical offset 80.
    const letterboxInfo = { scale: 1, offsetX: 0, offsetY: 80 };
    const out = decodeYoloOutput(makeOutput([[320, 320, 100, 100, 0.9]]), 640, 480, letterboxInfo);
    expect(out[0].box).toEqual([270, 190, 370, 290]);

    const edge = decodeYoloOutput(makeOutput([[5, 85, 100, 100, 0.9]]), 640, 480, letterboxInfo);
    expect(edge[0].box[0]).toBe(0); // clamped, not negative
    expect(edge[0].box[1]).toBe(0);
  });
});

describe('canvasToCHWTensor', () => {
  it('lays pixels out planar (all R, then all G, then all B) scaled to 0-1', () => {
    const size = 2;
    const rgba = new Uint8ClampedArray(size * size * 4);
    // pixel 0 = (255, 0, 51), the rest black
    rgba.set([255, 0, 51, 255], 0);
    const canvas = { width: size, height: size, getContext: () => ({ getImageData: () => ({ data: rgba }) }) };

    const tensor = canvasToCHWTensor(canvas);
    expect(tensor.dims).toEqual([1, 3, size, size]);
    expect(tensor.data[0]).toBeCloseTo(1, 6); // R plane, pixel 0
    expect(tensor.data[size * size + 0]).toBeCloseTo(0, 6); // G plane, pixel 0
    expect(tensor.data[2 * size * size + 0]).toBeCloseTo(51 / 255, 6); // B plane, pixel 0
  });
});

describe('detectPlastic', () => {
  it('runs the session once, decodes, and releases tensors and the letterbox canvas', async () => {
    const lbCanvas = {
      width: YOLO_INPUT_SIZE,
      height: YOLO_INPUT_SIZE,
      getContext: () => ({
        fillStyle: '',
        fillRect() {},
        drawImage() {},
        getImageData: () => ({ data: new Uint8ClampedArray(YOLO_INPUT_SIZE * YOLO_INPUT_SIZE * 4) }),
      }),
    };
    vi.stubGlobal('document', { createElement: () => lbCanvas });

    const dispose = vi.fn();
    const output = { ...makeOutput([[320, 320, 100, 100, 0.9]]), dispose };
    const session = { inputNames: ['images'], outputNames: ['output0'], run: vi.fn(async () => ({ output0: output })) };

    const detections = await detectPlastic(session, {}, 640, 640);
    expect(session.run).toHaveBeenCalledTimes(1);
    expect(detections).toHaveLength(1);
    expect(dispose).toHaveBeenCalled();
    expect(lbCanvas.width).toBe(0);
    vi.unstubAllGlobals();
  });
});
