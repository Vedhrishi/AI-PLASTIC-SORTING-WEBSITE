import { describe, expect, it } from 'vitest';
import { computeCoverProjection, iou, nms, overlapFraction, projectBox } from './geometry';

describe('iou', () => {
  it('is 1 for identical boxes and 0 for disjoint boxes', () => {
    expect(iou([0, 0, 10, 10], [0, 0, 10, 10])).toBe(1);
    expect(iou([0, 0, 10, 10], [20, 20, 30, 30])).toBe(0);
  });

  it('matches the hand-computed value for a half-overlap (50 / 150 = 1/3)', () => {
    expect(iou([0, 0, 10, 10], [5, 0, 15, 10])).toBeCloseTo(1 / 3, 10);
  });
});

describe('overlapFraction', () => {
  it('is the share of the FIRST box covered by the second (not symmetric)', () => {
    const plastic = [0, 0, 10, 10]; // area 100
    const person = [5, 0, 25, 10]; // covers x 5..10 of the plastic box
    expect(overlapFraction(plastic, person)).toBeCloseTo(0.5, 10);
    expect(overlapFraction(person, plastic)).toBeCloseTo(0.25, 10);
  });

  it('returns 0 for disjoint and for zero-area boxes', () => {
    expect(overlapFraction([0, 0, 10, 10], [50, 50, 60, 60])).toBe(0);
    expect(overlapFraction([5, 5, 5, 5], [0, 0, 10, 10])).toBe(0);
  });
});

describe('computeCoverProjection / projectBox (object-fit: cover)', () => {
  it('scales by the larger axis ratio and centre-crops the overflowing axis', () => {
    // 640x480 (4:3) video shown in a 16:9 box 800x450: width-limited, height is cropped.
    const projection = computeCoverProjection(640, 480, 800, 450);
    expect(projection.scale).toBeCloseTo(800 / 640, 10);
    expect(projection.offsetX).toBe(0);
    expect(projection.offsetY).toBeCloseTo((450 - 480 * (800 / 640)) / 2, 10); // negative = cropped
    expect(projection.offsetY).toBeLessThan(0);
  });

  it('is the identity when the box has the same size and aspect', () => {
    expect(computeCoverProjection(640, 480, 640, 480)).toEqual({ scale: 1, offsetX: 0, offsetY: 0 });
  });

  it('maps a native-space box onto the screen using point * scale + offset', () => {
    const projection = { scale: 2, offsetX: 10, offsetY: -20 };
    expect(projectBox([1, 2, 3, 4], projection)).toEqual([12, -16, 16, -12]);
  });
});

describe('nms', () => {
  it('keeps the best box and drops lower-scoring overlaps', () => {
    const boxes = [
      [0, 0, 10, 10],
      [1, 1, 11, 11],
      [100, 100, 110, 110],
    ];
    const scores = [0.6, 0.9, 0.5];
    expect(nms(boxes, scores, 0.45)).toEqual([1, 2]);
  });

  it('keeps everything when nothing overlaps beyond the threshold', () => {
    const boxes = [
      [0, 0, 10, 10],
      [20, 0, 30, 10],
    ];
    expect(nms(boxes, [0.5, 0.4]).sort()).toEqual([0, 1]);
  });
});
