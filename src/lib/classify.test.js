import { describe, expect, it, vi } from 'vitest';

import { CLASSIFIER_INPUT_SIZE, RESIN_CODES, canvasToNormalizedTensor, classifyResin, softmaxArgmax } from './classify';
import { Tensor } from './tensor';
import { DISPOSAL_RULES, CONTAMINATION_ORDER } from '../data/disposalRules';
import { RESIN_INFO } from '../data/resinInfo';

const PIXELS = CLASSIFIER_INPUT_SIZE * CLASSIFIER_INPUT_SIZE;

function solidCanvas(r, g, b, size = CLASSIFIER_INPUT_SIZE) {
  const rgba = new Uint8ClampedArray(size * size * 4);
  for (let i = 0; i < size * size; i++) rgba.set([r, g, b, 255], i * 4);
  return { width: size, height: size, getContext: () => ({ getImageData: () => ({ data: rgba }) }) };
}

describe('canvasToNormalizedTensor (ImageNet preprocessing)', () => {
  it('produces planar [1,3,224,224] data normalised with the ImageNet mean/std', () => {
    const tensor = canvasToNormalizedTensor(solidCanvas(255, 0, 51));
    expect(tensor.dims).toEqual([1, 3, 224, 224]);
    expect(tensor.data).toHaveLength(3 * PIXELS);

    // Reference values computed by hand: (value/255 - mean) / std per channel.
    expect(tensor.data[0]).toBeCloseTo((1 - 0.485) / 0.229, 5); // R plane
    expect(tensor.data[PIXELS]).toBeCloseTo((0 - 0.456) / 0.224, 5); // G plane
    expect(tensor.data[2 * PIXELS]).toBeCloseTo((51 / 255 - 0.406) / 0.225, 5); // B plane
    // Whole planes are uniform for a solid colour — proves no interleaving.
    expect(tensor.data[PIXELS - 1]).toBe(tensor.data[0]);
    expect(tensor.data[2 * PIXELS - 1]).toBe(tensor.data[PIXELS]);
  });

  it('fails loudly when the crop is not 224x224 instead of feeding ONNX a bad shape', () => {
    expect(() => canvasToNormalizedTensor(solidCanvas(10, 10, 10, 100))).toThrow(/expected 150528 elements/);
  });
});

describe('softmaxArgmax', () => {
  it('returns the argmax index and its softmax probability', () => {
    const { index, confidence } = softmaxArgmax(Float32Array.from([2, 1, 0, -1]));
    const expected = 1 / (1 + Math.exp(-1) + Math.exp(-2) + Math.exp(-3));
    expect(index).toBe(0);
    expect(confidence).toBeCloseTo(expected, 6);
  });

  it('is numerically stable for very large logits', () => {
    const { index, confidence } = softmaxArgmax(Float32Array.from([1000, 0, 0, 0]));
    expect(index).toBe(0);
    expect(confidence).toBeCloseTo(1, 6);
  });
});

describe('resin class mapping', () => {
  it('maps model output indices to the four supported resins and their ID codes', () => {
    expect(RESIN_CODES).toEqual({
      0: { code: 1, label: 'PET' },
      1: { code: 2, label: 'HDPE' },
      2: { code: 5, label: 'PP' },
      3: { code: 6, label: 'PS' },
    });
  });

  it('has disposal rules and facts for every resin the model can output', () => {
    for (const { label } of Object.values(RESIN_CODES)) {
      expect(RESIN_INFO[label], `RESIN_INFO.${label}`).toBeDefined();
      for (const level of CONTAMINATION_ORDER) {
        expect(DISPOSAL_RULES[label]?.[level], `${label} / ${level}`).toBeDefined();
      }
    }
  });
});

describe('classifyResin', () => {
  const makeSession = (logits) => ({
    inputNames: ['input'],
    outputNames: ['output'],
    run: vi.fn(async () => ({ output: new Tensor('float32', Float32Array.from(logits), [1, logits.length]) })),
  });

  it.each([
    [[5, 0, 0, 0], 'PET', 1],
    [[0, 5, 0, 0], 'HDPE', 2],
    [[0, 0, 5, 0], 'PP', 5],
    [[0, 0, 0, 5], 'PS', 6],
  ])('turns logits %j into %s (code %i)', async (logits, label, code) => {
    const result = await classifyResin(makeSession(logits), solidCanvas(120, 120, 120));
    expect(result.label).toBe(label);
    expect(result.code).toBe(code);
    expect(result.confidence).toBeGreaterThan(0.9);
  });

  it('reports Unknown (not a crash) if the model returns an unmapped class index', async () => {
    const result = await classifyResin(makeSession([0, 0, 0, 0, 9]), solidCanvas(1, 2, 3));
    expect(result.label).toBe('Unknown');
    expect(result.code).toBeNull();
  });

  it('rethrows inference failures after logging diagnostics', async () => {
    const spy = vi.spyOn(console, 'error').mockImplementation(() => {});
    const session = { inputNames: ['input'], outputNames: ['output'], run: vi.fn(async () => Promise.reject(new Error('boom'))) };
    await expect(classifyResin(session, solidCanvas(0, 0, 0))).rejects.toThrow('boom');
    expect(spy).toHaveBeenCalled();
    spy.mockRestore();
  });
});
