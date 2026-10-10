import { describe, expect, it, vi } from 'vitest';
import { selectTfBackend } from './tfBackend';

function fakeTf({ webgl = 'ok', cpu = 'ok' } = {}) {
  let active = null;
  const behaviours = { webgl, cpu };
  return {
    setBackend: vi.fn(async (name) => {
      const behaviour = behaviours[name];
      if (behaviour === 'throw') throw new Error(`${name} exploded`);
      if (behaviour === 'false') return false;
      active = name;
      return true;
    }),
    ready: vi.fn(async () => {}),
    getBackend: () => active,
  };
}

describe('selectTfBackend (WebGL availability)', () => {
  it('uses WebGL when it initialises', async () => {
    const tf = fakeTf();
    await expect(selectTfBackend(tf)).resolves.toBe('webgl');
    expect(tf.setBackend).toHaveBeenCalledTimes(1);
  });

  it('falls back to CPU when WebGL reports it is unavailable', async () => {
    const tf = fakeTf({ webgl: 'false' });
    await expect(selectTfBackend(tf)).resolves.toBe('cpu');
    expect(tf.setBackend.mock.calls.map((c) => c[0])).toEqual(['webgl', 'cpu']);
  });

  it('falls back to CPU when WebGL initialisation throws', async () => {
    await expect(selectTfBackend(fakeTf({ webgl: 'throw' }))).resolves.toBe('cpu');
  });

  it('fails with a clear, specific error when no backend works', async () => {
    const err = await selectTfBackend(fakeTf({ webgl: 'throw', cpu: 'false' })).catch((e) => e);
    expect(err.message).toMatch(/No TensorFlow\.js backend/);
    expect(err.message).toMatch(/webgl: webgl exploded/);
    expect(err.message).toMatch(/cpu: not available/);
  });
});
