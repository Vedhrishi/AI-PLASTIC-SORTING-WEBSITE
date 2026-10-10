import { beforeEach, describe, expect, it, vi } from 'vitest';

const create = vi.fn();
const env = { wasm: {} };
vi.mock('onnxruntime-web/wasm', () => ({ env, InferenceSession: { create: (...args) => create(...args) } }));

const { clearSessionCacheForTests, configureOrt, loadSession, resetOrtConfigForTests } = await import('./onnxSetup');

const modelBytes = () => new Uint8Array(4096);
const okFetch = () => vi.fn(async () => ({ ok: true, status: 200, headers: new Headers(), arrayBuffer: async () => modelBytes().buffer }));

beforeEach(() => {
  create.mockReset();
  clearSessionCacheForTests();
  resetOrtConfigForTests();
  for (const key of Object.keys(env.wasm)) delete env.wasm[key];
});

describe('configureOrt', () => {
  it('serves the runtime from this origin (no CDN); it is already inside our own worker, so no ORT proxy', () => {
    configureOrt({ baseUrl: '/', isolated: false, cores: 8 });
    expect(env.wasm.wasmPaths).toBe('/ort/');
    expect(env.wasm.proxy).toBe(false);
  });

  it('uses a single thread when the page is not cross-origin isolated', () => {
    expect(configureOrt({ isolated: false, cores: 16 }).threads).toBe(1);
  });

  it('uses up to 4 threads when isolated, never more than the cores available', () => {
    expect(configureOrt({ isolated: true, cores: 16 }).threads).toBe(4);
    resetOrtConfigForTests();
    expect(configureOrt({ isolated: true, cores: 2 }).threads).toBe(2);
  });
});

describe('loadSession', () => {
  it('creates one session per model and shares the in-flight promise between callers', async () => {
    create.mockResolvedValue({ id: 'session' });
    const fetchImpl = okFetch();
    const [a, b] = await Promise.all([
      loadSession('/m.onnx', { fetchImpl }),
      loadSession('/m.onnx', { fetchImpl }),
    ]);
    expect(a).toBe(b);
    expect(create).toHaveBeenCalledTimes(1);
    expect(fetchImpl).toHaveBeenCalledTimes(1);
  });

  it('falls back to WASM when the preferred execution provider cannot start', async () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    create.mockRejectedValueOnce(new Error('webgpu unavailable')).mockResolvedValueOnce({ id: 'wasm-session' });
    const session = await loadSession('/m.onnx', { executionProviders: ['webgpu', 'wasm'], fetchImpl: okFetch() });
    expect(session).toEqual({ id: 'wasm-session' });
    expect(create.mock.calls[0][1]).toEqual({ executionProviders: ['webgpu', 'wasm'] });
    expect(create.mock.calls[1][1]).toEqual({ executionProviders: ['wasm'] });
    warn.mockRestore();
  });

  it('reports an init error (not a hang) when even WASM cannot load the model', async () => {
    create.mockRejectedValue(new Error('invalid protobuf'));
    const err = await loadSession('/m.onnx', { fetchImpl: okFetch() }).catch((e) => e);
    expect(err).toMatchObject({ name: 'ModelLoadError', kind: 'init' });
    expect(err.message).toContain('invalid protobuf');
  });

  it('propagates download failures with their kind and lets a retry succeed', async () => {
    const failing = vi.fn(async () => ({ ok: false, status: 503, headers: new Headers() }));
    const err = await loadSession('/m.onnx', { fetchImpl: failing }).catch((e) => e);
    expect(err).toMatchObject({ kind: 'http', status: 503 });

    create.mockResolvedValue({ id: 'recovered' });
    await expect(loadSession('/m.onnx', { fetchImpl: okFetch() })).resolves.toEqual({ id: 'recovered' });
  });
});
