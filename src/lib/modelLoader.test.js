import { afterEach, describe, expect, it, vi } from 'vitest';
import { ModelLoadError, fetchModelBytes } from './modelLoader';

function streamResponse(chunks, { status = 200, headers = {} } = {}) {
  let i = 0;
  return {
    ok: status >= 200 && status < 300,
    status,
    headers: new Headers(headers),
    body: {
      getReader: () => ({
        read: async () => (i < chunks.length ? { done: false, value: chunks[i++] } : { done: true, value: undefined }),
      }),
    },
  };
}

afterEach(() => vi.unstubAllGlobals());

describe('fetchModelBytes', () => {
  it('concatenates streamed chunks and reports progress up to 1', async () => {
    const a = new Uint8Array(1500).fill(1);
    const b = new Uint8Array(1500).fill(2);
    const progress = [];
    const fetchImpl = vi.fn(async () => streamResponse([a, b], { headers: { 'content-length': '3000' } }));

    const bytes = await fetchModelBytes('/m.onnx', { fetchImpl, onProgress: (p) => progress.push(p) });
    expect(bytes).toHaveLength(3000);
    expect(bytes[0]).toBe(1);
    expect(bytes[1500]).toBe(2);
    expect(progress.at(-1)).toBe(1);
    expect(progress).toEqual([...progress].sort((x, y) => x - y)); // monotonic
  });

  it('never reports more than 100% when the server compresses (content-length < decoded size)', async () => {
    const big = new Uint8Array(5000);
    const progress = [];
    await fetchModelBytes('/m.onnx', {
      fetchImpl: async () => streamResponse([big], { headers: { 'content-length': '1000' } }),
      onProgress: (p) => progress.push(p),
    });
    expect(Math.max(...progress)).toBe(1);
  });

  it('classifies an HTTP error with its status', async () => {
    const err = await fetchModelBytes('/missing.onnx', { fetchImpl: async () => streamResponse([], { status: 404 }) }).catch((e) => e);
    expect(err).toBeInstanceOf(ModelLoadError);
    expect(err).toMatchObject({ kind: 'http', status: 404, url: '/missing.onnx' });
  });

  it('classifies a rejected fetch as a network error', async () => {
    vi.stubGlobal('navigator', { onLine: true });
    const err = await fetchModelBytes('/m.onnx', { fetchImpl: async () => Promise.reject(new TypeError('Failed to fetch')) }).catch((e) => e);
    expect(err).toMatchObject({ kind: 'network' });
  });

  it('classifies a rejected fetch as offline when the browser is offline', async () => {
    vi.stubGlobal('navigator', { onLine: false });
    const err = await fetchModelBytes('/m.onnx', { fetchImpl: async () => Promise.reject(new TypeError('Failed to fetch')) }).catch((e) => e);
    expect(err).toMatchObject({ kind: 'offline' });
  });

  it('classifies a connection that drops mid-download', async () => {
    vi.stubGlobal('navigator', { onLine: true });
    const response = streamResponse([]);
    response.body.getReader = () => ({ read: async () => Promise.reject(new Error('reset')) });
    const err = await fetchModelBytes('/m.onnx', { fetchImpl: async () => response }).catch((e) => e);
    expect(err).toMatchObject({ kind: 'network' });
  });

  it('rejects an empty or truncated body as corrupt', async () => {
    const err = await fetchModelBytes('/m.onnx', { fetchImpl: async () => streamResponse([new Uint8Array(10)]) }).catch((e) => e);
    expect(err).toMatchObject({ kind: 'corrupt' });
  });
});
