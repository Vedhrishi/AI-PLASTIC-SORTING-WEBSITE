/**
 * The slice of the Fetch API this loader uses (real `fetch` satisfies it).
 * @typedef {(url: string) => Promise<{ ok: boolean, status: number, headers: { get(name: string): string | null }, body?: any, arrayBuffer?: () => Promise<ArrayBuffer> }>} FetchLike
 */

export class ModelLoadError extends Error {
  /**
   * @param {string} message
   * @param {{ kind: 'offline' | 'network' | 'http' | 'corrupt' | 'init', url?: string, status?: number, cause?: unknown }} details
   */
  constructor(message, { kind, url, status, cause }) {
    super(message, { cause });
    this.name = 'ModelLoadError';
    this.kind = kind;
    this.url = url;
    this.status = status;
  }
}

const MIN_PLAUSIBLE_MODEL_BYTES = 1024;

/**
 * Downloads a model file, reporting progress, and classifies failures so the
 * UI can say *why* it failed (offline vs. HTTP error vs. dropped connection).
 * @param {string} url
 * @param {{ onProgress?: (fraction: number) => void, fetchImpl?: FetchLike }} [options]
 * @returns {Promise<Uint8Array>}
 */
export async function fetchModelBytes(url, { onProgress, fetchImpl = globalThis.fetch } = {}) {
  const isOffline = () => globalThis.navigator?.onLine === false;

  let response;
  try {
    response = await fetchImpl(url);
  } catch (cause) {
    throw new ModelLoadError(`Could not reach ${url}`, { kind: isOffline() ? 'offline' : 'network', url, cause });
  }
  if (!response.ok) {
    throw new ModelLoadError(`HTTP ${response.status} for ${url}`, { kind: 'http', url, status: response.status });
  }

  const total = Number(response.headers.get('content-length')) || 0;
  const chunks = [];
  let loaded = 0;

  try {
    if (response.body?.getReader) {
      const reader = response.body.getReader();
      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        chunks.push(value);
        loaded += value.byteLength;
        // content-length is the *compressed* size when the server gzips, so
        // the ratio can exceed 1 — clamp rather than report >100%.
        if (onProgress && total > 0) onProgress(Math.min(1, loaded / total));
      }
    } else {
      const buffer = new Uint8Array(await /** @type {() => Promise<ArrayBuffer>} */ (response.arrayBuffer)());
      chunks.push(buffer);
      loaded = buffer.byteLength;
    }
  } catch (cause) {
    throw new ModelLoadError(`Download of ${url} was interrupted`, { kind: isOffline() ? 'offline' : 'network', url, cause });
  }

  if (loaded < MIN_PLAUSIBLE_MODEL_BYTES) {
    throw new ModelLoadError(`${url} returned only ${loaded} bytes`, { kind: 'corrupt', url });
  }

  const bytes = new Uint8Array(loaded);
  let offset = 0;
  for (const chunk of chunks) {
    bytes.set(chunk, offset);
    offset += chunk.byteLength;
  }
  onProgress?.(1);
  return bytes;
}
