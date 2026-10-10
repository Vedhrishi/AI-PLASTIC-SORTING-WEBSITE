/**
 * TF.js reports download problems as plain Errors whose only signal is the
 * message text. Classifies them into the same kinds the ONNX loader uses so
 * the UI can say "offline" / "HTTP 404" instead of a raw error string.
 * @param {any} err
 * @param {boolean} [online]
 * @returns {{ kind?: 'offline' | 'network' | 'http', status?: number }}
 */
export function classifyTfLoadError(err, online = true) {
  const message = String(err?.message ?? err ?? '');
  const status = /status code (\d{3})/i.exec(message)?.[1];
  if (status) return { kind: 'http', status: Number(status) };
  if (/failed to fetch|networkerror|network request failed|load failed|err_internet|err_connection/i.test(message)) {
    return { kind: online ? 'network' : 'offline' };
  }
  return {};
}
