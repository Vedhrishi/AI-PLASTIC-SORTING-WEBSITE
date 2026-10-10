// Browser-capability detection and user-facing failure messages. Everything
// here is a pure function of the globals passed in (default: the real ones),
// so each failure path can be unit-tested without a browser.

/**
 * @typedef {{ kind: string, title: string, message: string, hint?: string }} Issue
 */

/**
 * Checks the hard requirements the app cannot run without.
 * @param {any} [g]
 * @returns {{ ok: boolean, problems: string[] }}
 */
export function detectSupport(g = globalThis) {
  const problems = [];
  if (typeof g.WebAssembly !== 'object') {
    problems.push('WebAssembly is not available (the neural-network runtime needs it).');
  }
  if (typeof g.fetch !== 'function') {
    problems.push('The Fetch API is not available (needed to download the models).');
  }
  const canvas = g.document?.createElement?.('canvas');
  if (!canvas || typeof canvas.getContext !== 'function' || !canvas.getContext('2d')) {
    problems.push('Canvas 2D is not available (needed to process images).');
  }
  return { ok: problems.length === 0, problems };
}

/**
 * Maps a getUserMedia failure (or a missing camera API) to a clear message.
 * The app stays usable for uploads and demo presets in every case.
 * @param {any} err
 * @param {any} [g]
 * @returns {Issue}
 */
export function describeCameraError(err, g = globalThis) {
  const name = err?.name ?? '';
  const apiMissing = !g.navigator?.mediaDevices?.getUserMedia;

  if (apiMissing && name !== 'NotAllowedError') {
    if (g.isSecureContext === false) {
      return {
        kind: 'insecure-context',
        title: 'Camera needs a secure connection',
        message: 'Browsers only allow camera access on HTTPS pages (or localhost).',
        hint: 'Open the HTTPS address of this app, or upload an image instead.',
      };
    }
    return {
      kind: 'unsupported',
      title: 'Camera not supported in this browser',
      message: 'This browser does not provide camera access to web pages.',
      hint: 'Upload an image or try a demo preset, or switch to a current Chrome, Edge, Firefox or Safari.',
    };
  }

  switch (name) {
    case 'NotAllowedError':
    case 'SecurityError':
    case 'PermissionDeniedError':
      return {
        kind: 'permission-denied',
        title: 'Camera permission denied',
        message: 'The browser blocked camera access for this site.',
        hint: 'Allow the camera from the lock icon in the address bar, then press Retry camera. You can also upload an image or try a demo preset.',
      };
    case 'NotFoundError':
    case 'DevicesNotFoundError':
    case 'OverconstrainedError':
      return {
        kind: 'no-camera',
        title: 'No camera found',
        message: 'No usable camera was detected on this device.',
        hint: 'Plug in or enable a camera and press Retry camera, or upload an image instead.',
      };
    case 'NotReadableError':
    case 'TrackStartError':
    case 'AbortError':
      return {
        kind: 'camera-busy',
        title: 'Camera is unavailable',
        message: 'The camera could not be started. Another app or tab may be using it.',
        hint: 'Close other apps that use the camera and press Retry camera, or upload an image instead.',
      };
    default:
      return {
        kind: 'camera-error',
        title: 'Camera could not be started',
        message: err?.message ? String(err.message) : 'An unknown camera error occurred.',
        hint: 'Press Retry camera, or upload an image instead.',
      };
  }
}

/**
 * Maps a model-download / model-initialisation failure to a clear message.
 * @param {any} err
 * @param {any} [g]
 * @returns {Issue}
 */
export function describeModelError(err, g = globalThis) {
  const offline = g.navigator?.onLine === false;
  const kind = err?.kind;

  if (offline || kind === 'offline') {
    return {
      kind: 'offline',
      title: "You're offline",
      message: 'The recognition models could not be downloaded because there is no internet connection.',
      hint: 'Reconnect and press Retry. Demo presets still work offline.',
    };
  }
  if (kind === 'http') {
    return {
      kind: 'http',
      title: 'Model download failed',
      message: `The server answered ${err.status ?? 'with an error'} while sending a model file${err.url ? ` (${err.url})` : ''}.`,
      hint: 'Press Retry. If it keeps failing, the deployment is missing its model files.',
    };
  }
  if (kind === 'network') {
    return {
      kind: 'network',
      title: 'Model download failed',
      message: 'The connection dropped while downloading the recognition models.',
      hint: 'Check your connection and press Retry.',
    };
  }
  if (kind === 'corrupt') {
    return {
      kind: 'corrupt',
      title: 'Model file is damaged',
      message: 'A downloaded model file was empty or truncated.',
      hint: 'Press Retry. If it keeps failing, redeploy the model files.',
    };
  }
  return {
    kind: 'model-init',
    title: 'Models could not be started',
    message: err?.message ? String(err.message) : 'The recognition models failed to initialise.',
    hint: 'Press Retry. If it keeps failing, try a different browser.',
  };
}
