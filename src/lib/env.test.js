import { describe, expect, it } from 'vitest';
import { describeCameraError, describeModelError, detectSupport } from './env';

const goodGlobals = () => ({
  WebAssembly: {},
  fetch: () => {},
  document: { createElement: () => ({ getContext: () => ({}) }) },
});

describe('detectSupport', () => {
  it('passes when WebAssembly, fetch and canvas are present', () => {
    expect(detectSupport(goodGlobals())).toEqual({ ok: true, problems: [] });
  });

  it('reports each missing requirement separately', () => {
    const { ok, problems } = detectSupport({ document: { createElement: () => ({ getContext: () => null }) } });
    expect(ok).toBe(false);
    expect(problems).toHaveLength(3);
    expect(problems.join(' ')).toMatch(/WebAssembly/);
    expect(problems.join(' ')).toMatch(/Fetch/);
    expect(problems.join(' ')).toMatch(/Canvas/);
  });
});

describe('describeCameraError', () => {
  const withCamera = { navigator: { mediaDevices: { getUserMedia: () => {} } }, isSecureContext: true };

  it.each([
    ['NotAllowedError', 'permission-denied'],
    ['SecurityError', 'permission-denied'],
    ['NotFoundError', 'no-camera'],
    ['OverconstrainedError', 'no-camera'],
    ['NotReadableError', 'camera-busy'],
    ['AbortError', 'camera-busy'],
    ['SomethingElse', 'camera-error'],
  ])('maps %s to %s with a title, message and a way forward', (name, kind) => {
    const issue = describeCameraError({ name, message: 'raw' }, withCamera);
    expect(issue.kind).toBe(kind);
    expect(issue.title).toMatch(/\S/);
    expect(issue.message).toMatch(/\S/);
    expect(issue.hint).toMatch(/upload|Retry/i);
  });

  it('explains a missing camera API on an insecure page versus an unsupported browser', () => {
    const insecure = describeCameraError(new TypeError('x'), { navigator: {}, isSecureContext: false });
    expect(insecure.kind).toBe('insecure-context');
    const unsupported = describeCameraError(new TypeError('x'), { navigator: {}, isSecureContext: true });
    expect(unsupported.kind).toBe('unsupported');
  });

  it('still treats a permission denial as a denial when the API is hidden by policy', () => {
    expect(describeCameraError({ name: 'NotAllowedError' }, { navigator: {}, isSecureContext: true }).kind).toBe('permission-denied');
  });
});

describe('describeModelError', () => {
  const online = { navigator: { onLine: true } };

  it('says "offline" when the browser reports no connection, whatever the underlying error', () => {
    expect(describeModelError(new Error('Failed to fetch'), { navigator: { onLine: false } }).kind).toBe('offline');
  });

  it.each([
    [{ kind: 'offline' }, 'offline'],
    [{ kind: 'http', status: 404, url: '/models/best.onnx' }, 'http'],
    [{ kind: 'network' }, 'network'],
    [{ kind: 'corrupt' }, 'corrupt'],
    [new Error('bad graph'), 'model-init'],
  ])('maps %j to %s', (err, kind) => {
    const issue = describeModelError(err, online);
    expect(issue.kind).toBe(kind);
    expect(issue.message).toMatch(/\S/);
    expect(issue.hint).toMatch(/Retry|presets/);
  });

  it('includes the HTTP status and file in the message', () => {
    const issue = describeModelError({ kind: 'http', status: 404, url: '/models/best.onnx' }, online);
    expect(issue.message).toContain('404');
    expect(issue.message).toContain('/models/best.onnx');
  });
});
