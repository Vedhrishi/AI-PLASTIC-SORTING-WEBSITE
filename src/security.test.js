import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';

const config = JSON.parse(readFileSync(new URL('../vercel.json', import.meta.url), 'utf8'));
const siteRule = config.headers.find((rule) => rule.source === '/(.*)');
const headers = Object.fromEntries(siteRule.headers.map(({ key, value }) => [key.toLowerCase(), value]));
const csp = Object.fromEntries(
  headers['content-security-policy']
    .split(';')
    .map((part) => part.trim().split(/\s+/))
    .filter(([name]) => name)
    .map(([name, ...sources]) => [name, sources])
);

describe('security headers (vercel.json)', () => {
  it('sends the baseline hardening headers on every route', () => {
    expect(headers['x-content-type-options']).toBe('nosniff');
    expect(headers['referrer-policy']).toBe('no-referrer');
    expect(headers['strict-transport-security']).toMatch(/max-age=\d{7,}/);
    expect(headers['x-frame-options']).toBe('DENY');
    expect(headers['cross-origin-opener-policy']).toBe('same-origin');
    expect(headers['cross-origin-embedder-policy']).toBe('require-corp');
  });

  it('allows the camera for this origin only and nothing else sensitive', () => {
    const policy = headers['permissions-policy'];
    expect(policy).toMatch(/camera=\(self\)/);
    expect(policy).toMatch(/microphone=\(\)/);
    expect(policy).toMatch(/geolocation=\(\)/);
  });

  it('enforces a CSP (not just report-only)', () => {
    expect(headers['content-security-policy']).toBeDefined();
    expect(headers['content-security-policy-report-only']).toBeUndefined();
  });

  it('keeps the CSP closed: no eval, no inline script, no wildcards, no remote origins', () => {
    expect(csp['default-src']).toEqual(["'self'"]);
    expect(csp['script-src']).toEqual(["'self'", "'wasm-unsafe-eval'"]);
    expect(csp['object-src']).toEqual(["'none'"]);
    expect(csp['frame-ancestors']).toEqual(["'none'"]);
    expect(csp['base-uri']).toEqual(["'self'"]);
    const allSources = Object.values(csp).flat();
    expect(allSources).not.toContain("'unsafe-eval'");
    expect(allSources).not.toContain('*');
    expect(allSources.filter((source) => /^https?:/.test(source))).toEqual([]);
    expect(csp['script-src']).not.toContain("'unsafe-inline'");
  });
});
