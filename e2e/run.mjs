// Real-browser verification of the PRODUCTION build (served by `vite preview`
// with the exact headers from vercel.json). Drives Chromium through the happy
// path and every failure path the app promises to handle, and fails (exit 1)
// if any expectation, console error or CSP violation is observed.
//
//   npm run build && npm run test:e2e
//
// Needs Chromium:  npx playwright install chromium
import { chromium } from '@playwright/test';
import { spawn } from 'node:child_process';
import { mkdirSync } from 'node:fs';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = fileURLToPath(new URL('..', import.meta.url));
const outDir = join(root, 'e2e', 'out');
mkdirSync(outDir, { recursive: true });

const PORT = 4173;
const URL_BASE = `http://localhost:${PORT}`;
const READY_TIMEOUT = 120_000;
const FAKE_CAMERA = ['--use-fake-device-for-media-stream', '--use-fake-ui-for-media-stream'];

const results = [];
let failures = 0;

function record(scenario, name, ok, detail = '') {
  results.push({ scenario, check: name, result: ok ? 'PASS' : 'FAIL', detail });
  if (!ok) failures += 1;
}

async function startPreview() {
  const proc = spawn(process.execPath, [join(root, 'node_modules', 'vite', 'bin', 'vite.js'), 'preview', '--port', String(PORT), '--strictPort'], {
    cwd: root,
    stdio: ['ignore', 'pipe', 'pipe'],
  });
  let log = '';
  proc.stdout.on('data', (d) => (log += d));
  proc.stderr.on('data', (d) => (log += d));
  for (let i = 0; i < 100; i++) {
    try {
      const res = await fetch(URL_BASE);
      if (res.ok) return proc;
    } catch {
      /* not up yet */
    }
    await new Promise((r) => setTimeout(r, 300));
  }
  proc.kill();
  throw new Error(`vite preview did not start:\n${log}`);
}

const CSP_PROBE = `
  window.__csp = [];
  window.__statusLog = [];
  const logStatus = () => {
    const s = document.querySelector('[data-testid="status-badge"]')?.getAttribute('data-status');
    if (s && window.__statusLog[window.__statusLog.length - 1] !== s) window.__statusLog.push(s);
  };
  new MutationObserver(logStatus).observe(document, { subtree: true, childList: true, attributes: true });
  window.__longTasks = [];
  try {
    new PerformanceObserver((list) => {
      for (const entry of list.getEntries()) window.__longTasks.push(entry.duration);
    }).observe({ type: 'longtask', buffered: true });
  } catch { /* longtask unsupported */ }
  window.__onlineEvents = 0;
  window.addEventListener('online', () => { window.__onlineEvents += 1; });
  document.addEventListener('securitypolicyviolation', (e) => {
    window.__csp.push(e.violatedDirective + ' <- ' + (e.blockedURI || e.sourceFile || 'inline'));
  });
`;

/**
 * @param {import('@playwright/test').Browser} browser
 * @param {{ contextOptions?: import('@playwright/test').BrowserContextOptions, initScript?: string }} [options]
 */
async function newPage(browser, { contextOptions = {}, initScript } = {}) {
  const context = await browser.newContext({ viewport: { width: 1366, height: 768 }, ...contextOptions });
  const page = await context.newPage();
  const consoleErrors = [];
  const consoleWarnings = [];
  page.on('console', (msg) => {
    if (msg.type() === 'error') consoleErrors.push(msg.text());
    if (msg.type() === 'warning') consoleWarnings.push(msg.text());
  });
  page.on('pageerror', (err) => consoleErrors.push(`pageerror: ${err.message}`));
  await page.addInitScript(CSP_PROBE);
  if (initScript) await page.addInitScript(initScript);
  return { context, page, consoleErrors, consoleWarnings };
}

const loadState = (page) => page.getAttribute('[data-testid="app-root"]', 'data-load-state');
const waitReady = (page) =>
  page.waitForFunction(() => document.querySelector('[data-testid="app-root"]')?.getAttribute('data-load-state') === 'ready', null, {
    timeout: READY_TIMEOUT,
  });
const cspViolations = (page) => page.evaluate(() => /** @type {any} */ (window).__csp);

// A deterministic synthetic test image (gradient + shapes), built in the page.
async function uploadSyntheticImage(page) {
  await page.evaluate(async () => {
    const canvas = document.createElement('canvas');
    canvas.width = 800;
    canvas.height = 600;
    const ctx = canvas.getContext('2d');
    const gradient = ctx.createLinearGradient(0, 0, 800, 600);
    gradient.addColorStop(0, '#d9e6f2');
    gradient.addColorStop(1, '#6b8fb3');
    ctx.fillStyle = gradient;
    ctx.fillRect(0, 0, 800, 600);
    ctx.fillStyle = 'rgba(40,120,200,0.85)';
    ctx.fillRect(330, 120, 140, 360);
    const blob = await new Promise((resolve) => canvas.toBlob(resolve, 'image/png'));
    const dt = new DataTransfer();
    dt.items.add(new File([blob], 'synthetic.png', { type: 'image/png' }));
    const input = /** @type {HTMLInputElement} */ (document.querySelector('input[type="file"]'));
    input.files = dt.files;
    input.dispatchEvent(new Event('change', { bubbles: true }));
  });
}

// Performs `action`, then waits for the scan it triggers to go analyzing -> outcome.
// Returns the outcome status. (Reading the badge right after the action would
// race: the previous scan's outcome is still showing until analysis starts.)
async function runScan(page, action, timeout = 60_000) {
  await page.evaluate(() => {
    /** @type {any} */ (window).__statusLog = [];
  });
  await action();
  await page.waitForFunction(
    () => {
      const log = /** @type {any} */ (window).__statusLog;
      const i = log.indexOf('analyzing');
      return i >= 0 && ['result', 'no-item', 'veto', 'classifier-error'].includes(log[log.length - 1]);
    },
    null,
    { timeout }
  );
  return page.evaluate(() => /** @type {any} */ (window).__statusLog.at(-1));
}

async function openInputSource(page) {
  await page.getByText('Input source & demo presets').click();
}

async function scenarioHappyPath(browser) {
  const name = 'happy path (fake camera, production build + headers)';
  const { context, page, consoleErrors, consoleWarnings } = await newPage(browser);
  await context.grantPermissions(['camera'], { origin: URL_BASE });
  /** @type {{ method: string, url: string, body: string | null }[]} */
  const requests = [];
  context.on('request', (req) => requests.push({ method: req.method(), url: req.url(), body: req.postData() }));
  const response = await page.goto(URL_BASE, { waitUntil: 'load' });

  const headers = response.headers();
  record(name, 'security headers present', Boolean(headers['x-content-type-options'] && headers['strict-transport-security'] && headers['permissions-policy'] && headers['x-frame-options']));
  record(name, 'CSP is enforced (not report-only)', Boolean(headers['content-security-policy']) && !headers['content-security-policy-report-only']);

  const t0 = Date.now();
  await waitReady(page);
  record(name, 'models loaded', true, `${Math.round((Date.now() - t0) / 100) / 10}s to ready`);
  record(name, 'cross-origin isolated (multi-thread WASM possible)', await page.evaluate(() => crossOriginIsolated));

  const engine = await page.locator('text=ENGINE').first().evaluate((el) => el.parentElement.textContent);
  record(name, 'diagnostics report the real engine', /WASM/.test(engine) && !/WebGL \(GPU\) → WASM/.test(engine), engine.replace(/\s+/g, ' ').trim());

  await page.waitForSelector('[data-testid="take-photo-button"]', { timeout: 20_000 });
  const cameraStatus = await runScan(page, () => page.click('[data-testid="take-photo-button"]'));
  record(name, 'camera snapshot runs the pipeline without error', cameraStatus !== 'classifier-error', `status=${cameraStatus}`);

  const upStart = Date.now();
  const uploadStatus = await runScan(page, () => uploadSyntheticImage(page));
  record(name, 'uploaded image runs the pipeline without error', uploadStatus !== 'classifier-error', `status=${uploadStatus}, ${Date.now() - upStart}ms wall`);
  const latency = await page.locator('text=LAST SCAN').first().evaluate((el) => el.textContent);
  record(name, 'latency reported', /\d+\s*ms/.test(latency), latency.replace(/\s+/g, ' ').trim());

  const longTasks = await page.evaluate(() => /** @type {any} */ (window).__longTasks);
  const longest = Math.round(Math.max(0, ...longTasks));
  record(name, 'main thread stays responsive during load + scans (longest task < 500 ms)', longest < 500, `longest task ${longest} ms, ${longTasks.length} tasks over 50 ms`);

  await openInputSource(page);
  await runScan(page, () => page.getByText('Clean PET Bottle').click());
  const [download] = await Promise.all([page.waitForEvent('download', { timeout: 30_000 }), page.getByText('Export audit receipt').click()]);
  await download.saveAs(join(outDir, 'receipt.png'));
  record(name, 'receipt export downloads a PNG', /^Scan-Receipt-\d+\.png$/.test(download.suggestedFilename()), download.suggestedFilename());

  // Privacy: frames are analysed locally. Every request the app made while
  // loading, scanning (camera + upload) and exporting was a body-less GET to
  // its own origin (or an in-memory blob:/data: URL).
  const external = requests.filter((r) => !/^(blob:|data:)/.test(r.url) && new URL(r.url).origin !== URL_BASE);
  const nonGet = requests.filter((r) => r.method !== 'GET' || r.body);
  record(name, 'no request leaves the origin and none uploads data', external.length === 0 && nonGet.length === 0, `${requests.length} requests; external=${external.length}; non-GET/with body=${nonGet.length}`);

  const violations = await cspViolations(page);
  record(name, 'no CSP violations (load, scans, receipt export)', violations.length === 0, violations.join(' | '));
  record(name, 'no console errors', consoleErrors.length === 0, consoleErrors.slice(0, 3).join(' | '));
  if (consoleWarnings.length) record(name, 'console warnings (informational)', true, consoleWarnings.slice(0, 3).join(' | '));
  await page.screenshot({ path: join(outDir, 'happy.png') });
  await context.close();
}

async function scenarioCamera(browser, { name, initScript, kind }) {
  const { context, page, consoleErrors } = await newPage(browser, { initScript });
  await page.goto(URL_BASE, { waitUntil: 'load' });
  await waitReady(page);
  const issue = page.locator('[data-testid="camera-issue"]');
  await issue.waitFor({ timeout: 15_000 });
  record(name, `clear message shown (kind=${kind})`, (await issue.getAttribute('data-issue-kind')) === kind, (await issue.innerText()).replace(/\s+/g, ' '));
  record(name, 'no blank screen: Take Photo hidden, retry offered', (await page.locator('[data-testid="take-photo-button"]').count()) === 0 && (await page.locator('[data-testid="retry-camera"]').count()) === 1);

  await openInputSource(page);
  await page.getByText('Clean PET Bottle').click();
  await page.waitForSelector('[data-testid="result-panel"]', { timeout: 10_000 });
  record(name, 'demo preset still works without a camera', (await page.textContent('[data-testid="result-resin"]')).includes('PET'));

  const uploadStatus = await runScan(page, () => uploadSyntheticImage(page));
  record(name, 'image upload still works without a camera', uploadStatus !== 'classifier-error', `status=${uploadStatus}`);
  record(name, 'no unexpected console errors', consoleErrors.length === 0, consoleErrors.slice(0, 3).join(' | '));
  await context.close();
}

async function scenarioModelDownload(browser) {
  const name = 'model download failure + retry';
  const { context, page } = await newPage(browser, { contextOptions: {} });
  await context.grantPermissions(['camera'], { origin: URL_BASE });
  await page.route('**/models/best.onnx', (route) => route.abort('connectionfailed'));
  await page.goto(URL_BASE, { waitUntil: 'load' });
  const err = page.locator('[data-testid="model-error"]');
  await err.waitFor({ timeout: 60_000 });
  record(name, 'network failure explained', (await err.getAttribute('data-issue-kind')) === 'network', (await err.innerText()).replace(/\s+/g, ' '));
  record(name, 'upload locked while models are missing', await page.evaluate(() => /** @type {HTMLInputElement} */ (document.querySelector('input[type="file"]')).disabled));

  await page.unroute('**/models/best.onnx');
  await page.route('**/models/best.onnx', (route) => route.fulfill({ status: 404, body: 'nope' }));
  await page.click('[data-testid="retry-models"]');
  await page.locator('[data-testid="model-error"][data-issue-kind="http"]').waitFor({ timeout: 60_000 });
  record(name, 'HTTP 404 explained', true, (await page.locator('[data-testid="model-error"]').innerText()).replace(/\s+/g, ' '));

  await page.unroute('**/models/best.onnx');
  await page.click('[data-testid="retry-models"]');
  await waitReady(page);
  record(name, 'Retry recovers once the files are reachable', (await loadState(page)) === 'ready');
  await context.close();
}

async function scenarioOffline(browser) {
  const name = 'offline';
  const { context, page } = await newPage(browser);
  await context.grantPermissions(['camera'], { origin: URL_BASE });
  await page.route('**/models/best.onnx', (route) => route.abort('connectionfailed'));
  await page.goto(URL_BASE, { waitUntil: 'load' });
  await page.locator('[data-testid="model-error"]').waitFor({ timeout: 60_000 });
  await page.unroute('**/models/best.onnx');
  await context.setOffline(true);
  await page.click('[data-testid="retry-models"]');
  await page.locator('[data-testid="model-error"][data-issue-kind="offline"]').waitFor({ timeout: 60_000 });
  record(name, 'offline is named as the cause', true, (await page.locator('[data-testid="model-error"]').innerText()).replace(/\s+/g, ' '));
  await openInputSource(page);
  await page.getByText('Dirty PP Cup').click();
  await page.waitForSelector('[data-testid="result-panel"]', { timeout: 10_000 });
  record(name, 'demo presets still work offline', true);

  await context.setOffline(false);
  await page.waitForTimeout(2000);
  // Headless Chromium does not always emit "online" when emulation is switched
  // off; if it did not, dispatch it so the app's listener is still exercised.
  const browserFiredOnline = (await page.evaluate(() => /** @type {any} */ (window).__onlineEvents)) > 0;
  if (!browserFiredOnline) await page.evaluate(() => window.dispatchEvent(new Event('online')));
  await waitReady(page); // the "online" event retries automatically
  record(name, 'recovers automatically when the connection returns', (await loadState(page)) === 'ready', browserFiredOnline ? 'browser fired the online event' : 'online event dispatched manually (Playwright did not emit it)');
  await context.close();
}

async function scenarioNoWebGL() {
  const name = 'WebGL / GPU disabled';
  const browser = await chromium.launch({
    args: [...FAKE_CAMERA, '--disable-gpu', '--disable-software-rasterizer', '--disable-webgl', '--disable-webgl2', '--disable-3d-apis'],
  });
  const { context, page, consoleErrors } = await newPage(browser);
  await context.grantPermissions(['camera'], { origin: URL_BASE });
  await page.goto(URL_BASE, { waitUntil: 'load' });
  await waitReady(page);
  const hasWebGL = await page.evaluate(() => Boolean(document.createElement('canvas').getContext('webgl')));
  const engine = await page.locator('text=ENGINE').first().evaluate((el) => el.parentElement.textContent);
  record(name, 'browser really has no WebGL', hasWebGL === false);
  record(name, 'app still loads, using the CPU backend', /CPU/i.test(engine), engine.replace(/\s+/g, ' ').trim());
  const status = await runScan(page, () => uploadSyntheticImage(page));
  record(name, 'inference still runs', status !== 'classifier-error', `status=${status}`);
  record(name, 'no console errors', consoleErrors.length === 0, consoleErrors.slice(0, 3).join(' | '));

  // The documented escape hatch for broken GPU drivers.
  const forced = await newPage(browser);
  await forced.context.grantPermissions(['camera'], { origin: URL_BASE });
  await forced.page.goto(`${URL_BASE}/?tfBackend=cpu`, { waitUntil: 'load' });
  await waitReady(forced.page);
  const forcedEngine = await forced.page.locator('text=ENGINE').first().evaluate((el) => el.parentElement.textContent);
  record(name, '?tfBackend=cpu forces the CPU backend', /CPU/i.test(forcedEngine), forcedEngine.replace(/\s+/g, ' ').trim());
  await browser.close();
}

async function scenarioStress(browser) {
  const name = 'stress: 25 scans + 4x CPU throttle';
  const { context, page } = await newPage(browser);
  await context.grantPermissions(['camera'], { origin: URL_BASE });
  await page.goto(URL_BASE, { waitUntil: 'load' });
  await waitReady(page);

  const cdp = await context.newCDPSession(page);
  const heapMb = async () => {
    await cdp.send('HeapProfiler.collectGarbage');
    const { metrics } = await cdp.send('Performance.getMetrics');
    return metrics.find((m) => m.name === 'JSHeapUsedSize').value / 1024 / 1024;
  };
  await cdp.send('Performance.enable');

  await runScan(page, () => uploadSyntheticImage(page)); // warm-up (allocations, JIT)
  const before = await heapMb();
  const nodesBefore = (await cdp.send('Performance.getMetrics')).metrics.find((m) => m.name === 'Nodes').value;
  for (let i = 0; i < 25; i++) await runScan(page, () => uploadSyntheticImage(page));
  const after = await heapMb();
  const nodesAfter = (await cdp.send('Performance.getMetrics')).metrics.find((m) => m.name === 'Nodes').value;
  record(name, 'main-thread JS heap does not grow over 25 scans (< 10 MB)', after - before < 10, `${before.toFixed(1)} MB -> ${after.toFixed(1)} MB`);
  record(name, 'DOM node count stays flat', nodesAfter - nodesBefore < 200, `${nodesBefore} -> ${nodesAfter} nodes`);

  await cdp.send('Emulation.setCPUThrottlingRate', { rate: 4 });
  await runScan(page, () => uploadSyntheticImage(page));
  const t0 = Date.now();
  const status = await runScan(page, () => uploadSyntheticImage(page));
  const wall = Date.now() - t0;
  const reported = await page.locator('text=LAST SCAN').first().evaluate((el) => el.textContent);
  record(name, 'a scan still completes on a 4x slower CPU (low-end phone proxy)', status !== 'classifier-error' && wall < 30_000, `${wall} ms wall; ${reported.replace(/\s+/g, ' ').trim()}`);
  await context.close();
}

async function scenarioUnsupported(browser) {
  const name = 'unsupported browser (no WebAssembly)';
  const { context, page } = await newPage(browser, { initScript: 'delete window.WebAssembly;' });
  await page.goto(URL_BASE, { waitUntil: 'load' });
  const panel = page.locator('[data-testid="unsupported-browser"]');
  await panel.waitFor({ timeout: 15_000 });
  record(name, 'explains what is missing instead of a blank page', /WebAssembly/.test(await panel.innerText()), (await panel.innerText()).replace(/\s+/g, ' ').slice(0, 120));
  await context.close();
}

async function scenarioMobile(browser) {
  const name = 'mobile viewport 375x812';
  const { context, page, consoleErrors } = await newPage(browser, {
    contextOptions: { viewport: { width: 375, height: 812 }, isMobile: true, hasTouch: true, deviceScaleFactor: 2 },
  });
  await context.grantPermissions(['camera'], { origin: URL_BASE });
  await page.goto(URL_BASE, { waitUntil: 'load' });
  await waitReady(page);
  const overflow = await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth);
  record(name, 'no horizontal overflow', overflow <= 1, `overflow=${overflow}px`);
  await openInputSource(page);
  await page.getByText('Soiled HDPE Jug').click();
  await page.waitForSelector('[data-testid="result-panel"]', { timeout: 10_000 });
  const overflowAfter = await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth);
  record(name, 'no horizontal overflow with a result shown', overflowAfter <= 1, `overflow=${overflowAfter}px`);
  record(name, 'no console errors', consoleErrors.length === 0, consoleErrors.slice(0, 3).join(' | '));
  record(name, 'no CSP violations', (await cspViolations(page)).length === 0, (await cspViolations(page)).join(' | '));
  await page.screenshot({ path: join(outDir, 'mobile.png'), fullPage: true });
  await context.close();
}

const server = await startPreview();
let exitCode = 0;
try {
  const browser = await chromium.launch({ args: FAKE_CAMERA });
  await scenarioHappyPath(browser);
  await scenarioCamera(browser, {
    name: 'camera permission denied',
    kind: 'permission-denied',
    initScript: `navigator.mediaDevices.getUserMedia = () => Promise.reject(new DOMException('Permission denied', 'NotAllowedError'));`,
  });
  await scenarioCamera(browser, {
    name: 'no camera present',
    kind: 'no-camera',
    initScript: `navigator.mediaDevices.getUserMedia = () => Promise.reject(new DOMException('Requested device not found', 'NotFoundError'));`,
  });
  await scenarioCamera(browser, {
    name: 'camera API missing',
    kind: 'unsupported',
    initScript: `Object.defineProperty(navigator, 'mediaDevices', { value: undefined, configurable: true });`,
  });
  await scenarioModelDownload(browser);
  await scenarioOffline(browser);
  await scenarioUnsupported(browser);
  await scenarioMobile(browser);
  await scenarioStress(browser);
  await browser.close();
  await scenarioNoWebGL();
} catch (err) {
  console.error('E2E run crashed:', err);
  record('harness', 'completed without crashing', false, String(err?.message ?? err).split('\n')[0]);
} finally {
  server.kill();
  console.table(results);
  console.log(failures === 0 ? `\nALL ${results.length} CHECKS PASSED` : `\n${failures} OF ${results.length} CHECKS FAILED`);
  exitCode = failures === 0 ? 0 : 1;
}
process.exit(exitCode);
