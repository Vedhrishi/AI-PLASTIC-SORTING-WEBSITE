// Plastic Decoder — automated accuracy evaluation suite.
//
// Launches the local Vite dev server, drives the app in headless Chromium
// (with a fake camera device so the app's camera-gated "ready" state still
// resolves in a headless CI environment), then uploads every image in
// ./test-images/ through the app's own file-upload UI — exercising the
// exact same Stage 1-4 pipeline (YOLO detection -> snapshot -> resin +
// contamination ResNet18 classification) a real user hits.
//
// Usage: npm run test:eval

import { chromium } from '@playwright/test';
import { spawn } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const APP_URL = 'http://localhost:5173';
const TEST_IMAGES_DIR = path.join(__dirname, 'test-images');
const RESULT_TIMEOUT_MS = 20000;
const SERVER_READY_TIMEOUT_MS = 30000;
const APP_READY_TIMEOUT_MS = 30000;

const IMAGE_EXT_RE = /\.(png|jpe?g|webp|gif|bmp)$/i;

function log(msg) {
  console.log(msg);
}

async function waitForServer(url, timeoutMs) {
  const start = Date.now();
  while (Date.now() - start < timeoutMs) {
    try {
      const res = await fetch(url);
      if (res.ok || res.status === 404) return;
    } catch {
      // not up yet
    }
    await new Promise((r) => setTimeout(r, 300));
  }
  throw new Error(`Dev server did not become ready at ${url} within ${timeoutMs}ms`);
}

async function startDevServer() {
  log('Starting Vite dev server...');
  const proc = spawn('npm', ['run', 'dev', '--', '--port', '5173', '--strictPort'], {
    cwd: __dirname,
    shell: true,
  });

  let output = '';
  proc.stdout?.on('data', (d) => {
    output += d.toString();
  });
  proc.stderr?.on('data', (d) => {
    output += d.toString();
  });

  try {
    await waitForServer(APP_URL, SERVER_READY_TIMEOUT_MS);
  } catch (err) {
    proc.kill();
    throw new Error(`${err.message}\n--- dev server output ---\n${output}`);
  }

  log('Dev server is up.\n');
  return proc;
}

async function evaluateImage(page, filePath) {
  const filename = path.basename(filePath);
  const startTime = Date.now();

  await page.setInputFiles('input[type="file"]', filePath);

  try {
    // "Analyzing" is the status while Stage 3/4 are running on the captured
    // snapshot; wait for it to leave that state (either 'result' or
    // 'classifier-error' or 'no-item' if Stage 2 never found the object).
    await page.waitForFunction(
      () => {
        const badge = document.querySelector('[data-testid="status-badge"]');
        const s = badge?.getAttribute('data-status');
        return s && s !== 'scanning' && s !== 'locking' && s !== 'analyzing';
      },
      { timeout: RESULT_TIMEOUT_MS }
    );

    const inferenceMs = Date.now() - startTime;
    const finalStatus = await page.getAttribute('[data-testid="status-badge"]', 'data-status');

    if (finalStatus !== 'result') {
      const noteByStatus = {
        'classifier-error': 'Classifier error',
        veto: 'Veto (person overlap detected)',
        'no-item': 'No plastic detected',
      };
      return {
        filename,
        resin: '—',
        contamination: '—',
        inferenceMs,
        note: noteByStatus[finalStatus] ?? `Unexpected status: ${finalStatus}`,
      };
    }

    const resinEl = await page.$('[data-testid="result-resin"]');
    const contaminationEl = await page.$('[data-testid="result-contamination"]');
    const resinConfEl = await page.$('[data-testid="result-resin-confidence"]');
    const contamConfEl = await page.$('[data-testid="result-contamination-confidence"]');

    const resin = (await resinEl?.textContent())?.trim() ?? '—';
    const contamination = (await contaminationEl?.textContent())?.trim() ?? '—';
    const resinConf = (await resinConfEl?.textContent())?.trim() ?? '';
    const contamConf = (await contamConfEl?.textContent())?.trim() ?? '';

    return {
      filename,
      resin: resinConf ? `${resin} (${resinConf})` : resin,
      contamination: contamConf ? `${contamination} (${contamConf})` : contamination,
      inferenceMs,
      note: '',
    };
  } catch (err) {
    return {
      filename,
      resin: '—',
      contamination: '—',
      inferenceMs: Date.now() - startTime,
      note: `Timed out (${err.message})`,
    };
  }
}

async function main() {
  if (!fs.existsSync(TEST_IMAGES_DIR)) {
    console.error(
      `\nNo ./test-images/ directory found.\n` +
        `Create it and drop in photos of real (ideally deformed/dirty/crushed) plastic items, then rerun "npm run test:eval".\n`
    );
    process.exit(1);
  }

  const files = fs
    .readdirSync(TEST_IMAGES_DIR)
    .filter((f) => IMAGE_EXT_RE.test(f))
    .map((f) => path.join(TEST_IMAGES_DIR, f))
    .sort();

  if (files.length === 0) {
    console.error(
      `\n./test-images/ is empty.\n` +
        `Add some .jpg/.png/.webp photos of plastic items and rerun "npm run test:eval".\n`
    );
    process.exit(1);
  }

  console.log('\n▲ Plastic Decoder — Evaluation Suite');
  console.log(`Found ${files.length} test image(s).\n`);

  let devServer;
  let browser;

  try {
    devServer = await startDevServer();

    browser = await chromium.launch({
      args: ['--use-fake-device-for-media-stream', '--use-fake-ui-for-media-stream'],
    });
    const context = await browser.newContext({ viewport: { width: 1280, height: 900 } });
    await context.grantPermissions(['camera'], { origin: APP_URL });
    const page = await context.newPage();

    log('Loading app...');
    await page.goto(APP_URL, { waitUntil: 'networkidle', timeout: APP_READY_TIMEOUT_MS });

    // Wait for models + (fake) camera to finish loading before the first
    // upload — the app gates its upload pipeline on loadState === 'ready'.
    await page.waitForFunction(
      () => document.querySelector('[data-testid="app-root"]')?.getAttribute('data-load-state') === 'ready',
      { timeout: APP_READY_TIMEOUT_MS }
    );
    log('App ready. Running evaluation...\n');

    const results = [];
    for (const filePath of files) {
      const filename = path.basename(filePath);
      process.stdout.write(`  Evaluating ${filename}... `);
      const result = await evaluateImage(page, filePath);
      results.push(result);
      console.log(result.note ? `${result.note} (${result.inferenceMs}ms)` : `${result.resin} / ${result.contamination} (${result.inferenceMs}ms)`);
    }

    console.log('\n=== Evaluation Results ===\n');
    console.table(
      results.map((r) => ({
        Filename: r.filename,
        'Predicted Resin': r.resin,
        'Predicted Contamination': r.contamination,
        'Inference Time (ms)': r.inferenceMs,
        Note: r.note || '—',
      }))
    );

    const classified = results.filter((r) => !r.note).length;
    const avgMs = Math.round(results.reduce((sum, r) => sum + r.inferenceMs, 0) / results.length);
    console.log(`Summary: ${classified}/${results.length} images classified. Average time per image: ${avgMs}ms.\n`);
  } finally {
    if (browser) await browser.close();
    if (devServer) devServer.kill();
  }
}

main().catch((err) => {
  console.error('\nEvaluation script failed:', err);
  process.exit(1);
});
