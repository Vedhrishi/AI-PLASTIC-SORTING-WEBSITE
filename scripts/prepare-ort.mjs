// Copies the exact onnxruntime-web WASM runtime that package-lock.json pins
// into public/ort/, so the app serves it from its own origin instead of a
// third-party CDN (no unpinned remote code, no CSP/SRI exceptions, works
// offline once cached). Runs automatically before dev/build/preview.
import { copyFileSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = join(dirname(fileURLToPath(import.meta.url)), '..');
const pkgDir = join(root, 'node_modules', 'onnxruntime-web');
const pkgPath = join(pkgDir, 'package.json');
const distDir = join(pkgDir, 'dist');
const outDir = join(root, 'public', 'ort');
const FILES = ['ort-wasm-simd-threaded.wasm', 'ort-wasm-simd-threaded.mjs'];

mkdirSync(outDir, { recursive: true });
for (const file of FILES) copyFileSync(join(distDir, file), join(outDir, file));

const { version } = JSON.parse(readFileSync(pkgPath, 'utf8'));
writeFileSync(join(outDir, 'VERSION'), `onnxruntime-web ${version}\n`);
console.log(`prepare-ort: copied ${FILES.join(', ')} (onnxruntime-web ${version}) -> public/ort/`);
