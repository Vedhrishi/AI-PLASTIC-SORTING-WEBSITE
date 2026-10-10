// Records size + SHA-256 of every shipped model file in public/models/MANIFEST.json.
// The test suite recomputes these, so a swapped, truncated or corrupted model
// fails CI instead of silently changing predictions. Run after intentionally
// replacing a model:  npm run models:manifest
import { createHash } from 'node:crypto';
import { readdirSync, readFileSync, statSync, writeFileSync } from 'node:fs';
import { join, relative, sep } from 'node:path';
import { fileURLToPath } from 'node:url';

const modelsDir = join(fileURLToPath(new URL('..', import.meta.url)), 'public', 'models');

function walk(dir) {
  return readdirSync(dir, { withFileTypes: true }).flatMap((entry) => {
    const full = join(dir, entry.name);
    return entry.isDirectory() ? walk(full) : [full];
  });
}

const files = walk(modelsDir)
  .filter((file) => !file.endsWith('MANIFEST.json'))
  .map((file) => relative(modelsDir, file).split(sep).join('/'))
  .sort();

const manifest = {};
for (const name of files) {
  const bytes = readFileSync(join(modelsDir, name));
  manifest[name] = { bytes: statSync(join(modelsDir, name)).size, sha256: createHash('sha256').update(bytes).digest('hex') };
}
writeFileSync(join(modelsDir, 'MANIFEST.json'), `${JSON.stringify(manifest, null, 2)}\n`);
console.log(`models manifest: ${files.length} files`);
