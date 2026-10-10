import { readFileSync } from 'node:fs'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'
import { defaultClientConditions, defineConfig } from 'vite'

// `vite preview` serves the production build with the exact security headers
// Vercel will send (vercel.json is the single source of truth), so what is
// tested locally is what ships.
const vercelConfig = JSON.parse(readFileSync(new URL('./vercel.json', import.meta.url), 'utf8'))
const siteHeaders = Object.fromEntries(
  vercelConfig.headers.find((rule) => rule.source === '/(.*)').headers.map(({ key, value }) => [key, value]),
)

// https://vite.dev/config/
export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: {
    // Use onnxruntime-web's "extern wasm" build: the runtime files come from
    // our own /ort/ directory (scripts/prepare-ort.mjs) instead of being
    // bundled a second time into dist/assets (a 14 MB duplicate).
    conditions: ['onnxruntime-web-use-extern-wasm', ...defaultClientConditions],
  },
  optimizeDeps: {
    exclude: ['onnxruntime-web'],
  },
  server: {
    // Cross-origin isolation (SharedArrayBuffer -> multi-threaded WASM) in dev.
    headers: {
      'Cross-Origin-Opener-Policy': 'same-origin',
      'Cross-Origin-Embedder-Policy': 'require-corp',
    },
  },
  worker: { format: 'es' },
  preview: {
    headers: siteHeaders,
  },
})
