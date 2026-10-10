# Security

Plastic Decoder is a static web app: there is no server-side code, no database, no accounts and no API keys.
This document lists what is protected, how that is checked, and the risks that remain.

## Reporting a vulnerability

Open a private security advisory on the GitHub repository (Security > Advisories > Report a vulnerability),
or contact the repository owner directly. Please do not file public issues for vulnerabilities.

## What is protected

| Area | Protection | How it is checked |
| --- | --- | --- |
| Camera and image privacy | Frames and uploaded images are analysed inside the browser (workers). The source contains no `fetch`/XHR/beacon/WebSocket/analytics code that sends data anywhere, and every request the page makes is a body-less `GET` to its own origin. | `npm run test:e2e` records every request during load, camera scans, uploads and receipt export and fails on any cross-origin request, non-GET request or request body. |
| Third-party code | No CDN or remote script, model, font or stylesheet is loaded at runtime. The ONNX Runtime WASM files, COCO-SSD weights and fonts are served from this origin, pinned by `package-lock.json` or committed. | CSP (below) blocks any other origin; `public/models/MANIFEST.json` pins the SHA-256 of every model and a test recomputes it. |
| Content Security Policy | Enforced: `default-src 'self'`, `script-src 'self' 'wasm-unsafe-eval'` (no inline script, no `eval`), `connect-src 'self' blob:`, `object-src 'none'`, `frame-ancestors 'none'`, one SHA-256 hash for the single inline style block html2canvas injects. | `src/security.test.js` asserts the policy stays closed; `npm run test:e2e` runs the production build under the enforced policy and fails on any violation or console error. |
| Transport and headers | `Strict-Transport-Security`, `X-Content-Type-Options: nosniff`, `Referrer-Policy: no-referrer`, `X-Frame-Options: DENY`, `Permissions-Policy` (camera for this origin only; microphone, geolocation, payment, USB off), `Cross-Origin-Opener-Policy`, `Cross-Origin-Embedder-Policy`, `Cross-Origin-Resource-Policy`. Defined once in `vercel.json` and also applied by `vite preview`. | Unit test on `vercel.json`; e2e checks the headers on a real response. |
| Supply chain | `npm audit --audit-level=high` runs in CI on every push and pull request (currently 0 vulnerabilities at any severity). Dependabot opens weekly PRs for npm packages and GitHub Actions. CI installs with `npm ci` from the lockfile. | `.github/workflows/ci.yml`, `.github/dependabot.yml`. |
| Secrets | The app needs none. `.env*` files are git-ignored. The full git history was scanned for private keys, cloud/GitHub/Slack/Stripe-style tokens and key/secret/password assignments: nothing found. | One-off scan; re-run before making the repository public. |
| Crash safety | A React error boundary and explicit messages for camera, model, offline and unsupported-browser failures mean users never see a blank page. | e2e scenarios for each failure. |

## Known limitations and remaining risks

- **Hosting headers apply on Vercel.** `vercel.json` is only honoured by Vercel. If the app is moved to another host, copy the
  same headers there (they are plain name/value pairs).
- **`'wasm-unsafe-eval'` is required** by WebAssembly (ONNX Runtime). It does not allow JavaScript `eval`.
- **One inline-style hash.** html2canvas-pro injects a fixed `<style>` block when exporting a receipt; its SHA-256 is allowed in
  `style-src`. A html2canvas-pro upgrade can change the hash: the e2e suite will fail with the new value to copy.
- **Cross-origin isolation** (COOP/COEP) is enabled so WASM can use several threads. Embedding third-party resources
  (images, iframes) in this app would require them to opt in with CORP/CORS headers.
- **Model supply chain.** `best.onnx` and `resin-resnet18.onnx` were trained outside this repository; their provenance and
  training data are not documented here. Integrity (not authenticity) is enforced through the checksum manifest.
- **Model inputs are untrusted.** Uploaded images are decoded by the browser and downscaled; a 25 MB size limit applies.
  A maliciously crafted image can at worst make a scan fail with an error message.
- **Receipts contain the analysed photo.** The downloaded PNG includes the captured frame; users should treat it as they would the photo.
- **No automated browser tests in CI.** `npm run test:e2e` needs Chromium and is run locally before releases.
- **The repository is a GitHub template**; derivative projects inherit this configuration and must keep their own headers and dependencies current.
