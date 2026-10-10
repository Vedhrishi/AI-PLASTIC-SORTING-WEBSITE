# Plastic Decoder

Point a camera at a plastic item (or upload a photo) and get its likely **resin type** (PET, HDPE, PP, PS) with a confidence
score, plus **disposal guidance** for the Indian Plastic Waste Management context, shaped by a contamination level you choose.

**Live demo:** https://plastic-decoder-project.vercel.app

Everything runs **in your browser**. Photos are never uploaded: see [Privacy](#privacy) and [SECURITY.md](SECURITY.md).

## How it works

1. **Person / hand veto** — COCO-SSD (TF.js) looks for people and bottles/cups. A plastic box that overlaps a person is blocked.
2. **Plastic detection** — a YOLOv8 ONNX model (`best.onnx`, 640x640, one class) finds the item. Confidence gate 0.25, NMS IoU 0.45.
3. **Resin classification** — a ResNet-18 ONNX model (`resin-resnet18.onnx`, 224x224, 4 classes) labels the crop as PET / HDPE / PP / PS.
4. **Disposal rules** — a lookup table keyed by resin x contamination (`src/data/disposalRules.js`). Contamination is a manual
   three-way control (Clean / Moderate / Heavy), not a model prediction.

Scans are **snapshot based** (press *Take Photo* or upload an image), not a continuous video loop.

### Architecture

| Piece | Where | Notes |
| --- | --- | --- |
| UI | `src/App.jsx`, React 19, Tailwind 4, Framer Motion | single-screen dashboard on large screens |
| ONNX inference | `src/workers/inference.worker.js` | onnxruntime-web WASM build, up to 4 threads when cross-origin isolated |
| Person detector | `src/workers/person.worker.js` | TF.js: WebGL on an OffscreenCanvas, CPU fallback |
| Worker clients | `src/lib/inferenceClient.js`, `personClient.js`, `workerRpc.js` | session-like wrappers, transferable buffers |
| Pre/post-processing | `src/lib/yolo.js`, `classify.js`, `geometry.js` | letterbox, ImageNet normalisation, NMS, softmax |
| Failure messages | `src/lib/env.js`, `modelLoader.js`, `tfErrors.js` | camera, download, offline, unsupported browser |
| Models | `public/models/` | `best.onnx` 44.7 MB, `resin-resnet18.onnx` 11.2 MB, COCO-SSD 18.6 MB; SHA-256 in `MANIFEST.json` |
| ONNX Runtime WASM | `public/ort/` (generated) | copied from `node_modules` by `scripts/prepare-ort.mjs` before dev/build/preview |

Nothing is loaded from a CDN at runtime: runtime, models and fonts are all served from the app's own origin.

## Requirements

- **Node.js 22.12 or newer** (see `.nvmrc`). Check with `node -v`.
- A current Chrome, Edge, Firefox or Safari with WebAssembly. A camera is optional (upload and demo presets work without one).
- Camera access needs HTTPS (or `localhost`).

## Run it

```bash
npm ci
npm run dev        # http://localhost:5173
```

Production build served with the exact production headers (CSP etc.):

```bash
npm run build
npm run preview    # http://localhost:4173
```

## Quality checks

```bash
npm run check      # typecheck + lint + tests + build
npm run test:e2e   # real-browser checks of the production build (needs: npx playwright install chromium)
npm run audit:ci   # npm audit, fails on high/critical
```

- **Unit tests** (`npm test`, 87 tests): preprocessing against hand-computed values, YOLO decoding, resin-class mapping, the
  disposal matrix, loaders and worker clients, error messages, security headers.
- **Golden model tests**: run the shipped `.onnx` files with onnxruntime-node and compare against recorded reference outputs and
  SHA-256 checksums. After intentionally replacing a model, run `npm run models:manifest` and update the reference values.
- **E2E** (`npm run test:e2e`, 48 checks): happy path, camera denied / missing / unsupported, model download failure and retry,
  offline and recovery, WebGL unavailable, unsupported browser, mobile viewport, 25-scan leak check, 4x CPU throttle, receipt
  export, enforced CSP, and a check that no request leaves the origin.
- `npm run test:eval` scores your own photos placed in `test-images/` (see that folder's README).

## Deploy

The repository ships a `vercel.json` with the security headers and cache rules. Deploy with the Vercel CLI
(`npx vercel --prod`) or connect the repository in Vercel. On other hosts, send the same headers listed in `vercel.json`.
The build command is `npm run build`; the output directory is `dist`.

## Performance (measured, headless Chromium on a development PC)

| Measure | Result |
| --- | --- |
| Download size of models | about 75 MB (44.7 + 11.2 + 18.6 MB) plus the 14 MB ONNX Runtime WASM, cached by the browser |
| Time until ready (localhost) | about 6 s |
| Scan latency (4 WASM threads) | about 0.9-1.1 s |
| Scan latency with 4x CPU throttling | about 1.1 s reported (the heavy work runs in workers) |
| Longest main-thread task during load and scans | about 0.2 s (inference runs in workers) |
| Main-thread JS heap after 25 scans | flat (3.3 MB to 4.0 MB) |

These numbers come from a software-rendered headless browser; real phones will differ. They were not measured on physical low-end devices.

## Privacy

- Camera frames and uploaded images are processed in the browser only. The source contains no code that sends them anywhere, the
  Content-Security-Policy blocks any other origin, and the end-to-end test fails if a request leaves the origin or carries a body.
- The downloadable audit receipt is generated locally and includes the analysed photo.
- Fonts, models and the ONNX runtime are self-hosted, so visiting the page does not contact third parties.

## Accuracy limits (please read)

- The classifier only knows **four resins**: PET (1), HDPE (2), PP (5), PS (6). PVC, LDPE, "other" and multi-layer packaging will be
  forced into one of the four, often with misleadingly high confidence. There is no "unknown" class.
- Accuracy depends on lighting, angle, crushing and labels. Below 80% confidence the app shows a warning, but the confidence
  value is a softmax score, not a calibrated probability.
- **No accuracy figure is published here**: this repository has no labelled test set. The golden tests prove the models are
  intact and the pipeline is consistent, not that the predictions are correct. Measure it on your own photos with `npm run test:eval`.
- Contamination is entered by the user and is not detected. Disposal text is general guidance for the Indian PWM context, not
  legal or municipal advice; local collection rules vary.
- Visual resin identification is no substitute for the resin identification code stamped on the item.
- Demo presets are simulated results, not detections.

## Troubleshooting

| You see | What to do |
| --- | --- |
| "Camera permission denied" | Allow the camera in the address-bar lock menu, press *Retry camera*, or upload an image. |
| "No camera found" / "Camera is unavailable" | Plug in or free the camera (close other apps), press *Retry camera*, or upload an image. |
| "You're offline" / "Model download failed" | Reconnect and press *Retry* (it also retries automatically when the connection returns). |
| Slow or glitchy person detection | Open the app with `?tfBackend=cpu` to skip WebGL. |

## Credits

COCO-SSD (`ssdlite_mobilenet_v2`) weights come from the TensorFlow.js models project (Apache-2.0) and are committed unchanged under
`public/models/coco-ssd/`. Inter and JetBrains Mono fonts come from the Fontsource packages (OFL-1.1).
