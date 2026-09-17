import { useEffect, useRef, useState, useCallback } from 'react';
import * as cocoSsd from '@tensorflow-models/coco-ssd';
import * as tf from '@tensorflow/tfjs';
import { AnimatePresence, motion } from 'framer-motion';
// html2canvas-pro, not html2canvas: Tailwind v4's default palette emits
// oklch()/oklab() colors, which stock html2canvas (last released for the
// pre-CSS-Color-4 web) cannot parse — every capture throws
// "unsupported color function". This fork is API-compatible and adds that
// support, which is the actual fix rather than a workaround.
import html2canvas from 'html2canvas-pro';
import {
  Camera,
  AlertTriangle,
  Recycle,
  Loader2,
  X,
  ShieldAlert,
  Volume2,
  VolumeX,
  Upload,
  Video,
  Sparkles,
  Download,
  Activity,
  RotateCcw,
  ChevronDown,
} from 'lucide-react';

import { loadSession } from './lib/onnxSetup';
import { detectPlastic } from './lib/yolo';
import { classifyResin, cropAndResize } from './lib/classify';
import { overlapFraction, computeCoverProjection, projectBox } from './lib/geometry';
import { DISPOSAL_RULES, getDisposalRule } from './data/disposalRules';
import { RESIN_INFO } from './data/resinInfo';
import * as audio from './lib/audioManager';

const VIDEO_WIDTH = 640;
const VIDEO_HEIGHT = 480;
const PERSON_VETO_IOU = 0.2;
const YOLO_FALLBACK_MIN_SCORE = 0.3;
const CONFIDENCE_THRESHOLD = 0.8;
// Stage 2 (YOLOv8) score gate. Kept low so heavily crushed/deformed items —
// which produce weaker, less bottle-shaped activations — still clear the
// bar and reach Stage 3, where ResNet's texture-based classification is
// more robust to deformation than YOLO's shape-based detection.
const YOLO_CONFIDENCE_THRESHOLD = 0.25;
const YOLO_NMS_IOU_THRESHOLD = 0.45;

const RESIN_ORDER = ['PET', 'HDPE', 'PP', 'PS'];
const CONTAMINATION_ORDER = ['Clean/Light Soiling', 'Moderate Contamination', 'Heavy Contamination'];

const DEMO_PRESETS = [
  { key: 'pet-clean', label: 'Clean PET Bottle', resin: 'PET', code: 1, resinConf: 0.974, contamination: 'Clean/Light Soiling', contamConf: 0.951 },
  { key: 'hdpe-soiled', label: 'Soiled HDPE Jug', resin: 'HDPE', code: 2, resinConf: 0.932, contamination: 'Moderate Contamination', contamConf: 0.881 },
  { key: 'pp-dirty', label: 'Dirty PP Cup', resin: 'PP', code: 5, resinConf: 0.908, contamination: 'Heavy Contamination', contamConf: 0.864 },
];

// Freeze-frames a video/image source into a w x h canvas using the same
// crop/scale math as CSS `object-fit: cover`, so the frozen frame the user
// sees is exactly what the live viewfinder was showing. Detection then runs
// on this canonical canvas, so box coordinates need no further reprojection
// beyond the final live-container-size scale in drawOverlay.
function coverFitCanvas(source, w, h) {
  const canvas = document.createElement('canvas');
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext('2d');
  const sw = source.videoWidth || source.naturalWidth || source.width;
  const sh = source.videoHeight || source.naturalHeight || source.height;
  const scale = Math.max(w / sw, h / sh);
  const drawW = sw * scale;
  const drawH = sh * scale;
  const dx = (w - drawW) / 2;
  const dy = (h - drawH) / 2;
  ctx.drawImage(source, dx, dy, drawW, drawH);
  return canvas;
}

// appState is the coarse three-way view the UI switches on; `status` (below)
// carries the finer-grained outcome once appState leaves 'live'.
function deriveAppState(status) {
  if (status === 'idle') return 'live';
  if (status === 'analyzing') return 'analyzing';
  return 'results';
}

export default function App() {
  const videoRef = useRef(null);
  const overlayRef = useRef(null);
  const frozenCanvasRef = useRef(null);
  const containerRef = useRef(null);
  const modelsRef = useRef(null);
  const streamRef = useRef(null);
  const isProcessingRef = useRef(false);
  const rowRefs = useRef({});
  const tableScrollRef = useRef(null);

  const [loadState, setLoadState] = useState('loading'); // loading | ready | error
  const [loadError, setLoadError] = useState(null);
  const [status, setStatus] = useState('idle'); // idle | analyzing | veto | result | no-item | classifier-error
  const [result, setResult] = useState(null);
  const [drawerRule, setDrawerRule] = useState(null);
  const [muted, setMuted] = useState(false);
  const [sourceMode, setSourceMode] = useState('camera'); // camera | upload | preset
  const [isDragActive, setIsDragActive] = useState(false);
  const [diagnostics, setDiagnostics] = useState({ latencyMs: 0, engine: 'ONNX Runtime Web · WebGL (GPU) → WASM' });
  const [landedRowKey, setLandedRowKey] = useState(null);
  const [classifierErrorMessage, setClassifierErrorMessage] = useState(null);
  // Contamination is now a manual, user-controlled call — not a model
  // output. Defaults to the cleanest state whenever a fresh resin result
  // lands; the user can then move the 3-way slider freely.
  const [manualContamination, setManualContamination] = useState('Clean/Light Soiling');
  const [videoDevices, setVideoDevices] = useState([]);
  const [selectedDeviceId, setSelectedDeviceId] = useState(null);

  const appState = deriveAppState(status);
  // Shared by both the visible ResultCard and the hidden export template
  // below, so they never drift out of sync with the slider.
  const currentRule = result ? getDisposalRule(result.resin.label, manualContamination) : null;

  useEffect(() => {
    audio.setMuted(muted);
  }, [muted]);

  // Starts (or switches to) a camera. Pass a deviceId to target a specific
  // device (from the dropdown); omit it for the initial best-guess pick
  // (rear/environment camera on phones, whatever default on laptops).
  // Device *labels* are only populated by enumerateDevices() after
  // permission has been granted at least once, so we always re-enumerate
  // right after a successful getUserMedia call.
  const startCamera = useCallback(async (deviceId) => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
    }

    const stream = await navigator.mediaDevices.getUserMedia({
      video: deviceId
        ? { deviceId: { exact: deviceId }, width: { ideal: VIDEO_WIDTH }, height: { ideal: VIDEO_HEIGHT } }
        : { width: { ideal: VIDEO_WIDTH }, height: { ideal: VIDEO_HEIGHT }, facingMode: { ideal: 'environment' } },
      audio: false,
    });

    streamRef.current = stream;
    if (videoRef.current) {
      videoRef.current.srcObject = stream;
      await videoRef.current.play();
    }

    try {
      const devices = await navigator.mediaDevices.enumerateDevices();
      const videoInputs = devices.filter((d) => d.kind === 'videoinput');
      setVideoDevices(videoInputs);
      const activeId = stream.getVideoTracks()[0]?.getSettings()?.deviceId ?? deviceId ?? videoInputs[0]?.deviceId ?? null;
      setSelectedDeviceId(activeId);
    } catch (err) {
      console.warn('enumerateDevices failed (camera list unavailable)', err);
    }
  }, []);

  const switchCamera = useCallback(
    async (deviceId) => {
      try {
        await startCamera(deviceId);
      } catch (err) {
        console.error('Failed to switch camera', err);
      }
    },
    [startCamera]
  );

  // ---- load webcam + models in parallel (camera issues shouldn't block model loading) ----
  useEffect(() => {
    let cancelled = false;

    async function initModels() {
      try {
        await tf.setBackend('webgl');
      } catch (err) {
        console.warn('tfjs WebGL backend unavailable, falling back to default backend', err);
      }
      await tf.ready();

      const [personModel, plasticSession, resinSession] = await Promise.all([
        cocoSsd.load({ base: 'lite_mobilenet_v2' }),
        // YOLO is the heavy compute stage (640x640 input) — worth the GPU.
        loadSession('/models/best.onnx'),
        // ResNet18 classifier is small/fast on WASM alone; pinning it to
        // WASM avoids WebGL op-coverage/precision edge cases silently
        // corrupting classification output on some mobile GPUs.
        // Stage 4 (contamination-resnet18.onnx) is no longer loaded at all —
        // contamination is now a manual user selection (see
        // manualContamination state), not a model inference, so this heavy
        // model download/init is dropped entirely for faster load and less
        // memory/compute pressure on mobile.
        loadSession('/models/resin-resnet18.onnx', { executionProviders: ['wasm'] }),
      ]);
      if (cancelled) return;
      modelsRef.current = { personModel, plasticSession, resinSession };
    }

    async function init() {
      try {
        await Promise.all([startCamera(), initModels()]);
        if (cancelled) return;
        setLoadState('ready');
      } catch (err) {
        console.error(err);
        if (!cancelled) {
          setLoadError(err.message || String(err));
          setLoadState('error');
        }
      }
    }

    init();

    return () => {
      cancelled = true;
      if (streamRef.current) streamRef.current.getTracks().forEach((t) => t.stop());
    };
  }, [startCamera]);

  // Draws detection boxes onto the overlay canvas, scaled from the
  // VIDEO_WIDTH x VIDEO_HEIGHT canonical frame (every captured/uploaded
  // source is normalized to this box via coverFitCanvas before detection
  // ever runs) into the container's live on-screen size.
  const drawOverlay = useCallback((plasticBox, personBoxes, vetoed, extraLabel) => {
    const canvas = overlayRef.current;
    const container = containerRef.current;
    if (!canvas || !container) return;

    const clientW = container.clientWidth;
    const clientH = container.clientHeight;
    if (clientW === 0 || clientH === 0) return;

    if (canvas.width !== clientW || canvas.height !== clientH) {
      canvas.width = clientW;
      canvas.height = clientH;
    }

    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    ctx.lineWidth = 2;
    ctx.font = '12px ui-monospace, monospace';

    const projection = computeCoverProjection(VIDEO_WIDTH, VIDEO_HEIGHT, clientW, clientH);
    const toScreen = (box) => projectBox(box, projection);

    for (const p of personBoxes) {
      const [x1, y1, x2, y2] = toScreen(p);
      ctx.strokeStyle = vetoed ? '#f87171' : 'rgba(248,113,113,0.5)';
      ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
      ctx.fillStyle = ctx.strokeStyle;
      ctx.fillText('PERSON', x1 + 4, y1 + 16);
    }

    if (plasticBox) {
      const [x1, y1, x2, y2] = toScreen(plasticBox);
      const w = x2 - x1;
      const h = y2 - y1;
      const color = vetoed ? '#f87171' : '#34d399';
      ctx.strokeStyle = color;
      ctx.shadowColor = color;
      ctx.shadowBlur = 8;
      ctx.strokeRect(x1, y1, w, h);

      const r = Math.min(18, w / 3, h / 3);
      ctx.lineWidth = 3;
      const corners = [
        [x1, y1, r, r],
        [x2, y1, -r, r],
        [x1, y2, r, -r],
        [x2, y2, -r, -r],
      ];
      for (const [cx, cy, dx, dy] of corners) {
        ctx.beginPath();
        ctx.moveTo(cx, cy + dy);
        ctx.lineTo(cx, cy);
        ctx.lineTo(cx + dx, cy);
        ctx.stroke();
      }
      ctx.shadowBlur = 0;

      ctx.fillStyle = color;
      const label = extraLabel ?? (vetoed ? 'BLOCKED' : 'PLASTIC');
      const ty = y1 - 8 < 12 ? y1 + 18 : y1 - 8;
      ctx.fillText(label, x1 + 2, ty);
    }
  }, []);

  const clearOverlay = useCallback(() => {
    const canvas = overlayRef.current;
    if (!canvas) return;
    canvas.getContext('2d').clearRect(0, 0, canvas.width, canvas.height);
  }, []);

  // The full 4-stage pipeline, run exactly once on a single already-captured
  // VIDEO_WIDTH x VIDEO_HEIGHT canvas (a camera snapshot, an uploaded image,
  // or nothing at all for the simulated presets). No continuous loop, no
  // stability timer — one click/upload, one analysis pass.
  const analyzeSnapshot = useCallback(
    async (canvas) => {
      if (isProcessingRef.current) return;
      isProcessingRef.current = true;

      const { personModel, plasticSession, resinSession } = modelsRef.current;
      const analysisStart = performance.now();

      setStatus('analyzing');
      setResult(null);
      setClassifierErrorMessage(null);
      audio.playScanBeep();

      try {
        const [detections, plasticDetections] = await Promise.all([
          personModel.detect(canvas),
          detectPlastic(plasticSession, canvas, VIDEO_WIDTH, VIDEO_HEIGHT, {
            confThreshold: YOLO_CONFIDENCE_THRESHOLD,
            iouThreshold: YOLO_NMS_IOU_THRESHOLD,
          }),
        ]);

        const personBoxes = detections
          .filter((p) => p.class === 'person')
          .map((p) => [p.bbox[0], p.bbox[1], p.bbox[0] + p.bbox[2], p.bbox[1] + p.bbox[3]]);

        let best = plasticDetections.length > 0 ? plasticDetections.reduce((a, b) => (b.score > a.score ? b : a)) : null;
        let plasticBox = best?.box ?? null;
        let usedFallback = false;

        if (!best || best.score < YOLO_FALLBACK_MIN_SCORE) {
          const fallback = detections.find((p) => (p.class === 'bottle' || p.class === 'cup') && p.score > 0.3);
          if (fallback) {
            const [x, y, w, h] = fallback.bbox;
            plasticBox = [x, y, x + w, y + h];
            usedFallback = true;
          }
        }

        // Stage 2 found nothing — abort, skip Stage 3/4 entirely.
        if (!plasticBox) {
          drawOverlay(null, personBoxes, false);
          setStatus('no-item');
          setDiagnostics((d) => ({ ...d, latencyMs: Math.round(performance.now() - analysisStart) }));
          return;
        }

        const vetoed = personBoxes.some((personBox) => overlapFraction(plasticBox, personBox) > PERSON_VETO_IOU);
        drawOverlay(plasticBox, personBoxes, vetoed, usedFallback && !vetoed ? 'PLASTIC (fallback)' : undefined);

        // Stage 1 veto — abort immediately, skip Stage 3/4.
        if (vetoed) {
          audio.playVetoAlarm();
          setStatus('veto');
          setDiagnostics((d) => ({ ...d, latencyMs: Math.round(performance.now() - analysisStart) }));
          return;
        }

        audio.playLockOn();
        const cropped = cropAndResize(canvas, plasticBox);

        let resin;
        try {
          resin = await classifyResin(resinSession, cropped);
        } catch (err) {
          const message = err?.message ?? String(err);
          console.error(`[analyzeSnapshot] Stage 3 failed: ${message}`, err);
          setStatus('classifier-error');
          setClassifierErrorMessage(message);
          setDiagnostics((d) => ({ ...d, latencyMs: Math.round(performance.now() - analysisStart) }));
          return;
        }

        const lowConfidence = resin.confidence < CONFIDENCE_THRESHOLD;

        audio.playSuccess();
        // Stage 4 is gone — contamination is a manual call, always starting
        // from the cleanest state for a freshly-identified item.
        setManualContamination('Clean/Light Soiling');
        setResult({
          resin,
          box: plasticBox,
          simulated: false,
          lowConfidence,
          // Included in the Results Card so the visual receipt export
          // (html2canvas) captures the actual analyzed frame, not just text.
          frameDataUrl: canvas.toDataURL('image/jpeg', 0.85),
        });
        setStatus('result');
        setDiagnostics((d) => ({ ...d, latencyMs: Math.round(performance.now() - analysisStart) }));
      } catch (err) {
        const message = err?.message ?? String(err);
        console.error(`[analyzeSnapshot] Stage 1/2 failed: ${message}`, err);
        setStatus('classifier-error');
        setClassifierErrorMessage(message);
      } finally {
        isProcessingRef.current = false;
      }
    },
    [drawOverlay]
  );

  // ---- manual capture: the entire replacement for the old continuous loop ----
  const handleTakePhoto = useCallback(async () => {
    if (isProcessingRef.current || !videoRef.current) return;

    const frame = coverFitCanvas(videoRef.current, VIDEO_WIDTH, VIDEO_HEIGHT);
    const frozen = frozenCanvasRef.current;
    if (frozen) {
      frozen.width = VIDEO_WIDTH;
      frozen.height = VIDEO_HEIGHT;
      frozen.getContext('2d').drawImage(frame, 0, 0);
    }
    videoRef.current.pause();

    await analyzeSnapshot(frame);
  }, [analyzeSnapshot]);

  const scanAgain = useCallback(() => {
    isProcessingRef.current = false;
    setResult(null);
    setClassifierErrorMessage(null);
    setManualContamination('Clean/Light Soiling');
    setStatus('idle');
    clearOverlay();
    if (sourceMode === 'camera') {
      videoRef.current?.play();
    }
  }, [sourceMode, clearOverlay]);

  // ---- auto-scroll + landed pulse whenever the result OR the manually
  // selected contamination level changes, so moving the slider re-highlights
  // the newly-matching disposal table row every time. ----
  useEffect(() => {
    if (status !== 'result' || !result) return;
    const key = `${result.resin.label}-${manualContamination}`;
    const el = rowRefs.current[key];
    if (el) {
      el.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
      setLandedRowKey(key);
      const timeout = setTimeout(() => setLandedRowKey(null), 1200);
      return () => clearTimeout(timeout);
    }
  }, [status, result, manualContamination]);

  const handleFiles = useCallback(
    (files) => {
      const file = files?.[0];
      if (!file || !file.type.startsWith('image/')) return;
      const img = new Image();
      img.onload = () => {
        setSourceMode('upload');
        const frame = coverFitCanvas(img, VIDEO_WIDTH, VIDEO_HEIGHT);
        const frozen = frozenCanvasRef.current;
        if (frozen) {
          frozen.width = VIDEO_WIDTH;
          frozen.height = VIDEO_HEIGHT;
          frozen.getContext('2d').drawImage(frame, 0, 0);
        }
        analyzeSnapshot(frame);
      };
      img.src = URL.createObjectURL(file);
    },
    [analyzeSnapshot]
  );

  const applyPreset = useCallback((preset) => {
    setSourceMode('preset');
    setStatus('analyzing');
    setResult(null);
    setTimeout(() => {
      audio.playLockOn();
      setTimeout(() => {
        audio.playSuccess();
        // Presets keep their own designed contamination level (that's the
        // point of "Soiled HDPE Jug" vs. "Clean PET Bottle") — the slider
        // still overrides it freely afterward, same as a real scan.
        setManualContamination(preset.contamination);
        setResult({
          resin: { label: preset.resin, code: preset.code, confidence: preset.resinConf },
          simulated: true,
          lowConfidence: preset.resinConf < CONFIDENCE_THRESHOLD,
        });
        setStatus('result');
      }, 400);
    }, 500);
  }, []);

  const returnToCamera = useCallback(() => {
    isProcessingRef.current = false;
    setSourceMode('camera');
    setStatus('idle');
    setResult(null);
    setClassifierErrorMessage(null);
    setManualContamination('Clean/Light Soiling');
    clearOverlay();
    videoRef.current?.play();
  }, [clearOverlay]);

  return (
    <div className="min-h-screen bg-[#05060a] text-gray-100 print:bg-white print:text-black" data-testid="app-root" data-load-state={loadState}>
      <header className="border-b border-slate-800/60 backdrop-blur-xl bg-slate-950/40 px-4 sm:px-6 py-3 sm:py-4 flex items-center justify-between gap-2 sm:gap-3 sticky top-0 z-30">
        <div className="flex items-center gap-2 sm:gap-3 min-w-0">
          <div className="relative shrink-0">
            <Recycle className="text-emerald-400" size={24} />
            <span className="absolute inset-0 blur-md text-emerald-400 opacity-60">
              <Recycle size={24} />
            </span>
          </div>
          <div className="min-w-0">
            <h1 className="text-base sm:text-xl font-semibold tracking-tight flex items-center gap-1.5 sm:gap-2">
              <span className="bg-gradient-to-r from-emerald-300 via-cyan-300 to-emerald-300 bg-clip-text text-transparent whitespace-nowrap">
                Plastic Decoder
              </span>
              <span className="text-[9px] sm:text-[10px] font-mono font-medium tracking-wide text-cyan-300 border border-cyan-500/30 bg-cyan-500/10 rounded-full px-1.5 sm:px-2 py-0.5 shrink-0">
                v2030
              </span>
            </h1>
            <p className="hidden sm:block text-sm text-gray-500 truncate">Cyber-HUD Molecular Decoder — in-browser resin ID &amp; disposal guidance</p>
          </div>
        </div>

        <button
          onClick={() => setMuted((m) => !m)}
          aria-label={muted ? 'Unmute' : 'Mute'}
          className="flex items-center gap-1.5 sm:gap-2 text-xs text-gray-400 hover:text-gray-200 border border-slate-700/50 rounded-full px-2.5 sm:px-3 py-1.5 backdrop-blur-sm bg-slate-900/60 transition-all duration-150 hover:scale-[1.02] active:scale-[0.98] shrink-0 whitespace-nowrap"
        >
          {muted ? <VolumeX size={14} /> : <Volume2 size={14} />}
          <span className="hidden xs:inline">{muted ? 'Muted' : 'Audio on'}</span>
        </button>
      </header>

      <main className="max-w-6xl mx-auto px-4 sm:px-6 py-4 sm:py-6 grid grid-cols-1 xl:grid-cols-[640px_1fr] xl:grid-rows-[minmax(0,1fr)] gap-6 xl:h-[calc(100vh-81px)] xl:overflow-hidden">
        <section className="space-y-3 min-w-0 xl:min-h-0 xl:overflow-hidden">
          <VideoHud
            containerRef={containerRef}
            videoRef={videoRef}
            overlayRef={overlayRef}
            frozenCanvasRef={frozenCanvasRef}
            loadState={loadState}
            loadError={loadError}
            status={status}
            appState={appState}
            sourceMode={sourceMode}
            isDragActive={isDragActive}
            onDragActive={setIsDragActive}
            onFiles={handleFiles}
            onTakePhoto={handleTakePhoto}
            onScanAgain={sourceMode === 'camera' ? scanAgain : undefined}
          />

          <StatusBadge status={status} sourceMode={sourceMode} />

          <DiagnosticsBar diagnostics={diagnostics} />

          <SourceControls
            sourceMode={sourceMode}
            onFiles={handleFiles}
            onPreset={applyPreset}
            onReturnToCamera={returnToCamera}
            videoDevices={videoDevices}
            selectedDeviceId={selectedDeviceId}
            onSwitchCamera={switchCamera}
          />
        </section>

        <section className="space-y-4 min-w-0 xl:h-full xl:min-h-0 xl:overflow-y-auto xl:pr-1">
          <div className="bg-slate-950 p-6 rounded-xl flex flex-col gap-6 w-full max-w-3xl mx-auto">
            <AnimatePresence mode="wait">
              {status === 'veto' ? (
                <VetoAlert key="veto" onScanAgain={sourceMode === 'camera' ? scanAgain : undefined} />
              ) : status === 'classifier-error' ? (
                <ClassifierErrorAlert
                  key="classifier-error"
                  message={classifierErrorMessage}
                  onScanAgain={sourceMode === 'camera' ? scanAgain : undefined}
                />
              ) : status === 'result' && result ? (
                <ResultCard
                  key="result"
                  result={result}
                  rule={currentRule}
                  contaminationLabel={manualContamination}
                  onContaminationChange={setManualContamination}
                  onScanAgain={sourceMode === 'camera' ? scanAgain : undefined}
                />
              ) : (
                <EmptyPanel key="empty" status={status} />
              )}
            </AnimatePresence>
          </div>

          <details className="group rounded-2xl border border-white/10 backdrop-blur-2xl bg-slate-900/40 shadow-[0_8px_32px_rgba(0,0,0,0.4)] w-full max-w-3xl mx-auto print:hidden">
            <summary className="flex items-center justify-between gap-2 cursor-pointer select-none px-6 py-4 list-none [&::-webkit-details-marker]:hidden">
              <span className="text-sm font-semibold text-gray-300">View Full Disposal Matrix</span>
              <ChevronDown size={16} className="text-gray-500 transition-transform duration-200 group-open:rotate-180" />
            </summary>
            <div className="px-6 pb-6">
              <DisposalRuleTable
                activeResin={result?.resin.label}
                activeContamination={result ? manualContamination : undefined}
                onSelectRow={setDrawerRule}
                rowRefs={rowRefs}
                landedRowKey={landedRowKey}
                tableScrollRef={tableScrollRef}
              />
            </div>
          </details>
        </section>
      </main>

      {/* Hidden, fixed-width (800px), plain-inline-style receipt template —
          deliberately NOT the responsive Tailwind UI above. html2canvas
          renders CSS Color 4 (oklch) and complex responsive grid/flex rules
          unreliably; this static, single-breakpoint layout sidesteps that
          entirely and is what "Export audit receipt" actually captures. */}
      {result && currentRule && (
        <div
          id="receipt-export-template"
          style={{
            position: 'absolute',
            left: '-9999px',
            top: '-9999px',
            width: '800px',
            backgroundColor: '#0f172a',
            padding: '40px',
            fontFamily: 'sans-serif',
            color: '#f8fafc',
            borderRadius: '12px',
          }}
        >
          <div style={{ fontSize: '28px', fontWeight: 'bold' }}>Plastic Decoder Audit Receipt</div>
          <div style={{ fontSize: '13px', color: '#94a3b8', marginTop: '4px' }}>
            {new Date().toLocaleString()}
          </div>

          {result.frameDataUrl && (
            <img
              src={result.frameDataUrl}
              alt="Captured item"
              style={{
                width: '100%',
                height: '400px',
                objectFit: 'contain',
                backgroundColor: '#000',
                borderRadius: '8px',
                marginTop: '20px',
              }}
            />
          )}

          <div style={{ marginTop: '24px', display: 'flex', flexDirection: 'column', gap: '6px' }}>
            <div style={{ fontSize: '22px', fontWeight: 'bold', color: '#34d399' }}>
              Resin Type: {result.resin.label}
            </div>
            <div style={{ fontSize: '16px', color: '#cbd5e1' }}>
              Contamination Level: {manualContamination}
            </div>
          </div>

          <div
            style={{
              marginTop: '24px',
              paddingTop: '20px',
              borderTop: '1px solid #334155',
              display: 'flex',
              flexDirection: 'column',
              gap: '16px',
            }}
          >
            <div>
              <div style={{ fontSize: '12px', fontWeight: 'bold', textTransform: 'uppercase', letterSpacing: '0.05em', color: '#94a3b8' }}>
                Action
              </div>
              <div style={{ fontSize: '15px', marginTop: '4px' }}>{currentRule.action}</div>
            </div>
            <div>
              <div style={{ fontSize: '12px', fontWeight: 'bold', textTransform: 'uppercase', letterSpacing: '0.05em', color: '#94a3b8' }}>
                Route
              </div>
              <div style={{ fontSize: '15px', marginTop: '4px' }}>{currentRule.route}</div>
            </div>
            <div>
              <div style={{ fontSize: '12px', fontWeight: 'bold', textTransform: 'uppercase', letterSpacing: '0.05em', color: '#94a3b8' }}>
                Reuse
              </div>
              <div style={{ fontSize: '15px', marginTop: '4px' }}>{currentRule.reuse}</div>
            </div>
          </div>
        </div>
      )}

      <RuleDrawer rule={drawerRule} onClose={() => setDrawerRule(null)} />
    </div>
  );
}

function VideoHud({
  containerRef,
  videoRef,
  overlayRef,
  frozenCanvasRef,
  loadState,
  loadError,
  status,
  appState,
  sourceMode,
  isDragActive,
  onDragActive,
  onFiles,
  onTakePhoto,
  onScanAgain,
}) {
  const showLiveVideo = sourceMode === 'camera' && appState === 'live';
  const showFrozenFrame = sourceMode !== 'preset' && !showLiveVideo;
  const showTakePhoto = sourceMode === 'camera' && appState === 'live' && loadState === 'ready';
  const showScanAgain = sourceMode === 'camera' && appState !== 'live' && onScanAgain;

  return (
    <div
      ref={containerRef}
      className={`relative rounded-2xl overflow-hidden border border-slate-700/50 bg-black shadow-2xl ${isDragActive ? 'dropzone-active' : ''}`}
      style={{ width: VIDEO_WIDTH, maxWidth: '100%', aspectRatio: `${VIDEO_WIDTH}/${VIDEO_HEIGHT}` }}
      onDragOver={(e) => {
        e.preventDefault();
        onDragActive(true);
      }}
      onDragLeave={() => onDragActive(false)}
      onDrop={(e) => {
        e.preventDefault();
        onDragActive(false);
        onFiles(e.dataTransfer.files);
      }}
    >
      <video
        ref={videoRef}
        muted
        playsInline
        className="absolute inset-0 w-full h-full object-cover"
        style={{ display: showLiveVideo ? 'block' : 'none' }}
      />
      <canvas
        ref={frozenCanvasRef}
        className="absolute inset-0 w-full h-full"
        style={{ display: showFrozenFrame ? 'block' : 'none' }}
      />
      {sourceMode === 'preset' && (
        <div className="absolute inset-0 flex flex-col items-center justify-center gap-2 bg-gradient-to-br from-slate-900 to-slate-950 text-center px-8">
          <Sparkles className="text-cyan-400" size={30} />
          <span className="text-sm text-gray-400">Simulated telemetry — no live imagery required for this preset</span>
        </div>
      )}
      <canvas
        ref={overlayRef}
        width={VIDEO_WIDTH}
        height={VIDEO_HEIGHT}
        className="absolute inset-0 w-full h-full"
        style={{ display: sourceMode === 'preset' ? 'none' : 'block' }}
      />

      {loadState === 'ready' && showLiveVideo && <div className="scan-line" />}

      {(status === 'analyzing' || status === 'result') && sourceMode === 'camera' && (
        <div className="absolute top-3 left-1/2 -translate-x-1/2 flex items-center gap-1.5 text-[10px] font-mono tracking-wide uppercase text-cyan-200 bg-slate-950/80 border border-cyan-500/30 rounded-full px-2.5 py-1 backdrop-blur-sm z-10">
          <span className="w-1.5 h-1.5 rounded-full bg-cyan-400" />
          Frame captured
        </div>
      )}

      {status === 'result' && sourceMode !== 'preset' && (
        <svg className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-10 h-10 text-emerald-400/50 pointer-events-none crosshair-lock" viewBox="0 0 40 40">
          <circle cx="20" cy="20" r="16" fill="none" stroke="currentColor" strokeWidth="1" strokeDasharray="4 6" />
        </svg>
      )}

      {isDragActive && (
        <div className="absolute inset-0 flex items-center justify-center bg-cyan-950/40 backdrop-blur-sm text-cyan-200 text-sm font-medium">
          Drop image to inspect
        </div>
      )}

      {loadState !== 'ready' && (
        <div data-testid="loading-overlay" className="absolute inset-0 flex flex-col items-center justify-center gap-3 backdrop-blur-xl bg-slate-950/80 text-sm">
          {loadState === 'loading' && (
            <>
              <Loader2 className="animate-spin text-emerald-400" size={28} />
              <span>Loading camera &amp; models…</span>
            </>
          )}
          {loadState === 'error' && (
            <>
              <AlertTriangle className="text-red-400" size={28} />
              <span className="text-red-300 px-6 text-center">{loadError}</span>
            </>
          )}
        </div>
      )}

      {/* HUD viewfinder corner brackets — always-on chrome, framed above the loading overlay */}
      <div className="hud-corner top-3 left-3 border-t-[3px] border-l-[3px] rounded-tl-md" />
      <div className="hud-corner top-3 right-3 border-t-[3px] border-r-[3px] rounded-tr-md" />
      <div className="hud-corner bottom-3 left-3 border-b-[3px] border-l-[3px] rounded-bl-md" />
      <div className="hud-corner bottom-3 right-3 border-b-[3px] border-r-[3px] rounded-br-md" />

      <AnimatePresence>
        {status === 'veto' && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="absolute inset-0 bg-red-950/30 border-2 border-red-500/60 pointer-events-none"
          />
        )}
      </AnimatePresence>

      {/* Primary capture control — prominent, glassmorphic, thumb-friendly */}
      {showTakePhoto && (
        <motion.button
          data-testid="take-photo-button"
          onClick={onTakePhoto}
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          whileTap={{ scale: 0.94 }}
          className="absolute bottom-4 left-1/2 -translate-x-1/2 z-20 flex items-center gap-2.5 min-h-14 pl-5 pr-6 rounded-full backdrop-blur-2xl bg-emerald-500/15 border border-emerald-400/40 shadow-[0_8px_32px_rgba(0,0,0,0.5),0_0_24px_rgba(52,211,153,0.25)] hover:bg-emerald-500/25 hover:border-emerald-400/60 hover:scale-[1.04] active:scale-[0.96] transition-colors"
        >
          <span className="flex items-center justify-center w-9 h-9 rounded-full bg-emerald-400 text-slate-950 shadow-inner">
            <Camera size={18} strokeWidth={2.5} />
          </span>
          <span className="text-sm font-semibold text-emerald-100 tracking-wide">Take Photo</span>
        </motion.button>
      )}

      {showScanAgain && (
        <motion.button
          data-testid="scan-new-item-button"
          onClick={onScanAgain}
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          whileTap={{ scale: 0.94 }}
          className="absolute bottom-4 left-1/2 -translate-x-1/2 z-20 flex items-center gap-2.5 min-h-14 pl-5 pr-6 rounded-full backdrop-blur-2xl bg-slate-900/60 border border-cyan-400/40 shadow-[0_8px_32px_rgba(0,0,0,0.5),0_0_24px_rgba(34,211,238,0.2)] hover:bg-slate-900/80 hover:border-cyan-400/60 hover:scale-[1.04] active:scale-[0.96] transition-colors"
        >
          <span className="flex items-center justify-center w-9 h-9 rounded-full bg-cyan-400 text-slate-950 shadow-inner">
            <RotateCcw size={18} strokeWidth={2.5} />
          </span>
          <span className="text-sm font-semibold text-cyan-100 tracking-wide">Scan New Item</span>
        </motion.button>
      )}
    </div>
  );
}

// Devices only get descriptive labels once permission has been granted; a
// light heuristic upgrades generic labels to "Front/Back Camera" where the
// browser's own label hints at it (common on phones), and falls back to a
// numbered placeholder for devices with no label at all.
function describeDevice(device, index) {
  const label = device.label || `Camera ${index + 1}`;
  if (/back|rear|environment/i.test(label)) return `Back Camera — ${label}`;
  if (/front|user|facetime/i.test(label)) return `Front Camera — ${label}`;
  return label;
}

function SourceControls({ sourceMode, onFiles, onPreset, onReturnToCamera, videoDevices, selectedDeviceId, onSwitchCamera }) {
  return (
    <details className="group rounded-2xl border border-white/10 backdrop-blur-2xl bg-slate-900/40 shadow-[0_8px_32px_rgba(0,0,0,0.4)]">
      <summary className="flex items-center justify-between gap-2 cursor-pointer select-none px-4 py-3 list-none [&::-webkit-details-marker]:hidden">
        <span className="text-xs uppercase tracking-wide text-gray-500 flex items-center gap-1.5">
          <Video size={13} /> Input source &amp; demo presets
        </span>
        <ChevronDown size={14} className="text-gray-500 transition-transform duration-200 group-open:rotate-180" />
      </summary>

      <div className="px-4 pb-4 space-y-3">
        {sourceMode !== 'camera' && (
          <button
            onClick={onReturnToCamera}
            className="w-full text-xs text-cyan-300 hover:text-cyan-200 border border-cyan-500/30 rounded-full px-3 py-1.5 transition-all duration-150 hover:scale-[1.01] active:scale-[0.98]"
          >
            Return to live camera
          </button>
        )}

        {sourceMode === 'camera' && videoDevices.length > 1 && (
          <div className="relative">
            <select
              value={selectedDeviceId ?? ''}
              onChange={(e) => onSwitchCamera(e.target.value)}
              className="w-full appearance-none text-xs text-gray-300 bg-slate-800/60 border border-slate-700/50 hover:border-cyan-500/40 rounded-xl pl-3 pr-8 py-2.5 cursor-pointer transition-colors focus:outline-none focus:ring-1 focus:ring-cyan-500/50"
            >
              {videoDevices.map((device, i) => (
                <option key={device.deviceId} value={device.deviceId}>
                  {describeDevice(device, i)}
                </option>
              ))}
            </select>
            <ChevronDown size={14} className="pointer-events-none absolute right-3 top-1/2 -translate-y-1/2 text-gray-500" />
          </div>
        )}

        <label className="flex items-center gap-2 text-xs text-gray-400 hover:text-gray-200 border border-dashed border-slate-700/60 hover:border-cyan-500/50 rounded-xl px-3 py-2.5 cursor-pointer transition-all duration-150 hover:scale-[1.01] active:scale-[0.99]">
          <Upload size={14} />
          Inspect an image — click or drag &amp; drop onto the viewport
          <input type="file" accept="image/*" className="hidden" onChange={(e) => onFiles(e.target.files)} />
        </label>

        <div>
          <span className="text-xs uppercase tracking-wide text-gray-500">Quick-demo presets</span>
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-2 mt-2">
            {DEMO_PRESETS.map((preset) => (
              <button
                key={preset.key}
                onClick={() => onPreset(preset)}
                className="text-xs text-left rounded-xl border border-slate-700/50 hover:border-emerald-500/50 hover:bg-slate-800/60 hover:scale-[1.02] active:scale-[0.98] transition-all duration-200 px-3 py-2.5 text-gray-300"
              >
                <div className="font-medium text-gray-200">{preset.label}</div>
                <div className="text-gray-500 mt-0.5">{preset.resin} · {preset.contamination}</div>
              </button>
            ))}
          </div>
        </div>
      </div>
    </details>
  );
}

function StatusBadge({ status, sourceMode }) {
  const config = {
    idle: { text: 'Live — frame the item and tap Take Photo', color: 'bg-slate-800/80 text-slate-300 border-slate-700/50', icon: Camera },
    analyzing: {
      text: sourceMode === 'preset' ? 'Simulating scan…' : 'Frame captured — analyzing resin & contamination…',
      color: 'bg-emerald-500/15 text-emerald-300 border-emerald-500/30',
      icon: Loader2,
    },
    veto: {
      text: 'CONTAMINATION RISK: Biometric hand overlap detected',
      color: 'bg-red-500/20 text-red-300 border-red-500/40',
      icon: AlertTriangle,
    },
    'no-item': { text: 'No plastic item detected', color: 'bg-slate-800/80 text-slate-300 border-slate-700/50', icon: Camera },
    result: { text: 'Item identified', color: 'bg-emerald-500/15 text-emerald-300 border-emerald-500/30', icon: Recycle },
    'classifier-error': {
      text: 'Classification failed — check console for details',
      color: 'bg-amber-500/15 text-amber-300 border-amber-500/30',
      icon: AlertTriangle,
    },
  }[status] ?? { text: 'Idle', color: 'bg-slate-800/80 text-slate-300 border-slate-700/50', icon: Camera };

  const Icon = config.icon;
  const showRadar = status === 'idle' || status === 'no-item';

  return (
    <div className="relative inline-flex items-center">
      {showRadar && (
        <span className="absolute -left-1.5 top-1/2 -translate-y-1/2 w-3 h-3">
          <span className="radar-ring" />
          <span className="absolute inset-0 rounded-full bg-slate-400/70" />
        </span>
      )}
      <motion.div
        key={status}
        data-testid="status-badge"
        data-status={status}
        initial={{ opacity: 0, y: 6 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.25 }}
        className={`inline-flex items-center gap-2 rounded-full border px-4 py-2 text-sm font-medium backdrop-blur-sm ${config.color} ${showRadar ? 'pl-6' : ''}`}
      >
        <Icon size={16} className={status === 'analyzing' ? 'animate-spin' : ''} />
        {config.text}
      </motion.div>
    </div>
  );
}

function DiagnosticsBar({ diagnostics }) {
  return (
    <div className="rounded-xl border border-slate-800/60 bg-slate-950/60 backdrop-blur-sm px-4 py-2.5 flex flex-wrap items-center gap-x-5 gap-y-1 text-[11px] font-mono text-gray-500">
      <span className="flex items-center gap-1.5 text-cyan-400/80">
        <Activity size={12} /> DIAGNOSTICS
      </span>
      <span>LAST SCAN <span className="text-gray-300">{diagnostics.latencyMs || '--'} ms</span></span>
      <span>ENGINE <span className="text-gray-300">{diagnostics.engine}</span></span>
      <span>TENSORS <span className="text-gray-300">640×640 → 224×224</span></span>
    </div>
  );
}

function ConfidenceRing({ label, value, ringClass = 'text-emerald-400', testId, labelTestId }) {
  const pct = Math.round(value * 100);
  const radius = 26;
  const circumference = 2 * Math.PI * radius;
  const offset = circumference - (pct / 100) * circumference;

  return (
    <div className="flex items-center gap-3">
      <div className="relative shrink-0" style={{ width: 64, height: 64 }}>
        <svg width="64" height="64" viewBox="0 0 64 64" className="-rotate-90">
          <circle cx="32" cy="32" r={radius} fill="none" stroke="rgba(148,163,184,0.15)" strokeWidth="6" />
          <motion.circle
            cx="32"
            cy="32"
            r={radius}
            fill="none"
            stroke="currentColor"
            strokeWidth="6"
            strokeLinecap="round"
            className={ringClass}
            strokeDasharray={circumference}
            initial={{ strokeDashoffset: circumference }}
            animate={{ strokeDashoffset: offset }}
            transition={{ duration: 0.8, ease: 'easeOut' }}
          />
        </svg>
        <span data-testid={testId} className="absolute inset-0 flex items-center justify-center text-sm font-semibold text-gray-100">
          {pct}%
        </span>
      </div>
      <div className="min-w-0">
        <div data-testid={labelTestId} className="text-[10px] uppercase tracking-wide text-gray-500">{label}</div>
      </div>
    </div>
  );
}

const CONTAMINATION_SLIDER_OPTIONS = [
  { key: 'Clean/Light Soiling', short: 'Clean', bg: 'bg-emerald-400', glow: 'shadow-[0_0_18px_rgba(52,211,153,0.55)]', text: 'text-slate-950' },
  { key: 'Moderate Contamination', short: 'Moderate', bg: 'bg-amber-400', glow: 'shadow-[0_0_18px_rgba(251,191,36,0.55)]', text: 'text-slate-950' },
  { key: 'Heavy Contamination', short: 'Heavy', bg: 'bg-rose-400', glow: 'shadow-[0_0_18px_rgba(251,113,133,0.55)]', text: 'text-slate-950' },
];

// Manual 3-way contamination control — replaces the old Stage 4 ML gauge.
// A shared framer-motion layoutId animates the colored highlight sliding
// between segments instead of hand-rolled transform math.
function ContaminationSlider({ value, onChange }) {
  return (
    <div>
      <div className="flex items-baseline justify-between text-xs text-gray-400 mb-2">
        <span className="uppercase tracking-wide">Contamination level</span>
        <span className="text-[9px] uppercase tracking-wide text-gray-600">Manual</span>
      </div>
      <div data-testid="contamination-slider" className="grid grid-cols-3 gap-1 rounded-xl border border-slate-700/50 bg-slate-800/40 p-1">
        {CONTAMINATION_SLIDER_OPTIONS.map((opt) => {
          const active = value === opt.key;
          return (
            <button
              key={opt.key}
              type="button"
              data-testid={`contamination-option-${opt.short.toLowerCase()}`}
              aria-pressed={active}
              onClick={() => onChange(opt.key)}
              className="relative py-2.5 text-xs font-semibold rounded-lg transition-colors"
            >
              {active && (
                <motion.span
                  layoutId="contamination-highlight"
                  className={`absolute inset-0 rounded-lg ${opt.bg} ${opt.glow}`}
                  transition={{ type: 'spring', stiffness: 500, damping: 32 }}
                />
              )}
              <span className={`relative z-10 ${active ? opt.text : 'text-gray-400 hover:text-gray-200'}`}>{opt.short}</span>
            </button>
          );
        })}
      </div>
    </div>
  );
}

function EmptyPanel({ status }) {
  const text =
    status === 'analyzing'
      ? 'Frame captured. Running the resin and contamination models once on this snapshot…'
      : 'Frame a single plastic item in the viewfinder, away from hands, and tap Take Photo — or upload an image / try a demo preset — to get a resin ID, contamination level, and disposal recommendation.';

  return (
    <motion.div
      layout
      initial={{ opacity: 0, y: 20, scale: 0.95 }}
      animate={{ opacity: 1, y: 0, scale: 1 }}
      exit={{ opacity: 0, y: -10, scale: 0.97 }}
      transition={{ duration: 0.3 }}
      className="rounded-2xl border border-white/10 backdrop-blur-2xl bg-slate-900/40 shadow-[0_8px_32px_rgba(0,0,0,0.4)] p-6 text-gray-400 text-sm"
    >
      {text}
    </motion.div>
  );
}

function VetoAlert({ onScanAgain }) {
  return (
    <motion.div
      layout
      initial={{ opacity: 0, y: 20, scale: 0.95 }}
      animate={{ opacity: 1, y: 0, scale: 1 }}
      exit={{ opacity: 0, y: -10, scale: 0.97 }}
      transition={{ duration: 0.3 }}
      className="shake rounded-2xl border border-red-500/50 backdrop-blur-xl bg-red-950/40 shadow-2xl shadow-red-950/50 p-6 space-y-4"
    >
      <div className="flex items-start gap-3">
        <ShieldAlert className="text-red-400 shrink-0 mt-0.5" size={22} />
        <div>
          <h2 className="text-red-300 font-semibold">Containment barrier engaged</h2>
          <p className="text-sm text-red-200/80 mt-1">
            CONTAMINATION RISK: Biometric hand overlap detected in the captured frame. Isolate the item from human
            contact and take another photo.
          </p>
        </div>
      </div>
      {onScanAgain && (
        <button
          onClick={onScanAgain}
          className="w-full flex items-center justify-center gap-2 text-xs font-medium rounded-xl border border-red-500/30 hover:border-red-400/50 hover:bg-red-900/40 transition-all duration-150 hover:scale-[1.01] active:scale-[0.98] px-3 py-2.5 text-red-200"
        >
          <RotateCcw size={14} />
          Scan new item
        </button>
      )}
    </motion.div>
  );
}

function ClassifierErrorAlert({ message, onScanAgain }) {
  return (
    <motion.div
      layout
      initial={{ opacity: 0, y: 20, scale: 0.95 }}
      animate={{ opacity: 1, y: 0, scale: 1 }}
      exit={{ opacity: 0, y: -10, scale: 0.97 }}
      transition={{ duration: 0.3 }}
      className="rounded-2xl border border-amber-500/40 backdrop-blur-xl bg-amber-950/30 shadow-2xl shadow-amber-950/40 p-6 space-y-4"
    >
      <div className="flex items-start gap-3">
        <AlertTriangle className="text-amber-400 shrink-0 mt-0.5" size={22} />
        <div>
          <h2 className="text-amber-300 font-semibold">Resin/contamination classifier failed</h2>
          <p className="text-sm text-amber-200/80 mt-1">
            Stage 2 (plastic detection) succeeded, but Stage 3/4 inference on the captured photo threw an error.
          </p>
          {message && (
            <code className="block mt-2 text-xs text-amber-300/90 bg-amber-950/60 border border-amber-500/20 rounded-lg px-3 py-2 font-mono break-words">
              {message}
            </code>
          )}
        </div>
      </div>
      {onScanAgain && (
        <button
          onClick={onScanAgain}
          className="w-full flex items-center justify-center gap-2 text-xs font-medium rounded-xl border border-amber-500/30 hover:border-amber-400/50 hover:bg-amber-900/40 transition-all duration-150 hover:scale-[1.01] active:scale-[0.98] px-3 py-2.5 text-amber-200"
        >
          <RotateCcw size={14} />
          Scan new item
        </button>
      )}
    </motion.div>
  );
}

function ResultCard({ result, rule, contaminationLabel, onContaminationChange, onScanAgain }) {
  const { resin, simulated, lowConfidence, frameDataUrl } = result;
  const info = RESIN_INFO[resin.label];
  const [exporting, setExporting] = useState(false);

  // Captures the hidden #receipt-export-template — a static, fixed-width,
  // plain-inline-style layout built specifically for html2canvas — rather
  // than any part of the live responsive Tailwind UI. html2canvas renders
  // oklch() colors and responsive grid/flex rules unreliably; a dedicated
  // off-screen template sidesteps that class of bug entirely.
  const handleExportReceipt = useCallback(async () => {
    if (exporting) return;
    const receiptElement = document.getElementById('receipt-export-template');
    if (!receiptElement) return;
    setExporting(true);
    try {
      const canvas = await html2canvas(receiptElement, {
        scale: 2,
        backgroundColor: '#0f172a',
        useCORS: true,
        logging: false,
      });
      const link = document.createElement('a');
      link.download = `Scan-Receipt-${Date.now()}.png`;
      link.href = canvas.toDataURL('image/png');
      link.click();
    } catch (err) {
      console.error('[ResultCard] receipt export failed', err);
    } finally {
      setExporting(false);
    }
  }, [exporting]);

  return (
    <motion.div
      layout
      data-testid="result-panel"
      initial={{ opacity: 0, y: 20, scale: 0.95 }}
      animate={{ opacity: 1, y: 0, scale: 1 }}
      exit={{ opacity: 0, y: -10, scale: 0.97 }}
      transition={{ duration: 0.3 }}
      className={`flicker-in rounded-2xl backdrop-blur-2xl bg-slate-900/40 shadow-[0_8px_32px_rgba(0,0,0,0.4)] p-6 space-y-5 border ${
        lowConfidence ? 'border-amber-500/30 shadow-amber-950/30' : 'border-emerald-500/30 shadow-emerald-950/30'
      }`}
    >
      {frameDataUrl && (
        <img
          src={frameDataUrl}
          alt="Captured item"
          className="w-full aspect-[4/3] object-contain bg-black rounded-xl border border-slate-700/50"
        />
      )}

      <div className="flex items-baseline justify-between">
        <h2 data-testid="result-resin" data-resin-code={resin.code} className={`text-lg font-semibold ${lowConfidence ? 'text-amber-300' : 'text-emerald-300'}`}>
          {resin.label}
        </h2>
        <div className="flex items-center gap-2">
          {simulated && (
            <span className="text-[10px] uppercase tracking-wide text-cyan-300 border border-cyan-500/40 rounded-full px-2 py-0.5">
              Simulated demo
            </span>
          )}
          <span className="text-xs text-gray-500">resin code {resin.code}</span>
        </div>
      </div>

      {lowConfidence && (
        <div className="flex items-center gap-2 text-xs text-amber-300 bg-amber-500/10 border border-amber-500/30 rounded-lg px-3 py-2">
          <AlertTriangle size={14} className="shrink-0" />
          Resin confidence below 80% — consider rescanning with better lighting or a closer angle.
        </div>
      )}

      <div className="flex items-center justify-center py-1">
        <ConfidenceRing label="Resin confidence" value={resin.confidence} ringClass="text-emerald-400" testId="result-resin-confidence" />
      </div>

      <ContaminationSlider value={contaminationLabel} onChange={onContaminationChange} />

      {info && (
        <div className="grid grid-cols-3 gap-2 text-center">
          <FactChip label="Melting point" value={info.meltingPoint} />
          <FactChip label="Density" value={info.density} />
          <FactChip label="Viability" value={info.viability} />
        </div>
      )}

      <div className="grid grid-cols-1 gap-3 pt-3 border-t border-slate-700/50">
        <RuleRow label="Action" value={rule.action} />
        <RuleRow label="Recycling route" value={rule.route} />
        <RuleRow label="Reuse suggestion" value={rule.reuse} />
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
        {onScanAgain && (
          <button
            onClick={onScanAgain}
            className="flex items-center justify-center gap-2 text-xs font-medium rounded-xl border border-cyan-500/30 hover:border-cyan-400/50 hover:bg-cyan-500/10 transition-all duration-150 hover:scale-[1.01] active:scale-[0.98] px-3 py-2.5 text-cyan-200"
          >
            <RotateCcw size={14} />
            Scan new item
          </button>
        )}
        <button
          onClick={handleExportReceipt}
          disabled={exporting}
          className="flex items-center justify-center gap-2 text-xs font-medium rounded-xl border border-slate-700/50 hover:border-emerald-500/40 hover:bg-slate-800/60 transition-all duration-150 hover:scale-[1.01] active:scale-[0.98] px-3 py-2.5 text-gray-300 disabled:opacity-50 disabled:hover:scale-100"
        >
          <Download size={14} />
          {exporting ? 'Capturing…' : 'Export audit receipt'}
        </button>
      </div>
    </motion.div>
  );
}

function FactChip({ label, value }) {
  return (
    <div className="rounded-lg bg-slate-800/50 border border-slate-700/40 px-2 py-2">
      <div className="text-[9px] uppercase tracking-wide text-gray-500">{label}</div>
      <div className="text-[11px] text-gray-200 mt-0.5 leading-tight">{value}</div>
    </div>
  );
}

function RuleRow({ label, value }) {
  return (
    <div className="rounded-xl bg-slate-800/40 border border-slate-700/40 px-4 py-3">
      <div className="text-[10px] font-semibold uppercase tracking-wide text-gray-500">{label}</div>
      <div className="text-base text-gray-100 mt-1 leading-snug">{value}</div>
    </div>
  );
}

function DisposalRuleTable({ activeResin, activeContamination, onSelectRow, rowRefs, landedRowKey, tableScrollRef }) {
  return (
    <div className="print:break-inside-avoid">
      <div ref={tableScrollRef} className="max-h-80 overflow-y-auto overflow-x-auto rounded-lg">
        <table className="w-full text-sm border-collapse">
          <colgroup>
            <col className="w-[9%]" />
            <col className="w-[16%]" />
            <col className="w-[22%]" />
            <col className="w-[27%]" />
            <col className="w-[26%]" />
          </colgroup>
          <thead className="sticky top-0 bg-slate-900/95 backdrop-blur-sm">
            <tr className="text-left text-slate-500">
              <th className="p-3 sm:px-3 sm:py-2 whitespace-nowrap">Resin</th>
              <th className="p-3 sm:px-3 sm:py-2 whitespace-nowrap">Contamination</th>
              <th className="p-3 sm:px-3 sm:py-2">Action</th>
              <th className="p-3 sm:px-3 sm:py-2">Route</th>
              <th className="p-3 sm:px-3 sm:py-2">Reuse</th>
            </tr>
          </thead>
          <tbody>
            {RESIN_ORDER.map((resin) =>
              CONTAMINATION_ORDER.map((contamination) => {
                const rule = DISPOSAL_RULES[resin][contamination];
                const key = `${resin}-${contamination}`;
                const isActive = resin === activeResin && contamination === activeContamination;
                return (
                  <tr
                    key={key}
                    ref={(el) => {
                      if (rowRefs) rowRefs.current[key] = el;
                    }}
                    onClick={() => onSelectRow({ resin, contamination, ...rule })}
                    className={`border-t border-slate-800/60 cursor-pointer transition-all duration-200 hover:scale-[1.01] active:scale-[0.995] hover:bg-slate-800/60 ${
                      isActive
                        ? 'bg-emerald-500/10 text-emerald-300 ring-2 ring-emerald-400 shadow-[0_0_20px_rgba(52,211,153,0.3)]'
                        : 'text-slate-300'
                    } ${landedRowKey === key ? 'row-land' : ''}`}
                  >
                    <td className="p-3 sm:px-3 sm:py-2 align-top text-left font-medium whitespace-nowrap">{resin}</td>
                    <td className="p-3 sm:px-3 sm:py-2 align-top text-left whitespace-normal break-words">{contamination}</td>
                    <td className="p-3 sm:px-3 sm:py-2 align-top text-left whitespace-normal break-words">{rule.action}</td>
                    <td className="p-3 sm:px-3 sm:py-2 align-top text-left whitespace-normal break-words">{rule.route}</td>
                    <td className="p-3 sm:px-3 sm:py-2 align-top text-left whitespace-normal break-words">{rule.reuse}</td>
                  </tr>
                );
              })
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function RuleDrawer({ rule, onClose }) {
  const info = rule ? RESIN_INFO[rule.resin] : null;

  return (
    <AnimatePresence>
      {rule && (
        <>
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={onClose}
            className="fixed inset-0 bg-black/60 backdrop-blur-sm z-40"
          />
          <motion.div
            initial={{ x: '100%' }}
            animate={{ x: 0 }}
            exit={{ x: '100%' }}
            transition={{ type: 'spring', damping: 28, stiffness: 260 }}
            className="fixed top-0 right-0 h-full w-full max-w-sm z-50 backdrop-blur-xl bg-slate-900/95 border-l border-slate-700/50 shadow-2xl p-6 overflow-y-auto"
          >
            <div className="flex items-start justify-between mb-6">
              <div>
                <h2 className="text-lg font-semibold text-emerald-300">{rule.resin}</h2>
                <p className="text-sm text-gray-500">{rule.contamination}</p>
                <p className="text-xs text-gray-600 mt-0.5">Polymer Circular Economy Pathways</p>
              </div>
              <button
                onClick={onClose}
                className="text-gray-400 hover:text-gray-200 transition-colors rounded-full p-1 hover:bg-slate-800/80"
              >
                <X size={20} />
              </button>
            </div>

            <div className="space-y-5">
              <DrawerField label="Recommended action" value={rule.action} />
              <DrawerField label="Recycling route" value={rule.route} />
              <DrawerField label="Reuse suggestion" value={rule.reuse} />

              {info && (
                <>
                  <div className="pt-2 border-t border-slate-700/50">
                    <div className="text-xs uppercase tracking-wide text-gray-500 mb-2">Chemical recycling pathways</div>
                    <div className="space-y-3">
                      <DrawerField label="Mechanical" value={info.pathways.mechanical} />
                      <DrawerField label="Pyrolysis" value={info.pathways.pyrolysis} />
                      <DrawerField label="Glycolysis" value={info.pathways.glycolysis} />
                      <DrawerField label="Waste-to-energy" value={info.pathways.wasteToEnergy} />
                    </div>
                  </div>
                  <div className="grid grid-cols-2 gap-2">
                    <DrawerField label="Melting point" value={info.meltingPoint} />
                    <DrawerField label="Density" value={info.density} />
                  </div>
                </>
              )}
            </div>
          </motion.div>
        </>
      )}
    </AnimatePresence>
  );
}

function DrawerField({ label, value }) {
  return (
    <div className="rounded-xl border border-slate-700/50 bg-slate-800/40 p-4">
      <div className="text-xs uppercase tracking-wide text-gray-500 mb-1">{label}</div>
      <div className="text-sm text-gray-200">{value}</div>
    </div>
  );
}
