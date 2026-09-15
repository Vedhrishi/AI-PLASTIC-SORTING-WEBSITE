import { useEffect, useRef, useState, useCallback } from 'react';
import * as cocoSsd from '@tensorflow-models/coco-ssd';
import '@tensorflow/tfjs';
import { AnimatePresence, motion } from 'framer-motion';
import {
  Camera,
  ScanLine,
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
} from 'lucide-react';

import { loadSession } from './lib/onnxSetup';
import { detectPlastic } from './lib/yolo';
import { classifyResin, classifyContamination, cropAndResize } from './lib/classify';
import { overlapFraction, mapCoverBox } from './lib/geometry';
import { DISPOSAL_RULES, getDisposalRule } from './data/disposalRules';
import { RESIN_INFO, CONTAMINATION_INFO } from './data/resinInfo';
import { buildInspectionCertificate, downloadCertificate } from './lib/audit';
import * as audio from './lib/audioManager';

const VIDEO_WIDTH = 640;
const VIDEO_HEIGHT = 480;
const PERSON_VETO_IOU = 0.2;
const YOLO_FALLBACK_MIN_SCORE = 0.3;

const RESIN_ORDER = ['PET', 'HDPE', 'PP', 'PS'];
const CONTAMINATION_ORDER = ['Clean/Light Soiling', 'Moderate Contamination', 'Heavy Contamination'];

const DEMO_PRESETS = [
  { key: 'pet-clean', label: 'Clean PET Bottle', resin: 'PET', code: 1, resinConf: 0.974, contamination: 'Clean/Light Soiling', contamConf: 0.951 },
  { key: 'hdpe-soiled', label: 'Soiled HDPE Jug', resin: 'HDPE', code: 2, resinConf: 0.932, contamination: 'Moderate Contamination', contamConf: 0.881 },
  { key: 'pp-dirty', label: 'Dirty PP Cup', resin: 'PP', code: 5, resinConf: 0.908, contamination: 'Heavy Contamination', contamConf: 0.864 },
];

function coverFitCanvas(img, w, h) {
  const canvas = document.createElement('canvas');
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext('2d');
  const iw = img.naturalWidth || img.width;
  const ih = img.naturalHeight || img.height;
  const scale = Math.max(w / iw, h / ih);
  const drawW = iw * scale;
  const drawH = ih * scale;
  const dx = (w - drawW) / 2;
  const dy = (h - drawH) / 2;
  ctx.drawImage(img, dx, dy, drawW, drawH);
  return canvas;
}

export default function App() {
  const videoRef = useRef(null);
  const overlayRef = useRef(null);
  const uploadCanvasRef = useRef(null);
  const modelsRef = useRef(null);
  const isProcessingRef = useRef(false);
  const lockedRef = useRef(false);
  const lastVetoAudioRef = useRef(0);
  const lastTickAtRef = useRef(0);
  const rowRefs = useRef({});
  const tableScrollRef = useRef(null);

  const [loadState, setLoadState] = useState('loading'); // loading | ready | error
  const [loadError, setLoadError] = useState(null);
  const [status, setStatus] = useState('idle'); // idle | scanning | veto | result | no-item
  const [result, setResult] = useState(null);
  const [drawerRule, setDrawerRule] = useState(null);
  const [muted, setMuted] = useState(false);
  const [sourceMode, setSourceMode] = useState('camera'); // camera | upload | preset
  const [uploadImg, setUploadImg] = useState(null);
  const [isDragActive, setIsDragActive] = useState(false);
  const [diagnostics, setDiagnostics] = useState({ fps: 0, latencyMs: 0, engine: 'ONNX Runtime Web · WebGL (GPU) → WASM' });
  const [landedRowKey, setLandedRowKey] = useState(null);
  const [exporting, setExporting] = useState(false);

  useEffect(() => {
    audio.setMuted(muted);
  }, [muted]);

  // ---- load webcam + models in parallel (camera issues shouldn't block model loading) ----
  useEffect(() => {
    let stream;
    let cancelled = false;

    async function initCamera() {
      stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: VIDEO_WIDTH },
          height: { ideal: VIDEO_HEIGHT },
          facingMode: { ideal: 'environment' },
        },
        audio: false,
      });
      if (cancelled) return;
      videoRef.current.srcObject = stream;
      await videoRef.current.play();
    }

    async function initModels() {
      const [personModel, plasticSession, resinSession, contamSession] = await Promise.all([
        cocoSsd.load({ base: 'lite_mobilenet_v2' }),
        // YOLO is the heavy compute stage (640x640 input) — worth the GPU.
        loadSession('/models/best.onnx'),
        // ResNet18 classifiers are small/fast on WASM alone; pinning them
        // to WASM avoids WebGL op-coverage/precision edge cases silently
        // corrupting classification output on some mobile GPUs.
        loadSession('/models/resin-resnet18.onnx', { executionProviders: ['wasm'] }),
        loadSession('/models/contamination-resnet18.onnx', { executionProviders: ['wasm'] }),
      ]);
      if (cancelled) return;
      modelsRef.current = { personModel, plasticSession, resinSession, contamSession };
    }

    async function init() {
      try {
        await Promise.all([initCamera(), initModels()]);
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
      if (stream) stream.getTracks().forEach((t) => t.stop());
    };
  }, []);

  const drawOverlay = useCallback((plasticBox, personBoxes, vetoed, extraLabel, nativeW, nativeH) => {
    const canvas = overlayRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    ctx.lineWidth = 2;
    ctx.font = '12px ui-monospace, monospace';

    // Native frame -> displayed (object-fit: cover) container space. A no-op
    // when the source is already container-sized (upload/preset canvases).
    const toScreen = (box) =>
      nativeW && nativeH ? mapCoverBox(box, nativeW, nativeH, VIDEO_WIDTH, VIDEO_HEIGHT) : box;

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

      // corner reticles
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

  const runPipeline = useCallback(
    async (source, srcW, srcH) => {
      const { personModel, plasticSession, resinSession, contamSession } = modelsRef.current;
      const tickStart = performance.now();

      const now = performance.now();
      if (lastTickAtRef.current) {
        const fps = 1000 / (now - lastTickAtRef.current);
        setDiagnostics((d) => ({ ...d, fps: Math.round(fps * 10) / 10 }));
      }
      lastTickAtRef.current = now;

      setStatus('scanning');
      audio.playScanBeep();

      const [detections, plasticDetections] = await Promise.all([
        personModel.detect(source),
        detectPlastic(plasticSession, source, srcW, srcH, { confThreshold: 0.4 }),
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

      // No plastic found (Stage 2) — skip Stage 3/4 entirely and save the compute.
      if (!plasticBox) {
        setStatus('no-item');
        setResult(null);
        lockedRef.current = false;
        drawOverlay(null, personBoxes, false, undefined, srcW, srcH);
        setDiagnostics((d) => ({ ...d, latencyMs: Math.round(performance.now() - tickStart) }));
        return;
      }

      const vetoed = personBoxes.some((personBox) => overlapFraction(plasticBox, personBox) > PERSON_VETO_IOU);

      drawOverlay(plasticBox, personBoxes, vetoed, usedFallback && !vetoed ? 'PLASTIC (fallback)' : undefined, srcW, srcH);

      // Hand/person overlap veto — also skip Stage 3/4 here.
      if (vetoed) {
        setStatus('veto');
        setResult(null);
        lockedRef.current = false;
        if (now - lastVetoAudioRef.current > 1400) {
          audio.playVetoAlarm();
          lastVetoAudioRef.current = now;
        }
        setDiagnostics((d) => ({ ...d, latencyMs: Math.round(performance.now() - tickStart) }));
        return;
      }

      if (!lockedRef.current) {
        audio.playLockOn();
        lockedRef.current = true;
      }

      const cropped = cropAndResize(source, plasticBox);

      let resin, contamination;
      try {
        [resin, contamination] = await Promise.all([
          classifyResin(resinSession, cropped),
          classifyContamination(contamSession, cropped),
        ]);
      } catch (err) {
        console.error('Stage 3/4 classification failed', err);
        setStatus('classifier-error');
        setResult(null);
        setDiagnostics((d) => ({ ...d, latencyMs: Math.round(performance.now() - tickStart) }));
        return;
      }

      const rule = getDisposalRule(resin.label, contamination.label);

      audio.playSuccess();
      setResult({ resin, contamination, rule, box: plasticBox, simulated: false });
      setStatus('result');
      setDiagnostics((d) => ({ ...d, latencyMs: Math.round(performance.now() - tickStart) }));
    },
    [drawOverlay]
  );

  // ---- continuous detection loop (camera mode only) ----
  // Self-scheduling via rAF: the next frame is only grabbed once the full
  // 4-stage pipeline for the previous one has resolved. isProcessingRef is
  // the hard lock — a frame in flight is never interrupted or doubled up,
  // which is what keeps this smooth instead of piling up work on a slow
  // mobile GPU/CPU.
  useEffect(() => {
    if (loadState !== 'ready' || sourceMode !== 'camera') return;
    let cancelled = false;
    let rafId;

    async function tick() {
      if (cancelled) return;

      if (isProcessingRef.current) {
        rafId = requestAnimationFrame(tick);
        return;
      }

      const video = videoRef.current;
      if (!video || video.readyState < 2 || !video.videoWidth) {
        rafId = requestAnimationFrame(tick);
        return;
      }

      isProcessingRef.current = true;
      try {
        await runPipeline(video, video.videoWidth, video.videoHeight);
      } catch (err) {
        console.error('pipeline error', err);
      } finally {
        isProcessingRef.current = false;
        if (!cancelled) rafId = requestAnimationFrame(tick);
      }
    }

    rafId = requestAnimationFrame(tick);

    return () => {
      cancelled = true;
      cancelAnimationFrame(rafId);
    };
  }, [loadState, sourceMode, runPipeline]);

  // ---- single-shot pipeline for an uploaded image ----
  useEffect(() => {
    if (sourceMode !== 'upload' || !uploadImg || loadState !== 'ready') return;
    let cancelled = false;

    async function run() {
      const canvas = coverFitCanvas(uploadImg, VIDEO_WIDTH, VIDEO_HEIGHT);
      const displayCanvas = uploadCanvasRef.current;
      if (displayCanvas) {
        displayCanvas.width = VIDEO_WIDTH;
        displayCanvas.height = VIDEO_HEIGHT;
        displayCanvas.getContext('2d').drawImage(canvas, 0, 0);
      }
      if (cancelled) return;
      isProcessingRef.current = true;
      try {
        await runPipeline(canvas, VIDEO_WIDTH, VIDEO_HEIGHT);
      } catch (err) {
        console.error('pipeline error', err);
      } finally {
        isProcessingRef.current = false;
      }
    }

    run();
    return () => {
      cancelled = true;
    };
  }, [sourceMode, uploadImg, loadState, runPipeline]);

  // ---- auto-scroll + landed pulse when a new result lands ----
  useEffect(() => {
    if (status !== 'result' || !result) return;
    const key = `${result.resin.label}-${result.contamination.label}`;
    const el = rowRefs.current[key];
    if (el) {
      el.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
      setLandedRowKey(key);
      const timeout = setTimeout(() => setLandedRowKey(null), 1200);
      return () => clearTimeout(timeout);
    }
  }, [status, result]);

  const handleFiles = useCallback((files) => {
    const file = files?.[0];
    if (!file || !file.type.startsWith('image/')) return;
    const img = new Image();
    img.onload = () => {
      setSourceMode('upload');
      setUploadImg(img);
    };
    img.src = URL.createObjectURL(file);
  }, []);

  const applyPreset = useCallback((preset) => {
    setSourceMode('preset');
    setUploadImg(null);
    setStatus('scanning');
    setResult(null);
    setTimeout(() => {
      const rule = getDisposalRule(preset.resin, preset.contamination);
      audio.playLockOn();
      setTimeout(() => {
        audio.playSuccess();
        setResult({
          resin: { label: preset.resin, code: preset.code, confidence: preset.resinConf },
          contamination: { label: preset.contamination, level: CONTAMINATION_INFO[preset.contamination].index, confidence: preset.contamConf },
          rule,
          simulated: true,
        });
        setStatus('result');
      }, 400);
    }, 500);
  }, []);

  const returnToCamera = useCallback(() => {
    setSourceMode('camera');
    setUploadImg(null);
    setStatus('idle');
    setResult(null);
    lockedRef.current = false;
  }, []);

  const handleExport = useCallback(async () => {
    if (!result) return;
    setExporting(true);
    try {
      const cert = await buildInspectionCertificate(result, { simulated: !!result.simulated });
      downloadCertificate(cert);
    } finally {
      setExporting(false);
    }
  }, [result]);

  return (
    <div className="min-h-screen bg-[#05060a] text-gray-100">
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

      <main className="max-w-6xl mx-auto px-4 sm:px-6 py-6 sm:py-8 grid gap-6 sm:gap-8 lg:grid-cols-[640px_1fr]">
        <section className="space-y-4">
          <VideoHud
            videoRef={videoRef}
            overlayRef={overlayRef}
            uploadCanvasRef={uploadCanvasRef}
            loadState={loadState}
            loadError={loadError}
            status={status}
            sourceMode={sourceMode}
            isDragActive={isDragActive}
            onDragActive={setIsDragActive}
            onFiles={handleFiles}
          />

          <StatusBadge status={status} sourceMode={sourceMode} />

          <DiagnosticsBar diagnostics={diagnostics} status={status} />

          <SourceControls
            sourceMode={sourceMode}
            onFiles={handleFiles}
            onPreset={applyPreset}
            onReturnToCamera={returnToCamera}
          />
        </section>

        <section className="space-y-6">
          <AnimatePresence mode="wait">
            {status === 'veto' ? (
              <VetoAlert key="veto" />
            ) : status === 'classifier-error' ? (
              <ClassifierErrorAlert key="classifier-error" />
            ) : status === 'result' && result ? (
              <ResultCard key="result" result={result} onExport={handleExport} exporting={exporting} />
            ) : (
              <EmptyPanel key="empty" status={status} />
            )}
          </AnimatePresence>

          <DisposalRuleTable
            activeResin={result?.resin.label}
            activeContamination={result?.contamination.label}
            onSelectRow={setDrawerRule}
            rowRefs={rowRefs}
            landedRowKey={landedRowKey}
            tableScrollRef={tableScrollRef}
          />
        </section>
      </main>

      <RuleDrawer rule={drawerRule} onClose={() => setDrawerRule(null)} />
    </div>
  );
}

function VideoHud({ videoRef, overlayRef, uploadCanvasRef, loadState, loadError, status, sourceMode, isDragActive, onDragActive, onFiles }) {
  return (
    <div
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
        width={VIDEO_WIDTH}
        height={VIDEO_HEIGHT}
        muted
        playsInline
        className="absolute inset-0 w-full h-full object-cover"
        style={{ display: sourceMode === 'camera' ? 'block' : 'none' }}
      />
      <canvas
        ref={uploadCanvasRef}
        className="absolute inset-0 w-full h-full"
        style={{ display: sourceMode === 'upload' ? 'block' : 'none' }}
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

      {loadState === 'ready' && status !== 'veto' && sourceMode !== 'preset' && <div className="scan-line" />}

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
        <div className="absolute inset-0 flex flex-col items-center justify-center gap-3 backdrop-blur-xl bg-slate-950/80 text-sm">
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
    </div>
  );
}

function SourceControls({ sourceMode, onFiles, onPreset, onReturnToCamera }) {
  return (
    <div className="rounded-2xl border border-white/10 backdrop-blur-2xl bg-slate-900/40 shadow-[0_8px_32px_rgba(0,0,0,0.4)] p-4 space-y-3">
      <div className="flex items-center justify-between">
        <span className="text-xs uppercase tracking-wide text-gray-500 flex items-center gap-1.5">
          <Video size={13} /> Input source
        </span>
        {sourceMode !== 'camera' && (
          <button
            onClick={onReturnToCamera}
            className="text-xs text-cyan-300 hover:text-cyan-200 border border-cyan-500/30 rounded-full px-3 py-1 transition-all duration-150 hover:scale-[1.02] active:scale-[0.98]"
          >
            Return to live camera
          </button>
        )}
      </div>

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
  );
}

function StatusBadge({ status, sourceMode }) {
  const config = {
    idle: { text: 'Idle', color: 'bg-slate-800/80 text-slate-300 border-slate-700/50', icon: Camera },
    scanning: { text: sourceMode === 'preset' ? 'Simulating scan…' : 'Scanning…', color: 'bg-cyan-500/15 text-cyan-300 border-cyan-500/30', icon: ScanLine },
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
        initial={{ opacity: 0, y: 6 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.25 }}
        className={`inline-flex items-center gap-2 rounded-full border px-4 py-2 text-sm font-medium backdrop-blur-sm ${config.color} ${showRadar ? 'pl-6' : ''}`}
      >
        <Icon size={16} />
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
      <span>FPS <span className="text-gray-300">{diagnostics.fps || '--'}</span></span>
      <span>LATENCY <span className="text-gray-300">{diagnostics.latencyMs || '--'} ms</span></span>
      <span>ENGINE <span className="text-gray-300">{diagnostics.engine}</span></span>
      <span>TENSORS <span className="text-gray-300">640×640 → 224×224</span></span>
    </div>
  );
}

function ConfidenceMeter({ label, value, colorClass = 'from-emerald-400 to-cyan-400' }) {
  const pct = Math.round(value * 100);
  return (
    <div>
      <div className="flex items-baseline justify-between text-xs text-gray-400 mb-1">
        <span className="uppercase tracking-wide">{label}</span>
        <span className="text-gray-300 font-medium">{pct}%</span>
      </div>
      <div className="h-2 rounded-full bg-slate-800/80 overflow-hidden">
        <motion.div
          initial={{ width: 0 }}
          animate={{ width: `${pct}%` }}
          transition={{ duration: 0.6, ease: 'easeOut' }}
          className={`h-full rounded-full bg-gradient-to-r ${colorClass}`}
        />
      </div>
    </div>
  );
}

function ConfidenceRing({ label, value, ringClass = 'text-emerald-400' }) {
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
        <span className="absolute inset-0 flex items-center justify-center text-sm font-semibold text-gray-100">
          {pct}%
        </span>
      </div>
      <div className="min-w-0">
        <div className="text-[10px] uppercase tracking-wide text-gray-500">{label}</div>
      </div>
    </div>
  );
}

function EmptyPanel({ status }) {
  const text =
    status === 'scanning'
      ? 'Analyzing the frame…'
      : 'Point the camera at a single plastic item, away from hands, or upload an image / try a demo preset to get a resin ID, contamination level, and disposal recommendation.';

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

function VetoAlert() {
  return (
    <motion.div
      layout
      initial={{ opacity: 0, y: 20, scale: 0.95 }}
      animate={{ opacity: 1, y: 0, scale: 1 }}
      exit={{ opacity: 0, y: -10, scale: 0.97 }}
      transition={{ duration: 0.3 }}
      className="shake rounded-2xl border border-red-500/50 backdrop-blur-xl bg-red-950/40 shadow-2xl shadow-red-950/50 p-6"
    >
      <div className="flex items-start gap-3">
        <ShieldAlert className="text-red-400 shrink-0 mt-0.5" size={22} />
        <div>
          <h2 className="text-red-300 font-semibold">Containment barrier engaged</h2>
          <p className="text-sm text-red-200/80 mt-1">
            CONTAMINATION RISK: Biometric hand overlap detected. Isolate the item from human contact and hold steady.
          </p>
        </div>
      </div>
    </motion.div>
  );
}

function ClassifierErrorAlert() {
  return (
    <motion.div
      layout
      initial={{ opacity: 0, y: 20, scale: 0.95 }}
      animate={{ opacity: 1, y: 0, scale: 1 }}
      exit={{ opacity: 0, y: -10, scale: 0.97 }}
      transition={{ duration: 0.3 }}
      className="rounded-2xl border border-amber-500/40 backdrop-blur-xl bg-amber-950/30 shadow-2xl shadow-amber-950/40 p-6"
    >
      <div className="flex items-start gap-3">
        <AlertTriangle className="text-amber-400 shrink-0 mt-0.5" size={22} />
        <div>
          <h2 className="text-amber-300 font-semibold">Resin/contamination classifier failed</h2>
          <p className="text-sm text-amber-200/80 mt-1">
            Stage 2 (plastic detection) succeeded, but Stage 3/4 inference threw an error. Check the browser console for
            the logged exception — the scan will retry automatically on the next frame.
          </p>
        </div>
      </div>
    </motion.div>
  );
}

function ResultCard({ result, onExport, exporting }) {
  const { resin, contamination, rule, simulated } = result;
  const info = RESIN_INFO[resin.label];

  return (
    <motion.div
      layout
      initial={{ opacity: 0, y: 20, scale: 0.95 }}
      animate={{ opacity: 1, y: 0, scale: 1 }}
      exit={{ opacity: 0, y: -10, scale: 0.97 }}
      transition={{ duration: 0.3 }}
      className="flicker-in rounded-2xl border border-emerald-500/30 backdrop-blur-2xl bg-slate-900/40 shadow-[0_8px_32px_rgba(0,0,0,0.4)] shadow-emerald-950/30 p-6 space-y-5"
    >
      <div className="flex items-baseline justify-between">
        <h2 className="text-lg font-semibold text-emerald-300">{resin.label}</h2>
        <div className="flex items-center gap-2">
          {simulated && (
            <span className="text-[10px] uppercase tracking-wide text-cyan-300 border border-cyan-500/40 rounded-full px-2 py-0.5">
              Simulated demo
            </span>
          )}
          <span className="text-xs text-gray-500">resin code {resin.code}</span>
        </div>
      </div>

      <div className="flex items-center justify-around gap-4 py-1">
        <ConfidenceRing label="Resin confidence" value={resin.confidence} ringClass="text-emerald-400" />
        <ConfidenceRing
          label={contamination.label}
          value={contamination.confidence}
          ringClass={
            contamination.level === 0
              ? 'text-emerald-400'
              : contamination.level === 1
                ? 'text-amber-400'
                : 'text-red-400'
          }
        />
      </div>

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

      <button
        onClick={onExport}
        disabled={exporting}
        className="w-full flex items-center justify-center gap-2 text-xs font-medium rounded-xl border border-slate-700/50 hover:border-emerald-500/40 hover:bg-slate-800/60 transition-all duration-150 hover:scale-[1.01] active:scale-[0.98] px-3 py-2.5 text-gray-300 disabled:opacity-50 disabled:hover:scale-100"
      >
        <Download size={14} />
        {exporting ? 'Generating certificate…' : 'Export inspection audit'}
      </button>
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
    <div>
      <div className="text-xs uppercase tracking-wide text-gray-500">{label}</div>
      <div className="text-sm text-gray-200">{value}</div>
    </div>
  );
}

function DisposalRuleTable({ activeResin, activeContamination, onSelectRow, rowRefs, landedRowKey, tableScrollRef }) {
  return (
    <div className="rounded-2xl border border-white/10 backdrop-blur-2xl bg-slate-900/40 shadow-[0_8px_32px_rgba(0,0,0,0.4)] p-6">
      <h3 className="text-sm font-semibold mb-3 text-gray-300">Stage 3 disposal rules</h3>
      <div ref={tableScrollRef} className="max-h-80 overflow-y-auto overflow-x-auto rounded-lg">
        <table className="w-full min-w-[640px] text-xs border-collapse">
          <thead className="sticky top-0 bg-slate-900/95 backdrop-blur-sm">
            <tr className="text-left text-gray-500">
              <th className="py-2 pr-3 whitespace-nowrap">Resin</th>
              <th className="py-2 pr-3 whitespace-nowrap">Contamination</th>
              <th className="py-2 pr-3 whitespace-nowrap">Action</th>
              <th className="py-2 pr-3">Route</th>
              <th className="py-2">Reuse</th>
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
                        : 'text-gray-400'
                    } ${landedRowKey === key ? 'row-land' : ''}`}
                  >
                    <td className="py-2.5 pr-3 align-top font-medium whitespace-nowrap">{resin}</td>
                    <td className="py-2.5 pr-3 align-top whitespace-nowrap">{contamination}</td>
                    <td className="py-2.5 pr-3 align-top whitespace-nowrap">{rule.action}</td>
                    <td className="py-2.5 pr-3 align-top">{rule.route}</td>
                    <td className="py-2.5 align-top">{rule.reuse}</td>
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
