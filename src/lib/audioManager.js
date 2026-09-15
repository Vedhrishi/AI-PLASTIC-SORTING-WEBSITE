// Procedural sound synthesis via the Web Audio API — no external audio files.
let ctx = null;
let muted = false;

function getCtx() {
  if (!ctx) {
    const AudioCtx = window.AudioContext || window.webkitAudioContext;
    ctx = new AudioCtx();
  }
  if (ctx.state === 'suspended') ctx.resume();
  return ctx;
}

function tone(freq, startTime, duration, { type = 'sine', gain = 0.15, glideTo = null } = {}) {
  const audioCtx = getCtx();
  const osc = audioCtx.createOscillator();
  const gainNode = audioCtx.createGain();
  osc.type = type;
  osc.frequency.setValueAtTime(freq, startTime);
  if (glideTo) osc.frequency.linearRampToValueAtTime(glideTo, startTime + duration);

  gainNode.gain.setValueAtTime(0, startTime);
  gainNode.gain.linearRampToValueAtTime(gain, startTime + 0.01);
  gainNode.gain.exponentialRampToValueAtTime(0.001, startTime + duration);

  osc.connect(gainNode).connect(audioCtx.destination);
  osc.start(startTime);
  osc.stop(startTime + duration + 0.02);
}

export function setMuted(value) {
  muted = value;
}

export function isMuted() {
  return muted;
}

export function playScanBeep() {
  if (muted) return;
  const t = getCtx().currentTime;
  tone(880, t, 0.06, { type: 'sine', gain: 0.05 });
}

export function playLockOn() {
  if (muted) return;
  const t = getCtx().currentTime;
  tone(660, t, 0.09, { type: 'square', gain: 0.08 });
  tone(1320, t + 0.07, 0.12, { type: 'square', gain: 0.08 });
}

export function playVetoAlarm() {
  if (muted) return;
  const t = getCtx().currentTime;
  tone(220, t, 0.18, { type: 'sawtooth', gain: 0.1, glideTo: 180 });
  tone(220, t + 0.25, 0.18, { type: 'sawtooth', gain: 0.1, glideTo: 180 });
}

export function playSuccess() {
  if (muted) return;
  const t = getCtx().currentTime;
  [523.25, 659.25, 783.99, 1046.5].forEach((freq, i) => {
    tone(freq, t + i * 0.09, 0.14, { type: 'triangle', gain: 0.09 });
  });
}
