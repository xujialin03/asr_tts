// Silero VAD (browser) for hands-free utterance detection.
//
// Uses @ricky0123/vad-web (Silero v5 via onnxruntime-web WASM). Model, worklet
// and wasm are served from /public/vad/ so it works offline.
//
// We use Silero's per-frame speech probability and do endpointing ourselves.
// This is more controllable than relying on vad-web's onSpeechEnd event: when
// probability stays slightly above its negative threshold, library onSpeechEnd
// may never fire. Here, an utterance ends after sustained low-ish probability.

import { MicVAD } from "@ricky0123/vad-web";

const SAMPLE_RATE = 16000;
const START_THRESHOLD = 0.5;
const END_THRESHOLD = 0.45;
const END_SILENCE_MS = 1100;
const MIN_SPEECH_MS = 250;
const PRE_ROLL_MS = 800;
const MAX_UTTERANCE_MS = 15000;
const PRE_ROLL_SAMPLES = SAMPLE_RATE * (PRE_ROLL_MS / 1000);

let vad = null;
let sharedStream = null;
let preRoll = [];
let preRollSamples = 0;
let speechActive = false;
let segmentFrames = [];
let segmentMs = 0;
let speechMs = 0;
let silenceMs = 0;
let callbacks = null;

function concatFrames(frames) {
  const total = frames.reduce((n, f) => n + f.length, 0);
  const out = new Float32Array(total);
  let off = 0;
  for (const f of frames) {
    out.set(f, off);
    off += f.length;
  }
  return out;
}

function resetSegment() {
  speechActive = false;
  segmentFrames = [];
  segmentMs = 0;
  speechMs = 0;
  silenceMs = 0;
}

function pushPreRollFrame(frame) {
  preRoll.push(frame);
  preRollSamples += frame.length;
  while (preRollSamples > PRE_ROLL_SAMPLES && preRoll.length > 1) {
    preRollSamples -= preRoll[0].length;
    preRoll.shift();
  }
}

function clearPreRoll() {
  preRoll = [];
  preRollSamples = 0;
}

function endSegment() {
  if (!speechActive) return;
  const audio = concatFrames(segmentFrames);
  const accepted = speechMs >= MIN_SPEECH_MS && audio.length > 0;
  resetSegment();
  clearPreRoll();
  if (accepted) callbacks?.onSpeechEnd?.(audio);
}

function normalizeFrame(frame) {
  if (frame instanceof Float32Array) return frame;
  if (frame instanceof ArrayBuffer) return new Float32Array(frame);
  if (ArrayBuffer.isView(frame)) {
    return new Float32Array(frame.buffer, frame.byteOffset, frame.byteLength / 4);
  }
  return null;
}

function processFrame(probs, rawFrame) {
  const frame = normalizeFrame(rawFrame);
  if (!frame || !frame.length) return;

  const p = Number(probs?.isSpeech ?? 0);
  const computedFrameMs = (frame.length / SAMPLE_RATE) * 1000;
  const frameMs = Number.isFinite(computedFrameMs) && computedFrameMs > 0 ? computedFrameMs : 32;

  if (!speechActive) {
    if (p >= START_THRESHOLD) {
      speechActive = true;
      segmentFrames = [...preRoll, frame];
      segmentMs = frameMs + (preRollSamples / SAMPLE_RATE) * 1000;
      speechMs = frameMs;
      silenceMs = 0;
      clearPreRoll();
      callbacks?.onSpeechStart?.();
    } else {
      pushPreRollFrame(frame);
    }
    callbacks?.onFrame?.(p, speechActive, silenceMs, segmentMs);
    return;
  }

  segmentFrames.push(frame);
  segmentMs += frameMs;

  if (p >= START_THRESHOLD) {
    speechMs += frameMs;
    silenceMs = 0;
  } else if (p < END_THRESHOLD) {
    silenceMs += frameMs;
  }

  const shouldEnd = silenceMs >= END_SILENCE_MS || segmentMs >= MAX_UTTERANCE_MS;
  callbacks?.onFrame?.(p, speechActive, silenceMs, segmentMs);
  if (shouldEnd) endSegment();
}

async function ensureStream() {
  if (!sharedStream || sharedStream.getTracks().every((t) => t.readyState === "ended")) {
    sharedStream = await navigator.mediaDevices.getUserMedia({
      audio: {
        channelCount: 1,
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true,
      },
      video: false,
    });
  }
  return sharedStream;
}

export function releaseSharedStream() {
  if (sharedStream) {
    sharedStream.getTracks().forEach((t) => t.stop());
    sharedStream = null;
  }
  resetSegment();
  clearPreRoll();
}

/**
 * Start the VAD.
 * @param {object} cb { onSpeechStart, onSpeechEnd(audio), onFrame(prob, active) }
 */
export async function startVad(cb) {
  if (vad) return;
  callbacks = cb;
  const stream = await ensureStream();

  vad = await MicVAD.new({
    model: "v5",
    baseAssetPath: "/vad/",
    onnxWASMBasePath: "/vad/",
    // Keep library thresholds permissive; endpointing is controlled above.
    positiveSpeechThreshold: START_THRESHOLD,
    negativeSpeechThreshold: END_THRESHOLD,
    redemptionMs: END_SILENCE_MS,
    minSpeechMs: MIN_SPEECH_MS,
    preSpeechPadMs: PRE_ROLL_MS,
    startOnLoad: false,
    getStream: async () => stream,
    pauseStream: async () => {},
    resumeStream: async () => stream,
    onFrameProcessed: processFrame,
    // We intentionally do not use library onSpeechStart/onSpeechEnd.
    onSpeechStart: () => {},
    onSpeechEnd: () => {},
  });
  vad.start();
}

export function stopVad() {
  if (vad) {
    vad.destroy();
    vad = null;
  }
  callbacks = null;
  resetSegment();
  clearPreRoll();
}
