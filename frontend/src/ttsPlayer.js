// Streaming PCM audio player for the TTS endpoint.
//
// The backend /api/tts returns raw 16-bit signed little-endian PCM, mono, at the
// sample rate advertised in the `X-Sample-Rate` response header. We read the
// response body as a stream, decode each chunk, and schedule AudioBufferSources
// back-to-back on the AudioContext timeline so playback is gapless.

let ctx = null;
let current = null; // active playback handle, so a new one can cancel the old

function audioCtx() {
  if (!ctx) ctx = new (window.AudioContext || window.webkitAudioContext)();
  return ctx;
}

// Stop whatever is currently playing.
export function stopPlayback() {
  if (current) {
    current.cancelled = true;
    for (const src of current.sources) {
      try {
        src.stop();
      } catch (_) {}
    }
    current = null;
  }
}

/**
 * Fetch /api/tts for `text` and stream-play the PCM.
 * @returns {Promise<void>} resolves when playback finishes (or is cancelled).
 */
export async function playTts(text, { onState } = {}) {
  stopPlayback();

  const handle = { cancelled: false, sources: [] };
  current = handle;

  const ac = audioCtx();
  if (ac.state === "suspended") await ac.resume();

  const state = (s) => onState && onState(s);
  state("synthesizing");

  const res = await fetch("/api/tts", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text }),
  });
  if (!res.ok) throw new Error(`tts failed: ${res.status} ${await res.text()}`);

  const sampleRate = parseInt(res.headers.get("X-Sample-Rate") || "24000", 10);
  const reader = res.body.getReader();

  // Schedule buffers back-to-back. Small lead-in avoids underruns.
  let nextTime = ac.currentTime + 0.1;
  let leftover = new Uint8Array(0); // carry an odd trailing byte across chunks
  let firstScheduled = false;

  while (true) {
    const { done, value } = await reader.read();
    if (handle.cancelled) {
      try {
        reader.cancel();
      } catch (_) {}
      return;
    }
    if (done) break;

    // Merge leftover byte so we always decode whole 16-bit samples.
    const bytes = new Uint8Array(leftover.length + value.length);
    bytes.set(leftover, 0);
    bytes.set(value, leftover.length);

    const usable = bytes.length - (bytes.length % 2);
    leftover = bytes.slice(usable); // keep the odd byte for next chunk

    if (usable === 0) continue;

    const int16 = new Int16Array(bytes.buffer, 0, usable / 2);
    const float32 = new Float32Array(int16.length);
    for (let i = 0; i < int16.length; i++) float32[i] = int16[i] / 32768;

    const buf = ac.createBuffer(1, float32.length, sampleRate);
    buf.copyToChannel(float32, 0);

    const src = ac.createBufferSource();
    src.buffer = buf;
    src.connect(ac.destination);

    // If we fell behind (underrun), restart from "now" instead of in the past.
    if (nextTime < ac.currentTime) nextTime = ac.currentTime + 0.02;
    src.start(nextTime);
    nextTime += buf.duration;
    handle.sources.push(src);

    if (!firstScheduled) {
      firstScheduled = true;
      state("playing");
    }
  }

  // Wait until the last scheduled buffer finishes.
  const waitMs = Math.max(0, (nextTime - ac.currentTime) * 1000);
  await new Promise((resolve) => setTimeout(resolve, waitMs));
  if (!handle.cancelled) state("done");
  if (current === handle) current = null;
}
