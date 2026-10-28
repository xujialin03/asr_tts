// TTS manager: buffers text, splits long replies into short segments, and
// synthesizes them serially so playback is gapless and interruption is cheap.
//
// Why split? A single long reply means: (1) a slow first chunk from the remote
// TTS, (2) the whole utterance blocks on one request, and (3) barge-in has to
// abort the entire clip. The backend also holds a global TTS_LOCK for the whole
// stream, so concurrent requests get 409 — segments MUST be requested serially.
//
// The win: while segment N is *playing*, segment N+1 is already being fetched
// and scheduled, so the pipeline stays full and a barge-in only drops the tail.
//
// The backend /api/tts returns raw 16-bit signed little-endian PCM, mono, at the
// sample rate advertised in the `X-Sample-Rate` response header. We read each
// segment's body as a stream, decode it, and schedule AudioBufferSources
// back-to-back on a shared AudioContext timeline so the segments stitch together.

let ctx = null;
let current = null; // active session handle, so a new one can cancel the old

// Segmentation tuning. Each segment is an independent zero-shot synthesis, so
// every boundary resets prosody and can drift the timbre slightly. We therefore
// only split when a reply is long enough that the latency/latency tradeoff is
// worth the seams: replies under NO_SPLIT_UNDER_CHARS play as one clip (most
// stable voice), and long replies split at a larger cap so there are fewer
// boundaries. MIN avoids tiny fragments ("好的。" style).
const NO_SPLIT_UNDER_CHARS = 80;
const MAX_SEGMENT_CHARS = 70;
const SOFT_BREAKS = ["。", "！", "？", "；", "!", "?", ";", "\n"];

// A segment whose headers take longer than this is flagged as slow so it shows up
// in the timing log / perf records rather than hiding inside "synthesizing".
const SLOW_SEGMENT_MS = 1500;

// The backend holds a global TTS_LOCK (upstream is single-concurrency), so a
// request can get 409 if another tab or a race grabs it. One short retry usually
// clears it; a genuine persistent 409 still surfaces as an error.
const TTS_BUSY_RETRY_DELAY_MS = 400;

function audioCtx() {
  if (!ctx) ctx = new (window.AudioContext || window.webkitAudioContext)();
  return ctx;
}

// Split one reply into short, speakable segments. Strategy: first cut at
// sentence-final punctuation (so boundaries never fall mid-sentence), then
// greedily pack consecutive sentences into segments up to maxChars. Keeps
// punctuation attached to the text before it. Never returns empty strings.
export function splitSegments(text, maxChars = MAX_SEGMENT_CHARS) {
  const clean = (text || "").replace(/\s+/g, " ").trim();
  if (!clean) return [];
  // Short replies stay as one clip: one synthesis = one continuous voice, no
  // boundary prosody resets. Only long replies pay the seam cost for latency.
  if (clean.length <= NO_SPLIT_UNDER_CHARS) return [clean];

  // 1. Cut into sentences at soft breaks (punctuation stays with its sentence).
  const sentences = [];
  let buf = "";
  for (const ch of clean) {
    buf += ch;
    if (SOFT_BREAKS.includes(ch)) {
      sentences.push(buf);
      buf = "";
    }
  }
  if (buf.trim()) sentences.push(buf);

  // 2. Greedily pack sentences into segments up to maxChars. A single sentence
  //    longer than maxChars is emitted whole (we never cut mid-sentence).
  const segments = [];
  let seg = "";
  for (const sentence of sentences) {
    if (seg && (seg + sentence).trim().length > maxChars) {
      segments.push(seg.trim());
      seg = "";
    }
    seg += sentence;
  }
  if (seg.trim()) segments.push(seg.trim());
  return segments;
}

// Stop the whole session: cancel scheduling, abort the in-flight fetch, and drop
// any not-yet-requested segments.
export function stopPlayback() {
  if (current) {
    current.cancelled = true;
    try {
      current.reader?.cancel();
    } catch (_) {}
    try {
      current.abort?.abort();
    } catch (_) {}
    for (const src of current.sources) {
      try {
        src.stop();
      } catch (_) {}
    }
    current = null;
  }
}

// Fetch + stream + schedule ONE segment onto the shared cursor. Mutates handle
// (sources, reader, abort) and returns nothing; throws on non-abort errors.
async function streamSegment(handle, text, segmentIndex, { instruction, onDuration, onTiming }) {
  const ac = audioCtx();
  const startedAt = performance.now();

  handle.abort = new AbortController();
  let res = await fetch("/api/tts", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text, instruction }),
    signal: handle.abort.signal,
  });
  if (res.status === 409 && !handle.cancelled) {
    // Upstream TTS is busy (backend lock held elsewhere). Back off once.
    console.info(`[Voice timing] TTS seg#${segmentIndex} busy (409), retrying in ${TTS_BUSY_RETRY_DELAY_MS}ms`);
    onTiming?.("tts_busy_retry", TTS_BUSY_RETRY_DELAY_MS, { segmentIndex });
    await new Promise((resolve) => setTimeout(resolve, TTS_BUSY_RETRY_DELAY_MS));
    if (!handle.cancelled) {
      res = await fetch("/api/tts", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text, instruction }),
        signal: handle.abort.signal,
      });
    }
  }
  if (!res.ok) throw new Error(`tts failed: ${res.status} ${await res.text()}`);

  const requestId = res.headers.get("X-TTS-Request-ID") || "?";
  const headersElapsedMs = performance.now() - startedAt;
  console.info(
    `[Voice timing] TTS #${requestId} seg#${segmentIndex} headers ${Math.round(headersElapsedMs)}ms len=${text.length}`
  );
  if (headersElapsedMs > SLOW_SEGMENT_MS) {
    console.warn(
      `[Voice timing] TTS #${requestId} seg#${segmentIndex} SLOW headers ${Math.round(headersElapsedMs)}ms`
    );
    onTiming?.("tts_slow", headersElapsedMs, { requestId, segmentIndex, textLength: text.length });
  }
  onTiming?.("tts_headers", headersElapsedMs, { requestId, segmentIndex });

  const sampleRate = parseInt(res.headers.get("X-Sample-Rate") || "24000", 10);
  const reader = res.body.getReader();
  handle.reader = reader;

  let leftover = new Uint8Array(0); // carry an odd trailing byte across chunks
  let firstScheduled = false;

  while (true) {
    const { done, value } = await reader.read();
    if (handle.cancelled) return;
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
    if (handle.cursor < ac.currentTime) handle.cursor = ac.currentTime + 0.02;
    src.start(handle.cursor);
    handle.cursor += buf.duration;
    handle.totalScheduled += buf.duration;
    handle.sources.push(src);
    onDuration?.({
      totalSeconds: handle.totalScheduled,
      remainingSeconds: Math.max(0, handle.cursor - ac.currentTime),
    });

    if (!firstScheduled) {
      firstScheduled = true;
      const firstChunkElapsedMs = performance.now() - startedAt;
      console.info(
        `[Voice timing] TTS #${requestId} seg#${segmentIndex} first-chunk ${Math.round(firstChunkElapsedMs)}ms`
      );
      // Only the very first segment's first chunk is the user-perceived latency.
      if (segmentIndex === 0) {
        onTiming?.("tts_first_chunk", firstChunkElapsedMs, { requestId, segmentIndex });
      }
    }
  }
}

/**
 * Synthesize + stream-play `text`, split into short segments requested serially.
 * `instruction` is the per-reply voice direction produced by the assistant agent.
 * @returns {Promise<void>} resolves when playback finishes (or is cancelled).
 */
export async function playTts(text, { instruction = "", onState, onDuration, onTiming } = {}) {
  stopPlayback();

  const handle = {
    cancelled: false,
    sources: [],
    abort: null,
    reader: null,
    cursor: null,
    totalScheduled: 0,
  };
  current = handle;

  const ac = audioCtx();
  if (ac.state === "suspended") await ac.resume();
  // Shared timeline cursor across segments; small lead-in avoids underruns.
  handle.cursor = ac.currentTime + 0.1;

  const state = (s) => onState && onState(s);
  state("synthesizing");

  const segments = splitSegments(text);
  if (!segments.length) {
    if (current === handle) current = null;
    return;
  }
  if (segments.length > 1) {
    console.info(`[Voice timing] TTS split into ${segments.length} segments (len=${text.length})`);
  }

  try {
    for (let i = 0; i < segments.length; i++) {
      if (handle.cancelled) return;
      if (i > 0) state("playing"); // first segment flips to playing on its first chunk
      await streamSegment(handle, segments[i], i, { instruction, onDuration, onTiming });
    }

    // Wait until the last scheduled buffer finishes.
    const waitMs = Math.max(0, (handle.cursor - ac.currentTime) * 1000);
    await new Promise((resolve) => setTimeout(resolve, waitMs));
    if (!handle.cancelled) state("done");
  } catch (err) {
    if (err.name === "AbortError" || handle.cancelled) return;
    throw err;
  } finally {
    if (current === handle) current = null;
  }
}
