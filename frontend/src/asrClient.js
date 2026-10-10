// Client for the backend voice-chat API.
//
// Backend endpoints (see backend/main.py):
//   POST /api/session/start   -> { session_id, asr_session_id }
//   POST /api/chunk?session_id=<id>   body: raw float32 PCM @16k mono -> {language, text}
//   POST /api/finish?session_id=<id>  -> {language, text}
//   POST /api/chat  {session_id, text} -> {reply}

const BASE = "/api";

export async function startSession() {
  const r = await fetch(`${BASE}/session/start`, { method: "POST" });
  if (!r.ok) throw new Error(`start failed: ${r.status} ${await r.text()}`);
  const j = await r.json();
  if (!j.session_id) throw new Error("start returned no session_id");
  return j.session_id;
}

export async function pushChunk(sessionId, float32_16k) {
  const r = await fetch(
    `${BASE}/chunk?session_id=${encodeURIComponent(sessionId)}`,
    {
      method: "POST",
      headers: { "Content-Type": "application/octet-stream" },
      body: float32_16k.buffer,
    }
  );
  if (!r.ok) throw new Error(`chunk failed: ${r.status} ${await r.text()}`);
  return await r.json();
}

export async function finishSession(sessionId) {
  const r = await fetch(
    `${BASE}/finish?session_id=${encodeURIComponent(sessionId)}`,
    { method: "POST" }
  );
  if (!r.ok) throw new Error(`finish failed: ${r.status} ${await r.text()}`);
  return await r.json();
}

export async function transcribeUtterance(sessionId, float32_16k) {
  const r = await fetch(
    `${BASE}/transcribe?session_id=${encodeURIComponent(sessionId)}`,
    {
      method: "POST",
      headers: { "Content-Type": "application/octet-stream" },
      body: float32_16k.buffer,
    }
  );
  if (!r.ok) throw new Error(`transcribe failed: ${r.status} ${await r.text()}`);
  return await r.json();
}

export async function chat(sessionId, text) {
  const r = await fetch(`${BASE}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: sessionId, text }),
  });
  if (!r.ok) throw new Error(`chat failed: ${r.status} ${await r.text()}`);
  return (await r.json()).reply;
}

// Linear resample a Float32Array from srcSr to dstSr.
export function resampleLinear(input, srcSr, dstSr) {
  if (srcSr === dstSr) return input;
  const ratio = dstSr / srcSr;
  const outLen = Math.max(0, Math.round(input.length * ratio));
  const out = new Float32Array(outLen);
  for (let i = 0; i < outLen; i++) {
    const x = i / ratio;
    const x0 = Math.floor(x);
    const x1 = Math.min(x0 + 1, input.length - 1);
    const t = x - x0;
    out[i] = input[x0] * (1 - t) + input[x1] * t;
  }
  return out;
}
