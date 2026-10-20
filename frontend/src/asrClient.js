// Client for the backend voice-chat API.

const BASE = "/api";

async function jsonResponse(response, label) {
  if (!response.ok) throw new Error(`${label} failed: ${response.status} ${await response.text()}`);
  return await response.json();
}

export async function startSession() {
  const response = await fetch(`${BASE}/session/start`, { method: "POST" });
  return await jsonResponse(response, "start");
}

export async function getSession(sessionId) {
  const response = await fetch(`${BASE}/session/${encodeURIComponent(sessionId)}`);
  return await jsonResponse(response, "session state");
}

export async function endSession(sessionId) {
  const response = await fetch(`${BASE}/session/${encodeURIComponent(sessionId)}`, {
    method: "DELETE",
  });
  if (!response.ok) throw new Error(`end session failed: ${response.status} ${await response.text()}`);
}

export async function beginTurn(sessionId, assistantSpeaking = false) {
  const response = await fetch(
    `${BASE}/session/${encodeURIComponent(sessionId)}/turns`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ source: "vad", assistant_speaking: assistantSpeaking }),
    }
  );
  return await jsonResponse(response, "begin turn");
}

export async function evaluateTurn(sessionId, turnId, float32_16k, metadata = {}) {
  const form = new FormData();
  form.append(
    "audio",
    new Blob([float32_16k.buffer], { type: "application/octet-stream" }),
    "utterance.pcm"
  );
  form.append("text", metadata.text || "");
  form.append("language", metadata.language || "");
  form.append("kws_detected", metadata.kws?.detected ? "true" : "false");
  if (metadata.kws?.keyword) form.append("kws_keyword", metadata.kws.keyword);
  if (metadata.kws?.available !== undefined) {
    form.append("kws_available", metadata.kws.available ? "true" : "false");
  }
  const response = await fetch(
    `${BASE}/session/${encodeURIComponent(sessionId)}/turns/${encodeURIComponent(turnId)}/evaluate`,
    { method: "POST", body: form }
  );
  return await jsonResponse(response, "evaluate turn");
}

export async function transcribeUtterance(sessionId, float32_16k) {
  const response = await fetch(
    `${BASE}/transcribe?session_id=${encodeURIComponent(sessionId)}`,
    {
      method: "POST",
      headers: { "Content-Type": "application/octet-stream" },
      body: float32_16k.buffer,
    }
  );
  return await jsonResponse(response, "transcribe");
}

export async function detectKeyword(float32_16k) {
  const response = await fetch(`${BASE}/kws/detect`, {
    method: "POST",
    headers: { "Content-Type": "application/octet-stream" },
    body: float32_16k.buffer,
  });
  return await jsonResponse(response, "KWS detect");
}

export async function touchSession(sessionId, { seconds = null, force = false } = {}) {
  const response = await fetch(`${BASE}/touch`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: sessionId, seconds, force }),
  });
  return await jsonResponse(response, "touch");
}

export async function recordPerformance(sessionId, event) {
  const response = await fetch(
    `${BASE}/session/${encodeURIComponent(sessionId)}/performance`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(event),
    }
  );
  if (!response.ok) throw new Error(`performance log failed: ${response.status}`);
}

export async function getPerformance(sessionId) {
  const response = await fetch(
    `${BASE}/session/${encodeURIComponent(sessionId)}/performance`
  );
  return await jsonResponse(response, "performance log");
}

export async function chat(sessionId, turnId) {
  const response = await fetch(`${BASE}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: sessionId, turn_id: turnId }),
  });
  return await jsonResponse(response, "chat");
}
