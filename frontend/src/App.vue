<script setup>
import { nextTick, onBeforeUnmount, onMounted, ref } from "vue";
import {
  startSession,
  pushChunk,
  finishSession,
  transcribeUtterance,
  decide,
  touchSession,
  chat,
  resampleLinear,
} from "./asrClient.js";
import { playTts, stopPlayback } from "./ttsPlayer.js";
import { startVad, stopVad, releaseSharedStream } from "./vad.js";

const TARGET_SR = 16000;
const CHUNK_MS = 500;
const POST_REPLY_ACTIVE_SECONDS = 20;

const status = ref("idle"); // idle | listening | finishing | error | done
const statusText = ref("未开始");
const language = ref("—");
const asrText = ref(""); // live ASR recognition
const messages = ref([]); // {role, content} conversation
const sending = ref(false);
const error = ref("");
const gateNotice = ref("");
const aiActive = ref(false);
const activeRemaining = ref(0);
const logEl = ref(null);
const asrEl = ref(null);

const autoPlay = ref(true); // auto-speak each assistant reply
const ttsState = ref("idle"); // idle | synthesizing | playing | done
const playingIndex = ref(-1); // which message is currently speaking
const vadEnabled = ref(false); // hands-free Silero VAD (opt-in; headphones recommended)
const vadProb = ref(0);
const vadActive = ref(false);
const vadSilenceMs = ref(0);
const vadSegmentMs = ref(0);

let audioCtx = null;
let processor = null;
let source = null;
let mediaStream = null;
let sessionId = null;
let running = false;
let buf = new Float32Array(0);
let pushing = false;
let holdActive = false; // pointer/key currently held for push-to-talk
let wantRecording = false; // unified record intent (button OR VAD)

function setStatus(s, label) {
  status.value = s;
  statusText.value = label;
}

function updateActive(info) {
  aiActive.value = !!info?.active;
  activeRemaining.value = Number(info?.active_remaining || 0);
}

let lastPlaybackTouchAt = 0;
let lastPlaybackExtension = 0;
async function extendActiveForPlayback(remainingSeconds, force = false) {
  if (!sessionId) return;
  const extension = Math.ceil(remainingSeconds + POST_REPLY_ACTIVE_SECONDS);
  const now = Date.now();
  // TTS yields many small chunks. Only sync when the estimated deadline moves
  // meaningfully, or at most once every two seconds.
  if (!force && extension <= lastPlaybackExtension + 1 && now - lastPlaybackTouchAt < 2000) return;
  lastPlaybackTouchAt = now;
  lastPlaybackExtension = extension;
  try {
    const info = await touchSession(sessionId, { seconds: extension, force: true });
    updateActive(info);
  } catch (_) {}
}

function concatFloat32(a, b) {
  const out = new Float32Array(a.length + b.length);
  out.set(a, 0);
  out.set(b, a.length);
  return out;
}

async function ensureSession() {
  if (!sessionId) sessionId = await startSession();
  return sessionId;
}

async function scrollLog() {
  await nextTick();
  if (logEl.value) logEl.value.scrollTop = logEl.value.scrollHeight;
}

async function scrollAsr() {
  await nextTick();
  if (asrEl.value) asrEl.value.scrollTop = asrEl.value.scrollHeight;
}

// Stop any TTS playback.
function stopSpeak() {
  stopPlayback();
  playingIndex.value = -1;
  ttsState.value = "idle";
}

// True only while the assistant is actively synthesizing/playing (not "done").
function isSpeaking() {
  return (
    playingIndex.value !== -1 ||
    ttsState.value === "playing" ||
    ttsState.value === "synthesizing"
  );
}

// Synthesize + stream-play one message's text.
async function speak(index) {
  const msg = messages.value[index];
  if (!msg) return;
  stopPlayback();
  playingIndex.value = index;
  lastPlaybackTouchAt = 0;
  lastPlaybackExtension = 0;
  // Keep the session active while synthesis is waiting for its first chunk.
  extendActiveForPlayback(0, true);
  try {
    await playTts(msg.content, {
      onState: (s) => {
        ttsState.value = s;
      },
      onDuration: ({ remainingSeconds }) => {
        extendActiveForPlayback(remainingSeconds);
      },
    });
  } catch (err) {
    error.value = err.message;
    ttsState.value = "idle";
  } finally {
    // Start a fresh post-reply window from actual playback completion/cancel.
    extendActiveForPlayback(0, true);
    if (playingIndex.value === index) {
      playingIndex.value = -1;
      ttsState.value = "idle";
    }
  }
}

// Send recognized text to the LLM and show + speak the reply.
async function sendToLLM(text) {
  const content = (text || "").trim();
  if (!content) return;
  error.value = "";
  gateNotice.value = "";
  sending.value = true;
  messages.value.push({ role: "user", content });
  // Clear the live-recognition box once the text has been sent.
  asrText.value = "";
  scrollLog();
  try {
    const sid = await ensureSession();
    const result = await chat(sid, content);
    const reply = result.reply;
    updateActive(result);
    messages.value.push({ role: "assistant", content: reply });
    scrollLog();
    if (autoPlay.value) speak(messages.value.length - 1);
  } catch (err) {
    error.value = err.message;
    setStatus("error", "LLM 调用失败");
  } finally {
    sending.value = false;
  }
}

async function stopPipeline() {
  try {
    if (processor) {
      processor.disconnect();
      processor.onaudioprocess = null;
    }
    if (source) source.disconnect();
    if (audioCtx) await audioCtx.close();
    if (mediaStream) mediaStream.getTracks().forEach((t) => t.stop());
  } catch (_) {}
  processor = source = audioCtx = mediaStream = null;
}

async function pump() {
  if (pushing) return;
  pushing = true;
  const chunkSamples = Math.round(TARGET_SR * (CHUNK_MS / 1000));
  try {
    while (running && buf.length >= chunkSamples) {
      const chunk = buf.slice(0, chunkSamples);
      buf = buf.slice(chunkSamples);
      const j = await pushChunk(sessionId, chunk);
      language.value = j.language || "—";
      asrText.value = j.text || "";
      scrollAsr();
    }
  } catch (err) {
    error.value = err.message;
    setStatus("error", "后端错误");
    running = false;
  } finally {
    pushing = false;
  }
}

async function startRecording() {
  if (running) return;
  // Barge-in: if the assistant is speaking, cut it off before we record, so the
  // mic doesn't pick up the TTS output.
  if (playingIndex.value !== -1 || ttsState.value !== "idle") stopSpeak();
  error.value = "";
  asrText.value = "";
  language.value = "—";
  buf = new Float32Array(0);

  try {
    setStatus("listening", "启动中…");
    await ensureSession();

    mediaStream = await navigator.mediaDevices.getUserMedia({
      audio: {
        channelCount: 1,
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true,
      },
      video: false,
    });

    audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    source = audioCtx.createMediaStreamSource(mediaStream);
    processor = audioCtx.createScriptProcessor(4096, 1, 1);

    processor.onaudioprocess = (e) => {
      if (!running) return;
      const input = e.inputBuffer.getChannelData(0);
      const resampled = resampleLinear(input, audioCtx.sampleRate, TARGET_SR);
      buf = concatFloat32(buf, resampled);
      if (!pushing) pump();
    };

    source.connect(processor);
    processor.connect(audioCtx.destination);

    // The user may have released the button while the stream was resolving.
    if (!wantRecording) {
      await stopPipeline();
      return;
    }

    running = true;
    setStatus("listening", "正在说话…");
  } catch (err) {
    error.value = err.message;
    setStatus("error", "启动失败");
    running = false;
    await stopPipeline();
  }
}

async function stopRecording() {
  if (!running) return;
  running = false;
  setStatus("finishing", "识别中…");
  await stopPipeline();
  try {
    // Flush any tail audio left in the buffer (pump only sends full 500ms
    // chunks, so the last partial chunk would otherwise be dropped).
    while (pushing) await new Promise((r) => setTimeout(r, 50));
    if (buf.length > 0) {
      const j = await pushChunk(sessionId, buf);
      language.value = j.language || "—";
      asrText.value = j.text || "";
      buf = new Float32Array(0);
    }
    const j = await finishSession(sessionId);
    language.value = j.language || "—";
    asrText.value = j.text || "";
    setStatus("done", "已识别");
    if (asrText.value.trim()) await sendToLLM(asrText.value);
  } catch (err) {
    error.value = err.message;
    setStatus("error", "收尾失败");
  } finally {
    buf = new Float32Array(0);
    pushing = false;
  }
}

// ---- Push-to-talk: press to start, release to finish ----
function onHoldStart() {
  if (vadEnabled.value) {
    // In hands-free mode, VAD already owns the microphone continuously. Pressing
    // the button is only used as an emergency barge-in to stop playback.
    if (isSpeaking()) stopSpeak();
    return;
  }
  if (holdActive) return;
  holdActive = true;
  wantRecording = true;
  startRecording();
}
function onHoldEnd() {
  if (vadEnabled.value) return;
  if (!holdActive) return;
  holdActive = false;
  wantRecording = false;
  stopRecording();
}

// ---- Hands-free VAD ----
let vadSegmentActive = false;
let vadProcessing = false;

async function touchActiveWindow() {
  try {
    const sid = await ensureSession();
    const info = await touchSession(sid);
    updateActive(info);
  } catch (_) {}
}

async function handleVadAudio(audio) {
  if (vadProcessing || !audio?.length) return;
  vadProcessing = true;
  error.value = "";
  setStatus("finishing", "识别中…");
  try {
    const sid = await ensureSession();
    const j = await transcribeUtterance(sid, audio);
    language.value = j.language || "—";
    asrText.value = j.text || "";
    setStatus("done", "已识别");
    const text = asrText.value.trim();
    if (!text) return;
    const decision = await decide(sid, text, "vad");
    updateActive(decision);
    if (decision.should_reply) {
      gateNotice.value = "";
      await sendToLLM(text);
    } else {
      setStatus("idle", "未触发回复");
      gateNotice.value = decision.reason || "未触发回复";
    }
  } catch (err) {
    error.value = err.message;
    asrText.value = "识别失败：远程 ASR 服务异常。";
    scrollAsr();
    setStatus("error", "识别失败");
  } finally {
    vadProcessing = false;
  }
}

async function onVadToggle() {
  if (vadEnabled.value) {
    try {
      await startVad({
        onSpeechStart: () => {
          // In hands-free mode, VAD owns recording. We only update UI and stop
          // playback if the user starts talking while assistant is speaking.
          if (isSpeaking()) stopSpeak();
          vadSegmentActive = true;
          vadActive.value = true;
          touchActiveWindow();
          asrText.value = "";
          language.value = "—";
          setStatus("listening", "正在说话…");
        },
        onSpeechEnd: (audio) => {
          if (!vadSegmentActive) return;
          vadSegmentActive = false;
          vadActive.value = false;
          handleVadAudio(audio);
        },
        onFrame: (prob, active, silenceMs, segmentMs) => {
          vadProb.value = prob;
          vadActive.value = active;
          vadSilenceMs.value = silenceMs || 0;
          vadSegmentMs.value = segmentMs || 0;
        },
      });
    } catch (err) {
      error.value = "VAD 启动失败: " + err.message;
      vadEnabled.value = false;
      stopVad();
      releaseSharedStream();
    }
  } else {
    vadSegmentActive = false;
    vadActive.value = false;
    vadProb.value = 0;
    vadSilenceMs.value = 0;
    vadSegmentMs.value = 0;
    stopVad();
    releaseSharedStream();
    if (!running) setStatus("idle", "未开始");
  }
}

// Keyboard hold (Space) mirrors the button for desktop users.
function onKey(e) {
  const tag = (e.target.tagName || "").toLowerCase();
  if (tag === "input" || tag === "textarea") return;
  if (e.code === "Space") {
    e.preventDefault();
    if (e.type === "keydown" && !e.repeat) onHoldStart();
    else if (e.type === "keyup") onHoldEnd();
  } else if (e.code === "Escape") {
    e.preventDefault();
    stopSpeak();
  }
}

function clearAll() {
  messages.value = [];
  asrText.value = "";
  language.value = "—";
  error.value = "";
  gateNotice.value = "";
  aiActive.value = false;
  activeRemaining.value = 0;
}

let activeTimer = null;

onMounted(() => {
  window.addEventListener("keydown", onKey);
  window.addEventListener("keyup", onKey);
  activeTimer = window.setInterval(() => {
    if (activeRemaining.value > 0) {
      activeRemaining.value = Math.max(0, activeRemaining.value - 1);
      if (activeRemaining.value === 0) aiActive.value = false;
    }
  }, 1000);
});
onBeforeUnmount(() => {
  window.removeEventListener("keydown", onKey);
  window.removeEventListener("keyup", onKey);
  if (activeTimer) window.clearInterval(activeTimer);
  running = false;
  stopPipeline();
  stopPlayback();
  stopVad();
  releaseSharedStream();
});
</script>

<template>
  <div class="wrap">
    <div class="card">
      <h1>语音助手 · ASR → LLM → TTS</h1>

      <div class="row">
        <span class="pill" :class="status">{{ statusText }}</span>
        <span class="pill">语言: {{ language }}</span>
        <span class="pill" :class="aiActive ? 'listening' : ''">
          {{ aiActive ? `AI 已激活 · ${Math.round(activeRemaining)}s` : "AI 未激活" }}
        </span>
        <span
          v-if="ttsState !== 'idle'"
          class="pill"
          :class="ttsState === 'playing' ? 'listening' : 'finishing'"
        >
          🔊 {{ ttsState === "synthesizing" ? "合成中…" : ttsState === "playing" ? "播放中…" : "播放完成" }}
        </span>
      </div>

      <div class="ptt-area">
        <button
          class="ptt"
          :class="{ recording: running }"
          @pointerdown.prevent="onHoldStart"
          @pointerup.prevent="onHoldEnd"
          @pointerleave="onHoldEnd"
          @pointercancel="onHoldEnd"
          @contextmenu.prevent
        >
          <span class="ptt-icon">{{ running ? "🔴" : vadEnabled ? "👂" : "🎙" }}</span>
          <span class="ptt-label">{{ running ? "松开结束" : vadEnabled ? "正在监听" : "按住说话" }}</span>
        </button>
        <div class="asr-col">
          <textarea
            ref="asrEl"
            id="live-asr"
            readonly
            rows="3"
            :value="asrText"
            placeholder="实时识别内容会显示在这里…"
          ></textarea>
          <div class="hint">
            {{ vadEnabled ? "免手持模式会持续监听麦克风，Mac 显示占用是正常的 · 说话自动识别，停顿后自动回复" : "按住说话，松开即识别并回复 · 播放中按住可打断 · 键盘长按空格" }}
          </div>
        </div>
      </div>

      <div class="row">
        <button @click="clearAll">清空对话</button>
        <label class="toggle">
          <input type="checkbox" v-model="autoPlay" />
          自动朗读回复
        </label>
        <label class="toggle" title="免手持：检测到说话自动开始录音，停顿自动结束。外放时 TTS 可能被误识别，建议戴耳机。">
          <input type="checkbox" v-model="vadEnabled" @change="onVadToggle" />
          免手持语音（Silero VAD）
        </label>
        <span v-if="vadEnabled" class="pill warn">建议戴耳机，避免外放回声误触发</span>
        <span v-if="vadEnabled" class="pill" :class="vadActive ? 'listening' : ''">
          VAD {{ vadActive ? "说话中" : "监听中" }} · p={{ vadProb.toFixed(2) }} · 静音{{ Math.round(vadSilenceMs) }}ms · 段{{ Math.round(vadSegmentMs) }}ms
        </span>
      </div>

      <div v-if="gateNotice" class="notice">{{ gateNotice }}</div>

      <div class="panel chatpanel">
        <div class="label">对话</div>
        <div id="log" ref="logEl">
          <div v-if="!messages.length" class="empty">
            按住下方按钮说话，LLM 会以「台湾知心朋友」角色回复…
          </div>
          <div v-for="(m, i) in messages" :key="i" class="msg" :class="m.role">
            <span class="who">{{ m.role === "user" ? "我" : "她" }}</span>
            <span class="body">{{ m.content }}</span>
            <button
              v-if="m.role === 'assistant'"
              class="playbtn"
              :class="{ active: playingIndex === i }"
              :disabled="playingIndex !== -1 && playingIndex !== i"
              :title="playingIndex === i ? '播放中…' : '播放这条回复'"
              @click="speak(i)"
            >
              {{ playingIndex === i ? "🔊" : "▶" }}
            </button>
          </div>
        </div>
      </div>

      <div v-if="error" class="err">⚠ {{ error }}</div>
    </div>
  </div>
</template>

<style scoped>
.wrap {
  height: 100vh;
  padding: 16px;
  box-sizing: border-box;
  display: flex;
}
.card {
  width: 100%;
  max-width: 860px;
  height: 100%;
  margin: 0 auto;
  background: #fff;
  border: 1px solid #e5e7eb;
  border-radius: 14px;
  padding: 16px;
  box-sizing: border-box;
  box-shadow: 0 10px 30px rgba(0, 0, 0, 0.06);
  display: flex;
  flex-direction: column;
  gap: 12px;
  overflow: hidden;
}
h1 {
  font-size: 18px;
  margin: 0;
}
.row {
  display: flex;
  gap: 12px;
  align-items: center;
  flex-wrap: wrap;
}
button {
  border: 1px solid #e5e7eb;
  border-radius: 12px;
  padding: 10px 14px;
  cursor: pointer;
  color: #0f172a;
  background: #f8fafc;
  font-weight: 700;
}
button:hover {
  background: #f1f5f9;
}
button:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}
.pill {
  font-size: 12px;
  padding: 6px 10px;
  border-radius: 999px;
  border: 1px solid #e5e7eb;
  color: #5b6472;
  background: #f8fafc;
}
.pill.listening {
  color: #065f46;
  border-color: rgba(5, 150, 105, 0.35);
  background: rgba(5, 150, 105, 0.1);
}
.pill.finishing {
  color: #92400e;
  border-color: rgba(217, 119, 6, 0.35);
  background: rgba(217, 119, 6, 0.1);
}
.pill.error {
  color: #9f1239;
  border-color: rgba(225, 29, 72, 0.35);
  background: rgba(225, 29, 72, 0.1);
}
.ptt-area {
  display: flex;
  align-items: center;
  gap: 16px;
  padding: 4px 0;
}
.ptt {
  flex: none;
  display: inline-flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 4px;
  width: 110px;
  height: 110px;
  border-radius: 50%;
  border: 2px solid rgba(5, 150, 105, 0.35);
  background: rgba(5, 150, 105, 0.1);
  color: #065f46;
  font-weight: 800;
  user-select: none;
  touch-action: none;
  transition: transform 0.08s ease, background 0.15s ease;
}
.ptt:hover {
  background: rgba(5, 150, 105, 0.16);
}
.ptt:active {
  transform: scale(0.97);
}
.ptt.recording {
  border-color: rgba(225, 29, 72, 0.5);
  background: rgba(225, 29, 72, 0.12);
  color: #9f1239;
}
.ptt-icon {
  font-size: 30px;
}
.ptt-label {
  font-size: 13px;
}
.asr-col {
  flex: 1;
  min-width: 0;
  display: flex;
  flex-direction: column;
  gap: 6px;
}
.hint {
  font-size: 12px;
  color: #94a3b8;
  text-align: center;
}
#live-asr {
  width: 100%;
  resize: vertical;
  border: 1px solid #e5e7eb;
  border-radius: 10px;
  padding: 10px 12px;
  font-size: 15px;
  line-height: 1.6;
  font-family: inherit;
  box-sizing: border-box;
  background: #f8fafc;
  color: #0f172a;
}
#live-asr::placeholder {
  color: #94a3b8;
}
.hint kbd {
  background: #f1f5f9;
  border: 1px solid #e5e7eb;
  border-radius: 5px;
  padding: 1px 6px;
  font-family: ui-monospace, Menlo, monospace;
  font-size: 11px;
}
.toggle {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  font-size: 13px;
  color: #5b6472;
  cursor: pointer;
  user-select: none;
}
.panel {
  border: 1px solid #e5e7eb;
  border-radius: 12px;
  background: #fff;
  padding: 12px;
}
.chatpanel {
  flex: 1;
  min-height: 0;
  display: flex;
  flex-direction: column;
}
.label {
  color: #5b6472;
  font-size: 12px;
  margin-bottom: 6px;
}
#log {
  flex: 1;
  min-height: 0;
  overflow-y: auto;
  display: flex;
  flex-direction: column;
  gap: 10px;
}
.empty {
  color: #94a3b8;
}
.msg {
  display: flex;
  gap: 8px;
  align-items: flex-start;
}
.msg .who {
  flex: none;
  font-size: 12px;
  font-weight: 700;
  padding: 2px 8px;
  border-radius: 8px;
  margin-top: 2px;
}
.msg.user .who {
  background: rgba(5, 150, 105, 0.12);
  color: #065f46;
}
.msg.assistant .who {
  background: rgba(59, 130, 246, 0.12);
  color: #1e40af;
}
.msg .body {
  white-space: pre-wrap;
  line-height: 1.6;
  font-size: 15px;
  flex: 1;
}
.playbtn {
  flex: none;
  border: 1px solid #e5e7eb;
  border-radius: 8px;
  width: 30px;
  height: 30px;
  padding: 0;
  cursor: pointer;
  background: #f8fafc;
  font-size: 14px;
  line-height: 1;
}
.playbtn:hover {
  background: #f1f5f9;
}
.playbtn.active {
  border-color: rgba(59, 130, 246, 0.4);
  background: rgba(59, 130, 246, 0.12);
}
.playbtn:disabled {
  opacity: 0.4;
  cursor: not-allowed;
}
.err {
  color: #9f1239;
  font-size: 13px;
}
.notice {
  color: #64748b;
  font-size: 13px;
}
</style>
