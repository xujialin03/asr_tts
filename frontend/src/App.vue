<script setup>
import { nextTick, onBeforeUnmount, ref } from "vue";
import {
  startSession,
  pushChunk,
  finishSession,
  chat,
  resampleLinear,
} from "./asrClient.js";
import { playTts, stopPlayback } from "./ttsPlayer.js";

const TARGET_SR = 16000;
const CHUNK_MS = 500;

const status = ref("idle"); // idle | listening | finishing | error | done
const statusText = ref("未开始");
const language = ref("—");
const asrText = ref(""); // live ASR recognition
const inputText = ref(""); // textarea (editable before sending)
const messages = ref([]); // {role, content} conversation
const sending = ref(false);
const error = ref("");
const logEl = ref(null);

const autoPlay = ref(true); // auto-speak each assistant reply
const ttsState = ref("idle"); // idle | synthesizing | playing | done
const playingIndex = ref(-1); // which message is currently speaking

let audioCtx = null;
let processor = null;
let source = null;
let mediaStream = null;
let sessionId = null;
let running = false;
let buf = new Float32Array(0);
let pushing = false;

function setStatus(s, label) {
  status.value = s;
  statusText.value = label;
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

// Synthesize + stream-play one message's text.
async function speak(index) {
  const msg = messages.value[index];
  if (!msg) return;
  stopPlayback();
  playingIndex.value = index;
  try {
    await playTts(msg.content, {
      onState: (s) => {
        ttsState.value = s;
      },
    });
  } catch (err) {
    error.value = err.message;
    ttsState.value = "idle";
  } finally {
    if (playingIndex.value === index) playingIndex.value = -1;
  }
}

function stopSpeak() {
  stopPlayback();
  playingIndex.value = -1;
  ttsState.value = "idle";
}

// Send text (from ASR or typed) to the LLM and show the reply.
async function sendToLLM(text) {
  const content = (text ?? inputText.value).trim();
  if (!content) return;
  error.value = "";
  sending.value = true;
  messages.value.push({ role: "user", content });
  inputText.value = "";
  scrollLog();
  try {
    const sid = await ensureSession();
    const reply = await chat(sid, content);
    messages.value.push({ role: "assistant", content: reply });
    scrollLog();
    if (autoPlay.value) {
      // Fire-and-forget: playback streams in the background.
      speak(messages.value.length - 1);
    }
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

    running = true;
    setStatus("listening", "识别中…");
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
  setStatus("finishing", "收尾中…");
  await stopPipeline();
  try {
    const j = await finishSession(sessionId);
    language.value = j.language || "—";
    asrText.value = j.text || "";
    setStatus("done", "已停止");
    // Auto-send the recognized text to the LLM.
    if (asrText.value.trim()) await sendToLLM(asrText.value);
  } catch (err) {
    error.value = err.message;
    setStatus("error", "收尾失败");
  } finally {
    buf = new Float32Array(0);
    pushing = false;
  }
}

// ---- File upload path ----
const fileInput = ref(null);
const fileBusy = ref(false);

async function onFilePicked(e) {
  const file = e.target.files?.[0];
  if (!file) return;
  error.value = "";
  asrText.value = "";
  language.value = "—";
  fileBusy.value = true;
  setStatus("listening", "上传识别中…");

  try {
    const arrayBuf = await file.arrayBuffer();
    const decodeCtx = new (window.AudioContext || window.webkitAudioContext)();
    const audioBuf = await decodeCtx.decodeAudioData(arrayBuf);
    await decodeCtx.close();

    const ch = audioBuf.numberOfChannels;
    const len = audioBuf.length;
    const mono = new Float32Array(len);
    for (let c = 0; c < ch; c++) {
      const data = audioBuf.getChannelData(c);
      for (let i = 0; i < len; i++) mono[i] += data[i] / ch;
    }
    const pcm16k = resampleLinear(mono, audioBuf.sampleRate, TARGET_SR);

    const sid = await ensureSession();
    const chunkSamples = Math.round(TARGET_SR * (CHUNK_MS / 1000));
    for (let off = 0; off < pcm16k.length; off += chunkSamples) {
      const chunk = pcm16k.slice(off, off + chunkSamples);
      const j = await pushChunk(sid, chunk);
      language.value = j.language || "—";
      asrText.value = j.text || "";
    }
    const j = await finishSession(sid);
    language.value = j.language || "—";
    asrText.value = j.text || "";
    setStatus("done", "识别完成");
    if (asrText.value.trim()) await sendToLLM(asrText.value);
  } catch (err) {
    error.value = err.message;
    setStatus("error", "识别失败");
  } finally {
    fileBusy.value = false;
    if (fileInput.value) fileInput.value.value = "";
  }
}

function clearAll() {
  messages.value = [];
  asrText.value = "";
  inputText.value = "";
  language.value = "—";
  error.value = "";
}

onBeforeUnmount(() => {
  running = false;
  stopPipeline();
  stopPlayback();
});
</script>

<template>
  <div class="wrap">
    <div class="card">
      <h1>语音助手 · ASR → LLM → TTS</h1>

      <div class="row">
        <span class="pill" :class="status">{{ statusText }}</span>
        <span class="pill">语言: {{ language }}</span>
        <span v-if="ttsState !== 'idle'" class="pill" :class="ttsState === 'playing' ? 'listening' : 'finishing'">
          🔊 {{ ttsState === "synthesizing" ? "合成中…" : ttsState === "playing" ? "播放中…" : "播放完成" }}
        </span>
      </div>

      <div class="row">
        <button class="primary" :disabled="running || fileBusy" @click="startRecording">
          🎙 开始录音
        </button>
        <button class="danger" :disabled="!running" @click="stopRecording">
          ⏹ 停止
        </button>
        <label class="filebtn">
          📁 上传音频
          <input
            ref="fileInput"
            type="file"
            accept="audio/*"
            :disabled="running || fileBusy"
            @change="onFilePicked"
          />
        </label>
        <button @click="clearAll">清空</button>
        <label class="toggle">
          <input type="checkbox" v-model="autoPlay" />
          自动朗读回复
        </label>
        <button class="danger" :disabled="playingIndex === -1" @click="stopSpeak">
          ⏹ 停止播放
        </button>
      </div>

      <div class="panel">
        <div class="label">识别结果（可编辑后发送）</div>
        <textarea
          id="asr"
          v-model="inputText"
          rows="3"
          placeholder="录音停止后识别结果会自动填入，也可以直接打字…"
        ></textarea>
        <div class="row" style="margin-top: 8px">
          <button class="primary" :disabled="sending || !inputText.trim()" @click="sendToLLM()">
            {{ sending ? "思考中…" : "发送 → LLM" }}
          </button>
          <span v-if="asrText && asrText !== inputText" class="pill">
            原始识别: {{ asrText }}
          </span>
        </div>
      </div>

      <div class="panel chatpanel">
        <div class="label">对话</div>
        <div id="log" ref="logEl">
          <div v-if="!messages.length" class="empty">
            录音或打字后发送，LLM 会以「台湾知心朋友」角色回复…
          </div>
          <div
            v-for="(m, i) in messages"
            :key="i"
            class="msg"
            :class="m.role"
          >
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
  min-height: 100vh;
  padding: 16px;
  box-sizing: border-box;
  display: flex;
}
.card {
  width: 100%;
  max-width: 860px;
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
button.primary {
  border-color: rgba(5, 150, 105, 0.35);
  background: rgba(5, 150, 105, 0.1);
}
button.danger {
  border-color: rgba(225, 29, 72, 0.35);
  background: rgba(225, 29, 72, 0.1);
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
.filebtn {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  border: 1px solid #e5e7eb;
  border-radius: 12px;
  padding: 10px 14px;
  cursor: pointer;
  background: #f8fafc;
  font-weight: 700;
  color: #0f172a;
}
.filebtn input {
  display: none;
}
.panel {
  border: 1px solid #e5e7eb;
  border-radius: 12px;
  background: #fff;
  padding: 12px;
}
.chatpanel {
  flex: 1;
  display: flex;
  flex-direction: column;
  min-height: 240px;
}
.label {
  color: #5b6472;
  font-size: 12px;
  margin-bottom: 6px;
}
#asr {
  width: 100%;
  resize: vertical;
  border: 1px solid #e5e7eb;
  border-radius: 8px;
  padding: 10px;
  font-size: 15px;
  font-family: inherit;
  box-sizing: border-box;
}
#log {
  flex: 1;
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
.toggle {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  font-size: 13px;
  color: #5b6472;
  cursor: pointer;
  user-select: none;
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
</style>
