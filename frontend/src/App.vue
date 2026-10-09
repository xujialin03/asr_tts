<script setup>
import { onBeforeUnmount, ref } from "vue";
import {
  startSession,
  pushChunk,
  finishSession,
  resampleLinear,
} from "./asrClient.js";

const TARGET_SR = 16000; // remote ASR expects 16 kHz mono float32
const CHUNK_MS = 500; // send a chunk every 500 ms

const status = ref("idle"); // idle | listening | finishing | error | done
const statusText = ref("未开始");
const language = ref("—");
const text = ref("");
const error = ref("");

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

// Sequentially drain buffered audio to the backend.
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
      text.value = j.text || "";
    }
  } catch (err) {
    setStatus("error", "后端错误");
    error.value = err.message;
    running = false;
  } finally {
    pushing = false;
  }
}

async function startRecording() {
  if (running) return;
  error.value = "";
  text.value = "";
  language.value = "—";
  buf = new Float32Array(0);

  try {
    setStatus("listening", "启动中…");
    sessionId = await startSession();

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
    sessionId = null;
    await stopPipeline();
  }
}

async function stopRecording() {
  if (!running) return;
  running = false;
  setStatus("finishing", "收尾中…");
  await stopPipeline();
  try {
    if (sessionId) {
      const j = await finishSession(sessionId);
      language.value = j.language || "—";
      text.value = j.text || "";
    }
    setStatus("done", "已停止");
  } catch (err) {
    error.value = err.message;
    setStatus("error", "收尾失败");
  } finally {
    sessionId = null;
    buf = new Float32Array(0);
    pushing = false;
  }
}

// ---- File upload path: decode -> resample -> stream -> finish ----
const fileInput = ref(null);
const fileBusy = ref(false);

async function onFilePicked(e) {
  const file = e.target.files?.[0];
  if (!file) return;
  error.value = "";
  text.value = "";
  language.value = "—";
  fileBusy.value = true;
  setStatus("listening", "上传识别中…");

  try {
    const arrayBuf = await file.arrayBuffer();
    const decodeCtx = new (window.AudioContext || window.webkitAudioContext)();
    const audioBuf = await decodeCtx.decodeAudioData(arrayBuf);
    await decodeCtx.close();

    // mix to mono
    const ch = audioBuf.numberOfChannels;
    const len = audioBuf.length;
    const mono = new Float32Array(len);
    for (let c = 0; c < ch; c++) {
      const data = audioBuf.getChannelData(c);
      for (let i = 0; i < len; i++) mono[i] += data[i] / ch;
    }
    const pcm16k = resampleLinear(mono, audioBuf.sampleRate, TARGET_SR);

    sessionId = await startSession();
    const chunkSamples = Math.round(TARGET_SR * (CHUNK_MS / 1000));
    for (let off = 0; off < pcm16k.length; off += chunkSamples) {
      const chunk = pcm16k.slice(off, off + chunkSamples);
      const j = await pushChunk(sessionId, chunk);
      language.value = j.language || "—";
      text.value = j.text || "";
    }
    const j = await finishSession(sessionId);
    language.value = j.language || "—";
    text.value = j.text || "";
    setStatus("done", "识别完成");
  } catch (err) {
    error.value = err.message;
    setStatus("error", "识别失败");
  } finally {
    fileBusy.value = false;
    sessionId = null;
    if (fileInput.value) fileInput.value.value = "";
  }
}

function clearText() {
  text.value = "";
  language.value = "—";
  error.value = "";
}

onBeforeUnmount(() => {
  running = false;
  stopPipeline();
});
</script>

<template>
  <div class="wrap">
    <div class="card">
      <h1>Qwen3-ASR · 语音转文字</h1>

      <div class="row">
        <span class="pill" :class="status">{{ statusText }}</span>
        <span class="pill">语言: {{ language }}</span>
      </div>

      <div class="row">
        <button
          class="primary"
          :disabled="running || fileBusy"
          @click="startRecording"
        >
          🎙 开始录音
        </button>
        <button class="danger" :disabled="!running" @click="stopRecording">
          ⏹ 停止
        </button>
        <button @click="clearText">清空</button>
      </div>

      <div class="row">
        <label class="filebtn">
          📁 上传音频文件
          <input
            ref="fileInput"
            type="file"
            accept="audio/*"
            :disabled="running || fileBusy"
            @change="onFilePicked"
          />
        </label>
        <span v-if="fileBusy" class="pill warn">处理中…</span>
      </div>

      <div class="panel textpanel">
        <div class="label">识别结果</div>
        <div id="text" :class="{ empty: !text }">
          {{ text || "点击「开始录音」实时识别，或上传音频文件转写…" }}
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
.pill.finishing,
.pill.warn {
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
.textpanel {
  flex: 1;
  display: flex;
  flex-direction: column;
  min-height: 220px;
}
.label {
  color: #5b6472;
  font-size: 12px;
  margin-bottom: 6px;
}
#text {
  flex: 1;
  white-space: pre-wrap;
  font-size: 16px;
  line-height: 1.6;
  color: #0f172a;
}
#text.empty {
  color: #94a3b8;
}
.err {
  color: #9f1239;
  font-size: 13px;
}
</style>
