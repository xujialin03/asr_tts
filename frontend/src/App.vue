<script setup>
import { nextTick, onBeforeUnmount, onMounted, ref } from "vue";
import {
  startSession,
  endSession,
  beginTurn,
  evaluateTurn,
  transcribeUtterance,
  detectKeyword,
  touchSession,
  recordPerformance,
  chat,
} from "./asrClient.js";
import { playTts, stopPlayback } from "./ttsPlayer.js";
import { startVad, stopVad, releaseSharedStream } from "./vad.js";

const POST_REPLY_ACTIVE_SECONDS = 20;

const status = ref("idle"); // idle | listening | finishing | error | done
const statusText = ref("未开始");
const language = ref("—");
const asrText = ref(""); // live ASR recognition
const messages = ref([]); // {role, content, ttsInstruction?} conversation
const sending = ref(false);
const error = ref("");
const gateNotice = ref("");
const decisionDebug = ref(null);
const aiActive = ref(false);
const activeRemaining = ref(0);
const speakerId = ref("");
const speakerCreated = ref(false);
const voiceEvents = ref([]); // diagnostic history of recognized voices
const voiceLogEl = ref(null);
const logEl = ref(null);
const asrEl = ref(null);

const autoPlay = ref(true); // auto-speak each assistant reply
// 声纹识别侧栏暂时隐藏（功能与状态保留，后续在此区域做别的调度界面）。
const showVoiceSidebar = ref(false);
// 右侧决策模型调度面板：输入 → 判断 → 结论。
const showDecisionPanel = ref(true);
const ttsState = ref("idle"); // idle | synthesizing | playing | done
const playingIndex = ref(-1); // which message is currently speaking
const vadEnabled = ref(false); // hands-free Silero VAD (opt-in; headphones recommended)
const vadProb = ref(0);
const vadActive = ref(false);
const vadSilenceMs = ref(0);
const vadSegmentMs = ref(0);

const SPEAKER_COLORS = [
  "#2563eb",
  "#059669",
  "#d97706",
  "#7c3aed",
  "#db2777",
  "#0891b2",
  "#dc2626",
  "#4f46e5",
];

function speakerColor(id) {
  let hash = 0;
  for (const char of id || "") hash = (hash * 31 + char.charCodeAt(0)) >>> 0;
  return SPEAKER_COLORS[hash % SPEAKER_COLORS.length];
}

function formatScore(score) {
  return Number.isFinite(Number(score)) ? Number(score).toFixed(3) : "首次建档";
}

function formatEventTime(date = new Date()) {
  return new Intl.DateTimeFormat("zh-CN", {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    hour12: false,
  }).format(date);
}

async function addVoiceEvent(identity, text) {
  const id = identity?.speaker_id;
  if (!id) return;
  voiceEvents.value.push({
    id: `${Date.now()}-${voiceEvents.value.length}`,
    speakerId: id,
    text: (text || "").trim() || "（无识别文本）",
    score: identity.score,
    created: !!identity.created,
    time: formatEventTime(),
  });
  await nextTick();
  if (voiceLogEl.value) voiceLogEl.value.scrollTop = 0;
}

let sessionId = null;

function recordTiming(stage, startedAt, { turnId = null, detail = {} } = {}) {
  if (!sessionId) return;
  const elapsedMs = performance.now() - startedAt;
  recordPerformance(sessionId, { stage, elapsed_ms: elapsedMs, turn_id: turnId, detail }).catch(() => {});
}

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

async function ensureSession() {
  if (!sessionId) {
    const state = await startSession();
    sessionId = state.session_id;
    applySessionState(state);
  }
  return sessionId;
}

function applySessionState(state) {
  if (!state) return;
  updateActive(state);
}

function speakerName(speaker) {
  return speaker?.preferred_name || speaker?.display_name || `声纹 ${(speaker?.speaker_id || "").slice(0, 8)}`;
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
      instruction: msg.ttsInstruction || "",
      onState: (s) => {
        ttsState.value = s;
      },
      onDuration: ({ remainingSeconds }) => {
        extendActiveForPlayback(remainingSeconds);
      },
      onTiming: (stage, elapsedMs, detail) => {
        if (!sessionId) return;
        recordPerformance(sessionId, { stage, elapsed_ms: elapsedMs, detail }).catch(() => {});
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

function appendUserMessage(text, speaker = null) {
  const content = (text || "").trim();
  if (!content) return "";
  messages.value.push({
    role: "user",
    content,
    speakerLabel: speaker ? speakerName(speaker) : "我",
  });
  asrText.value = "";
  scrollLog();
  return content;
}

// Send recognized text to the LLM and show + speak the reply.
async function sendToLLM(turnId, speaker) {
  if (!turnId) return;
  error.value = "";
  sending.value = true;
  const chatStartedAt = performance.now();
  try {
    const sid = await ensureSession();
    const result = await chat(sid, turnId);
    const reply = result.reply;
    console.info(`[Voice timing] Hermes ${Math.round(performance.now() - chatStartedAt)}ms`);
    recordTiming("hermes", chatStartedAt, { turnId });
    updateActive(result);
    messages.value.push({
      role: "assistant",
      content: reply,
      ttsInstruction: result.tts_instruction || "",
    });
    scrollLog();
    if (autoPlay.value) speak(messages.value.length - 1);
  } catch (err) {
    error.value = err.message;
    setStatus("error", "LLM 调用失败");
  } finally {
    sending.value = false;
  }
}

// ---- Hands-free VAD ----
let vadSegmentActive = false;
let vadTurnPromise = null;
const vadQueue = [];
let vadQueueRunning = false;
const MAX_VAD_QUEUE = 4;

async function handleEvaluatedTurn(decision, recognizedText) {
  applySessionState(decision.session);
  const identity = decision.speaker;
  if (identity) {
    speakerId.value = identity.speaker_id || "";
    speakerCreated.value = !!identity.created;
    await addVoiceEvent(identity, recognizedText);
  }
  gateNotice.value = decision.decision_reason || decision.reason || "";
  decisionDebug.value = {
    text: recognizedText,
    should_reply: decision.should_reply,
    addressed: decision.is_addressed_to_ai,
    response_type: decision.response_type,
    confidence: decision.confidence,
    decision_reason: decision.decision_reason || "",
    gate_reason: decision.reason || "",
    fallback: decision.decision_fallback,
    wake: decision.wake_event,
    interrupt: decision.interrupt,
    speaker_id: identity?.speaker_id || "",
    speaker_name: identity
      ? identity.preferred_name || identity.display_name || ""
      : "",
    session_active: decision.session?.active ?? null,
    active_remaining: decision.session?.active_remaining ?? null,
    at: new Date().toLocaleTimeString(),
  };
  if (decision.interrupt) stopSpeak();
  if (!decision.should_reply) {
    const decisionLabel = decision.decision_fallback
      ? "决策模型异常，已使用传统门控"
      : decision.response_type === "silent"
        ? "决策模型判定无需回应"
        : "未触发回复";
    setStatus("idle", decision.interrupt ? "已打断" : decisionLabel);
    return;
  }

  const replyText = (decision.reply_text || recognizedText).trim();
  appendUserMessage(replyText, identity);
  await sendToLLM(decision.turn_id, identity);
}

function queueVadAudio(item) {
  if (vadQueue.length >= MAX_VAD_QUEUE) {
    gateNotice.value = "语音处理队列已满，本段未处理";
    return;
  }
  vadQueue.push(item);
  drainVadQueue();
}

async function drainVadQueue() {
  if (vadQueueRunning) return;
  vadQueueRunning = true;
  while (vadQueue.length) {
    const item = vadQueue.shift();
    await handleVadAudio(item.audio, item.turnPromise);
  }
  vadQueueRunning = false;
}

async function handleVadAudio(audio, turnPromise) {
  if (!audio?.length) return;
  const turnStartedAt = performance.now();
  let stageStartedAt = turnStartedAt;
  let processingStage = "ASR";
  error.value = "";
  setStatus("finishing", "识别中…");
  try {
    const sid = await ensureSession();
    const [asrResult, kwsResult] = await Promise.all([
      transcribeUtterance(sid, audio),
      detectKeyword(audio).catch((err) => {
        console.warn("KWS detect failed", err);
        return { available: false, detected: false, keyword: null };
      }),
    ]);
    console.info(`[Voice timing] ASR+KWS ${Math.round(performance.now() - stageStartedAt)}ms`);
    recordTiming("asr_kws", stageStartedAt);
    language.value = asrResult.language || "—";
    asrText.value = asrResult.text || "";
    setStatus("done", "已识别");
    const text = asrText.value.trim();
    processingStage = "声纹识别";
    stageStartedAt = performance.now();
    const turn = await turnPromise;
    const decision = await evaluateTurn(sid, turn.turn_id, audio, {
      text,
      language: asrResult.language || "",
      kws: kwsResult,
    });
    console.info(`[Voice timing] Voice gate ${Math.round(performance.now() - stageStartedAt)}ms`);
    recordTiming("voice_gate", stageStartedAt, {
      turnId: turn.turn_id,
      detail: {
        shouldRespond: decision.should_reply,
        responseType: decision.response_type,
        fallback: decision.decision_fallback,
      },
    });
    console.info(
      `[Voice timing] Turn evaluated ${Math.round(performance.now() - turnStartedAt)}ms after VAD end`
    );
    await handleEvaluatedTurn(decision, text || kwsResult.keyword || "");
  } catch (err) {
    error.value = `${processingStage}失败：${err.message}`;
    if (processingStage === "ASR") {
      asrText.value = "语音识别失败，请确认远程 ASR 服务状态。";
      scrollAsr();
    }
    setStatus("error", `${processingStage}失败`);
  }
}

async function onVadToggle() {
  if (vadEnabled.value) {
    try {
      await startVad({
        onSpeechStart: () => {
          const duringPlayback = isSpeaking();
          vadSegmentActive = true;
          vadActive.value = true;
          vadTurnPromise = ensureSession().then((sid) =>
            beginTurn(sid, duringPlayback)
          );
          asrText.value = "";
          language.value = "—";
          setStatus("listening", "正在说话…");
        },
        onSpeechEnd: (audio) => {
          if (!vadSegmentActive) return;
          vadSegmentActive = false;
          vadActive.value = false;
          const turnPromise = vadTurnPromise;
          vadTurnPromise = null;
          queueVadAudio({ audio, turnPromise });
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
    setStatus("idle", "未开始");
  }
}

function onKey(e) {
  const tag = (e.target.tagName || "").toLowerCase();
  if (tag === "input" || tag === "textarea") return;
  if (e.code === "Escape") {
    e.preventDefault();
    stopSpeak();
  }
}

function clearVoiceEvents() {
  voiceEvents.value = [];
}

async function clearAll() {
  stopSpeak();
  if (sessionId) {
    try {
      await endSession(sessionId);
    } catch (_) {}
  }
  sessionId = null;
  voiceEvents.value = [];
  messages.value = [];
  asrText.value = "";
  language.value = "—";
  error.value = "";
  gateNotice.value = "";
  aiActive.value = false;
  activeRemaining.value = 0;
  speakerId.value = "";
  speakerCreated.value = false;
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
  stopPlayback();
  stopVad();
  releaseSharedStream();
});
</script>

<template>
  <div class="app-shell" :class="{ 'single-column': !showVoiceSidebar && !showDecisionPanel }">
    <main class="card assistant-card">
      <header class="app-header">
        <div>
          <h1>语音助手</h1>
          <p>ASR · KWS · 声纹 · Hermes · TTS</p>
        </div>
        <div class="row status-row">
          <span class="pill" :class="status">{{ statusText }}</span>
          <span class="pill">语言: {{ language }}</span>
          <span class="pill" :class="aiActive ? 'listening' : ''">
            {{ aiActive ? `AI 已激活 · ${Math.round(activeRemaining)}s` : "AI 未激活" }}
          </span>
          <span v-if="speakerId" class="pill" :class="speakerCreated ? 'finishing' : 'listening'">
            {{ speakerCreated ? "新声纹" : "已识别" }} · {{ speakerId.slice(0, 8) }}
          </span>
          <span
            v-if="ttsState !== 'idle'"
            class="pill"
            :class="ttsState === 'playing' ? 'listening' : 'finishing'"
          >
            🔊 {{ ttsState === "synthesizing" ? "合成中…" : ttsState === "playing" ? "播放中…" : "播放完成" }}
          </span>
        </div>
      </header>

      <div class="vad-area">
        <div class="vad-indicator" :class="{ active: vadActive, enabled: vadEnabled }">
          <span class="vad-icon">{{ vadActive ? "🗣" : vadEnabled ? "👂" : "🎙" }}</span>
          <span>{{ vadActive ? "检测到说话" : vadEnabled ? "正在监听" : "VAD 已关闭" }}</span>
        </div>
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
            {{ vadEnabled ? "持续监听麦克风 · 说话自动识别，停顿后自动处理" : "打开下方 VAD 开关后即可开始免手持语音会话" }}
          </div>
        </div>
      </div>

      <div class="row controls-row">
        <button @click="clearAll">清空对话</button>
        <label class="toggle">
          <input type="checkbox" v-model="autoPlay" />
          自动朗读回复
        </label>
        <label class="toggle" title="免手持：检测到说话自动开始录音，停顿自动结束。外放时 TTS 可能被误识别，建议戴耳机。">
          <input type="checkbox" v-model="vadEnabled" @change="onVadToggle" />
          Silero VAD 语音监听
        </label>
        <span v-if="vadEnabled" class="pill warn">建议戴耳机</span>
        <span v-if="vadEnabled" class="pill" :class="vadActive ? 'listening' : ''">
          VAD {{ vadActive ? "说话中" : "监听中" }} · p={{ vadProb.toFixed(2) }} · 静音{{ Math.round(vadSilenceMs) }}ms · 段{{ Math.round(vadSegmentMs) }}ms
        </span>
      </div>

      <div v-if="gateNotice" class="notice">{{ gateNotice }}</div>

      <div class="panel chatpanel">
        <div class="panel-title">对话</div>
        <div id="log" ref="logEl">
          <div v-if="!messages.length" class="empty">
            开启 VAD 后持续侦听，每段语音都会交给决策模型判断是否回应。
          </div>
          <div v-for="(m, i) in messages" :key="i" class="msg" :class="m.role">
            <span class="who">{{ m.role === "user" ? (m.speakerLabel || "我") : "她" }}</span>
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
    </main>

    <aside v-if="showVoiceSidebar" class="card voice-sidebar">
      <header class="sidebar-header">
        <div>
          <h2>声纹识别</h2>
          <p>开启 VAD 后持续侦听到的声音</p>
        </div>
        <button class="ghost-button" @click="clearAll">结束会话</button>
      </header>

      <details class="history-details">
        <summary>声纹识别记录（{{ voiceEvents.length }}）</summary>
        <button class="ghost-button" :disabled="!voiceEvents.length" @click="clearVoiceEvents">清空记录</button>
        <div ref="voiceLogEl" class="voice-log compact">
          <article
            v-for="event in [...voiceEvents].reverse()"
            :key="event.id"
            class="voice-event"
            :style="{ '--speaker-color': speakerColor(event.speakerId) }"
          >
            <div class="voice-event-head">
              <div class="speaker-mark"><span class="speaker-dot"></span><span>声纹 {{ event.speakerId.slice(0, 8) }}</span></div>
              <time>{{ event.time }}</time>
            </div>
            <p class="voice-text">{{ event.text }}</p>
          </article>
        </div>
      </details>
    </aside>

    <aside v-if="showDecisionPanel" class="card decision-panel">
      <header class="sidebar-header">
        <div>
          <h2>决策模型调度</h2>
          <p>输入 → 判断 → 结论（最近一轮）</p>
        </div>
        <span class="pill" :class="decisionDebug ? 'listening' : ''">
          {{ decisionDebug ? decisionDebug.at : "等待语音" }}
        </span>
      </header>

      <div v-if="!decisionDebug" class="dp-empty">
        开启 VAD 说话后，这里会展示决策模型对这一轮的完整判断链路。
      </div>

      <div v-else class="dp-flow">
        <!-- 输入 -->
        <section class="dp-stage dp-input">
          <div class="dp-stage-head"><span class="dp-badge">1</span><h3>输入</h3></div>
          <div class="dp-quote">“{{ decisionDebug.text || "（无识别文本）" }}”</div>
          <div class="dp-grid">
            <span class="dp-k">说话人</span>
            <span class="dp-v">
              {{ decisionDebug.speaker_name || (decisionDebug.speaker_id ? `声纹 ${decisionDebug.speaker_id.slice(0, 8)}` : "—") }}
            </span>
            <span class="dp-k">唤醒事件</span>
            <span :class="['dp-v', decisionDebug.wake ? 'dp-yes' : 'dp-no']">
              {{ decisionDebug.wake ? "冷启动唤醒" : "无" }}
            </span>
            <span class="dp-k">会话状态</span>
            <span :class="['dp-v', decisionDebug.session_active ? 'dp-yes' : 'dp-no']">
              {{ decisionDebug.session_active ? "已激活" : "未激活" }}
              <template v-if="decisionDebug.active_remaining != null">
                · {{ decisionDebug.active_remaining }}s
              </template>
            </span>
          </div>
        </section>

        <div class="dp-arrow">↓</div>

        <!-- 判断 -->
        <section class="dp-stage dp-judge">
          <div class="dp-stage-head"><span class="dp-badge">2</span><h3>判断</h3></div>
          <div class="dp-grid">
            <span class="dp-k">面向 AI</span>
            <span :class="['dp-v', decisionDebug.addressed ? 'dp-yes' : 'dp-no']">
              {{ decisionDebug.addressed ? "是" : "否" }}
            </span>
            <span class="dp-k">回应类型</span>
            <span class="dp-v">{{ decisionDebug.response_type }}</span>
            <span class="dp-k">置信度</span>
            <span class="dp-v">
              <span class="dp-bar"><span class="dp-bar-fill" :style="{ width: `${Math.round((decisionDebug.confidence ?? 0) * 100)}%` }"></span></span>
              {{ decisionDebug.confidence ?? "—" }}
            </span>
            <span class="dp-k">决策来源</span>
            <span :class="['dp-v', decisionDebug.fallback ? 'dp-no' : 'dp-yes']">
              {{ decisionDebug.fallback ? "回退门控（模型异常）" : "决策模型" }}
            </span>
          </div>
          <div v-if="decisionDebug.decision_reason" class="dp-reason">
            模型理由：{{ decisionDebug.decision_reason }}
          </div>
        </section>

        <div class="dp-arrow">↓</div>

        <!-- 结论 -->
        <section class="dp-stage dp-result" :class="decisionDebug.should_reply ? 'is-reply' : 'is-silent'">
          <div class="dp-stage-head"><span class="dp-badge">3</span><h3>结论</h3></div>
          <div class="dp-verdict">
            <span class="dp-verdict-tag">{{ decisionDebug.should_reply ? "回应" : "沉默" }}</span>
            <span class="dp-verdict-desc">
              {{
                decisionDebug.interrupt
                  ? "命中打断，停止播放"
                  : decisionDebug.should_reply
                    ? "送入 Hermes 生成回复"
                    : decisionDebug.fallback
                      ? "决策模型异常，已用传统门控"
                      : "仅显示识别文本，不调用 Hermes"
              }}
            </span>
          </div>
          <div v-if="decisionDebug.gate_reason" class="dp-reason">
            门控理由：{{ decisionDebug.gate_reason }}
          </div>
        </section>
      </div>
    </aside>
  </div>
</template>

<style scoped>
.app-shell {
  width: 100vw;
  height: 100vh;
  padding: 14px;
  display: grid;
  grid-template-columns: minmax(0, 1fr) minmax(340px, 28vw);
  gap: 14px;
  overflow: hidden;
}
.app-shell.single-column {
  grid-template-columns: minmax(0, 1fr);
}
.card {
  min-width: 0;
  min-height: 0;
  background: #fff;
  border: 1px solid #e2e8f0;
  border-radius: 16px;
  box-shadow: 0 10px 32px rgba(15, 23, 42, 0.07);
  overflow: hidden;
}
.assistant-card {
  padding: 18px;
  display: flex;
  flex-direction: column;
  gap: 12px;
}
.app-header {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 20px;
}
.app-header h1,
.sidebar-header h2 {
  margin: 0;
  color: #0f172a;
}
.app-header h1 {
  font-size: 22px;
  letter-spacing: -0.02em;
}
.app-header p,
.sidebar-header p {
  margin: 4px 0 0;
  color: #94a3b8;
  font-size: 12px;
}
.status-row {
  justify-content: flex-end;
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
.vad-area {
  display: flex;
  align-items: center;
  gap: 16px;
  padding: 4px 0;
}
.vad-indicator {
  flex: none;
  display: inline-flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 5px;
  width: 110px;
  height: 110px;
  border-radius: 50%;
  border: 2px solid #cbd5e1;
  background: #f8fafc;
  color: #64748b;
  font-size: 12px;
  font-weight: 800;
  text-align: center;
}
.vad-indicator.enabled {
  border-color: rgba(5, 150, 105, 0.35);
  background: rgba(5, 150, 105, 0.1);
  color: #065f46;
}
.vad-indicator.active {
  border-color: rgba(225, 29, 72, 0.5);
  background: rgba(225, 29, 72, 0.12);
  color: #9f1239;
}
.vad-icon {
  font-size: 30px;
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
.panel-title {
  color: #475569;
  font-size: 13px;
  font-weight: 700;
  margin-bottom: 8px;
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
.member-sections {
  flex: 1;
  min-height: 0;
  overflow-y: auto;
  padding: 14px;
  display: flex;
  flex-direction: column;
  gap: 18px;
}
.member-section h3 {
  margin: 0 0 9px;
  font-size: 13px;
  color: #475569;
}
.member-empty {
  padding: 16px;
  border: 1px dashed #cbd5e1;
  border-radius: 12px;
  text-align: center;
  color: #94a3b8;
  font-size: 12px;
}
.member-card {
  position: relative;
  margin-bottom: 9px;
  padding: 12px 12px 10px 15px;
  border: 1px solid #e2e8f0;
  border-radius: 12px;
  background: #fff;
  overflow: hidden;
}
.member-card::before {
  content: "";
  position: absolute;
  inset: 0 auto 0 0;
  width: 4px;
  background: var(--speaker-color);
}
.member-head {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 10px;
}
.member-head strong,
.member-head small {
  display: block;
}
.member-head strong {
  color: #334155;
  font-size: 13px;
}
.member-head small {
  margin-top: 4px;
  color: #94a3b8;
  font-size: 10px;
}
.member-action {
  padding: 6px 10px;
  border-radius: 8px;
  font-size: 11px;
}
.member-action.join {
  color: #065f46;
  border-color: rgba(5, 150, 105, 0.3);
  background: rgba(5, 150, 105, 0.08);
}
.member-action.danger {
  color: #9f1239;
  border-color: rgba(225, 29, 72, 0.25);
  background: rgba(225, 29, 72, 0.06);
}
.history-details {
  border-top: 1px solid #e2e8f0;
  padding: 12px 14px;
  color: #64748b;
  font-size: 12px;
}
.history-details summary {
  cursor: pointer;
  font-weight: 700;
}
.voice-log.compact {
  max-height: 220px;
  padding: 10px 0 0;
}
.voice-sidebar {
  display: flex;
  flex-direction: column;
  background: #f8fafc;
}
.sidebar-header {
  padding: 18px 18px 14px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  border-bottom: 1px solid #e2e8f0;
  background: #fff;
}
.sidebar-header h2 {
  font-size: 17px;
}
.ghost-button {
  padding: 7px 11px;
  border-radius: 9px;
  font-size: 12px;
  color: #64748b;
  background: transparent;
}
.voice-summary {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 10px;
  padding: 12px 14px;
  border-bottom: 1px solid #e2e8f0;
}
.summary-item {
  padding: 12px;
  border: 1px solid #e2e8f0;
  border-radius: 12px;
  background: #fff;
  display: flex;
  align-items: baseline;
  gap: 8px;
}
.summary-item strong {
  font-size: 22px;
  line-height: 1;
  color: #0f172a;
}
.summary-item span {
  font-size: 12px;
  color: #64748b;
}
.voice-log {
  flex: 1;
  min-height: 0;
  padding: 14px;
  overflow-y: auto;
  display: flex;
  flex-direction: column;
  gap: 10px;
}
.voice-empty {
  flex: 1;
  min-height: 220px;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 8px;
  text-align: center;
  color: #94a3b8;
  font-size: 13px;
  line-height: 1.6;
  padding: 24px;
}
.voice-empty strong {
  color: #64748b;
  font-size: 14px;
}
.voice-empty-icon {
  width: 46px;
  height: 46px;
  display: grid;
  place-items: center;
  border-radius: 50%;
  border: 1px dashed #cbd5e1;
  font-size: 24px;
  color: #64748b;
}
.voice-event {
  position: relative;
  padding: 13px 13px 12px 16px;
  border: 1px solid #e2e8f0;
  border-radius: 13px;
  background: #fff;
  box-shadow: 0 3px 12px rgba(15, 23, 42, 0.04);
  overflow: hidden;
}
.voice-event::before {
  content: "";
  position: absolute;
  inset: 0 auto 0 0;
  width: 4px;
  background: var(--speaker-color);
}
.voice-event-head,
.voice-meta {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 10px;
}
.voice-event-head time,
.voice-meta {
  color: #94a3b8;
  font-size: 11px;
}
.speaker-mark {
  min-width: 0;
  display: flex;
  align-items: center;
  gap: 7px;
  color: #334155;
  font-size: 13px;
  font-weight: 800;
}
.speaker-dot {
  width: 8px;
  height: 8px;
  flex: none;
  border-radius: 50%;
  background: var(--speaker-color);
  box-shadow: 0 0 0 3px color-mix(in srgb, var(--speaker-color) 16%, transparent);
}
.speaker-full-id {
  display: block;
  margin-top: 7px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  color: #94a3b8;
  font-size: 10px;
  background: transparent;
}
.voice-text {
  margin: 10px 0;
  color: #0f172a;
  font-size: 14px;
  line-height: 1.55;
  word-break: break-word;
}
.voice-meta {
  padding-top: 9px;
  border-top: 1px solid #f1f5f9;
}
.err {
  color: #9f1239;
  font-size: 13px;
}
.notice {
  color: #64748b;
  font-size: 13px;
}
.decision-debug {
  padding: 10px 12px;
  font-size: 12px;
}
.decision-debug .panel-title {
  margin-bottom: 6px;
}
.dd-text {
  color: #334155;
  margin-bottom: 6px;
}
.dd-grid {
  display: grid;
  grid-template-columns: auto 1fr;
  gap: 2px 10px;
}
.dd-k {
  color: #94a3b8;
}
.dd-v {
  color: #334155;
}
.dd-yes {
  color: #16a34a;
}
.dd-no {
  color: #dc2626;
}
.dd-reason {
  margin-top: 6px;
  color: #64748b;
}
.decision-panel {
  display: flex;
  flex-direction: column;
  background: #f8fafc;
}
.dp-empty {
  flex: 1;
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 24px;
  color: #94a3b8;
  font-size: 13px;
  text-align: center;
}
.dp-flow {
  flex: 1;
  min-height: 0;
  overflow-y: auto;
  padding: 14px;
  display: flex;
  flex-direction: column;
  gap: 4px;
}
.dp-stage {
  border: 1px solid #e2e8f0;
  border-radius: 12px;
  background: #fff;
  padding: 12px 14px;
}
.dp-stage-head {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 8px;
}
.dp-stage-head h3 {
  margin: 0;
  font-size: 14px;
  color: #0f172a;
}
.dp-badge {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 20px;
  height: 20px;
  border-radius: 50%;
  font-size: 12px;
  font-weight: 700;
  color: #fff;
  background: #64748b;
}
.dp-input .dp-badge { background: #2563eb; }
.dp-judge .dp-badge { background: #7c3aed; }
.dp-result .dp-badge { background: #16a34a; }
.dp-result.is-silent .dp-badge { background: #94a3b8; }
.dp-quote {
  color: #334155;
  font-size: 14px;
  margin-bottom: 10px;
  padding: 8px 10px;
  background: #f1f5f9;
  border-radius: 8px;
  border-left: 3px solid #2563eb;
}
.dp-grid {
  display: grid;
  grid-template-columns: auto 1fr;
  gap: 4px 12px;
  font-size: 13px;
}
.dp-k { color: #94a3b8; }
.dp-v { color: #334155; display: flex; align-items: center; gap: 8px; }
.dp-yes { color: #16a34a; }
.dp-no { color: #dc2626; }
.dp-bar {
  flex: 1;
  max-width: 120px;
  height: 6px;
  border-radius: 3px;
  background: #e2e8f0;
  overflow: hidden;
}
.dp-bar-fill {
  display: block;
  height: 100%;
  background: #7c3aed;
}
.dp-reason {
  margin-top: 8px;
  font-size: 12px;
  color: #64748b;
}
.dp-arrow {
  text-align: center;
  color: #cbd5e1;
  font-size: 16px;
  line-height: 1;
}
.dp-verdict {
  display: flex;
  align-items: center;
  gap: 10px;
}
.dp-verdict-tag {
  font-size: 18px;
  font-weight: 800;
  color: #16a34a;
}
.dp-result.is-silent .dp-verdict-tag { color: #64748b; }
.dp-verdict-desc {
  font-size: 13px;
  color: #475569;
}
.dp-result.is-reply { border-color: #bbf7d0; background: #f0fdf4; }
.dp-result.is-silent { border-color: #e2e8f0; }
@media (max-width: 980px) {
  .app-shell {
    height: auto;
    min-height: 100vh;
    grid-template-columns: 1fr;
    overflow: visible;
  }
  .assistant-card {
    min-height: calc(100vh - 28px);
  }
  .voice-sidebar {
    min-height: 520px;
  }
  .decision-panel {
    min-height: 520px;
  }
  .app-header {
    flex-direction: column;
  }
  .status-row {
    justify-content: flex-start;
  }
}
@media (max-width: 620px) {
  .app-shell {
    padding: 8px;
    gap: 8px;
  }
  .assistant-card {
    padding: 12px;
    border-radius: 12px;
  }
  .vad-area {
    align-items: stretch;
  }
  .vad-indicator {
    width: 86px;
    height: 86px;
  }
  .controls-row {
    align-items: flex-start;
  }
}
</style>
