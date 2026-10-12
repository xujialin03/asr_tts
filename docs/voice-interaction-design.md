# 语音交互详细设计

本文记录语音助手的交互架构、状态机和关键实现约束。README 只保留项目介绍与运行方法；涉及 VAD、ASR、会话激活、打断、LLM、TTS 的设计决策统一维护在本文。

## 1. 总体链路

```text
麦克风
  ↓
Silero VAD（免手持模式）或 Push-to-talk（手动模式）
  ↓
Qwen3-ASR
  ↓
唤醒词 / 连续对话门控
  ├─ 未触发：只显示识别文本，不调用 LLM
  └─ 已触发：Qwen LLM
                ↓
             Breeze-TTS-2
                ↓
             Web Audio 流式播放
```

前端负责麦克风、VAD、播放和用户交互；FastAPI 后端负责 Session、门控状态、远程服务代理和对话历史。

## 2. 输入模式

### 2.1 Push-to-talk

用户按住按钮开始录音，松开后结束：

1. 浏览器采集单声道音频；
2. 重采样为 16kHz float32 PCM；
3. 每 500ms 推送一个 ASR chunk；
4. 松开时发送尾部不足 500ms 的缓冲，并调用 ASR finish；
5. Push-to-talk 是显式操作，因此识别文本默认视为对 AI 说话，不要求唤醒词。

关键实现：

- `frontend/src/App.vue`：`onHoldStart`、`onHoldEnd`、`startRecording`、`stopRecording`
- `frontend/src/asrClient.js`：`pushChunk`、`finishSession`

### 2.2 Silero VAD 免手持模式

Silero VAD 持续监听麦克风，根据每帧 `isSpeech` 概率切分完整 utterance：

- 开始阈值：`0.5`
- 结束阈值：`0.45`
- 连续静音：`1100ms`
- 最短语音：`250ms`
- 前置音频：`800ms`
- 单段最长：`15s`

VAD 在 `onSpeechEnd(audio)` 产生完整 16kHz float32 PCM 片段，前端将整段发给 `/api/transcribe`。免手持模式不再另起一套录音状态机，避免 VAD 与录音 start/stop 竞态。

关键实现：

- `frontend/src/vad.js`：Silero 模型加载、概率判断、endpointing
- `frontend/src/App.vue`：`onVadToggle`、`handleVadAudio`
- `backend/main.py`：`POST /api/transcribe`

免手持模式开启时，浏览器会持续占用麦克风，macOS 显示麦克风使用状态属于预期行为。关闭免手持模式或离开页面时必须调用 `stopVad` 和 `releaseSharedStream`。

## 3. 唤醒词与连续对话

免手持模式识别到环境人声后，不应直接调用 LLM。文本先经过门控：

```text
ASR 文本
  ↓
是否命中唤醒词？ ──是→ 激活并回复
  ↓ 否
是否在活跃窗口内？ ──是→ 续期并回复
  ↓ 否
是否为明显问题/指令？ ──是→ 激活并回复
  ↓ 否
静默丢弃，只显示识别文本
```

默认唤醒词：

- 志玲
- 志玲姐姐
- 姐姐
- 小助手

由于 ASR 可能把“志玲”识别成“志宁”，可在 `config.yaml` 中加入常见误识别词。

门控实现：

- `backend/session.py`：`should_reply`、`activate`、`is_active`
- `backend/main.py`：`POST /api/decide`
- `frontend/src/App.vue`：`handleVadAudio`

## 4. AI 激活窗口

激活窗口不是固定从 LLM 回复产生时开始倒计时，而是覆盖完整播放过程：

```text
激活剩余时间 = TTS 尚未播放的时长 + 播放结束后的等待时间
```

默认播放结束后等待时间为 20 秒。

前端根据收到的 PCM 数据和 24kHz、16-bit、单声道格式计算已排队的音频时长，并调用 `/api/touch` 动态延长后端 Session：

```text
当前时间 + 剩余播放时长 + 20 秒
```

用户在倒计时结束前开始说话时，VAD `onSpeechStart` 会立即调用 `/api/touch` 续租，避免用户说话、静音检测和 ASR 耗时造成窗口在判断前过期。

后端只允许延长，不会用较短的新窗口缩短已有窗口：

```python
active_until = max(active_until, now + requested_seconds)
```

关键实现：

- `backend/session.py`：`activate`、`active_remaining`
- `backend/main.py`：`POST /api/touch`
- `frontend/src/App.vue`：`touchActiveWindow`、`extendActiveForPlayback`
- `frontend/src/ttsPlayer.js`：`onDuration`

前端顶部显示：

```text
AI 未激活
AI 已激活 · 32s
```

## 5. 打断（Barge-in）逻辑

### 5.1 触发入口

当前支持三种打断：

1. VAD 检测到用户开始说话；
2. 用户在播放时按下 Push-to-talk 按钮；
3. 用户按 `Esc`。

VAD 自动打断位于 `frontend/src/App.vue` 的 `onVadToggle`：

```js
onSpeechStart: () => {
  if (isSpeaking()) stopSpeak()
}
```

`isSpeaking` 在以下状态返回 true：

- TTS 正在合成；
- TTS 正在播放；
- 某条 assistant 消息仍标记为正在播放。

### 5.2 统一停止入口

所有打断最终调用：

```js
stopSpeak()
  → stopPlayback()
  → playingIndex = -1
  → ttsState = idle
```

关键实现：`frontend/src/App.vue` 的 `stopSpeak`。

### 5.3 HTTP 流取消传播

仅停止 WebAudio 不足以中止 TTS 推理。`frontend/src/ttsPlayer.js` 的 `stopPlayback` 同时执行：

1. `reader.cancel()`：停止读取 PCM 流；
2. `AbortController.abort()`：中止 `/api/tts` 请求；
3. `AudioBufferSource.stop()`：停止已排入 WebAudio 时间线的音频；
4. 清理当前播放句柄。

取消链路：

```text
用户打断
  ↓
stopSpeak
  ↓
reader.cancel + fetch abort + WebAudio stop
  ↓
浏览器断开本地 /api/tts
  ↓
FastAPI 关闭远程 HTTP 流并释放 TTS_LOCK
  ↓
Breeze-TTS-2 _InferenceStreamingResponse 检测断开
  ↓
关闭模型生成器并释放 _request_lock
```

### 5.4 误打断风险

当前实现偏向低延迟：Silero 初步判断 speech start 后立即打断。因此清嗓子、咳嗽、耳麦摩擦可能误触发。

若误打断明显，可改为确认人声持续 250～400ms 后再调用 `stopSpeak`。代价是打断延迟增加约 250～400ms。

## 6. TTS 流式播放与并发

Breeze-TTS-2 官方 API 是单并发服务，同一时间只允许一个 inference。并发请求会返回：

```http
409 Conflict
{"detail":"An inference request is already running."}
```

本地代理使用 `TTS_LOCK` 防止本项目内并发请求。新版 Breeze-TTS-2 使用 `_InferenceStreamingResponse`，在客户端断开、取消、发送失败或模型异常时关闭生成器并释放远程锁。

音频格式：

- 裸 PCM，无 WAV 头
- 24kHz
- signed 16-bit little-endian
- 单声道
- chunked HTTP 流

前端收到 PCM chunk 后转换成 Float32Array，并把 AudioBufferSource 按 AudioContext 时间线连续排列，实现边收边播。

关键实现：

- `backend/main.py`：`POST /api/tts`、`TTS_LOCK`
- `frontend/src/ttsPlayer.js`：PCM 解码、排队播放、取消

## 7. 音色稳定

每次 TTS 请求都携带固定的参考音频和准确 transcript：

```yaml
tts:
  ref_audio: assets/ref_voice.wav
  ref_text: "参考音频的准确文字稿"
```

Breeze-TTS-2 的参考音频要求：

- 干净、低噪声、低混响；
- 单一说话人；
- `ref_text` 与参考音频逐字一致。

当前模式属于 Voice Direction：参考音频锁定说话人身份，`instruction` 控制语气、速度和情绪。官方建议 `cfg_scale=4` 增强 instruction 遵循度。

## 8. Session 管理

后端每个 Session 保存：

- 本地 session id；
- 当前远程 ASR session id；
- 最近一次 ASR 累计结果；
- LLM user/assistant 历史；
- `active_until`。

Session 当前存放在进程内存中：

- 刷新页面会创建新 Session；
- 重启后端会丢失所有 Session；
- 多实例部署时需要改为 Redis 或数据库。

关键实现：`backend/session.py`。

## 9. 已知限制

1. VAD 免手持开启时会持续占用麦克风。
2. 浏览器麦克风要求 localhost 或 HTTPS。
3. Breeze-TTS-2 单并发，短时间内可能出现 409；持续 409 通常表示服务端版本旧或资源未释放。
4. 当前门控主要依赖 ASR 文本、唤醒词和 Session 状态，不是声学 KWS。
5. 环境语音仍会被 ASR 转写，但未触发门控时不会发给 LLM。
6. Session 尚未持久化。

## 10. 修改本文的时机

以下改动必须同步更新本文：

- VAD 阈值、静音时间、最长 utterance；
- 唤醒词和连续对话规则；
- 激活窗口计算方式；
- 打断触发条件或取消链路；
- ASR/TTS 音频格式；
- Session 生命周期与持久化方式；
- 远程服务并发和超时策略。
