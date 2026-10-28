# 语音交互详细设计

本文记录语音助手的交互架构、状态机和关键实现约束。README 只保留项目介绍与运行方法；涉及 VAD、ASR、会话激活、打断、direct-LLM 对话 Agent、TTS 的设计决策统一维护在本文。

## 1. 总体链路

```text
麦克风
  ↓
Silero VAD（唯一语音输入模式，可通过界面开关启停）
  ↓
Qwen3-ASR
  ↓
决策模型（上下文 / 说话人 / 唤醒信号）
  ├─ silent：仅显示识别文本，不调用对话 Agent
  └─ normal / brief：进程内 direct-LLM 对话 Agent
                         ↓
                      Breeze-TTS-2
                         ↓
                      Web Audio 流式播放
```

前端负责麦克风、VAD、播放和用户交互；FastAPI 后端负责本地 Session、决策状态、对话 Agent 和远程服务代理。每个 Session 独立维护最近五条真实用户/助理消息，决策模型仅读取该上下文并输出结构化交互决策；进程内 direct-LLM Agent 管理助理人设、对话上下文与技能调用（如 teleop.*）。Agent 的最终响应使用 `{"text":"...","instruction":"..."}`：`text` 用于页面展示和朗读，`instruction` 作为本轮动态 Voice Direction 传给 TTS；缺失时回退到 `tts.prompt`。

## 2. Silero VAD 输入模式

Silero VAD 是应用唯一的语音输入方式。用户通过界面开关开启或关闭持续监听；项目不再提供按住说话、空格键录音或流式 `/api/chunk`、`/api/finish` 的前端入口。

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

## 3. 会话参与者、声纹身份与连续对话

免手持模式区分两层状态，不能混为一谈：

- **持久声纹身份**：声纹服务的 `speaker_id`、模板、`preferred_name`，跨浏览器 Session 保存；
- **当前会话参与者**：本次 browser Session 中明确用唤醒词加入的 `speaker_id` 集合，只保存在 backend Session 内存。

这让房间可以有多个真人，但助手只服务主动进入本次对话的人：

```text
普通环境语音 / 未加入者发言
  → 调用只读 identify（不自动建档）
  → 未识别或不在当前参与者集合 → 静默忽略，不进入决策模型/对话 Agent

任何人说出唤醒词
  → resolve 声纹（已有身份复用；未知身份才建匿名档案）
  → 加入当前 Session 的 participant_ids
  → 该人本次起可自然连续对话，无需每句重复唤醒

已加入成员说普通话
  → identify 命中该成员 → 进入上下文、决策模型与回复流程

已入库、但未加入当前 Session 的身份
  → 即使被 identify 认出，仍静默忽略；必须带唤醒词才可加入
```

新加入的匿名身份（`访客 xxxxxxxx` 且没有 `preferred_name`）会在**加入时**自然被问一次“我该怎么称呼你？”。回答“我是许老板”后，直连 LLM 会调用受限的项目内声纹技能保存称呼；不回答不影响后续对话，也不会在同一个 Session 内反复追问。

参与者门控也适用于语音打断：只有已经加入本次会话的人能用“等一下”“停一下”等语音停止播放，旁人不能借环境人声打断助手。

当前多人支持指多人轮流发言。重叠讲话仍会被 VAD、ASR 和 CAM++ 当作一个混合片段；可靠支持重叠语音需要额外的 diarization、源分离或多麦克风能力。

关键实现：

- `backend/session.py`：`Participant`、`ObservedSpeaker`、`VoiceTurn`、`evaluate_turn`
- `backend/main.py`：turn begin/evaluate、参与人加入/踢出、按 turn 调用 chat
- `frontend/src/App.vue`：VAD 轮次队列、参与人和侦听列表

免手持模式识别到环境人声后，文本先完成声纹与参与资格校验，再由决策模型结合完整交互上下文决定是否面向 AI 及是否回应。VAD→ASR 后送入决策模型的决策状态按架构图的数据模型组装：

- `agent`：角色模型（AgentProfile），来自 `config.yaml` 的 `agent` 段（`name` / `role` / `persona`），让决策模型知道“助手是谁”。进程内 direct-LLM Agent 在 `conversation_agent.SYSTEM_PROMPT` 里另有一份人设，这里只用于交互判断。
- `current_message`：本轮 ASR 文本。
- `speaker`：声纹身份（`speaker_id`、显示名、参与人编号、是否参与人）。
- `session`：会话模型，含 `active`、`participant_count`、`topics`（当前话题列表）与 `recent_messages`（最近五条真实用户/助理消息）。
- `wake_word_detected`、`assistant_speaking`、`has_pending_task`：信号位。

`recent_messages` 是判断延续性追问的关键上下文。`topics` 目前是启发式：把最近被放行的用户发言按序去重保留，作为“房间在聊什么”的粗略信号。决策模型当前只做交互判断（是否面向 AI / 是否回应），暂不做话题归纳抽取；需要真正的 Topic 归纳时再接生成式后端（可在对话 Agent 内或独立的 Topic 模型中实现）。

延续性兜底：决策模型对“参与人在活跃窗口内、紧接助理回复之后的延续追问”（如天气回答后问“最近是不是有个台风啊”）处在判断边界，`should_respond` 会抖动甚至误判 silent。当决策模型判 `is_addressed_to_ai=True` 但 `should_respond=False`，且满足“参与人 + 会话活跃 + 窗口内有助理回复 + 文本像追问（命中 `assistant.followup_markers` 或以 `assistant.followup_sentence_finals` 结尾）”时，兜底强制进入回复流程。纯确认词（“嗯”“好的”）不命中追问特征，仍保持 silent；`is_addressed_to_ai=False`（人类互聊）不触发兜底。

唤醒词和激活窗口是决策信号及模型异常时的保守回退，不再是连续对话的唯一入口；未加入的声纹仍不能被决策模型放行到对话 Agent。

默认唤醒词：

- 志玲
- 志玲姐姐
- 姐姐
- 小助手

由于 ASR 可能把“志玲”识别成“志宁”，可在 `config.yaml` 中加入常见误识别词。

门控实现：

- `backend/session.py`：`decide_turn`、`should_reply`、`activate`、`is_active`
- `backend/decision.py`：决策模型 Schema、超时和回退
- `backend/main.py`：`POST /api/session/{session_id}/turns/{turn_id}/evaluate`
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

### 5.1 两阶段文本确认式打断

为降低咳嗽、耳麦摩擦和旁人说话造成的误打断，VAD 检测到人声时不再立即停止 TTS。流程为：

```text
AI 正在合成/播放
  ↓
VAD 检测到人声（继续播放，不立即打断）
  ↓
VAD 收集完整 utterance
  ↓
ASR 转写
  ↓
正则匹配 interrupt_patterns
  ├─ 未命中：视为环境干扰，不停止、不回复
  ├─ 命中且无剩余文本：停止 TTS，不调用 LLM
  └─ 命中且有剩余文本：停止 TTS，将剩余文本发给 LLM
```

示例：

```text
“等一下”                     → 只停止播放
“慢着，我想问厦门天气”       → 停止播放，并把“我想问厦门天气”发给 LLM
“旁边的人在聊天”             → 未命中打断词，继续播放且不回复
```

打断正则在 `config.yaml` 的 `assistant.interrupt_patterns` 配置。默认包括：

- 等一下、等下、等一等、等会儿、稍等一下
- 停一下、停一停、暂停一下、先停一下
- 慢着、慢一点
- 先别说了、别说了、不要说了
- 打住、闭嘴

后端 `Session.match_interrupt` 负责正则匹配和移除打断短语；`POST /api/decide` 根据 `assistant_speaking` 返回：

```json
{
  "interrupt": true,
  "should_reply": false,
  "reply_text": "",
  "reason": "命中纯打断指令"
}
```

### 5.2 手动打断

用户按 `Esc` 或点击单条回复的停止/播放控制属于明确操作，可以立即调用 `stopSpeak`，不经过语音文本匹配。

### 5.3 统一停止入口

所有打断最终调用：

```js
stopSpeak()
  → stopPlayback()
  → playingIndex = -1
  → ttsState = idle
```

关键实现：`frontend/src/App.vue` 的 `stopSpeak`。

### 5.4 HTTP 流取消传播

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

### 5.5 延迟与误判权衡

文本确认式打断需要等 VAD 切段和 ASR 转写后才停止 TTS，因此比“检测到声音立即停止”有更高延迟。收益是能过滤咳嗽、耳麦摩擦、旁人对话等干扰。

如后续需要更低延迟，可采用两级策略：持续人声达到 250～400ms 时先降低 TTS 音量，ASR 文本命中打断词后再完全停止；未命中则恢复音量。

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

当前模式属于 Voice Direction：参考音频锁定说话人身份，direct-LLM Agent 为每条回复生成的 `instruction` 控制语气、速度和情绪。该值沿 `/api/chat` → 前端消息 → `/api/tts` 传递；缺失时回退到 `config.yaml` 的 `tts.prompt`。`cfg_scale` 由 `tts.cfg_scale` 配置，默认使用官方建议值 `4` 以增强 instruction 遵循度。

## 8. Session 管理

后端每个 Session 只保存：

- 本地 session id；
- 当前远程 ASR session id；
- 最近一次 ASR 累计结果；
- `active_until`；
- 当前 Session 的对话 Agent 调用锁（`chat_lock`）。

助理的人设由 `conversation_agent.SYSTEM_PROMPT` 与 `agent` 段共同决定；对话上下文（最近 5 条）由 backend Session 持有，进程内 direct-LLM Agent 是无状态的，按轮次接收上下文调用模型。

Session 当前存放在进程内存中：

- 刷新页面会创建新 Session；
- 重启后端会丢失所有 Session 及其对话上下文；
- 多实例部署时需要改为 Redis 或数据库。

关键实现：`backend/session.py`、`backend/conversation_agent.py`。

## 9. 已知限制

1. VAD 免手持开启时会持续占用麦克风。
2. 浏览器麦克风要求 localhost 或 HTTPS。
3. Breeze-TTS-2 单并发，短时间内可能出现 409；持续 409 通常表示服务端版本旧或资源未释放。
4. 当前门控主要依赖 ASR 文本、唤醒词和 Session 状态，不是声学 KWS。
5. 环境语音仍会被 ASR 转写，但未触发门控时不会进入对话 Agent。
6. backend Session 尚未持久化；对话上下文随 Session 生命周期存亡。

## 10. 修改本文的时机

以下改动必须同步更新本文：

- VAD 阈值、静音时间、最长 utterance；
- 唤醒词和连续对话规则；
- 激活窗口计算方式；
- 打断触发条件或取消链路；
- ASR/TTS 音频格式；
- Session 生命周期与持久化方式；
- 远程服务并发和超时策略。
