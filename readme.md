# 知玲 Zhiling

一个浏览器端多人语音助手原型，串联 Qwen3-ASR、Silero VAD、CAM++ 声纹识别、决策模型（远程生成式）、进程内 direct-LLM 对话 Agent 与 Breeze-TTS-2。

名字取自助手人设“志玲”，也涵盖 ASR 的同音误识别（诗林/志宁/志凌）——叫得准不准，它都认得你是在叫它。

前端负责麦克风采集、免手持语音检测和流式音频播放；FastAPI 后端负责会话状态、决策模型调度、唤醒/打断门控、声纹解析以及远程服务代理。对话人设和历史由 backend 进程内 direct-LLM Agent 独占管理。

## 功能

- 👂 **纯 VAD 语音**：Silero VAD 是唯一语音输入模式；通过界面开关持续监听麦克风，自动检测说话起止并切分完整语音片段。
- 🧠 **决策模型调度**：每段语音经声纹解析后交给远程生成式决策模型，结合上下文与说话人判断“是否面向 AI / 是否回应”，输出 normal / brief / silent；页面右侧面板可视化「输入 → 判断 → 结论」全链路。
- 🔔 **关键词唤醒**：冷启动时由 sherpa-onnx 声学 KWS 或 ASR 文本唤醒词激活助手；KWS 不可用时可降级为文本唤醒。
- 💬 **连续对话**：激活后在配置的时间窗口内无需重复唤醒，TTS 播放期间会自动延长窗口。
- ✋ **语音打断**：AI 播放期间先完成 ASR，再根据打断词确认是否停止播放，减少环境声音造成的误打断。
- 🗣️ **对话 Agent**：进程内 direct-LLM Agent 管理助理人设、上下文和对话历史，并为每条回复输出适配内容的 TTS 语气指令。
- 🔊 **流式语音回复**：Breeze-TTS-2 返回裸 PCM，浏览器边接收边播放；支持自动朗读、单条重播和停止播放。
- 👥 **多人声纹侦听**：开启 VAD 后持续侦听，每段语音都会解析声纹身份；当前多人支持多人轮流发言，不含重叠讲话的说话人分离。

> 交互状态机和设计约束见 [语音交互详细设计](docs/voice-interaction-design.md)。

## 技术栈

- 前端：Vue 3、Vite、Web Audio API
- 浏览器 VAD：Silero VAD（`@ricky0123/vad-web`）
- 后端：FastAPI、uv、httpx
- 语音识别：OpenAI 兼容 Qwen3-ASR 网关（整段转写）
- 关键词唤醒：独立可选的 sherpa-onnx 中文 KWS 服务
- 声纹识别：CAM++ 服务 + PostgreSQL
- 交互决策：远程 OpenAI 兼容生成式决策模型（默认 `qwen3.6-35b-awq`，可换端点）
- 对话 Agent：进程内 direct-LLM Agent（调用 OpenAI 兼容聊天模型）
- 语音合成：远程 Breeze-TTS-2 流式合成服务
- 本地服务编排：Docker Compose

## 架构与语音链路

```text
浏览器（Vue 3）
  ├─ Silero VAD：唯一输入模式，开关控制持续监听和 utterance 切分
  └─ Web Audio：播放流式 PCM
          │
          ▼
FastAPI backend
  ├─ /api/transcribe ──────────────────▶ 远程 Qwen3-ASR（VAD 整段转写）
  ├─ /api/kws/detect ──────────────────▶ 本地 sherpa-onnx KWS
  ├─ /api/session/.../turns/evaluate ──▶ 声纹解析 + 决策模型判断 + 唤醒/打断门控
  ├─ /api/chat ────────────────────────▶ 进程内 direct-LLM 对话 Agent
  └─ /api/tts ─────────────────────────▶ 远程 Breeze-TTS-2
```

免手持模式下，ASR 与 KWS 会并行处理完整语音片段。每段语音都会先解析声纹身份，再交给决策模型结合上下文与说话人判断是否回应；决策模型异常时回退到唤醒词与连续对话窗口门控。当前“多人会话”支持多人轮流发言，不包含重叠讲话的说话人分离——重叠讲话会被 VAD、ASR 和 CAM++ 当作一个混合片段。

远程 ASR/TTS 不需要向浏览器开放 CORS，浏览器只访问 FastAPI；开发环境下 Vite 再将 `/api/*` 代理到 FastAPI。

## 目录结构

```text
backend/              FastAPI 后端、direct-LLM 对话 Agent 和上游服务代理
frontend/             Vue 3 + Vite 前端
voiceprint-service/   CAM++ 声纹识别与用户管理服务
kws-service/          sherpa-onnx 关键词唤醒服务
docs/                 交互、声纹和 KWS 详细文档
compose.yaml          KWS 与声纹服务容器配置
config.yaml.example   应用配置模板
.env.example          Docker Compose 环境变量模板
```

## 前置条件

- Python 3.10+ 与 [uv](https://docs.astral.sh/uv/)
- Node.js 与 npm
- Docker 与 Docker Compose（`compose.yaml` 自带 PostgreSQL、声纹与 KWS 服务）
- 可访问的 Qwen3-ASR 与 Breeze-TTS-2 服务
- 可访问的 OpenAI 兼容对话模型（direct-LLM Agent 使用）

如果只调试部分链路，KWS 可关闭。每段语音都会解析声纹身份；声纹服务不可用、音频过短或识别失败时，本段会 fail closed，不会进入对话 Agent。

## 配置

首次运行先复制配置模板：

```bash
cp config.yaml.example config.yaml
cp .env.example .env
```

### `config.yaml`

根目录 `config.yaml` 是后端的主要配置文件，包括：

- `services.asr`：OpenAI 兼容 ASR 网关（`base_url` 带 `/v1` 前缀，含 `model`、`api_key`、`timeout_seconds`）
- `services.tts`：Breeze-TTS-2 地址
- `services.voiceprint`：本地声纹服务地址
- `services.kws`：KWS 地址、启用状态与失败降级策略
- `database.postgresql`：项目使用的 PostgreSQL 连接信息
- `agent`：角色模型（AgentProfile），注入决策模型交互判断，让模型知道"助手是谁"；进程内 direct-LLM Agent 另有一份 `conversation_agent.SYSTEM_PROMPT` 人设
- `assistant`：唤醒词、连续对话窗口和打断正则
- `decision_model`：交互决策模型地址、模型名、超时、思考开关和重试次数
- `direct_llm`：对话 Agent 使用的 OpenAI 兼容聊天模型（`base_url`、`model`、`api_key`、`timeout_seconds`、`max_tokens`、`temperature`、`enable_thinking`）
- `teleop`：LeRobot SO101 局域网 MCP（`url`/`token`/`enabled`/`timeout_seconds`），同服务暴露 9 个工具（4 遥操 + 4 推理 + 1 双摄视觉问答），`enabled=true` 且 token 非空时以 `teleop.*` 技能注入系统提示词，由进程内 direct-LLM Agent 调用；token 写进 gitignored 的 `config.yaml`，或用环境变量 `TELEOP_MCP_TOKEN`
- `tts`：对话 Agent 未返回动态语气时的兜底提示词、`cfg_scale`、参考音频及其文本

后端默认读取仓库根目录的 `config.yaml`；可以通过 `CONFIG_PATH` 环境变量指定其他路径。`tts.ref_audio` 的相对路径以 `backend/` 为基准。

### `.env`

根目录 `.env` 供 Docker Compose 注入服务配置，主要包括：

- `POSTGRES_DB` / `POSTGRES_USER` / `POSTGRES_PASSWORD`：compose 内置 PostgreSQL 的库与账号（声纹服务自动复用，需与 `config.yaml` 的 `database.postgresql` 保持一致）
- `VOICEPRINT_IDENTIFY_THRESHOLD`
- `VOICEPRINT_UNCERTAIN_THRESHOLD`
- `KWS_KEYWORDS_SCORE`
- `KWS_KEYWORDS_THRESHOLD`
- `KWS_NUM_TRAILING_BLANKS`

声纹和 KWS 阈值只是初始值，应使用真实麦克风、房间和用户录音进行校准。不要提交包含真实密码或私有服务信息的 `config.yaml`、`.env`。

## 运行

### 1. 启动 PostgreSQL、声纹与 KWS 服务

`compose.yaml` 自带专属 PostgreSQL（宿主机端口 5433，避开其他项目的 5432）、声纹与 KWS 服务。先在 `.env` 里设置 `POSTGRES_PASSWORD`（需与 `config.yaml` 的 `database.postgresql.password` 一致），然后：

```bash
docker compose up -d --build
```

声纹服务启动时会自动执行表结构迁移，无需手动建库建表。

健康检查：

```bash
curl http://127.0.0.1:8003/health  # KWS
curl http://127.0.0.1:8002/health  # voiceprint
```

首次启动会下载 KWS 和 CAM++ 模型。更详细的部署、接口与调参说明见：

- [独立关键词唤醒服务](docs/kws-service.md)
- [本地声纹与用户服务](docs/voiceprint-service.md)

如果不使用声学 KWS，可在 `config.yaml` 中设置：

```yaml
services:
  kws:
    enabled: false
```

关闭后仍可使用 ASR 文本唤醒词。

### 2. 启动后端

```bash
cd backend
uv sync
uv run uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

后端健康检查：

```bash
curl http://127.0.0.1:8000/health
```

### 3. 启动前端

```bash
cd frontend
npm install
npm run dev
```

访问 <http://localhost:5173>。Vite 会把 `/api/*` 代理到 `http://127.0.0.1:8000`。

> 浏览器麦克风只能在安全上下文中使用，例如 `localhost` 或 HTTPS。

## 使用方式

### Silero VAD 语音

开启“Silero VAD 语音监听”后，浏览器会持续占用麦克风：

1. VAD 检测到停顿后提交完整语音；
2. ASR 与 KWS 并行处理；
3. 后端解析声纹身份（每段语音都会解析，无参与资格门槛）；
4. 决策模型结合上下文与说话人判断是否回应（normal / brief / silent）；
5. 命中回应时由进程内 direct-LLM Agent 生成回复，并流式 TTS 朗读；
6. 决策模型异常时回退到唤醒词与连续对话窗口门控。

外放的 TTS 可能被麦克风重新采集，建议开启 VAD 时使用耳机。关闭 VAD 开关后浏览器会释放麦克风。

## 后端 API 概览

| 方法 | 路径 | 说明 |
|---|---|---|
| `GET` | `/health` | 后端状态与上游服务地址 |
| `POST` | `/api/session/start` | 创建本地交互 session |
| `POST` | `/api/transcribe?session_id=<id>` | 转写 VAD 产生的完整语音片段（整段 WAV 一次性提交 OpenAI 兼容 ASR 网关） |
| `POST` | `/api/kws/detect` | 调用可选的本地 KWS 服务 |
| `GET` | `/api/session/{id}` | 查询激活状态与侦听到的声音 |
| `POST` | `/api/session/{id}/turns` | 在 speech-start 创建不可变语音轮次 |
| `POST` | `/api/session/{id}/turns/{turn_id}/evaluate` | 解析声纹、调用决策模型并执行唤醒/打断门控 |
| `POST` | `/api/chat` | 消费已获准的 `turn_id` 并交给进程内对话 Agent 生成回复 |
| `POST` | `/api/touch` | 延长连续对话激活窗口 |
| `POST` | `/api/tts` | 代理流式 TTS PCM |

FastAPI 后端还提供声纹用户和模板管理代理接口，详细契约见 [声纹服务文档](docs/voiceprint-service.md)。

## 上游音频协议

- ASR 输入：浏览器 VAD 产出 little-endian float32 PCM、16kHz、单声道片段；后端将其转成 16-bit PCM WAV，通过 `POST {asr.base_url}/audio/transcriptions`（multipart，带 `model` 字段）一次性提交转写。
- TTS 请求：对话 Agent 每轮返回 `text` 和动态 `instruction`；后端以 `multipart/form-data` 调用 `/v1/audio/speech`，并附带 `instruction`、配置的 `cfg_scale`、`ref_audio` 和 `ref_text`。Agent 未返回有效 `instruction` 时回退到 `tts.prompt`。
- TTS 响应：`audio/pcm` 裸 PCM、16-bit、单声道，采样率由上游 `/health` 返回（当前默认 24000Hz）；后端通过 `X-Sample-Rate` 响应头传给前端。
- Breeze-TTS-2 上游只支持单并发，后端通过锁避免同时发起多个合成请求。

## 已知限制与安全说明

- 免手持模式会持续占用麦克风；关闭该模式或离开页面后才会释放。
- 后端 session 当前保存在进程内存中，后端重启后会丢失，也不适合直接进行无粘性多实例部署。
- 声纹识别不具备防录音回放或防语音克隆能力，不能单独用于支付、开锁等高风险认证。
- 声纹服务 API 当前没有认证，只应绑定本机或可信内网，不应直接暴露公网。
- KWS 与声纹阈值依赖实际设备和环境，默认值不构成准确率或安全保证。
- KWS 默认 `fail_open: true` 表示服务异常时降级为精确文本唤醒，而不是绕过唤醒门控。

## 更新记录

### 2026-10-27 · 移除 Hermes worker，对话统一由进程内 direct-LLM Agent 提供

- **去除 Hermes worker 依赖**：删除 `backend/session.py` 中的 `_chat_hermes` 子进程调用路径与 `parse_hermes_reply`/`VOICE_RESPONSE_CONTRACT`，`_chat_reply` 直接走进程内 `VoiceConversationAgent`。`config.py` 移除 `HERMES_CLIENT_PATH`/`HERMES_TIMEOUT_SECONDS` 与 `CONVERSATION_PROVIDER` 切换；`config.yaml(.example)` 删除 `conversation:` 与 `hermes:` 段。
- **类型改名**：`HermesReply` → `ChatReply`（仅作内部数据载体；`conversation_agent.AgentReply` 是带 `preferred_name` 的更宽版本，session 层 `ChatReply` 只取 `text`/`instruction`）。
- **前端文案**：header 与决策面板的「Hermes」改为「LLM」，`recordTiming("hermes")` 改为 `recordTiming("chat")`，相关注释同步。
- **历史条目保留**：下方旧更新记录里的 Hermes 字样属于历史，未改写。

### 2026-10-27 · 语音助手扩展至 LeRobot 全部 9 个 MCP 工具

- **从 4 个遥操工具扩展到 9 个**：`backend/teleop_skill.py` 现注册全部 `teleop.*` 技能——遥操 4 个（`start_teleoperation`/`stop_teleoperation`/`get_teleoperation_status`/`get_joint_curves`）、推理 4 个（`start_inference`/`stop_inference`/`get_inference_status`/`unload_inference_model`）、双摄视觉问答 1 个（`ask_camera_vision`）。9 个工具共用同一 MCP 服务、同一 token、同一 `TELEOP_ENABLED` 开关。技能定义按线上 `tools/list` 的真实 inputSchema 对齐（含 `instruction`/`question` 等必填、字符串与布尔类型）。
- **参数校验泛化**：`_validate` 由「只支持数值 min/max」改为按 schema 类型分别校验 integer/number/string(minLength-maxLength)/boolean，并强制 `required`。`start_teleoperation` 兜底摘要兼容 v1.3 的 `start_status`/`feedback.parameters` 新返回结构。
- **推理与视觉结果转述**：`_narration_payload` 新增推理状态分支（只保留 `phase`/`running`/`resident`/`total_dispatches` 等关键字段，丢弃体量很大的 `log_tail`）和视觉问答分支（透传 `answer` 并继续脱敏 URL token）。
- **提示词安全规则扩展**：`TELEOP_RULES` 明确 `start_inference` 同样属通电运动危险操作、需先确认现场安全；`instruction` 仅传用户自然语言任务、不得替用户指定模型/Corrector；异常时遥操用 `stop_teleoperation`、推理用 `stop_inference`；`ask_camera_vision` 不动机械臂但会把截图发往外部视觉模型。
- `config.yaml.example` 与 `readme` 注释同步更新为「9 个工具」口径。

### 2026-10-27 · 接入 LeRobot SO101 遥操 MCP

- **direct_llm Agent 接入遥操技能**：`backend/teleop_skill.py` 以极简 JSON-RPC 客户端直连 MCP 服务（`http://192.168.200.30:8791/mcp`，Streamable-HTTP、无会话、每请求一次 POST），暴露 4 个项目技能：`teleop.start_teleoperation`、`teleop.stop_teleoperation`、`teleop.get_teleoperation_status`、`teleop.get_joint_curves`。不引入 MCP SDK，复用 httpx，与 `voiceprint_skill.py` 同构。
- **技能列表可插拔**：`conversation_agent.py` 的系统提示词按 `config.TELEOP_ENABLED` 动态注入技能定义与遥操安全规则；`_execute_actions` 按技能名路由到 `VoiceprintManagementSkill` 或 `TeleopMCPSkill`。
- **二段转述**：遥操查询/启动类动作会真实返回数据，原先单趟「文本+动作」拿不到结果。`respond` 在动作执行后追加一次模型转述（`_narrate_tool_result`），把 `structuredContent` 压缩成口语化播报；转述失败时回退到 skill 自带的摘要。`voiceprint` 动作仍为单趟无结果。
- **Token 与 URL 不泄漏**：视频流/看板地址含 Bearer token，转述前经 `_redact_tokens` 去除 `?token=`，模型被要求只说「可在综合看板查看」、不念 URL。真实 token 仅存 gitignored 的 `config.yaml`（或 `TELEOP_MCP_TOKEN` 环境变量）。
- **危险操作门控**：`start_teleoperation` 会让实体机械臂通电运动。提示词要求先向用户确认现场安全才可调用，参数范围（fps 1–60、servo_hz 1–120、seconds 1–120）在 skill 层强制校验，单轮最多一个动作。

### 2026-10-27 · ASR 整段转写化 + 进程内 Agent 抽离

- **ASR 改为 OpenAI 兼容网关整段转写**：移除自定义流式协议（`/api/start`、`/api/chunk`、`/api/finish` 端点与远程 ASR session）。VAD 产出的完整语音片段由后端转成 16-bit PCM WAV，一次性 POST 到 `{asr.base_url}/audio/transcriptions`。`config.yaml` 的 `services.asr` 相应改为 `base_url`（带 `/v1`）+ `model` + `api_key` + `timeout_seconds`。
- **direct_llm 对话逻辑抽离为进程内 Agent**：原先内联在 `Session` 里的系统提示词、消息拼装、模型调用与技能执行下沉到 `backend/conversation_agent.py` 的 `VoiceConversationAgent`；`Session` 只负责构造 `AgentTurn` 并按返回的 `preferred_name` 更新声纹画像。

### 2026-10-20 · 决策模型化 + 去参与者

- **交互决策改为远程生成式决策模型**：移除本地判别式 Laya 后端，统一由 OpenAI 兼容的生成式决策模型判断「是否面向 AI / 是否回应」，默认端点 `http://mo.sy.cn:8000/v1`、模型 `qwen3.6-35b-awq`（可在 `config.yaml` 的 `decision_model` 段更换）。唤醒词强制回应、活跃窗口内延续追问兜底保留；模型异常时回退到唤醒词与连续对话窗口门控。
- **去掉「参与者」概念**：不再有参与人编号、自动加入、手动加入/踢出，也没有参与资格门槛。开启 VAD 即持续侦听，每段语音都会解析声纹身份并全部流向决策模型，是否回应完全由模型决定。移除 `POST/DELETE /api/session/{id}/participants/{speaker_id}` 端点与 `membership` 字段。
- **新增决策调度面板**：页面右侧可视化最近一轮的「输入 → 判断 → 结论」全链路（说话人、唤醒事件、会话状态 → 面向 AI、回应类型、置信度、决策来源 → 回应/沉默）。
- 激活窗口语义调整为由决策模型决定回应时才开启（即"助手回复了才开窗"），不再由唤醒词或加入触发。

