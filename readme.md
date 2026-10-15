# Qwen3-ASR 语音助手

一个浏览器端语音助手原型，串联 Qwen3-ASR、Silero VAD、sherpa-onnx KWS、CAM++ 声纹识别、Hermes agent 与 Breeze-TTS-2。

前端负责麦克风采集、免手持语音检测和流式音频播放；FastAPI 后端负责会话状态、唤醒/打断门控、声纹解析以及远程服务代理。对话人设和历史由项目专用 Hermes agent 独占管理。

## 功能

- 🎙️ **按住说话**：按住按钮或空格录音，松开后完成识别并发送给 Hermes。
- 👂 **免手持语音**：Silero VAD 持续监听麦克风，自动检测说话起止并切分完整语音片段。
- 🔔 **关键词唤醒**：冷启动时由 sherpa-onnx 声学 KWS 或 ASR 文本唤醒词激活助手；KWS 不可用时可降级为文本唤醒。
- 💬 **连续对话**：激活后在配置的时间窗口内无需重复唤醒，TTS 播放期间会自动延长窗口。
- ✋ **语音打断**：AI 播放期间先完成 ASR，再根据打断词确认是否停止播放，减少环境声音造成的误打断。
- 🧠 **Hermes 对话**：项目专用 Hermes agent 管理助理人设、上下文和对话历史，并为每条回复输出适配内容的 TTS 语气指令。
- 🔊 **流式语音回复**：Breeze-TTS-2 返回裸 PCM，浏览器边接收边播放；支持自动朗读、单条重播和停止播放。
- 🗣️ **声纹识别**：有效唤醒后由 CAM++ 匹配已有身份，或自动创建匿名 `speaker_id`。
- 📋 **声纹记录**：页面侧栏展示本次会话识别到的声音数量、声纹 ID、识别文本、相似度和建档状态。

> 交互状态机和设计约束见 [语音交互详细设计](docs/voice-interaction-design.md)。

## 技术栈

- 前端：Vue 3、Vite、Web Audio API
- 浏览器 VAD：Silero VAD（`@ricky0123/vad-web`）
- 后端：FastAPI、uv、httpx
- 语音识别：远程 Qwen3-ASR 流式识别服务
- 关键词唤醒：独立可选的 sherpa-onnx 中文 KWS 服务
- 声纹识别：CAM++ 服务 + PostgreSQL
- 对话 Agent：项目专用 Hermes agent
- 语音合成：远程 Breeze-TTS-2 流式合成服务
- 本地服务编排：Docker Compose

## 架构与语音链路

```text
浏览器（Vue 3）
  ├─ Push-to-talk：按住说话，松开识别
  ├─ Silero VAD：持续监听，自动切分 utterance
  └─ Web Audio：播放流式 PCM
          │
          ▼
FastAPI backend
  ├─ /api/chunk + /api/finish ─────────▶ 远程 Qwen3-ASR（按住说话）
  ├─ /api/transcribe ──────────────────▶ 远程 Qwen3-ASR（VAD 整段转写）
  ├─ /api/kws/detect ──────────────────▶ 本地 sherpa-onnx KWS
  ├─ /api/decide ──────────────────────▶ 唤醒词 / 连续对话 / 打断门控
  ├─ /api/voiceprint/resolve ──────────▶ 本地 CAM++ ─▶ PostgreSQL
  ├─ /api/chat ────────────────────────▶ 项目专用 Hermes agent
  └─ /api/tts ─────────────────────────▶ 远程 Breeze-TTS-2
```

免手持模式下，ASR 与 KWS 会并行处理完整语音片段。只有门控判定需要回复后，才会解析声纹并调用 Hermes；未触发的环境语音只显示识别结果，不创建声纹身份。

远程 ASR/TTS 不需要向浏览器开放 CORS，浏览器只访问 FastAPI；开发环境下 Vite 再将 `/api/*` 代理到 FastAPI。

## 目录结构

```text
backend/              FastAPI 后端、Hermes 调用和上游服务代理
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
- Docker 与 Docker Compose
- 一个可用的 PostgreSQL 实例（当前 `compose.yaml` 不启动 PostgreSQL）
- 可访问的 Qwen3-ASR 与 Breeze-TTS-2 服务
- 可调用的项目专用 Hermes agent client

如果只调试部分链路，KWS 可关闭；声纹服务不可用时，对话会按匿名身份继续。

## 配置

首次运行先复制配置模板：

```bash
cp config.yaml.example config.yaml
cp .env.example .env
```

### `config.yaml`

根目录 `config.yaml` 是后端的主要配置文件，包括：

- `services.asr`：Qwen3-ASR 地址
- `services.tts`：Breeze-TTS-2 地址
- `services.voiceprint`：本地声纹服务地址
- `services.kws`：KWS 地址、启用状态与失败降级策略
- `database.postgresql`：项目使用的 PostgreSQL 连接信息
- `assistant`：唤醒词、连续对话窗口和打断正则
- `hermes`：Hermes client 路径和超时时间
- `tts`：Hermes 未返回动态语气时的兜底提示词、`cfg_scale`、参考音频及其文本

后端默认读取仓库根目录的 `config.yaml`；可以通过 `CONFIG_PATH` 环境变量指定其他路径。`tts.ref_audio` 的相对路径以 `backend/` 为基准。

### `.env`

根目录 `.env` 供 Docker Compose 注入声纹/KWS 服务配置，主要包括：

- `VOICEPRINT_DATABASE_URL`
- `VOICEPRINT_IDENTIFY_THRESHOLD`
- `VOICEPRINT_UNCERTAIN_THRESHOLD`
- `KWS_KEYWORDS_SCORE`
- `KWS_KEYWORDS_THRESHOLD`
- `KWS_NUM_TRAILING_BLANKS`

声纹和 KWS 阈值只是初始值，应使用真实麦克风、房间和用户录音进行校准。不要提交包含真实密码或私有服务信息的 `config.yaml`、`.env`。

## 运行

### 1. 准备 PostgreSQL

声纹服务复用已有 PostgreSQL。首次使用时，用管理员账号创建项目数据库和独立账号：

```bash
docker exec -i postgres psql -U root -d postgres \
  -v app_user=qwen3_asr \
  -v app_password='替换为随机强密码' \
  -v app_database=qwen3_asr \
  < voiceprint-service/migrations/000_create_database.psql
```

按实际环境替换容器名、管理员账号和密码，并确保 `.env` 中的 `VOICEPRINT_DATABASE_URL` 使用相同凭据。声纹服务启动时会自动执行表结构迁移。

### 2. 启动 KWS 与声纹服务

```bash
docker compose up -d --build kws voiceprint
```

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

### 3. 启动后端

```bash
cd backend
uv sync
uv run uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

后端健康检查：

```bash
curl http://127.0.0.1:8000/health
```

### 4. 启动前端

```bash
cd frontend
npm install
npm run dev
```

访问 <http://localhost:5173>。Vite 会把 `/api/*` 代理到 `http://127.0.0.1:8000`。

> 浏览器麦克风只能在安全上下文中使用，例如 `localhost` 或 HTTPS。

## 使用方式

### 按住说话

按住页面中的麦克风按钮或键盘空格开始录音，松开后完成识别并直接发送给 Hermes。该模式属于显式操作，不要求唤醒词；播放中按住可立即停止当前回复。

### 免手持语音

开启“免手持语音（Silero VAD）”后，浏览器会持续占用麦克风：

1. VAD 检测到停顿后提交完整语音；
2. ASR 与 KWS 并行处理；
3. 冷启动需命中声学 KWS 或配置的文本唤醒词；
4. 激活窗口内可连续对话；
5. 门控通过后执行声纹解析、Hermes 回复和流式 TTS。

外放的 TTS 可能被麦克风重新采集，建议免手持模式下使用耳机。

## 后端 API 概览

| 方法 | 路径 | 说明 |
|---|---|---|
| `GET` | `/health` | 后端状态与上游服务地址 |
| `POST` | `/api/session/start` | 创建本地交互 session，远程 ASR 按需懒启动 |
| `POST` | `/api/chunk?session_id=<id>` | 推送原始 float32 PCM（16kHz、单声道）到 ASR |
| `POST` | `/api/finish?session_id=<id>` | 结束一次流式 ASR |
| `POST` | `/api/transcribe?session_id=<id>` | 转写 VAD 产生的完整语音片段 |
| `POST` | `/api/kws/detect` | 调用可选的本地 KWS 服务 |
| `POST` | `/api/decide` | 执行唤醒、连续对话和打断门控 |
| `POST` | `/api/voiceprint/resolve?session_id=<id>` | 匹配或创建声纹身份 |
| `POST` | `/api/chat` | 将当前用户话语发送给 Hermes |
| `POST` | `/api/touch` | 延长连续对话激活窗口 |
| `POST` | `/api/tts` | 代理流式 TTS PCM |

FastAPI 后端还提供声纹用户和模板管理代理接口，详细契约见 [声纹服务文档](docs/voiceprint-service.md)。

## 上游音频协议

- ASR 输入：little-endian float32 PCM、16kHz、单声道；流式模式通常每 500ms 推送一个 chunk。
- TTS 请求：Hermes 每轮返回 `text` 和动态 `instruction`；后端以 `multipart/form-data` 调用 `/v1/audio/speech`，并附带 `instruction`、配置的 `cfg_scale`、`ref_audio` 和 `ref_text`。Hermes 未返回有效 `instruction` 时回退到 `tts.prompt`。
- TTS 响应：`audio/pcm` 裸 PCM、16-bit、单声道，采样率由上游 `/health` 返回（当前默认 24000Hz）；后端通过 `X-Sample-Rate` 响应头传给前端。
- Breeze-TTS-2 上游只支持单并发，后端通过锁避免同时发起多个合成请求。

## 已知限制与安全说明

- 免手持模式会持续占用麦克风；关闭该模式或离开页面后才会释放。
- 后端 session 当前保存在进程内存中，后端重启后会丢失，也不适合直接进行无粘性多实例部署。
- 声纹识别不具备防录音回放或防语音克隆能力，不能单独用于支付、开锁等高风险认证。
- 声纹服务 API 当前没有认证，只应绑定本机或可信内网，不应直接暴露公网。
- KWS 与声纹阈值依赖实际设备和环境，默认值不构成准确率或安全保证。
- KWS 默认 `fail_open: true` 表示服务异常时降级为精确文本唤醒，而不是绕过唤醒门控。
