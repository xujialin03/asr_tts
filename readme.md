# 技术栈
- 前端：Vue 3 + Vite
- 后端：FastAPI + uv
- 语音检测VAD: Silero VAD
- 语音识别：Qwen3-ASR 流式识别服务
- LLM：Qwen3.8-Flash-Next 
- 语音合成：Breeze-TTS-2 流式合成服务

# Qwen3-ASR 语音转文字

把语音实时转成文字的 Web 应用。前端 Vue 采集/上传音频，后端 FastAPI 代理转发到远程 Qwen3-ASR 流式识别服务。

```
浏览器 (Vue)  ──/api/*──▶  FastAPI 代理 (backend)  ──▶  远程 ASR 服务
```

远程服务不带 CORS 头，浏览器无法直连，所以中间加一层后端代理。

## 目录结构

```
backend/    FastAPI 代理（uv 管理）
frontend/   Vue 3 + Vite 前端
```

## 远程 ASR 协议

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/start` | 返回 `{session_id}` |
| POST | `/api/chunk?session_id=<id>` | body 为原始 float32 PCM, 16kHz, 单声道；返回 `{language, text}` |
| POST | `/api/finish?session_id=<id>` | 结束会话，返回最终 `{language, text}` |

## 远程 TTS 协议（流式）

TTS 同样是流式服务：响应为 `Transfer-Encoding: chunked`，可边生成边读取播放。

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/health` | 返回 `{status, sample_rate}`（当前 `sample_rate: 24000`） |
| POST | `/v1/audio/speech` | 合成语音，**multipart/form-data**；响应为裸 PCM 音频流 |

请求字段（multipart form）：

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `text` | string | ✅ | 要合成的文本 |
| `instruction` | string | ❌ | 音色/语气提示词（见 `config.yaml` 的 `tts.prompt`） |
| `ref_audio` | file | ❌ | 参考音频（克隆音色用） |
| `ref_text` | string | ❌ | 参考音频对应文本 |
| `cfg_scale` | number | ❌ | 默认 1.0 |
| `seed` | integer | ❌ | 默认 42 |

响应：`content-type: audio/pcm`，**裸 PCM（无 WAV 头），16-bit 单声道，24000Hz**，chunked 流式。播放前需补 WAV 头或按该格式解码。

curl 示例：

```bash
curl -X POST "http://<tts-host>:<tts-port>/v1/audio/speech" \
  -F 'text=你好，这是一段语音合成测试。' \
  -F 'instruction=一位温柔自信的年轻女性，声音清晰，语气亲切' \
  -o out.pcm
```

> 服务地址与端口见 `config.yaml`（`services.tts`）。

## 运行

### 1. 后端（uv）

```bash
cd backend
uv sync
uv run uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

远程地址在 `backend/.env` 里配置（`ASR_BASE_URL`）。

### 2. 前端（Vite）

```bash
cd frontend
npm install
npm run dev                   # http://localhost:5173
```

开发时 Vite 把 `/api/*` 代理到 `http://127.0.0.1:8000`（见 `frontend/vite.config.js`），浏览器只访问同一个源，无跨域问题。

## 功能

- 🎙 实时录音识别：麦克风 → 重采样 16kHz float32 → 每 500ms 推一个 chunk → 实时显示文字与语言。
- 📁 上传音频文件：解码任意音频 → 混合单声道 → 重采样 → 流式推送 → 输出转写。

> 浏览器麦克风需要安全上下文（localhost 或 HTTPS）。