
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