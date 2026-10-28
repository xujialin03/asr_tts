# 本地声纹与用户服务

该服务使用 CAM++ 模型 `iic/speech_campplus_sv_zh-cn_16k-common` 提取 192 维说话人向量。声纹服务只负责解析稳定身份；是否能参与当前对话由 FastAPI backend 的会话参与人状态决定。首位唤醒者会自动加入，激活期间识别到的其他身份只进入侦听列表，用户手动加入后才会进入对话 Agent。

## 设计

- 输入统一为裸 `float32 little-endian PCM`、16 kHz、单声道，与当前浏览器 VAD 输出一致。
- 一名用户可以注册多个声纹模板；识别时取该用户所有模板中的最高余弦相似度。
- PostgreSQL 保存用户资料、归一化 embedding、模型版本和音频质量元数据。
- 不保存注册或识别的原始音频。
- 默认只监听 `127.0.0.1:8002`（容器内为 `8001`），API 目前没有认证，不应直接暴露到公网。

初始阈值为：

- `score >= 0.55`：`matched`
- `0.45 <= score < 0.55`：`uncertain`
- `score < 0.45`：`unknown`

阈值只是初始值，必须用真实麦克风、房间和用户录音校准。普通声纹识别不具备防录音回放或防语音克隆能力，不能单独用于支付、开锁等高风险认证。

## 1. PostgreSQL

`compose.yaml` 自带专属 PostgreSQL（宿主机端口 5433，避开其他项目的 5432），并在 `.env` 的 `POSTGRES_DB/USER/PASSWORD` 中创建项目独立账号和数据库。声纹服务通过 compose 内部网络直接连到 `postgres:5432`，无需手动建库建表。

服务启动时会幂等执行 `001_initial.sql`，创建 `users` 和 `voiceprints` 表。

## 2. 配置与启动

```bash
cp .env.example .env
```

在 `.env` 中设置 `POSTGRES_PASSWORD`（需与 `config.yaml` 的 `database.postgresql.password` 一致），然后：

```bash
docker compose up -d --build voiceprint
docker compose logs -f voiceprint
```

首次启动会下载 CAM++ 模型到 Docker volume，耗时取决于网络。健康检查：

```bash
curl http://127.0.0.1:8002/health
```

接口文档：<http://127.0.0.1:8002/docs>

## 自动身份解析

现有前端在 VAD 取得完整语音、ASR 命中唤醒词或连续对话门控后，自动调用：

```text
POST /v1/resolve
Content-Type: application/octet-stream
Body: float32 PCM, 16 kHz, mono
```

- 匹配到已有声纹（`score >= accept_threshold`）：返回原有 `speaker_id`，`created: false`，`decision: "matched"`。
- 落在不确定区间（`uncertain_threshold <= score < accept_threshold`）：判定为**老用户声音漂移**（离麦远、嗓音状态变化等），默认把这段作为该身份的**又一个声纹模板**追加（`auto_enroll_uncertain=true`，见 `app/settings.py`），返回同一 `speaker_id`、`created: false`、`decision: "uncertain"`、`template_added: true`。这样同一个人越用越容易被认出，不会被重新建档成"访客"、导致助手反复追问身份。把 `auto_enroll_uncertain` 设为 `false` 可退回"不确定也新建"的旧行为。
- 低于不确定阈值（`score < uncertain_threshold`）或库为空：在一个数据库事务中创建匿名用户及首个声纹模板，返回新 `speaker_id`、`created: true`、`decision: "unknown"`。
- 未通过唤醒门控的环境语音不会调用解析，自然也不会建档。
- 身份建档由程序完成；对话 Agent 后续可以用 `speaker_id` 补充姓名、称呼和偏好，但不负责判定声纹是否相同。

> 冷启动：库为空时第一段被解析的声音即被建档为第一个身份（`created: true`），后续该人说话走 matched/uncertain。无需预先建档或 UI 登记——见下文"唤醒问候与姓名绑定"。


示例响应：

```json
{
  "speaker_id": "7d8180d5-e5e8-4fe0-bb68-2d8ad117c647",
  "created": true,
  "score": null,
  "speaker": {
    "id": "7d8180d5-e5e8-4fe0-bb68-2d8ad117c647",
    "username": "speaker-7d8180d5e5e84fe0bb682d8ad117c647",
    "display_name": "访客 7d8180d5",
    "preferred_name": null
  }
}
```

## 称呼管理（direct-LLM Agent 的声纹技能）

backend 进程内 direct-LLM Agent 通过受限的项目内 `voiceprint-management` skill 管理当前发言人称呼。Agent 每轮收到受信任的 `VOICE_CONTEXT`：

```text
[VOICE_CONTEXT]{"session_id":"...","turn_id":"...","wake_event":false,"joined_session":true,"speaker":{"speaker_id":"...","display_name":"访客 xxxx","preferred_name":null,"identity_named":false},"should_ask_name":true,"response_type":"normal"}[/VOICE_CONTEXT]
```

- `speaker.speaker_id` 表示当前已获准轮次的发言人；Agent 调用声纹技能时由 backend 绑定为该身份，模型不能传入、更换或枚举任意 `speaker_id`。
- `wake_event` / `joined_session`：本轮是否由唤醒词加入会话，用于决定是否首次询问称呼。
- `should_ask_name`：当前身份未命名且本会话未追问过时为 true，Agent 据此自然地问一次「我该怎么称呼你？」。
- 称呼持久化走声纹服务 `PATCH /v1/users/{speaker_id}`，不直连数据库。

当前只开放一个低风险动作：

```text
voiceprint.set_current_preferred_name(preferred_name)
```

- 仅当用户明确自报称呼（“我叫 X”“叫我 X”）时模型才可请求该动作；
- backend 校验称呼格式（1–32 个中英文/数字/下划线字符）后持久化 `preferred_name`，并同步当前 Session 的身份资料；
- 不开放删除用户、删除模板、添加模板、查询其他身份等动作。

支持的自然语言示例：

- “我是谁？”
- “我叫许老板，以后叫我许老板。”
- “有哪些声纹用户？”（未开放，会拒答）
- “删除我的全部声纹。”（未开放，会拒答）

因此也支持完整的自然闭环：匿名访客被唤醒 → 志玲问一次称呼 → 用户说“我是许老板” → 立即存为 `preferred_name: "许老板"` → 后续识别到同一声纹会使用该称呼。

### 人工维护 CLI

`voiceprintctl.py` 供人工查看和维护声纹数据（模型不会调用它）：

```bash
python3 backend/voiceprintctl.py list
python3 backend/voiceprintctl.py show --speaker-id '<speaker_id>'
python3 backend/voiceprintctl.py templates --speaker-id '<speaker_id>'
```

### 唤醒问候与姓名绑定

`VOICE_CONTEXT` 中的 `wake_event`、`identity_named`、`should_ask_name` 共同驱动问候与询问称呼：

```text
冷启动唤醒 + 未命名身份
  → 回应用户，并询问如何称呼（should_ask_name=true）
  → 下一轮“我是 X / 我叫 X / 叫我 X”
  → Agent 调用 voiceprint.set_current_preferred_name 更新同一个 speaker_id

冷启动唤醒 + 已命名身份
  → 使用 preferred_name（否则 display_name）自然问候

连续对话
  → 不重复问候，也不会每轮重复询问姓名
```

名称更新成功后，后端会同步当前内存 session 的身份资料，无需等待下一次声纹解析。`speaker_id` 始终不变，只更新用户名称字段。

## 后台用户管理

创建用户：

```bash
curl -X POST http://127.0.0.1:8002/v1/users \
  -H 'Content-Type: application/json' \
  -d '{
    "username": "owner",
    "display_name": "主人",
    "preferred_name": "老板",
    "locale": "zh-CN",
    "timezone": "Asia/Shanghai",
    "metadata": {"role": "owner"}
  }'
```

其他用户接口：

| 方法 | 路径 | 说明 |
|---|---|---|
| `GET` | `/v1/users` | 用户列表及有效声纹数量 |
| `GET` | `/v1/users/{user_id}` | 用户详情 |
| `PATCH` | `/v1/users/{user_id}` | 更新部分用户字段 |
| `DELETE` | `/v1/users/{user_id}` | 删除用户及其声纹 |

## 后台声纹注册与识别

音频文件必须是裸 float32 PCM，不是 WAV 文件。现有前端 VAD 的 `Float32Array.buffer` 可以直接作为请求 body。

注册模板：

```bash
curl -X POST \
  'http://127.0.0.1:8002/v1/users/<user_id>/voiceprints?label=客厅麦克风-1' \
  -H 'Content-Type: application/octet-stream' \
  --data-binary @enroll.f32
```

建议每名用户注册 5～10 段不同内容、每段 3～8 秒的自然语音。

识别已注册用户：

```bash
curl -X POST http://127.0.0.1:8002/v1/identify \
  -H 'Content-Type: application/octet-stream' \
  --data-binary @query.f32
```

验证是否为指定用户：

```bash
curl -X POST http://127.0.0.1:8002/v1/users/<user_id>/verify \
  -H 'Content-Type: application/octet-stream' \
  --data-binary @query.f32
```

声纹模板管理：

| 方法 | 路径 | 说明 |
|---|---|---|
| `GET` | `/v1/users/{user_id}/voiceprints` | 模板列表，不返回 embedding |
| `DELETE` | `/v1/users/{user_id}/voiceprints/{voiceprint_id}` | 删除模板 |

## 开发与测试

```bash
cd voiceprint-service
uv sync --extra cpu
uv run --extra cpu pytest
uv run --extra cpu uvicorn app.main:app --port 8001
```

CAM++ 网络结构源码取自 ModelScope 3D-Speaker，使用 Apache-2.0 许可证；许可证副本见 `THIRD_PARTY_LICENSE_3D-Speaker.txt`。
