# 本地声纹与用户服务

该服务使用 CAM++ 模型 `iic/speech_campplus_sv_zh-cn_16k-common` 提取 192 维说话人向量。正常交互无需显式注册：VAD 唤醒成功后调用解析接口，匹配已有身份或自动创建匿名身份并返回稳定的 `speaker_id`。用户管理和显式声纹接口保留用于后台维护。

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

## 1. 准备已有 PostgreSQL

当前 compose 只启动声纹服务，复用本机已运行并暴露 `5432` 的 PostgreSQL。首次使用时，用 PostgreSQL 管理员账号创建项目独立账号和数据库：

```bash
docker exec -i postgres psql -U root -d postgres \
  -v app_user=qwen3_asr \
  -v app_password='替换为随机强密码' \
  -v app_database=qwen3_asr \
  < voiceprint-service/migrations/000_create_database.psql
```

如果容器名或管理员账号不同，请相应替换。服务启动时会幂等执行 `001_initial.sql`，创建 `users` 和 `voiceprints` 表。

## 2. 配置与启动

```bash
cp .env.example .env
```

修改 `.env` 中连接串的密码，使其与上一步一致，然后：

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

- 匹配到已有声纹：返回原有 `speaker_id`，`created: false`。
- 没有达到匹配阈值：在一个数据库事务中创建匿名用户及首个声纹模板，返回新 `speaker_id`，`created: true`。
- 未通过唤醒门控的环境语音不会创建身份。
- 身份创建由程序完成；Hermes 后续可以用 `speaker_id` 补充姓名、称呼和偏好，但不负责判定声纹是否相同。

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

## Hermes 语音管理

当前项目使用的 Hermes `worker` profile 已安装 `voiceprint-management` skill。每次通过门控的语音在发送给 Hermes 前，后端会附加受信任的机器上下文：

```text
[VOICE_CONTEXT]{"speaker_id":"...","audio_ref":"..."}[/VOICE_CONTEXT]
```

- `speaker_id` 用于查询和更新当前身份。
- `audio_ref` 是当前 VAD 音频的短期不透明引用，约五分钟后失效，并且只能用于对应身份。
- Hermes 通过 `python3 backend/voiceprintctl.py` 调用后端，不直接连接 PostgreSQL。
- 添加当前声音、查询、命名和列出模板可以直接执行；删除模板、全部声纹或整个身份必须先说明影响并在下一轮取得明确确认。

支持的自然语言示例：

- “我是谁？”
- “我叫许老板，以后叫我许老板。”
- “把我现在这段声音添加为一个声纹模板，叫客厅麦克风。”
- “我有几个声纹模板？”
- “有哪些声纹用户？”
- “删除我的全部声纹。”（Hermes 会先确认）

`voiceprintctl.py` 也可供人工维护：

```bash
python3 backend/voiceprintctl.py list
python3 backend/voiceprintctl.py show --speaker-id '<speaker_id>'
python3 backend/voiceprintctl.py templates --speaker-id '<speaker_id>'
```

### 唤醒问候与姓名绑定

后端会在受信任的 `VOICE_CONTEXT` 中额外提供：

- `wake_event`：本轮是否从未激活状态刚刚被唤醒；连续对话中为 `false`。
- `identity_named`：当前身份是否已有用户提供的名称；自动生成的 `访客 xxxxxxxx` 不算已命名。
- `display_name` / `preferred_name`：当前可靠称呼。

Hermes 按以下流程处理：

```text
冷启动唤醒 + 未命名身份
  → 回应用户，并询问如何称呼
  → 下一轮“我是 X / 我叫 X / 叫我 X”
  → 调用 voiceprintctl rename 更新同一个 speaker_id

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
