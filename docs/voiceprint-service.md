# 本地声纹与用户服务

该服务使用 CAM++ 模型 `iic/speech_campplus_sv_zh-cn_16k-common` 提取 192 维说话人向量，提供用户管理、声纹注册、指定用户验证和已注册用户识别 API。

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

## 3. 用户管理

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

## 4. 声纹注册与识别

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
