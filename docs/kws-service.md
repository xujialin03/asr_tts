# 独立关键词唤醒服务

本模块使用 `sherpa-onnx` 和中文 WenetSpeech Zipformer KWS 模型检测原始音频中的唤醒词。它是一个独立、可选、可替换的 HTTP 服务；现有后端不依赖 sherpa-onnx 的 Python API，只依赖 `/v1/detect` 契约。

## 数据流

```text
浏览器 VAD 完整音频（float32 PCM / 16 kHz / mono）
  ├─ 远程 ASR：得到完整文本
  └─ 本地 KWS：检测“志玲/志玲姐姐”
       ↓
后端门控：KWS 命中 OR ASR 精确文本命中 OR 已在连续对话窗口
```

应用层没有 ASR 文本转拼音或编辑距离逻辑。WenetSpeech KWS 模型自身使用声母和带声调韵母作为建模单元，这是模型的声学 token 格式；运行时直接处理原始音频。

## 独立 HTTP 契约

### `GET /health`

返回引擎、模型和采样率。

### `POST /v1/detect`

请求：

```text
Content-Type: application/octet-stream
Body: little-endian float32 PCM, 16000 Hz, mono
```

响应：

```json
{
  "detected": true,
  "keyword": "志玲",
  "engine": "sherpa-onnx",
  "duration_ms": 1032
}
```

未来替换 openWakeWord 或其他模型时，只需提供同一契约。

## 配置关键词

[kws-service/keywords.txt](../kws-service/keywords.txt) 保存模型 token 序列和返回标签：

```text
zh ì l íng @志玲
zh ì l íng j iě j iě @志玲姐姐
```

修改后重建 KWS 容器。不要把“志明”“志宁”等 ASR 混淆文字加入文件；模型应根据声音决定是否与“志玲”接近。

## 启动

```bash
docker compose up -d --build kws
curl http://127.0.0.1:8003/health
```

首次启动下载约 31 MB 的官方模型归档到 `kws_models` Docker volume，并校验 SHA-256。服务绑定在 `127.0.0.1:8003`。

## 可选与移除

在根目录 `config.yaml` 中关闭：

```yaml
services:
  kws:
    enabled: false
```

关闭后后端 `/api/kws/detect` 返回 `available: false`，门控继续使用原有精确文本唤醒词。也可以删除 `compose.yaml` 的 `kws` service 与 `kws-service/`；ASR、声纹和对话 Agent 不需要修改。

KWS 服务运行失败时，默认 `fail_open: true`，自动降级到精确文本规则，不阻断语音对话。

## 阈值

`.env` 支持：

```env
KWS_KEYWORDS_SCORE=1.5
KWS_KEYWORDS_THRESHOLD=0.20
KWS_NUM_TRAILING_BLANKS=1
```

- `keywords_score` 越大，关键词路径越容易保留。
- `keywords_threshold` 越大，越难触发。
- 首轮值只是原型起点，需要用真实麦克风正样本和困难负样本校准。

建议测试集至少包括：

- 正样本：志玲、志玲姐姐、志玲在不在、志玲帮我……
- 困难近音：志明、志宁、志凌、智玲。
- 句中负样本：我昨天见到志明了、他叫志玲、电视播放相关人名。
- 不同说话人、距离、房间噪音和 TTS 回放。

## 模型来源

- sherpa-onnx 官方仓库：<https://github.com/k2-fsa/sherpa-onnx>
- 模型：`sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01`
- 下载：<https://github.com/k2-fsa/sherpa-onnx/releases/download/kws-models/sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01.tar.bz2>
- 许可证：Apache-2.0
