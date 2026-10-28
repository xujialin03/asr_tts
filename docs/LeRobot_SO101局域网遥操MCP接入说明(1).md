# LeRobot SO101 局域网遥操与推理 MCP 接入说明

**版本：v1.3**  
**适用范围：同一局域网内的 Hermes Agent 设备**

## 1. 功能简介

本 MCP 服务部署在连接 SO101 主从臂和两路摄像头的 Mac mini 上。局域网内的 Hermes Agent 接入后，可以远程控制遥操，也可以用“任务指令 + 模型是否长驻”启动本机 VLA 推理，还可以针对当前双摄画面向 DeepSeek V4.1 Flash 提问。服务端负责把自然语言任务映射为规范 VLA 提示词、选择对应权重，并自动使用该权重预绑定的 Corrector；调用方不能自行指定或替换 Corrector。

本次扩展为**新增工具**：原有 4 个遥操工具和 4 个推理工具的名称、参数、默认值和调用方式均未改变，旧接入配置继续有效；已接入的 Hermes 执行 `/reload-mcp` 或重启后即可发现视觉问答工具。

MCP 服务地址：

```text
http://192.168.200.30:8791/mcp
```

服务采用 Bearer Token 鉴权。本文不包含真实 Token，请向设备所有者单独获取；不要通过公开群聊或代码仓库传递 Token。

## 2. 能力清单

| MCP 工具 | 作用 |
|---|---|
| `start_teleoperation` | 启动 SO101 主从臂遥操；支持设置两路摄像头 FPS 和舵机控制频率 |
| `stop_teleoperation` | 停止遥操，并执行从臂扭矩关闭与状态回读 |
| `get_teleoperation_status` | 查询运行阶段、PID、参数、主从臂曲线、视频流和看板地址 |
| `get_joint_curves` | 获取最近 1～120 秒的主臂、从臂六轴时间序列 |
| `start_inference` | 输入自然语言任务和是否长驻；服务端选择 VLA 权重、规范提示词及该权重预绑定的 Corrector，然后提交推理启动请求 |
| `stop_inference` | 停止当前推理并执行硬件清理；长驻模式下可保留模型热加载 |
| `get_inference_status` | 查询推理阶段、模型、Corrector、长驻和硬件清理状态 |
| `unload_inference_model` | 在任务停止后显式卸载长驻模型 |
| `ask_camera_vision` | 获取当前 Front/Third 双摄截图，发送给 DeepSeek V4.1 Flash，并返回基于画面的回答；不会移动机械臂 |

Hermes 加载后对应的工具名：

```text
mcp_lerobot_teleop_start_teleoperation
mcp_lerobot_teleop_stop_teleoperation
mcp_lerobot_teleop_get_teleoperation_status
mcp_lerobot_teleop_get_joint_curves
mcp_lerobot_teleop_start_inference
mcp_lerobot_teleop_stop_inference
mcp_lerobot_teleop_get_inference_status
mcp_lerobot_teleop_unload_inference_model
mcp_lerobot_teleop_ask_camera_vision
```

## 3. 硬件与数据定义

### 3.1 摄像头

- `front`：机械臂腕部/第一人称摄像头
- `third`：场地全景/第三人称摄像头

服务返回两路 MJPEG 实时视频流和 JPEG 快照地址。

### 3.2 主从臂六轴

```text
shoulder_pan
shoulder_lift
elbow_flex
wrist_flex
wrist_roll
gripper
```

综合看板同时显示两路视频、主臂六轴实线、从臂六轴虚线和当前运行状态。

## 4. Hermes Agent 接入

打开目标设备的 Hermes 配置文件：

```text
~/.hermes/config.yaml
```

合并以下配置；将 `<由设备所有者提供的TOKEN>` 替换为真实 Token：

```yaml
mcp_servers:
  lerobot_teleop:
    url: "http://192.168.200.30:8791/mcp"
    headers:
      Authorization: "Bearer <由设备所有者提供的TOKEN>"
    timeout: 180
    connect_timeout: 30
```

注意：如果配置文件中已经存在 `mcp_servers`，只添加 `lerobot_teleop` 子项，不要覆盖其他 MCP 配置。

配置完成后测试连接：

```bash
hermes mcp test lerobot_teleop
```

预期结果：

```text
Connected
Tools discovered: 9
```

已打开的 Hermes 会话可执行：

```text
/reload-mcp
```

也可以重启 Hermes Agent。

## 5. 使用方法

### 5.1 默认参数启动

对 Hermes 说：

```text
通过 lerobot_teleop MCP 启动遥操，使用默认参数。
```

默认参数：

| 参数 | 默认值 | 允许范围 | 含义 |
|---|---:|---:|---|
| `front_fps` | 30 | 1～60 | 腕部摄像头 FPS |
| `third_fps` | 30 | 1～60 | 场地摄像头 FPS |
| `servo_hz` | 60 | 1～120 | 主从臂遥操循环频率 |

### 5.2 指定参数启动

```json
{
  "front_fps": 30,
  "third_fps": 30,
  "servo_hz": 60
}
```

启动调用会等待以下项目全部就绪后才返回成功：遥操硬件进程处于 `running`、PID 有效、主从臂遥测有效、Front/Third 最新画面文件有效。成功响应返回统一状态值 `start_status: "running"` 和 `feedback` 字段：

```json
{
  "start_status": "running",
  "feedback": {
  "event": "teleoperation_started",
  "ok": true,
  "ready": true,
  "message": "遥操已启动，主从臂和双摄遥测已就绪",
  "mode": "teleop",
  "phase": "running",
  "pid": 12345,
  "parameters": {
    "front_fps": 30,
    "third_fps": 30,
    "servo_hz": 60
  },
  "telemetry_timestamp": 1234567890.123
  }
}
```

如果 20 秒内未达到就绪条件，调用返回错误并执行安全停止，不会返回虚假的“启动成功”。现有工具名和启动参数均未改变，旧调用方式继续兼容。

示例指令：

```text
通过 lerobot_teleop 启动遥操，front 设为 24 FPS，third 设为 30 FPS，舵机设为 50 Hz。
```

### 5.3 查询状态和画面

```text
查询当前遥操状态，并返回综合看板地址。
```

返回内容包括：运行状态、PID、实际参数、主从臂六轴曲线、Front/Third 视频流地址和综合看板地址。

### 5.4 获取曲线

```text
获取最近 30 秒的主从臂六轴曲线。
```

`seconds` 支持 1～120 秒。

### 5.5 停止遥操

```text
通过 lerobot_teleop 停止遥操。
```

停止流程会优先正常中断；必要时逐级终止进程，并执行从臂扭矩关闭和回读验证。

### 5.6 启动 VLA 推理

`start_inference` 只接受两个输入，其中 `resident` 可省略并默认为 `false`：

```json
{
  "instruction": "把紫色的鸭子夹到纸盒里",
  "resident": false
}
```

Hermes 自然语言示例：

```text
通过 lerobot_teleop MCP 把紫色的鸭子夹到纸盒里，模型不要长驻。
```

服务端会解析为：

```json
{
  "vla_prompt": "Grasp the purple duck, place it inside the tan cardboard box, and return the arm to its home position",
  "model_id": "duck-purple-100000",
  "corrector_id": "duck-corrector-b",
  "resident": false
}
```

上述 `vla_prompt`、`model_id` 和 `corrector_id` 是服务端解析结果，不是客户端输入参数。Corrector 与权重在服务端注册表中预先绑定；客户端不能传入 `model_id`、`task` 或 `corrector_id` 覆盖绑定。当前只部署了紫鸭入盒权重，不支持的任务会明确报错，不会猜测权重或错误启动机械臂。

`start_inference` 返回统一状态值 `start_status` 和 `inference_start_accepted` 反馈。通常刚提交时为 `"queued"`；若 worker 已立即处理，也可能直接返回 `loading`、`warming` 等真实阶段。它表示启动请求已进入推理程序，并回传解析后的提示词、模型、Corrector 和长驻设置；这不等同于模型已完成加载或机械臂已经开始运动，可继续调用 `get_inference_status` 查看 `loading`、`warming`、`connecting`、`predicting`、`executing` 等阶段。

```json
{
  "start_status": "queued",
  "feedback": {
    "event": "inference_start_accepted",
    "accepted": true,
    "model_id": "duck-purple-100000",
    "corrector_id": "duck-corrector-b",
    "resident": false
  }
}
```

### 5.7 模型长驻

- `resident: false`（默认）：任务结束后卸载模型并退出推理 worker。
- `resident: true`：任务停止并确认硬件释放后，保留模型热加载，便于下一次任务减少加载时间。
- 长驻只保留模型，不持续占用机械臂、摄像头或串口。
- 不再需要长驻模型时，先停止任务并等待硬件释放，再调用 `unload_inference_model`。

### 5.8 停止和查询推理

```text
查询当前推理状态。
停止当前推理。
卸载长驻模型。
```

推理和遥操互斥：任一模式正在占用硬件时，另一模式的启动请求会被拒绝。

### 5.9 双摄视觉问答

`ask_camera_vision` 只接受一个参数 `question`：

```json
{
  "question": "你看到了什么？"
}
```

Hermes 自然语言示例：

```text
通过 lerobot_teleop MCP 看一下当前两路摄像头，告诉我你看到了什么。
```

服务会同时提供：

- `front`：机械臂腕部/第一人称截图；
- `third`：场地全景/第三人称截图。

遥操运行时复用遥操进程发布的新鲜画面；推理运行时复用当前推理录像中的同序号双摄画面；设备空闲时只临时打开摄像头采集一对截图，随后立即释放并删除临时截图。若摄像头已被任务占用但没有可用的新鲜成对画面，调用会明确失败，不会再打开一组摄像头。

返回示例：

```json
{
  "vision_status": "completed",
  "answer": "front看到机械臂夹爪，third看到桌面和纸盒。",
  "model": "deepseek-v4-1-flash-261017",
  "response_status": "completed",
  "cameras": ["front", "third"],
  "capture_source": "on_demand",
  "captured_at": 1234567890.123
}
```

注意：调用此工具会把当前两路摄像头截图发送给设备所有者配置的外部 DeepSeek V4.1 Flash 接口。工具不会启动、通电或移动机械臂。

## 6. 安全要求

1. `start_teleoperation` 和 `start_inference` 都会使实体机械臂通电并运动，调用前必须确认现场有人看护、机械臂周围无人员和障碍物。
2. 不得在机械臂运动时将手伸入工作区域。
3. 遥操出现碰撞风险、异常抖动、失控或摄像头画面冻结时，应立即调用 `stop_teleoperation`；推理异常时立即调用 `stop_inference`。
4. Bearer Token 等同于遥操和推理控制权限，不得上传 Git、公开网盘或公开聊天群。
5. 本服务仅供可信局域网使用，不应直接暴露到公网。
6. 不要在多个设备上同时启动遥操或推理；服务会检查冲突的 LeRobot 控制进程与硬件租约。
7. `ask_camera_vision` 会将当前双摄截图发送到外部模型接口；画面含敏感信息时不要调用。

## 7. 网络与运行要求

- 使用方设备需要能访问 `192.168.200.30:8791`。
- 使用方设备与 Mac mini 应处于同一局域网，或具备可信的内网路由。
- Mac mini 必须登录桌面用户，否则 macOS 摄像头权限会话不可用，遥操启动将被拒绝。
- 服务健康检查地址：

```text
http://192.168.200.30:8791/healthz
```

健康响应示例：

```json
{"ok": true, "server": "lerobot-teleop", "version": "1.3.0"}
```

## 8. 常见问题

### 8.1 `401 unauthorized`

Bearer Token 缺失或错误。检查 `Authorization` 是否为：

```text
Bearer <真实TOKEN>
```

### 8.2 MCP连接失败

依次检查：

1. 是否在同一局域网；
2. `192.168.200.30` 是否可达；
3. 端口 `8791` 是否可达；
4. Hermes 配置缩进是否正确；
5. Token 是否正确；
6. 执行 `hermes mcp test lerobot_teleop` 查看具体错误。

### 8.3 工具已发现，但启动失败

可能原因包括：Mac mini 桌面未登录、已有遥操/录制/推理进程、机械臂串口被占用、摄像头未连接或 macOS 摄像头权限不可用。读取工具返回的错误信息，不要通过强制杀进程或修改系统权限绕过检查。

### 8.4 视频地址无法访问

视频和看板地址带鉴权参数，必须使用 MCP 最新返回的完整地址。旧地址或缺少 Token 的地址无法访问。

### 8.5 推理任务无法匹配

当前服务只会选择已部署且在服务端配置了任务匹配规则的权重。若返回“没有匹配到已部署权重”，说明该任务尚无可用权重；应先部署并注册相应权重，不能通过客户端强制指定模型或 Corrector。

### 8.6 `inference_start_accepted` 后未开始运动

该反馈只表示请求已被推理程序接收。调用 `get_inference_status` 查看真实阶段和错误信息；只有进入 `executing` 且有实际下发记录，才表示已经开始执行动作。

### 8.7 视觉问答失败

确认 Mac mini 已登录桌面、两路摄像头已连接、没有其他未受管程序占用摄像头，并查看工具返回的截图或模型连接错误。视觉模型密钥只保存在服务端，不需要也不允许由 MCP 调用方传入。

## 9. 当前部署信息

- 部署版本：`v0.9.0-mcp-vision`
- MCP协议验证：Hermes原生客户端和 mcporter 均通过
- 服务端口：`8791`
- MCP服务版本：`1.3.0`
- 工具数量：9（4个遥操工具 + 4个推理工具 + 1个双摄视觉问答工具）
- 当前设计：PI0.5推理、录制和遥操互斥；自然语言任务在服务端映射为规范提示词和权重，Corrector仅按权重预绑定关系启用；视觉问答复用新鲜双摄画面且不控制机械臂

---

**文档说明：** 本文可发给接入方；真实 Bearer Token 请通过单独的安全渠道提供。