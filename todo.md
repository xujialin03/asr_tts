# 待办事项
- 输入缓冲区优化：已实现
    - 回复不重复发送：已有（turn.consumed + chat_lock + 串行队列）
    - 回复积压保留最新：`replyQueue`（上限 2 条，超期只回最新一条，clearAll 时清空）
    - 跨 turn 合并发送：评估后不做（决策语义绑定单 turn，VAD 切句合并已由后端 pending_buffer 覆盖）
- TTS 缓冲区优化：已实现
    - 忙时不重复发送：已有（后端 TTS_LOCK + 分段串行 + stopPlayback 抢占）
    - 409 退避重试一次：`ttsPlayer.js`（400ms 后重试，仍失败才报错）
    - 只播最新：已是抢占式（新回复 stopPlayback 掐旧播放）
