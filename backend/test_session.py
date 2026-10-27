import asyncio
import array
import wave

from decision import InteractionDecision
from session import HermesReply, Session, SpeakerIdentity, _pcm_to_wav, parse_hermes_reply


def _f32(frames):
    arr = array.array("f", frames)
    return arr.tobytes()


def test_pcm_to_wav_roundtrips_length_and_format():
    raw = _f32([0.0, 0.5, -0.5, 1.2, -1.5, 0.25])
    out = _pcm_to_wav(raw)
    # WAV header is 44 bytes; 6 samples at PCM16 = 12 bytes payload.
    assert len(out) == 44 + 12
    import io

    with wave.open(io.BytesIO(out), "rb") as wav:
        assert wav.getnchannels() == 1
        assert wav.getsampwidth() == 2
        assert wav.getframerate() == 16000
        assert wav.getnframes() == 6
        frames = wav.readframes(6)
    values = array.array("h", frames).tolist()
    # Clamped to int16 range, ~32767 scale.
    assert values[0] == 0
    assert values[1] == int(0.5 * 32767)
    assert values[3] == 32767  # 1.2 clamped up to 1.0
    assert values[4] == -32767  # -1.5 clamped down to -1.0


def test_pcm_to_wav_drops_trailing_partial_sample():
    raw = _f32([0.5, 0.5]) + b"\x01\x02"  # 2 full f32 + 2 stray bytes
    out = _pcm_to_wav(raw)
    import io

    with wave.open(io.BytesIO(out), "rb") as wav:
        assert wav.getnframes() == 2


def test_pcm_to_wav_empty():
    out = _pcm_to_wav(b"")
    assert len(out) == 44  # header only, zero frames


def identity(speaker_id: str) -> SpeakerIdentity:
    return SpeakerIdentity(
        speaker_id=speaker_id,
        display_name=f"访客 {speaker_id}",
        score=0.9,
    )


def evaluate(
    session: Session,
    speaker_id: str,
    text: str,
    *,
    active: bool | None = None,
    assistant_speaking: bool = False,
):
    if active is True:
        session.activate()
    elif active is False:
        session.active_until = 0
    turn = session.begin_turn("vad", assistant_speaking)
    return session.evaluate_turn(
        turn.id,
        identity(speaker_id),
        b"audio",
        text=text,
        language="zh",
        kws_detected=False,
        kws_keyword=None,
    )


def test_first_wake_turn_joins_session_and_is_provisionally_admitted():
    session = Session(id="s")

    turn = evaluate(session, "a", "志玲你好", active=False)

    assert turn.should_reply is True
    assert turn.wake_event is True
    assert turn.joined_session is True
    assert session.participant_ids == {"a"}
    assert "a" in session.observed_speakers


def test_speaker_can_continue_without_wake_while_active():
    session = Session(id="s")
    evaluate(session, "a", "志玲你好", active=False)
    # The assistant's reply opens the continuous-conversation window (decide_turn
    # calls activate() when the model responds).
    session.activate()

    turn = evaluate(session, "a", "今天天气怎么样")

    assert turn.started_while_active is True
    assert turn.wake_event is False
    assert turn.should_reply is True
    assert turn.reason == "在连续对话窗口内"


def test_queued_turn_uses_activation_created_before_evaluation():
    session = Session(id="s")
    evaluate(session, "a", "志玲你好", active=False)
    session.active_until = 0
    queued = session.begin_turn("vad", False)
    assert queued.started_while_active is False

    # An earlier queued turn or TTS playback activates the session before this
    # turn reaches evaluate_turn(). It must not require another wake word.
    session.activate()
    result = session.evaluate_turn(
        queued.id,
        identity("a"),
        b"audio",
        text="接着刚才的话题",
        language="zh",
        kws_detected=False,
        kws_keyword=None,
    )

    assert result.should_reply is True
    assert result.wake_event is False
    assert result.reason == "在连续对话窗口内"


def test_inactive_nonparticipant_without_wake_is_rejected():
    session = Session(id="s")

    turn = evaluate(session, "a", "继续聊聊", active=False)

    assert turn.should_reply is False
    assert turn.reason == "未加入当前会话：需要先说唤醒词"


def test_active_nonparticipant_is_filtered_until_they_wake():
    session = Session(id="s")
    evaluate(session, "a", "志玲你好", active=False)
    session.activate()

    turn = evaluate(session, "b", "你们在聊什么", active=True)

    assert turn.should_reply is False
    assert turn.reason == "未加入当前会话：需要先说唤醒词"
    assert "b" not in session.participant_ids
    assert "b" not in session.observed_speakers


def test_second_speaker_joins_with_wake_even_while_session_is_active():
    session = Session(id="s")
    evaluate(session, "a", "志玲你好", active=False)
    session.activate()

    turn = evaluate(session, "b", "志玲，我也想聊聊", active=True)

    assert turn.joined_session is True
    assert turn.should_reply is True
    assert session.participant_ids == {"a", "b"}
    assert "b" in session.observed_speakers


def test_interrupt_applies_only_to_joined_speakers():
    session = Session(id="s")
    evaluate(session, "a", "志玲你好", active=False)

    outsider = evaluate(session, "b", "停一下", active=True, assistant_speaking=True)
    member = evaluate(session, "a", "停一下", active=True, assistant_speaking=True)

    assert outsider.interrupt is False
    assert outsider.reason == "未加入当前会话：需要先说唤醒词"
    assert member.interrupt is True
    assert member.should_reply is False


def test_empty_room_ordinary_speech_is_not_observed_or_admitted():
    session = Session(id="s")

    turn = evaluate(session, "a", "你好", active=False)

    assert turn.wake_event is False
    assert turn.should_reply is False
    assert turn.reason == "未加入当前会话：需要先说唤醒词"
    assert session.observed_speakers == {}


def test_audio_refs_do_not_overwrite_each_other():
    session = Session(id="s")
    ref_a = session.store_voice_audio(b"a", "a", "turn-a")
    ref_b = session.store_voice_audio(b"b", "b", "turn-b")

    assert session.get_voice_audio("a", ref_a) == b"a"
    assert session.get_voice_audio("b", ref_b) == b"b"
    assert session.get_voice_audio("a", ref_b) is None


def test_parse_structured_reply():
    reply = parse_hermes_reply(
        '{"text":"我们慢慢来。","instruction":"用平静、安心的语气缓慢表达。"}'
    )

    assert reply == HermesReply(
        text="我们慢慢来。",
        instruction="用平静、安心的语气缓慢表达。",
    )


def test_parse_json_code_fence():
    reply = parse_hermes_reply(
        '```json\n{"text":"太好了！","instruction":"用明快、喜悦且富有感染力的语气表达。"}\n```'
    )

    assert reply.text == "太好了！"
    assert reply.instruction == "用明快、喜悦且富有感染力的语气表达。"


def test_parse_json_after_short_preamble():
    reply = parse_hermes_reply(
        '当然。\n{"text":"请留意前方。","instruction":"用克制、严肃的语气清晰表达，稍微放慢语速。"}'
    )

    assert reply.text == "请留意前方。"
    assert reply.instruction == "用克制、严肃的语气清晰表达，稍微放慢语速。"


def test_plain_text_reply_remains_compatible():
    reply = parse_hermes_reply("这是旧版 Hermes 的纯文本回复。")

    assert reply == HermesReply(text="这是旧版 Hermes 的纯文本回复。")


def test_invalid_instruction_does_not_break_reply():
    reply = parse_hermes_reply('{"text":"可以。","instruction":{"tone":"warm"}}')

    assert reply == HermesReply(text="可以。")


class StubDecisionClient:
    def __init__(self, decision: InteractionDecision | Exception):
        self.decision = decision
        self.states = []

    async def decide(self, state):
        self.states.append(state)
        if isinstance(self.decision, Exception):
            raise self.decision
        return self.decision


class ScriptedDecisionClient:
    """Return a decision per call, in order; record the current_message seen."""

    def __init__(self, decisions: list[InteractionDecision]):
        self.decisions = decisions
        self.messages = []

    async def decide(self, state):
        self.messages.append(state["current_message"])
        return self.decisions[len(self.messages) - 1]


def test_buffer_holds_incomplete_fragment_then_merges_next():
    client = ScriptedDecisionClient([
        InteractionDecision(True, False, "silent", 0.5, "残句", is_complete=False),
        InteractionDecision(True, True, "normal", 0.9, "完整提问", is_complete=True),
    ])
    session = Session(id="s", decision_client=client)

    first = evaluate(session, "a", "今天", active=True)
    first = asyncio.run(session.decide_turn(first))
    assert first.should_reply is False
    assert first.reason == "输入缓冲区：句子未说完，等待后续合并"
    assert session.pending_buffer and session.pending_buffer["text"] == "今天"

    second = evaluate(session, "a", "天气怎么样啊", active=True)
    second = asyncio.run(session.decide_turn(second))
    assert second.should_reply is True
    assert second.buffer_merged is True
    # The model saw the merged utterance, and the reply carries the merged text.
    assert client.messages[-1] == "今天 天气怎么样啊"
    assert second.reply_text == "今天 天气怎么样啊"
    assert session.pending_buffer is None


def test_buffer_flushes_when_next_segment_is_complete():
    client = ScriptedDecisionClient([
        InteractionDecision(True, False, "silent", 0.5, "残句", is_complete=False),
        InteractionDecision(True, True, "normal", 0.9, "完整", is_complete=True),
    ])
    session = Session(id="s", decision_client=client)
    asyncio.run(session.decide_turn(evaluate(session, "a", "那个", active=True)))
    turn = asyncio.run(session.decide_turn(evaluate(session, "a", "厦门天气怎么样", active=True)))
    assert turn.should_reply is True
    assert turn.reply_text == "那个 厦门天气怎么样"


def test_buffer_does_not_merge_across_speakers():
    client = ScriptedDecisionClient([
        InteractionDecision(True, False, "silent", 0.5, "残句", is_complete=False),
        InteractionDecision(True, True, "normal", 0.9, "完整", is_complete=True),
    ])
    session = Session(id="s", decision_client=client)
    asyncio.run(session.decide_turn(evaluate(session, "a", "今天", active=True)))
    turn = asyncio.run(session.decide_turn(evaluate(session, "b", "天气怎么样", active=True)))
    # Different speaker: the held fragment is dropped, not merged.
    assert turn.reply_text == "天气怎么样"
    assert turn.buffer_merged is False


def test_buffer_disabled_holds_nothing():
    import config as cfg
    original = cfg.BUFFER_ENABLED
    cfg.BUFFER_ENABLED = False
    try:
        client = StubDecisionClient(
            InteractionDecision(True, False, "silent", 0.5, "残句", is_complete=False)
        )
        session = Session(id="s", decision_client=client)
        turn = asyncio.run(session.decide_turn(evaluate(session, "a", "今天", active=True)))
        assert session.pending_buffer is None
        assert turn.should_reply is False
    finally:
        cfg.BUFFER_ENABLED = original


def test_decision_state_includes_agent_persona_and_session_model():
    client = StubDecisionClient(
        InteractionDecision(True, True, "normal", 0.9, "用户在延续对话")
    )
    session = Session(id="s", decision_client=client)
    first = evaluate(session, "a", "志玲，厦门天气怎么样", active=False)
    asyncio.run(session.decide_turn(first))
    turn = evaluate(session, "a", "那明天呢", active=True)

    asyncio.run(session.decide_turn(turn))

    state = client.states[-1]
    assert state["agent"]["name"]
    assert state["agent"]["persona"]
    assert state["current_message"] == "那明天呢"
    assert state["speaker"]["speaker_id"] == "a"
    # The current utterance is passed as current_message; topics holds the
    # already-established topics from prior admitted turns.
    assert state["session"]["topics"] == ["志玲，厦门天气怎么样"]
    assert [entry["text"] for entry in state["session"]["recent_messages"]] == [
        "志玲，厦门天气怎么样",
    ]


def test_continuation_followup_after_assistant_reply_is_forced_to_reply():
    client = StubDecisionClient(
        InteractionDecision(True, False, "silent", 0.0, "无实质信息或重复反馈")
    )
    session = Session(id="s", decision_client=client)
    first = evaluate(session, "a", "志玲，厦门今天天气怎么样", active=False)
    asyncio.run(session.decide_turn(first))
    # The assistant's weather answer lands in context (as chat() would do).
    session._append_context("assistant", "厦门今天大晴天，最高33度。")

    turn = evaluate(session, "a", "最近是不是有个台风啊", active=True)
    result = asyncio.run(session.decide_turn(turn))

    assert result.should_reply is True
    assert result.decision.reason == "活跃窗口内延续助理话题，兜底进入回复流程"


def test_bare_acknowledgment_after_assistant_reply_stays_silent():
    client = StubDecisionClient(
        InteractionDecision(True, False, "silent", 0.6, "无实质信息或重复反馈")
    )
    session = Session(id="s", decision_client=client)
    first = evaluate(session, "a", "志玲，厦门今天天气怎么样", active=False)
    asyncio.run(session.decide_turn(first))
    session._append_context("assistant", "厦门今天大晴天，最高33度。")

    turn = evaluate(session, "a", "嗯好的", active=True)
    result = asyncio.run(session.decide_turn(turn))

    assert result.should_reply is False
    assert result.decision.response_type == "silent"


def test_human_to_human_silent_is_not_forced_to_reply():
    client = StubDecisionClient(
        InteractionDecision(False, False, "silent", 0.9, "人类之间的对话")
    )
    session = Session(id="s", decision_client=client)
    first = evaluate(session, "a", "志玲，厦门今天天气怎么样", active=False)
    asyncio.run(session.decide_turn(first))
    session._append_context("assistant", "厦门今天大晴天，最高33度。")

    turn = evaluate(session, "a", "你晚上是不是有个局啊", active=True)
    result = asyncio.run(session.decide_turn(turn))

    # addressed=False means the decision model judged it human-to-human; the fallback must not fire.
    assert result.should_reply is False


def test_decision_model_silent_does_not_admit_turn_or_write_context():
    client = StubDecisionClient(
        InteractionDecision(False, False, "silent", 0.9, "两位用户正在互相交谈")
    )
    session = Session(id="s", decision_client=client)
    evaluate(session, "a", "志玲你好", active=False)
    turn = evaluate(session, "a", "晚上吃什么", active=True)

    result = asyncio.run(session.decide_turn(turn))

    assert result.should_reply is False
    assert result.decision.response_type == "silent"
    assert session.conversation_context == []
    assert client.states[0]["current_message"] == "晚上吃什么"


def test_decision_model_brief_admits_turn_and_records_user_context():
    client = StubDecisionClient(
        InteractionDecision(True, True, "brief", 0.8, "用户在确认上一轮提问")
    )
    session = Session(id="s", decision_client=client)
    initial = evaluate(session, "a", "志玲你好", active=False)
    asyncio.run(session.decide_turn(initial))
    turn = evaluate(session, "a", "音乐吧", active=True)

    result = asyncio.run(session.decide_turn(turn))

    assert result.should_reply is True
    assert result.decision.response_type == "brief"
    assert [entry["text"] for entry in session.conversation_context] == ["志玲你好", "音乐吧"]


def test_decision_model_failure_uses_existing_gate_as_fallback():
    session = Session(id="s", decision_client=StubDecisionClient(TimeoutError("slow model")))
    turn = evaluate(session, "a", "志玲你好", active=False)

    result = asyncio.run(session.decide_turn(turn))

    assert result.should_reply is True
    assert result.decision.fallback is True
    assert result.decision.response_type == "normal"


def test_wake_word_overrides_decision_model_silent():
    session = Session(
        id="s",
        decision_client=StubDecisionClient(
            InteractionDecision(True, False, "silent", 0.9, "无实质信息或重复反馈")
        ),
    )
    turn = evaluate(session, "a", "志玲姐姐，你在不在？", active=False)

    result = asyncio.run(session.decide_turn(turn))

    assert result.wake_detected is True
    assert result.should_reply is True
    assert result.decision.response_type == "normal"
    assert result.decision.reason == "命中唤醒词，强制进入回复流程"


def test_context_window_keeps_only_five_real_messages():
    session = Session(id="s")

    for index in range(6):
        session._append_context("user", str(index), "a")

    assert [entry["text"] for entry in session.conversation_context] == ["1", "2", "3", "4", "5"]


def test_real_reply_is_added_to_context(monkeypatch):
    session = Session(id="s")
    turn = evaluate(session, "a", "志玲你好", active=False)

    async def reply(_turn):
        return HermesReply(text="你好。", instruction="平静")

    monkeypatch.setattr(session, "_chat_reply", reply)
    result = asyncio.run(session.chat(turn.id))

    assert result.text == "你好。"
    assert session.conversation_context == [
        {"role": "assistant", "text": "你好。", "speaker_id": None, "timestamp": session.conversation_context[0]["timestamp"]}
    ]


def test_performance_events_are_bounded_and_preserve_metadata():
    session = Session(id="s")

    for index in range(205):
        session.record_performance(
            "asr_kws",
            index + 0.25,
            source="frontend",
            turn_id=f"turn-{index}",
            detail={"index": index},
        )

    events = session.performance_snapshot()
    assert len(events) == 200
    assert events[0]["turn_id"] == "turn-5"
    assert events[-1] == {
        "timestamp": events[-1]["timestamp"],
        "stage": "asr_kws",
        "elapsed_ms": 204.2,
        "source": "frontend",
        "turn_id": "turn-204",
        "detail": {"index": 204},
    }
