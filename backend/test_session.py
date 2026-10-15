from session import HermesReply, Session, SpeakerIdentity, parse_hermes_reply


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


def test_first_waker_becomes_participant_one():
    session = Session(id="s")

    turn = evaluate(session, "a", "志玲你好", active=False)

    assert turn.should_reply is True
    assert turn.auto_joined is True
    assert turn.participant_number == 1
    assert list(session.participants) == ["a"]


def test_participant_can_continue_without_wake_while_active():
    session = Session(id="s")
    evaluate(session, "a", "志玲你好", active=False)

    turn = evaluate(session, "a", "今天天气怎么样")

    assert turn.started_while_active is True
    assert turn.wake_event is False
    assert turn.should_reply is True
    assert turn.reason == "参与人在连续对话窗口内"


def test_queued_participant_turn_uses_activation_created_before_evaluation():
    session = Session(id="s")
    participant_identity = identity("a")
    session._add_participant(participant_identity, "wake")
    session.active_until = 0
    queued = session.begin_turn("vad", False)
    assert queued.started_while_active is False

    # An earlier queued turn or TTS playback activates the session before this
    # turn reaches evaluate_turn(). It must not require another wake word.
    session.activate()
    result = session.evaluate_turn(
        queued.id,
        participant_identity,
        b"audio",
        text="接着刚才的话题",
        language="zh",
        kws_detected=False,
        kws_keyword=None,
    )

    assert result.should_reply is True
    assert result.wake_event is False
    assert result.reason == "参与人在连续对话窗口内"


def test_participant_requires_wake_after_window_expires():
    session = Session(id="s")
    evaluate(session, "a", "志玲你好", active=False)
    session.active_until = 0

    turn = evaluate(session, "a", "继续聊聊", active=False)

    assert turn.should_reply is False
    assert turn.reason == "会话未激活，请先说唤醒词"


def test_stranger_during_active_session_is_only_observed():
    session = Session(id="s")
    evaluate(session, "a", "志玲你好", active=False)
    before = session.active_until

    turn = evaluate(session, "b", "你们在聊什么")

    assert turn.should_reply is False
    assert turn.participant_number is None
    assert "b" in session.observed_speakers
    assert "b" not in session.participants
    assert session.active_until == before


def test_manual_join_allows_next_turn_and_kick_blocks_it():
    session = Session(id="s")
    evaluate(session, "a", "志玲你好", active=False)
    evaluate(session, "b", "我也想聊", active=True)

    participant = session.add_participant("b")
    admitted = evaluate(session, "b", "现在可以了吗", active=True)
    session.remove_participant("b")
    blocked = evaluate(session, "b", "还能听见吗", active=True)

    assert participant.number == 2
    assert admitted.should_reply is True
    assert blocked.should_reply is False
    assert "b" in session.observed_speakers


def test_stranger_cannot_interrupt_playback():
    session = Session(id="s")
    evaluate(session, "a", "志玲你好", active=False)

    stranger = evaluate(
        session,
        "b",
        "停一下",
        active=True,
        assistant_speaking=True,
    )
    participant = evaluate(
        session,
        "a",
        "停一下",
        active=True,
        assistant_speaking=True,
    )

    assert stranger.interrupt is False
    assert participant.interrupt is True
    assert participant.should_reply is False


def test_empty_room_requires_wake_word():
    session = Session(id="s")

    turn = evaluate(session, "a", "你好", active=False)

    assert turn.auto_joined is False
    assert turn.should_reply is False
    assert not session.participants


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
