from session import HermesReply, parse_hermes_reply


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
