import asyncio

import pytest

from conversation_agent import AgentTurn, VoiceConversationAgent


def turn(**overrides) -> AgentTurn:
    values = {
        "session_id": "s",
        "turn_id": "t",
        "speaker_id": "speaker-a",
        "speaker": {
            "speaker_id": "speaker-a",
            "display_name": "访客 a",
            "preferred_name": None,
            "identity_named": False,
        },
        "text": "我是许老板",
        "wake_event": True,
        "joined_session": True,
        "should_ask_name": True,
        "response_type": "normal",
    }
    values.update(overrides)
    return AgentTurn(**values)


def test_messages_include_current_member_context_and_only_valid_history():
    agent = VoiceConversationAgent()
    messages = agent._messages(
        turn(),
        [
            {"role": "user", "text": "前一句", "speaker_id": "speaker-b"},
            {"role": "assistant", "text": "收到", "speaker_id": None},
            {"role": "system", "text": "must not pass through"},
        ],
    )

    assert messages[0]["role"] == "system"
    assert "voiceprint.set_current_preferred_name" in messages[0]["content"]
    assert messages[1] == {"role": "user", "content": "[用户 speaker-] 前一句"}
    assert messages[2] == {"role": "assistant", "content": "收到"}
    assert len(messages) == 4
    assert '"joined_session":true' in messages[-1]["content"]
    assert '"should_ask_name":true' in messages[-1]["content"]


def test_parse_reply_keeps_text_instruction_and_no_actions():
    text, instruction, actions = VoiceConversationAgent._parse_reply(
        '{"text":"你好呀","instruction":" 温柔 亲切 ","actions":[]}'
    )

    assert text == "你好呀"
    assert instruction == "温柔 亲切"
    assert actions == []


@pytest.mark.parametrize(
    "raw, error",
    [
        ('{"text":"好","actions":{}}', "actions must be an array"),
        ('{"text":"好","actions":[{},{}]}', "at most one"),
        ('{"text":"好","actions":["bad"]}', "action must be an object"),
        ('{"instruction":"温柔","actions":[]}', "text must be a non-empty string"),
    ],
)
def test_parse_reply_rejects_invalid_action_contract(raw, error):
    with pytest.raises(RuntimeError, match=error):
        VoiceConversationAgent._parse_reply(raw)


def test_execute_actions_is_bound_to_turn_speaker(monkeypatch):
    agent = VoiceConversationAgent()
    captured = {}

    async def execute(self, name, arguments):
        captured["speaker_id"] = self._speaker_id
        captured["name"] = name
        captured["arguments"] = arguments
        from voiceprint_skill import VoiceprintSkillResult
        return VoiceprintSkillResult(action=name, preferred_name="许老板")

    monkeypatch.setattr("conversation_agent.VoiceprintManagementSkill.execute", execute)
    name = asyncio.run(
        agent._execute_actions(
            turn(speaker_id="speaker-current"),
            [{
                "name": "voiceprint.set_current_preferred_name",
                "arguments": {"preferred_name": "许老板", "speaker_id": "attacker"},
            }],
        )
    )

    assert name == "许老板"
    assert captured["speaker_id"] == "speaker-current"
    assert captured["arguments"]["speaker_id"] == "attacker"
