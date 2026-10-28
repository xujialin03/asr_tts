import asyncio

import pytest

import config
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


def test_teleop_skills_injected_only_when_enabled(monkeypatch):
    agent = VoiceConversationAgent()
    monkeypatch.setattr(config, "TELEOP_ENABLED", False)
    assert "teleop." not in agent._system_message()

    monkeypatch.setattr(config, "TELEOP_ENABLED", True)
    system = agent._system_message()
    for name in (
        "start_teleoperation",
        "stop_teleoperation",
        "get_teleoperation_status",
        "get_joint_curves",
        "start_inference",
        "stop_inference",
        "get_inference_status",
        "unload_inference_model",
        "ask_camera_vision",
    ):
        assert f"teleop.{name}" in system
    assert "会使实体机械臂通电运动" in system
    # inference/vision safety rules are surfaced too
    assert "start_inference" in system and "ask_camera_vision" in system


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
    name, tool_text, tool_meta = asyncio.run(
        agent._execute_actions(
            turn(speaker_id="speaker-current"),
            [{
                "name": "voiceprint.set_current_preferred_name",
                "arguments": {"preferred_name": "许老板", "speaker_id": "attacker"},
            }],
        )
    )

    assert name == "许老板"
    assert tool_text == ""
    assert tool_meta is None
    assert captured["speaker_id"] == "speaker-current"
    assert captured["arguments"]["speaker_id"] == "attacker"


def test_execute_actions_routes_teleop_when_enabled(monkeypatch):
    from teleop_skill import TeleopSkillResult

    agent = VoiceConversationAgent()
    monkeypatch.setattr(config, "TELEOP_ENABLED", True)

    async def execute(self, action, arguments):
        return TeleopSkillResult(
            action=action,
            ok=True,
            structured={"status": {"running": True, "mode": "teleop", "teleop_config": {}}},
            fallback_summary="遥操已启动。",
        )

    monkeypatch.setattr("conversation_agent.TeleopMCPSkill.execute", execute)
    name, tool_text, tool_meta = asyncio.run(
        agent._execute_actions(
            turn(),
            [{"name": "teleop.start_teleoperation", "arguments": {}}],
        )
    )

    assert name is None
    assert tool_text == "遥操已启动。"
    assert tool_meta["action"] == "teleop.start_teleoperation"
    assert tool_meta["ok"] is True
    assert "running" in tool_meta["payload"]


def test_execute_actions_rejects_unknown_skill(monkeypatch):
    agent = VoiceConversationAgent()
    monkeypatch.setattr(config, "TELEOP_ENABLED", False)
    with pytest.raises(RuntimeError, match="unknown skill action"):
        asyncio.run(
            agent._execute_actions(turn(), [{"name": "teleop.start_teleoperation", "arguments": {}}])
        )


def test_narration_payload_redacts_tokens_and_caps_curves():
    from conversation_agent import _narration_payload, _redact_tokens

    assert _redact_tokens("http://h/d?token=SECRET&x=1") == "http://h/d?x=1"
    assert _redact_tokens("http://h/d?token=SECRET") == "http://h/d"
    assert "SECRET" not in _redact_tokens({"url": "http://h/s?token=SECRET"})

    curves_structured = {
        "seconds": 30,
        "timestamps": [1.0, 2.0, 3.0],
        "leader": {"gripper.pos": [1, 2, 3]},
        "follower": {"gripper.pos": [4, 5, 6]},
    }
    curves = _narration_payload("teleop.get_joint_curves", curves_structured)
    assert "30" in curves
    # Only the latest sample per axis is kept for narration.
    assert '"gripper.pos": 3' in curves

    status_structured = {
        "status": {"mode": "teleop", "running": True, "pid": 1, "error": "", "teleop_config": {"front_fps": 30}},
        "dashboard_url": "http://192.168.200.30:8791/dashboard?token=SECRET",
        "streams": {"front_mjpeg": "http://h/front.mjpg?token=SECRET"},
    }
    status = _narration_payload("teleop.get_teleoperation_status", status_structured)
    assert "SECRET" not in status
    assert '"has_dashboard": true' in status


def test_inference_status_payload_drops_log_tail():
    from conversation_agent import _narration_payload

    structured = {
        "status": {
            "phase": "executing", "running": True, "resident": False,
            "model_warm": True, "total_dispatches": 12, "error": "",
            "log_tail": ["[corrector_score] noise"] * 50,
        }
    }
    payload = _narration_payload("teleop.get_inference_status", structured)
    assert "executing" in payload and "log_tail" not in payload
    assert len(payload) <= 2000


def test_vision_payload_passes_answer_and_redacts_tokens():
    from conversation_agent import _narration_payload

    structured = {
        "vision_status": "completed",
        "answer": "front 看到夹爪，third 看到桌面和纸盒。",
        "capture_source": "on_demand",
        "image_url": "http://h/snap.jpg?token=SECRET",
    }
    payload = _narration_payload("teleop.ask_camera_vision", structured)
    assert "纸盒" in payload
    assert "SECRET" not in payload


def test_validator_handles_string_bool_and_required():
    from teleop_skill import TeleopMCPSkill

    skill = TeleopMCPSkill()
    # required + types
    assert skill._validate("teleop.start_inference", {"instruction": "把紫鸭夹到纸盒", "resident": True}) == {
        "instruction": "把紫鸭夹到纸盒", "resident": True,
    }
    with pytest.raises(ValueError, match="missing required"):
        skill._validate("teleop.start_inference", {})
    with pytest.raises(ValueError, match="must be a boolean"):
        skill._validate("teleop.start_inference", {"instruction": "x", "resident": "yes"})
    with pytest.raises(ValueError, match="at most"):
        skill._validate("teleop.ask_camera_vision", {"question": "q" * 2001})
    with pytest.raises(ValueError, match="at least"):
        skill._validate("teleop.ask_camera_vision", {"question": ""})
    # numeric range still enforced
    with pytest.raises(ValueError, match="must be >= 1"):
        skill._validate("teleop.get_joint_curves", {"seconds": 0})
    with pytest.raises(ValueError, match="unexpected arguments"):
        skill._validate("teleop.stop_inference", {"model_id": "x"})


def test_all_nine_actions_are_supported():
    from teleop_skill import ACTION_NAMES, TeleopMCPSkill

    assert len(ACTION_NAMES) == 9
    for action in ACTION_NAMES:
        assert TeleopMCPSkill.is_supported(action)
    # definitions cover every action so the model can see all nine
    assert {d["name"] for d in TeleopMCPSkill.definitions()} == set(ACTION_NAMES)
