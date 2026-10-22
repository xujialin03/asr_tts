import asyncio

import pytest

from voiceprint_skill import SET_CURRENT_PREFERRED_NAME, VoiceprintManagementSkill


def test_action_definition_is_scoped_to_current_speaker_name():
    definition = VoiceprintManagementSkill.action_definition()

    assert definition["name"] == SET_CURRENT_PREFERRED_NAME
    assert set(definition["arguments"]["properties"]) == {"preferred_name"}
    assert "speaker_id" not in definition["arguments"]["properties"]


def test_skill_rejects_unknown_action_without_http_request():
    skill = VoiceprintManagementSkill("current-speaker")

    with pytest.raises(ValueError, match="unsupported"):
        asyncio.run(skill.execute("voiceprint.delete_user", {}))


@pytest.mark.parametrize(
    "preferred_name",
    ["", "许老板🙂", "许老板!", "<script>", "a" * 33],
)
def test_skill_rejects_invalid_preferred_names_without_http_request(preferred_name):
    skill = VoiceprintManagementSkill("current-speaker")

    with pytest.raises(ValueError, match="preferred_name"):
        asyncio.run(skill.execute(SET_CURRENT_PREFERRED_NAME, {"preferred_name": preferred_name}))
