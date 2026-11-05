import asyncio

import httpx
import pytest

import config
from session_skill import INVITE_SPEAKER, InviteStatus, SessionInviteSkill


def _client_with_users(users):
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/users"
        return httpx.Response(200, json=users)

    return httpx.MockTransport(handler)


@pytest.fixture
def patch_transport(monkeypatch):
    monkeypatch.setattr(config, "VOICEPRINT_BASE_URL", "http://vp.test")
    real_client = httpx.AsyncClient

    def factory(*args, **kwargs):
        kwargs["transport"] = _patched["transport"]
        return real_client(*args, **kwargs)

    _patched = {"transport": None}
    monkeypatch.setattr(httpx, "AsyncClient", factory)
    return lambda transport: _patched.update(transport=transport)


def test_action_definition_takes_only_preferred_name():
    definition = SessionInviteSkill.action_definition()

    assert definition["name"] == INVITE_SPEAKER
    assert set(definition["arguments"]["properties"]) == {"preferred_name"}
    assert definition["arguments"]["required"] == ["preferred_name"]


def test_execute_rejects_unknown_action():
    skill = SessionInviteSkill()
    with pytest.raises(ValueError, match="unsupported"):
        asyncio.run(skill.execute("session.remove_speaker", {}))


@pytest.mark.parametrize("arguments", [None, "老周", []])
def test_execute_rejects_non_object_arguments_without_lookup(patch_transport, arguments):
    patch_transport(httpx.MockTransport(lambda request: pytest.fail("must not fetch users")))
    skill = SessionInviteSkill()

    with pytest.raises(ValueError, match="arguments must be an object"):
        asyncio.run(skill.execute(INVITE_SPEAKER, arguments))


@pytest.mark.parametrize("arguments", [{"preferred_name": 42}, {"preferred_name": "老周🙂"}, {"preferred_name": ""}])
def test_execute_returns_invalid_name_without_lookup(patch_transport, arguments):
    patch_transport(httpx.MockTransport(lambda request: pytest.fail("must not fetch users")))
    skill = SessionInviteSkill()

    result = asyncio.run(skill.execute(INVITE_SPEAKER, arguments))

    assert result.status is InviteStatus.INVALID_NAME


def test_invite_matches_preferred_name(patch_transport):
    patch_transport(
        httpx.MockTransport(
            lambda request: httpx.Response(
                200,
                json=[
                    {"id": "u-1", "display_name": "周哥", "preferred_name": "老周", "is_active": True},
                    {"id": "u-2", "display_name": "访客 abc", "preferred_name": None, "is_active": True},
                ],
            )
        )
    )

    result = asyncio.run(SessionInviteSkill().execute(INVITE_SPEAKER, {"preferred_name": " 老周 "}))

    assert result.status is InviteStatus.INVITED
    assert result.speaker_id == "u-1"


def test_invite_skips_disabled_accounts(patch_transport):
    patch_transport(
        httpx.MockTransport(
            lambda request: httpx.Response(
                200,
                json=[
                    {"id": "u-old", "display_name": "访客 old", "preferred_name": "老周", "is_active": False},
                    {"id": "u-new", "display_name": "访客 new", "preferred_name": "老周", "is_active": True},
                ],
            )
        )
    )

    result = asyncio.run(SessionInviteSkill().execute(INVITE_SPEAKER, {"preferred_name": "老周"}))

    assert result.status is InviteStatus.INVITED
    assert result.speaker_id == "u-new"


def test_invite_disabled_only_is_not_found(patch_transport):
    patch_transport(
        httpx.MockTransport(
            lambda request: httpx.Response(
                200,
                json=[{"id": "u-old", "display_name": "老周", "preferred_name": "老周", "is_active": False}],
            )
        )
    )

    result = asyncio.run(SessionInviteSkill().execute(INVITE_SPEAKER, {"preferred_name": "老周"}))

    assert result.status is InviteStatus.NOT_FOUND


def test_invite_matches_display_name(patch_transport):
    patch_transport(
        httpx.MockTransport(
            lambda request: httpx.Response(
                200, json=[{"id": "u-9", "display_name": "小敏", "preferred_name": None, "is_active": True}]
            )
        )
    )

    result = asyncio.run(SessionInviteSkill().execute(INVITE_SPEAKER, {"preferred_name": "小敏"}))

    assert result.status is InviteStatus.INVITED
    assert result.speaker_id == "u-9"


def test_invite_not_found(patch_transport):
    patch_transport(
        httpx.MockTransport(lambda request: httpx.Response(200, json=[{"id": "u-1", "display_name": "老周"}]))
    )

    result = asyncio.run(SessionInviteSkill().execute(INVITE_SPEAKER, {"preferred_name": "不存在的人"}))

    assert result.status is InviteStatus.NOT_FOUND
    assert result.speaker_id is None


def test_invite_ambiguous_same_name(patch_transport):
    patch_transport(
        httpx.MockTransport(
            lambda request: httpx.Response(
                200,
                json=[
                    {"id": "u-1", "display_name": "张伟", "preferred_name": None, "is_active": True},
                    {"id": "u-2", "display_name": "实验室的张伟", "preferred_name": "张伟", "is_active": True},
                ],
            )
        )
    )

    result = asyncio.run(SessionInviteSkill().execute(INVITE_SPEAKER, {"preferred_name": "张伟"}))

    assert result.status is InviteStatus.AMBIGUOUS
    assert result.match_count == 2
    assert result.speaker_id is None
