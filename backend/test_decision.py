import asyncio
import json

import httpx
import pytest

from decision import (
    DecisionModelClient,
    InteractionDecision,
    _extract_json,
    build_decision_client,
)


def valid_decision(**changes):
    result = {
        "is_addressed_to_ai": True,
        "should_respond": True,
        "response_type": "normal",
        "confidence": 0.9,
        "reason": "用户在延续对话",
    }
    result.update(changes)
    return result


def test_silent_normalizes_should_respond_to_false():
    decision = InteractionDecision.from_value(valid_decision(response_type="silent"))

    assert decision.should_respond is False
    assert decision.response_type == "silent"


def test_should_not_respond_normalizes_response_type_to_silent():
    decision = InteractionDecision.from_value(valid_decision(should_respond=False, response_type="brief"))

    assert decision.response_type == "silent"


@pytest.mark.parametrize(
    "value",
    [
        "not an object",
        valid_decision(confidence=1.2),
        valid_decision(response_type="long"),
        valid_decision(reason=1),
    ],
)
def test_invalid_decision_is_rejected(value):
    with pytest.raises(ValueError):
        InteractionDecision.from_value(value)


def test_extract_json_tolerates_surrounding_text():
    assert _extract_json('好的，这是结果：{"is_addressed_to_ai": true} 完毕') == {
        "is_addressed_to_ai": True
    }


def test_extract_json_rejects_no_object():
    with pytest.raises(ValueError):
        _extract_json("没有 JSON")


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        if self._payload.get("_status", 200) >= 400:
            raise httpx.HTTPStatusError("boom", request=None, response=None)

    def json(self):
        return self._payload


def _decision_model_client(monkeypatch, responses):
    """responses: list of _FakeResponse or Exception, consumed per attempt."""
    client = DecisionModelClient()
    calls = []
    it = iter(responses)

    async def fake_post(path, json=None):
        calls.append((path, json))
        item = next(it)
        if isinstance(item, Exception):
            raise item
        return item

    monkeypatch.setattr(client._client, "post", fake_post)
    monkeypatch.setattr("decision.config.DECISION_MODEL_MAX_RETRIES", len(responses))
    monkeypatch.setattr("decision.config.DECISION_MODEL_ENABLE_THINKING", False)
    return client, calls


def test_decision_model_payload_disables_thinking_and_requests_json(monkeypatch):
    client, calls = _decision_model_client(
        monkeypatch,
        [_FakeResponse({"choices": [{"message": {"content": json.dumps(valid_decision())}}]})],
    )
    result = asyncio.run(client.decide({"current_message": "诗林在不在？"}))

    assert result.should_respond is True
    payload = calls[0][1]
    assert payload["response_format"] == {"type": "json_object"}
    assert payload["chat_template_kwargs"] == {"enable_thinking": False}
    assert payload["temperature"] == 0


def test_decision_model_retries_then_succeeds(monkeypatch):
    client, calls = _decision_model_client(
        monkeypatch,
        [
            httpx.ConnectError("down"),
            _FakeResponse({"choices": [{"message": {"content": json.dumps(valid_decision())}}]}),
        ],
    )
    result = asyncio.run(client.decide({"current_message": "你好"}))

    assert result.should_respond is True
    assert len(calls) == 2


def test_decision_model_raises_after_exhausting_retries(monkeypatch):
    client, calls = _decision_model_client(
        monkeypatch,
        [httpx.ConnectError("down"), httpx.ConnectError("down")],
    )
    with pytest.raises(httpx.HTTPError):
        asyncio.run(client.decide({"current_message": "你好"}))
    assert len(calls) == 2


def test_build_decision_client_returns_decision_model():
    assert isinstance(build_decision_client(), DecisionModelClient)
