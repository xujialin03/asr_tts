import numpy as np
import pytest

from app.main import _best_match, _decision


def test_decision_bands():
    # accept=0.55, uncertain=0.45
    assert _decision(0.80, 0.55, 0.45) == "matched"
    assert _decision(0.55, 0.55, 0.45) == "matched"  # inclusive at accept
    assert _decision(0.50, 0.55, 0.45) == "uncertain"
    assert _decision(0.45, 0.55, 0.45) == "uncertain"  # inclusive at uncertain
    assert _decision(0.40, 0.55, 0.45) == "unknown"


def test_best_match_takes_max_per_user_and_global():
    def cand(user_id, emb):
        return {"user_id": user_id, "embedding": emb}

    a, b = "u-a", "u-b"
    # Unit vectors. a's best template scores 0.8, b peaks at 0.6 -> pick a.
    candidates = [
        cand(a, [0.6, 0.8]),
        cand(a, [0.8, 0.6]),
        cand(b, [0.6, 0.8]),
    ]
    best = _best_match([1.0, 0.0], candidates)
    assert best is not None
    score, chosen = best
    assert chosen["user_id"] == a
    assert score == pytest.approx(0.8)

    assert _best_match([1.0, 0.0], []) is None


class _FakeRequest:
    def __init__(self, body=b"pcm"):
        self._body = body
        self.headers = {"content-type": "application/octet-stream"}

    async def body(self):
        return self._body


def _audio_meta():
    return {
        "duration_ms": 2000,
        "audio_sha256": "0" * 64,
        "quality": {"rms": 0.1, "peak": 0.1},
    }


@pytest.fixture
def wired_resolve(monkeypatch):
    """Patch resolve's I/O seams: embedding + database. Returns (embeddings, calls)."""
    from app import main as m

    state = {
        "query": None,  # embedding returned by _audio_embedding
        "candidates": [],  # returned by active_embeddings
        "added": [],  # add_voiceprint calls
        "created": [],  # create_anonymous calls
    }

    async def fake_audio_embedding(request):
        return state["query"], _audio_meta()

    async def fake_active_embeddings(user_id=None):
        return state["candidates"]

    async def fake_add_voiceprint(user_id, label, embedding, **kw):
        state["added"].append({"user_id": user_id, "label": label})
        return {"id": "vp-added", "user_id": user_id, "label": label}

    async def fake_create_anonymous(embedding, **kw):
        state["created"].append(embedding)
        return {
            "id": "vp-new-user",
            "username": "speaker-new",
            "display_name": "访客 new",
            "preferred_name": None,
            "voiceprint_id": "vp-new",
        }

    monkeypatch.setattr(m, "_audio_embedding", fake_audio_embedding)
    monkeypatch.setattr(m.database, "active_embeddings", fake_active_embeddings)
    monkeypatch.setattr(m.database, "add_voiceprint", fake_add_voiceprint)
    monkeypatch.setattr(
        m.database, "create_anonymous_speaker_with_voiceprint", fake_create_anonymous
    )
    return m, state


async def test_resolve_matched_returns_existing_without_enroll(wired_resolve):
    m, state = wired_resolve
    # 45-degree unit vector, self-match score = 1.0 >= accept.
    v = [1.0 / np.sqrt(2), 1.0 / np.sqrt(2)]
    state["query"] = v
    state["candidates"] = [{"user_id": "u1", "embedding": v,
                            "username": "u", "display_name": "D", "preferred_name": "P"}]

    result = await m.resolve_speaker(_FakeRequest(), 0.55, 0.45)
    assert result["decision"] == "matched"
    assert result["created"] is False
    assert result["speaker_id"] == "u1"
    assert state["added"] == [] and state["created"] == []


async def test_resolve_uncertain_self_enrolls_same_identity(wired_resolve):
    m, state = wired_resolve
    # Query [1,0] vs candidate [0.5, sqrt(0.75)] -> cos = 0.5 -> uncertain band (0.45..0.55)
    q = [1.0, 0.0]
    c = [0.5, np.sqrt(0.75)]
    state["query"] = q
    state["candidates"] = [{"user_id": "u1", "embedding": c,
                            "username": "u", "display_name": "访客 u1", "preferred_name": None}]

    result = await m.resolve_speaker(_FakeRequest(), 0.55, 0.45)
    assert result["decision"] == "uncertain"
    assert result["created"] is False  # NOT a new person
    assert result["template_added"] is True
    assert result["speaker_id"] == "u1"
    assert len(state["added"]) == 1 and state["added"][0]["user_id"] == "u1"
    assert state["created"] == []  # did not spawn a new anonymous speaker


async def test_resolve_uncertain_creates_new_when_auto_enroll_off(wired_resolve, monkeypatch):
    m, state = wired_resolve
    monkeypatch.setattr(m.settings, "auto_enroll_uncertain", False)
    q = [1.0, 0.0]
    c = [0.5, np.sqrt(0.75)]  # cos 0.5 -> uncertain band
    state["query"] = q
    state["candidates"] = [{"user_id": "u1", "embedding": c,
                            "username": "u", "display_name": "访客 u1", "preferred_name": None}]

    result = await m.resolve_speaker(_FakeRequest(), 0.55, 0.45)
    # With auto-enroll off, uncertain falls through to a fresh identity (old behavior).
    assert result["decision"] == "unknown"
    assert result["created"] is True
    assert state["added"] == []
    assert len(state["created"]) == 1


async def test_resolve_unknown_low_score_creates_new(wired_resolve):
    m, state = wired_resolve
    q = [1.0, 0.0]
    c = [0.0, 1.0]  # cos = 0 -> unknown
    state["query"] = q
    state["candidates"] = [{"user_id": "u1", "embedding": c,
                            "username": "u", "display_name": "访客 u1", "preferred_name": None}]

    result = await m.resolve_speaker(_FakeRequest(), 0.55, 0.45)
    assert result["decision"] == "unknown"
    assert result["created"] is True
    assert len(state["created"]) == 1


async def test_resolve_empty_database_creates_first_identity(wired_resolve):
    m, state = wired_resolve
    state["query"] = [1.0, 0.0]
    state["candidates"] = []  # cold start: nobody enrolled yet

    result = await m.resolve_speaker(_FakeRequest(), 0.55, 0.45)
    assert result["decision"] == "unknown"
    assert result["created"] is True
    assert result["speaker_id"] == "vp-new-user"
    assert len(state["created"]) == 1
