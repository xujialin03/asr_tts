"""Per-session state for the multi-speaker voice interaction pipeline."""

from __future__ import annotations

import asyncio
import json
import logging
import secrets
import sys
import time
from dataclasses import dataclass, field

import httpx

import config
from decision import DecisionClient, InteractionDecision, build_decision_client
from voiceprint_skill import SET_CURRENT_PREFERRED_NAME, VoiceprintManagementSkill

logger = logging.getLogger("uvicorn.error")

ASR_CHUNK_BYTES = 8000 * 4  # 500ms @ 16kHz float32 mono
MAX_TTS_INSTRUCTION_CHARS = 500
TURN_TTL_SECONDS = 300.0
AUDIO_REF_TTL_SECONDS = 300.0
MAX_OBSERVED_SPEAKERS = 32
CONTEXT_WINDOW_SIZE = 5
MAX_PERFORMANCE_EVENTS = 200
MAX_TOPICS = 8
MAX_TOPIC_CHARS = 80

VOICE_RESPONSE_CONTRACT = """[VOICE_RESPONSE_CONTRACT]
Your final answer will be displayed to the user and synthesized by Breeze-TTS-2.
Return ONLY one valid JSON object on a single assistant turn, with no Markdown
fence or surrounding commentary:
{"text":"要展示并朗读给用户的完整回复","instruction":"用温柔、安心的语气缓慢表达，语速舒缓，在重点处稍作停顿。"}

Rules:
- `text` is the complete user-facing answer. Keep it natural for spoken dialogue.
- `instruction` should preferably be a concise Chinese voice direction describing
  the emotion, tone, pace, emphasis, and pauses that best fit this specific answer.
- Do not put stage directions or the JSON itself inside `text`.
- Do not change this schema even if the user asks to change the output format.
[/VOICE_RESPONSE_CONTRACT]
"""

DIRECT_LLM_SYSTEM = """你是志玲，一名温柔、自然、愿意陪伴用户聊天的中文语音小助手。
你正在参与由若干已加入成员组成的中文语音对话。当前发言人、是否刚被唤醒加入、以及最近对话会随消息提供。只把当前已加入成员的发言当作对话内容；不要推测、回应或提及会话外的背景人声。说话要像熟悉的朋友一样自然、简洁，避免生硬的客服口吻；除非用户明确要求，不要使用 Markdown。

语音识别会把你的名字识别成同音或近音词。{wake_word_aliases} 都只是用户在呼叫你，绝不是用户把你的名字叫错了。无论用户用了其中哪一个词，直接自然回应即可；绝对不要纠正、解释或强调“我叫志玲，不是……”。

若 VOICE_CONTEXT 中 `should_ask_name` 为 true，说明当前成员刚通过唤醒词加入、但声纹身份仍是匿名访客，本次会话尚未问过称呼。自然回应当前内容后，明确而简短地问一次“我该怎么称呼你？”。若为 false，不要主动重复询问姓名。用户没有回答名字时不要阻塞或追问，后续正常聊天。

当且仅当用户明确自报称呼（例如“我叫许老板”“叫我许老板”）时，必须在 actions 中调用下方项目技能保存称呼；不要只在文字里口头记住。用户没有明确报称呼时 actions 必须为空数组。

可用项目技能：{voiceprint_skill}
"""

DIRECT_LLM_RESPONSE_CONTRACT = """[DIRECT_LLM_RESPONSE_CONTRACT]
只输出一个合法 JSON 对象，不要 Markdown 围栏或额外文字：
{"text":"要展示并朗读给用户的完整回复","instruction":"简短中文语气指令","actions":[{"name":"项目技能名","arguments":{}}]}

规则：
- text 是完整、自然的口语回复。
- instruction 是简短中文 TTS 语气指令。
- actions 必须是数组；无动作时为 []。
- 仅可使用系统消息列出的项目技能；不可虚构技能或传入未定义参数。
[/DIRECT_LLM_RESPONSE_CONTRACT]
"""


@dataclass(frozen=True)
class HermesReply:
    text: str
    instruction: str = ""


def parse_hermes_reply(raw_reply: str) -> HermesReply:
    """Parse Hermes' voice response, falling back to legacy plain text."""
    raw_reply = (raw_reply or "").strip()
    decoder = json.JSONDecoder()
    candidates = [raw_reply]

    if raw_reply.startswith("```") and raw_reply.endswith("```"):
        lines = raw_reply.splitlines()
        if len(lines) >= 3:
            candidates.insert(0, "\n".join(lines[1:-1]).strip())

    candidates.extend(raw_reply[index:] for index, char in enumerate(raw_reply) if char == "{")
    for candidate in candidates:
        try:
            value, _ = decoder.raw_decode(candidate)
        except (json.JSONDecodeError, TypeError):
            continue
        if not isinstance(value, dict):
            continue
        text = value.get("text")
        instruction = value.get("instruction", "")
        if not isinstance(text, str) or not text.strip():
            continue
        if not isinstance(instruction, str):
            instruction = ""
        instruction = " ".join(instruction.split())[:MAX_TTS_INSTRUCTION_CHARS]
        return HermesReply(text=text.strip(), instruction=instruction)

    return HermesReply(text=raw_reply)


def _pad_asr_chunk(body: bytes) -> bytes:
    if not body:
        return body
    if len(body) < ASR_CHUNK_BYTES:
        return body + (b"\x00" * (ASR_CHUNK_BYTES - len(body)))
    rem = len(body) % 4
    if rem:
        return body + (b"\x00" * (4 - rem))
    return body


def _identity_named(display_name: str | None, preferred_name: str | None) -> bool:
    return bool(preferred_name or (display_name and not display_name.startswith("访客 ")))


@dataclass
class SpeakerIdentity:
    speaker_id: str
    display_name: str | None = None
    preferred_name: str | None = None
    identity_named: bool = False
    created: bool = False
    score: float | None = None

    @classmethod
    def from_result(cls, result: dict) -> "SpeakerIdentity":
        speaker = result.get("speaker") or {}
        display_name = speaker.get("display_name")
        preferred_name = speaker.get("preferred_name")
        return cls(
            speaker_id=str(result["speaker_id"]),
            display_name=display_name,
            preferred_name=preferred_name,
            identity_named=_identity_named(display_name, preferred_name),
            created=bool(result.get("created")),
            score=result.get("score"),
        )

    @classmethod
    def from_identify_result(cls, result: dict) -> "SpeakerIdentity | None":
        """Convert a non-mutating identify result, if it resolved a user."""
        user = result.get("user")
        if not isinstance(user, dict) or not user.get("id"):
            return None
        display_name = user.get("display_name")
        preferred_name = user.get("preferred_name")
        return cls(
            speaker_id=str(user["id"]),
            display_name=display_name,
            preferred_name=preferred_name,
            identity_named=_identity_named(display_name, preferred_name),
            created=False,
            score=result.get("score"),
        )

    def public(self) -> dict:
        return {
            "speaker_id": self.speaker_id,
            "display_name": self.display_name,
            "preferred_name": self.preferred_name,
            "identity_named": self.identity_named,
            "created": self.created,
            "score": self.score,
        }


@dataclass
class ObservedSpeaker:
    identity: SpeakerIdentity
    first_seen_at: float
    last_seen_at: float
    last_text: str = ""
    last_score: float | None = None
    observation_count: int = 1

    def public(self) -> dict:
        return {
            **self.identity.public(),
            "first_seen_at": self.first_seen_at,
            "last_seen_at": self.last_seen_at,
            "last_text": self.last_text,
            "last_score": self.last_score,
            "observation_count": self.observation_count,
        }


@dataclass
class VoiceTurn:
    id: str
    sequence: int
    source: str
    assistant_speaking: bool
    started_at: float
    started_while_active: bool
    expires_at: float
    text: str = ""
    language: str = ""
    speaker: SpeakerIdentity | None = None
    wake_event: bool = False
    wake_detected: bool = False
    joined_session: bool = False
    should_reply: bool = False
    interrupt: bool = False
    reply_text: str = ""
    decision: InteractionDecision | None = None
    reason: str = ""
    buffer_merged: bool = False
    audio_ref: str | None = None
    evaluated: bool = False
    consumed: bool = False

    def public(self) -> dict:
        return {
            "turn_id": self.id,
            "sequence": self.sequence,
            "source": self.source,
            "started_while_active": self.started_while_active,
            "speaker": self.speaker.public() if self.speaker else None,
            "wake_event": self.wake_event,
            "joined_session": self.joined_session,
            "should_reply": self.should_reply,
            "interrupt": self.interrupt,
            "reply_text": self.reply_text,
            "is_addressed_to_ai": self.decision.is_addressed_to_ai if self.decision else None,
            "response_type": self.decision.response_type if self.decision else "silent",
            "confidence": self.decision.confidence if self.decision else None,
            "is_complete": self.decision.is_complete if self.decision else True,
            "buffer_merged": self.buffer_merged,
            "decision_reason": self.decision.reason if self.decision else "",
            "decision_fallback": self.decision.fallback if self.decision else False,
            "reason": self.reason,
        }


@dataclass
class VoiceAudioRef:
    token: str
    speaker_id: str
    turn_id: str | None
    body: bytes = field(repr=False)
    expires_at: float = 0.0


@dataclass
class Session:
    id: str
    asr_session_id: str | None = None
    last_asr: dict = field(default_factory=lambda: {"language": "", "text": ""})
    active_until: float = 0.0
    observed_speakers: dict[str, ObservedSpeaker] = field(default_factory=dict)
    participant_ids: set[str] = field(default_factory=set)
    turns: dict[str, VoiceTurn] = field(default_factory=dict)
    voice_audio_refs: dict[str, VoiceAudioRef] = field(default_factory=dict)
    conversation_context: list[dict] = field(default_factory=list)
    topics: list[str] = field(default_factory=list)
    performance_events: list[dict] = field(default_factory=list)
    pending_buffer: dict | None = None
    name_prompted_speaker_ids: set[str] = field(default_factory=set)
    decision_client: DecisionClient | None = field(default=None, repr=False)
    next_turn_sequence: int = 1
    turn_lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)
    chat_lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)

    # ---- ASR lifecycle ----
    async def asr_start(self, client: httpx.AsyncClient) -> str:
        r = await client.post(
            f"{config.ASR_BASE_URL}/api/start",
            timeout=httpx.Timeout(10.0, connect=5.0),
        )
        r.raise_for_status()
        self.asr_session_id = r.json()["session_id"]
        self.last_asr = {"language": "", "text": ""}
        return self.asr_session_id

    async def asr_chunk(self, client: httpx.AsyncClient, body: bytes) -> dict:
        if not self.asr_session_id:
            await self.asr_start(client)
        r = await self._post_chunk(client, body)
        if r.status_code == 400:
            await self.asr_start(client)
            r = await self._post_chunk(client, body)
        r.raise_for_status()
        j = r.json()
        self.last_asr = {"language": j.get("language", ""), "text": j.get("text", "")}
        return j

    async def _post_chunk(self, client: httpx.AsyncClient, body: bytes):
        return await client.post(
            f"{config.ASR_BASE_URL}/api/chunk",
            params={"session_id": self.asr_session_id},
            content=_pad_asr_chunk(body),
            headers={"Content-Type": "application/octet-stream"},
            timeout=httpx.Timeout(15.0, connect=5.0),
        )

    async def asr_finish(self, client: httpx.AsyncClient) -> dict:
        if not self.asr_session_id:
            return dict(self.last_asr)
        try:
            r = await client.post(
                f"{config.ASR_BASE_URL}/api/finish",
                params={"session_id": self.asr_session_id},
                timeout=httpx.Timeout(5.0, connect=10.0),
            )
            r.raise_for_status()
            j = r.json()
        except (httpx.HTTPError, ValueError):
            j = dict(self.last_asr)
        self.asr_session_id = None
        return j

    # ---- activation and turns ----
    def activate(self, seconds: float | None = None) -> None:
        duration = config.ACTIVE_WINDOW_SECONDS if seconds is None else seconds
        duration = max(1.0, min(float(duration), 300.0))
        self.active_until = max(self.active_until, time.time() + duration)

    def is_active(self) -> bool:
        return time.time() <= self.active_until

    def active_remaining(self) -> float:
        return max(0.0, self.active_until - time.time())

    def cleanup_expired(self) -> None:
        now = time.time()
        self.turns = {
            turn_id: turn
            for turn_id, turn in self.turns.items()
            if turn.expires_at >= now or (turn.evaluated and not turn.consumed)
        }
        self.voice_audio_refs = {
            token: ref for token, ref in self.voice_audio_refs.items() if ref.expires_at >= now
        }

    def begin_turn(self, source: str, assistant_speaking: bool) -> VoiceTurn:
        self.cleanup_expired()
        now = time.time()
        turn = VoiceTurn(
            id=secrets.token_urlsafe(24),
            sequence=self.next_turn_sequence,
            source=source,
            assistant_speaking=assistant_speaking,
            started_at=now,
            started_while_active=self.is_active(),
            expires_at=now + TURN_TTL_SECONDS,
        )
        self.next_turn_sequence += 1
        self.turns[turn.id] = turn
        return turn

    def get_turn(self, turn_id: str) -> VoiceTurn | None:
        self.cleanup_expired()
        return self.turns.get(turn_id)

    def _observe(self, identity: SpeakerIdentity, text: str) -> ObservedSpeaker:
        now = time.time()
        observed = self.observed_speakers.get(identity.speaker_id)
        if observed:
            observed.identity = identity
            observed.last_seen_at = now
            observed.last_text = text
            observed.last_score = identity.score
            observed.observation_count += 1
        else:
            observed = ObservedSpeaker(
                identity=identity,
                first_seen_at=now,
                last_seen_at=now,
                last_text=text,
                last_score=identity.score,
            )
            self.observed_speakers[identity.speaker_id] = observed
        if len(self.observed_speakers) > MAX_OBSERVED_SPEAKERS:
            oldest = min(self.observed_speakers.values(), key=lambda item: item.last_seen_at)
            self.observed_speakers.pop(oldest.identity.speaker_id, None)
        return observed

    def store_voice_audio(
        self,
        body: bytes,
        speaker_id: str,
        turn_id: str | None,
        ttl_seconds: float = AUDIO_REF_TTL_SECONDS,
    ) -> str:
        self.cleanup_expired()
        token = secrets.token_urlsafe(24)
        self.voice_audio_refs[token] = VoiceAudioRef(
            token=token,
            speaker_id=speaker_id,
            turn_id=turn_id,
            body=body,
            expires_at=time.time() + ttl_seconds,
        )
        return token

    def get_voice_audio(self, speaker_id: str, audio_ref: str) -> bytes | None:
        self.cleanup_expired()
        for token, ref in self.voice_audio_refs.items():
            if secrets.compare_digest(token, audio_ref):
                return ref.body if ref.speaker_id == speaker_id else None
        return None

    def _wake_detected(self, text: str, kws_detected: bool) -> bool:
        return kws_detected or any(word and word in text for word in config.WAKE_WORDS)

    def record_performance(
        self,
        stage: str,
        elapsed_ms: float,
        *,
        source: str,
        turn_id: str | None = None,
        detail: dict | None = None,
    ) -> None:
        event = {
            "timestamp": time.time(),
            "stage": stage,
            "elapsed_ms": round(max(0.0, float(elapsed_ms)), 1),
            "source": source,
            "turn_id": turn_id,
            "detail": detail or {},
        }
        self.performance_events.append(event)
        del self.performance_events[:-MAX_PERFORMANCE_EVENTS]
        logger.info(
            "performance session=%s source=%s stage=%s elapsed_ms=%.1f turn=%s detail=%s",
            self.id,
            source,
            stage,
            event["elapsed_ms"],
            turn_id,
            event["detail"],
        )

    def performance_snapshot(self) -> list[dict]:
        return list(self.performance_events)

    def _append_context(self, role: str, text: str, speaker_id: str | None = None) -> None:
        self.conversation_context.append(
            {"role": role, "text": text, "speaker_id": speaker_id, "timestamp": time.time()}
        )
        del self.conversation_context[:-CONTEXT_WINDOW_SIZE]

    def _track_topic(self, text: str) -> None:
        """Record the current user utterance as the active topic.

        A lightweight heuristic for the Topic model in the architecture: the most
        recent admitted user turn is treated as the current topic, with earlier
        topics kept as history. A real topic model would summarize/merge; this
        keeps a bounded, de-duplicated list so the decision model sees what the
        room is on.
        """
        topic = " ".join((text or "").split())[:MAX_TOPIC_CHARS]
        if not topic:
            return
        if self.topics and self.topics[-1] == topic:
            return
        self.topics.append(topic)
        del self.topics[:-MAX_TOPICS]

    def _has_assistant_reply(self) -> bool:
        return any(entry["role"] == "assistant" for entry in self.conversation_context)

    def _looks_like_followup(self, text: str) -> bool:
        """Whether an utterance reads as a question / continuation follow-up.

        Guards the continuation fallback so it fires on real follow-ups
        ("最近是不是有个台风啊") but not on bare acknowledgments ("嗯", "好的",
        "谢谢") that the decision model correctly leaves silent.
        """
        text = (text or "").strip()
        if not text:
            return False
        if any(mark in text for mark in ("?", "？")):
            return True
        if any(word in text for word in config.FOLLOWUP_MARKERS):
            return True
        return text[-1] in config.FOLLOWUP_SENTENCE_FINALS

    def _fallback_decision(self, turn: VoiceTurn) -> InteractionDecision:
        should_respond = turn.should_reply
        return InteractionDecision(
            is_addressed_to_ai=should_respond,
            should_respond=should_respond,
            response_type="normal" if should_respond else "silent",
            confidence=0.0,
            reason="决策模型不可用，使用唤醒词与会话窗口回退",
            fallback=True,
        )

    def _decision_state(self, turn: VoiceTurn) -> dict:
        speaker = turn.speaker
        return {
            "agent": {
                "name": config.AGENT_NAME,
                "role": config.AGENT_ROLE,
                "persona": config.AGENT_PERSONA,
            },
            "current_message": turn.text,
            "speaker": {
                "speaker_id": speaker.speaker_id if speaker else None,
                "display_name": speaker.display_name if speaker else None,
                "preferred_name": speaker.preferred_name if speaker else None,
                "is_session_participant": bool(speaker and speaker.speaker_id in self.participant_ids),
            },
            "session": {
                "active": self.is_active(),
                "participant_count": len(self.participant_ids),
                "topics": list(self.topics),
                "recent_messages": list(self.conversation_context),
            },
            "wake_word_detected": turn.wake_detected,
            "assistant_speaking": turn.assistant_speaking,
            "has_pending_task": False,
        }

    def _buffer_merge(self, turn: VoiceTurn) -> tuple[str, bool]:
        """Prepend any held fragment from the same speaker; return (text, merged)."""
        buf = self.pending_buffer
        if not buf:
            return turn.text, False
        same_speaker = buf["speaker_id"] == turn.speaker.speaker_id
        expired = time.time() - buf["updated_at"] > config.BUFFER_MAX_WAIT_SECONDS
        if not same_speaker or expired:
            self.pending_buffer = None
            return turn.text, False
        merged = (buf["text"] + " " + turn.text).strip()
        self.pending_buffer = None
        return merged, True

    def _buffer_hold(self, turn: VoiceTurn, text: str) -> None:
        self.pending_buffer = {
            "speaker_id": turn.speaker.speaker_id,
            "text": text,
            "segments": (self.pending_buffer or {}).get("segments", 0) + 1,
            "updated_at": time.time(),
        }

    async def decide_turn(self, turn: VoiceTurn) -> VoiceTurn:
        if not turn.text or turn.interrupt or not turn.speaker:
            return turn
        started = time.monotonic()
        fallback = self._fallback_decision(turn)
        if self.decision_client is None:
            turn.decision = fallback
            return turn

        # Input buffer: merge a held fragment from the same speaker before judging,
        # so the model sees the whole utterance even when VAD split it.
        buffer_merged = False
        if config.BUFFER_ENABLED:
            turn.text, buffer_merged = self._buffer_merge(turn)
        turn.buffer_merged = buffer_merged

        try:
            decision = await self.decision_client.decide(self._decision_state(turn))
        except Exception as exc:
            logger.warning(
                "decision failed session=%s turn=%s error=%s: %s",
                self.id,
                turn.id,
                type(exc).__name__,
                exc,
            )
            decision = fallback
        turn.decision = decision

        # Hold an incomplete fragment so a following segment can complete the
        # thought ("今天" + "天气怎么样啊"). Only when the model is not already
        # going to respond — a bare wake word the model wants to ack briefly is
        # not held. Addressee is not required: a leading fragment often has no
        # signal about who it is for until the sentence finishes.
        if (
            config.BUFFER_ENABLED
            and not decision.fallback
            and not decision.is_complete
            and not decision.should_respond
            and not turn.wake_detected
            and len(turn.text) < config.BUFFER_MAX_CHARS
            and (self.pending_buffer or {}).get("segments", 0) < config.BUFFER_MAX_SEGMENTS
        ):
            self._buffer_hold(turn, turn.text)
            turn.should_reply = False
            turn.reply_text = ""
            turn.reason = "输入缓冲区：句子未说完，等待后续合并"
            self.record_performance(
                "decision_model",
                (time.monotonic() - started) * 1000,
                source="backend",
                turn_id=turn.id,
                detail={
                    "addressed": decision.is_addressed_to_ai,
                    "respond": False,
                    "response_type": "silent",
                    "fallback": decision.fallback,
                    "buffered": True,
                    "buffer_text": turn.text,
                },
            )
            return turn

        if turn.wake_detected and not decision.should_respond:
            decision = InteractionDecision(
                is_addressed_to_ai=True,
                should_respond=True,
                response_type="normal",
                confidence=decision.confidence,
                reason="命中唤醒词，强制进入回复流程",
                fallback=decision.fallback,
            )
            turn.decision = decision
        elif (
            not decision.should_respond
            and decision.is_addressed_to_ai
            and self.is_active()
            and self._has_assistant_reply()
            and self._looks_like_followup(turn.text)
        ):
            # The decision model marked the turn as addressed to the assistant
            # but chose not to respond. Inside the active window, right after the
            # assistant spoke, that is the classic continuation follow-up (e.g.
            # "最近是不是有个台风啊" after a weather answer). The model's
            # respond/confidence on such follow-ups sits on the decision boundary
            # and flips, so keep the conversation going.
            decision = InteractionDecision(
                is_addressed_to_ai=True,
                should_respond=True,
                response_type="normal",
                confidence=decision.confidence,
                reason="活跃窗口内延续助理话题，兜底进入回复流程",
                fallback=decision.fallback,
            )
            turn.decision = decision
        turn.should_reply = decision.should_respond
        if decision.should_respond:
            turn.reply_text = turn.text
            self.activate()
            self._append_context("user", turn.reply_text, turn.speaker.speaker_id)
            self._track_topic(turn.reply_text)
            turn.reason = decision.reason or turn.reason
        else:
            turn.reply_text = ""
            turn.reason = decision.reason or "决策模型判定无需回应"
        self.record_performance(
            "decision_model",
            (time.monotonic() - started) * 1000,
            source="backend",
            turn_id=turn.id,
            detail={
                "addressed": decision.is_addressed_to_ai,
                "respond": decision.should_respond,
                "response_type": decision.response_type,
                "fallback": decision.fallback,
                "complete": decision.is_complete,
                "buffer_merged": buffer_merged,
            },
        )
        logger.info(
            "decision session=%s turn=%s speaker=%s wake=%s context=%s addressed=%s respond=%s type=%s fallback=%s",
            self.id, turn.id, turn.speaker.speaker_id, turn.wake_detected, len(self.conversation_context),
            decision.is_addressed_to_ai, decision.should_respond, decision.response_type, decision.fallback,
        )
        return turn

    def evaluate_turn(
        self,
        turn_id: str,
        identity: SpeakerIdentity,
        audio: bytes,
        *,
        text: str,
        language: str,
        kws_detected: bool,
        kws_keyword: str | None,
    ) -> VoiceTurn:
        turn = self.get_turn(turn_id)
        if not turn:
            raise KeyError("turn not found or expired")
        if turn.evaluated:
            raise ValueError("turn already evaluated")

        text = (text or "").strip()
        wake_detected = self._wake_detected(text, kws_detected)
        # A turn normally snapshots activation at speech-start. Also honor the
        # current server-side window at evaluation time: queued VAD turns can be
        # created just before an earlier turn activates the session.
        active_for_turn = turn.started_while_active or self.is_active()
        turn.text = text
        turn.language = language
        turn.speaker = identity
        turn.wake_detected = wake_detected
        turn.wake_event = wake_detected and not active_for_turn
        is_participant = identity.speaker_id in self.participant_ids
        if wake_detected and not is_participant:
            self.participant_ids.add(identity.speaker_id)
            is_participant = True
            turn.joined_session = True
        turn.audio_ref = self.store_voice_audio(audio, identity.speaker_id, turn.id)

        # A speaker must explicitly enter this Session with a wake word. Once
        # joined, they can continue naturally; voices outside the participant
        # set are deliberately ignored even while the room is active.
        if not is_participant:
            turn.reason = "未加入当前会话：需要先说唤醒词"
            turn.evaluated = True
            return turn

        self._observe(identity, text)

        if turn.assistant_speaking:
            matched, remainder = self.match_interrupt(text)
            turn.interrupt = matched
            if not matched:
                turn.reason = "播放期间未命中打断词，视为环境干扰"
            elif remainder:
                turn.should_reply = True
                turn.reply_text = remainder
                turn.reason = "命中打断词并包含后续问题"
                self.activate()
            else:
                turn.reason = "命中纯打断指令"
                self.activate()
            turn.evaluated = True
            return turn

        # Provisional reply flag: used only as the fallback when the decision
        # model is unavailable. When the model responds it is the authority.
        if not text:
            turn.reason = "空识别结果"
        elif wake_detected or active_for_turn:
            turn.should_reply = True
            turn.reply_text = text
            if wake_detected:
                turn.reason = (
                    f"命中声学唤醒词：{kws_keyword or '已配置关键词'}"
                    if kws_detected
                    else "命中精确文本唤醒词"
                )
            else:
                turn.reason = "在连续对话窗口内"
        else:
            turn.reason = "会话未激活"

        turn.evaluated = True
        return turn

    def reject_turn(self, turn_id: str, reason: str) -> VoiceTurn:
        turn = self.get_turn(turn_id)
        if not turn:
            raise KeyError("turn not found or expired")
        turn.reason = reason
        turn.evaluated = True
        return turn

    def match_interrupt(self, text: str) -> tuple[bool, str]:
        normalized = (text or "").strip()
        for pattern in config.INTERRUPT_PATTERNS:
            match = pattern.search(normalized)
            if match:
                remainder = (normalized[: match.start()] + normalized[match.end() :]).strip(
                    " ，,。.!！?？、"
                )
                return True, remainder
        return False, normalized

    def snapshot(self) -> dict:
        observed = sorted(
            self.observed_speakers.values(), key=lambda item: item.last_seen_at, reverse=True
        )
        buffer = None
        if self.pending_buffer:
            buffer = {
                "text": self.pending_buffer["text"],
                "segments": self.pending_buffer["segments"],
                "waiting_seconds": round(time.time() - self.pending_buffer["updated_at"], 1),
            }
        return {
            "session_id": self.id,
            "active": self.is_active(),
            "active_remaining": round(self.active_remaining(), 1),
            "observed_speakers": [speaker.public() for speaker in observed],
            "buffer": buffer,
        }

    # ---- Hermes turn ----
    async def _chat_hermes(self, turn: VoiceTurn) -> str:
        context = {
            "session_id": self.id,
            "turn_id": turn.id,
            "turn_sequence": turn.sequence,
            "source": turn.source,
            "wake_event": turn.wake_event,
            "speaker": turn.speaker.public(),
            "audio_ref": turn.audio_ref,
            "response_type": turn.decision.response_type if turn.decision else "normal",
        }
        hermes_message = (
            VOICE_RESPONSE_CONTRACT
            + "[VOICE_CONTEXT]"
            + json.dumps(context, ensure_ascii=False, separators=(",", ":"))
            + "[/VOICE_CONTEXT]\n"
            + ("请仅作一句简短确认或追问。\n" if context["response_type"] == "brief" else "")
            + turn.reply_text
        )
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            config.HERMES_CLIENT_PATH,
            "ask",
            hermes_message,
            "--timeout",
            str(config.HERMES_TIMEOUT_SECONDS),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        spawn_at = time.monotonic()
        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(), timeout=config.HERMES_TIMEOUT_SECONDS + 10
            )
        except BaseException:
            if process.returncode is None:
                process.kill()
                try:
                    await asyncio.wait_for(process.communicate(), timeout=5.0)
                except asyncio.TimeoutError:
                    pass
            raise
        logger.info(
            "Hermes subprocess session=%s turn=%s subprocess=%.2fs",
            self.id,
            turn.id,
            time.monotonic() - spawn_at,
        )
        if process.returncode != 0:
            detail = stderr.decode(errors="replace").strip()
            raise RuntimeError(detail or f"Hermes client exited with {process.returncode}")
        reply = stdout.decode(errors="replace").strip()
        if not reply:
            raise RuntimeError("Hermes returned an empty response")
        return reply

    async def _execute_direct_llm_actions(self, turn: VoiceTurn, raw_reply: str) -> None:
        """Execute the small, session-scoped action allowlist returned by the LLM."""
        start = raw_reply.find("{")
        if start < 0:
            raise RuntimeError("direct LLM response did not contain a JSON object")
        try:
            value, _ = json.JSONDecoder().raw_decode(raw_reply[start:])
        except json.JSONDecodeError as exc:
            raise RuntimeError("direct LLM returned invalid JSON") from exc
        if not isinstance(value, dict):
            raise RuntimeError("direct LLM response must be a JSON object")
        actions = value.get("actions", [])
        if not isinstance(actions, list):
            raise RuntimeError("direct LLM actions must be an array")
        if len(actions) > 1:
            raise RuntimeError("direct LLM may execute at most one skill action per turn")
        if not actions:
            return
        action = actions[0]
        if not isinstance(action, dict):
            raise RuntimeError("direct LLM action must be an object")
        name = action.get("name")
        arguments = action.get("arguments")
        if not isinstance(name, str):
            raise RuntimeError("direct LLM action name must be a string")

        # The skill is bound to turn.speaker, never to an LLM-provided ID.
        skill = VoiceprintManagementSkill(turn.speaker.speaker_id)
        result = await skill.execute(name, arguments)
        if result.action == SET_CURRENT_PREFERRED_NAME:
            self.update_speaker_profile(
                turn.speaker.speaker_id,
                turn.speaker.display_name,
                result.preferred_name,
            )
            logger.info(
                "voiceprint skill session=%s turn=%s action=%s speaker=%s",
                self.id,
                turn.id,
                result.action,
                turn.speaker.speaker_id,
            )

    async def _chat_direct_llm(self, turn: VoiceTurn) -> str:
        """Ask the configured OpenAI-compatible model without involving Hermes."""
        if not config.DIRECT_LLM_MODEL:
            raise RuntimeError("direct_llm.model is required when conversation.provider is direct_llm")
        context = {
            "session_id": self.id,
            "turn_id": turn.id,
            "wake_event": turn.wake_event,
            "joined_session": turn.joined_session,
            "speaker": turn.speaker.public(),
            "should_ask_name": (
                turn.joined_session
                and not turn.speaker.identity_named
                and turn.speaker.speaker_id not in self.name_prompted_speaker_ids
            ),
            "response_type": turn.decision.response_type if turn.decision else "normal",
        }
        skill_definition = json.dumps(
            VoiceprintManagementSkill.action_definition(), ensure_ascii=False, separators=(",", ":")
        )
        system = DIRECT_LLM_SYSTEM.format(
            wake_word_aliases="、".join(f"“{word}”" for word in config.WAKE_WORDS if word)
            or "任何 ASR 对你名字的近音转写",
            voiceprint_skill=skill_definition,
        )
        messages = [
            {
                "role": "system",
                "content": system + "\n" + DIRECT_LLM_RESPONSE_CONTRACT,
            }
        ]
        for entry in self.conversation_context:
            role = entry["role"]
            if role not in {"user", "assistant"}:
                continue
            content = entry["text"]
            if role == "user" and entry.get("speaker_id"):
                content = f"[用户 {entry['speaker_id'][:8]}] {content}"
            messages.append({"role": role, "content": content})
        messages.append(
            {
                "role": "user",
                "content": (
                    "[VOICE_CONTEXT]"
                    + json.dumps(context, ensure_ascii=False, separators=(",", ":"))
                    + "[/VOICE_CONTEXT]\n"
                    + ("请仅作一句简短确认或追问。\n" if context["response_type"] == "brief" else "")
                    + turn.reply_text
                ),
            }
        )
        payload = {
            "model": config.DIRECT_LLM_MODEL,
            "messages": messages,
            "temperature": config.DIRECT_LLM_TEMPERATURE,
            "max_tokens": config.DIRECT_LLM_MAX_TOKENS,
            "response_format": {"type": "json_object"},
            "chat_template_kwargs": {"enable_thinking": config.DIRECT_LLM_ENABLE_THINKING},
        }
        headers = (
            {"Authorization": f"Bearer {config.DIRECT_LLM_API_KEY}"}
            if config.DIRECT_LLM_API_KEY
            else {}
        )
        started = time.monotonic()
        async with httpx.AsyncClient(
            base_url=config.DIRECT_LLM_BASE_URL,
            timeout=httpx.Timeout(config.DIRECT_LLM_TIMEOUT_SECONDS, connect=5.0),
            headers=headers,
        ) as client:
            response = await client.post("/chat/completions", json=payload)
            response.raise_for_status()
        logger.info(
            "direct LLM session=%s turn=%s elapsed=%.2fs",
            self.id,
            turn.id,
            time.monotonic() - started,
        )
        try:
            reply = response.json()["choices"][0]["message"]["content"]
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError("direct LLM returned an invalid chat response") from exc
        if not isinstance(reply, str) or not reply.strip():
            raise RuntimeError("direct LLM returned an empty response")
        await self._execute_direct_llm_actions(turn, reply)
        return reply

    async def _chat_reply(self, turn: VoiceTurn) -> str:
        if config.CONVERSATION_PROVIDER == "direct_llm":
            return await self._chat_direct_llm(turn)
        return await self._chat_hermes(turn)

    async def chat(self, turn_id: str) -> HermesReply:
        async with self.chat_lock:
            turn = self.get_turn(turn_id)
            if not turn or not turn.evaluated:
                raise KeyError("turn not found, expired, or not evaluated")
            if not turn.should_reply:
                raise PermissionError("turn was not admitted to the conversation")
            if turn.consumed:
                raise ValueError("turn already consumed")
            raw_reply = await self._chat_reply(turn)
            turn.consumed = True
            reply = parse_hermes_reply(raw_reply)
            if turn.joined_session and turn.speaker and not turn.speaker.identity_named:
                self.name_prompted_speaker_ids.add(turn.speaker.speaker_id)
            self._append_context("assistant", reply.text)
            return reply

    def update_speaker_profile(
        self,
        speaker_id: str,
        display_name: str | None,
        preferred_name: str | None,
    ) -> None:
        identity = None
        if speaker_id in self.observed_speakers:
            identity = self.observed_speakers[speaker_id].identity
        if identity:
            identity.display_name = display_name
            identity.preferred_name = preferred_name
            identity.identity_named = _identity_named(display_name, preferred_name)
        for turn in self.turns.values():
            if turn.speaker and turn.speaker.speaker_id == speaker_id:
                turn.speaker.display_name = display_name
                turn.speaker.preferred_name = preferred_name
                turn.speaker.identity_named = _identity_named(display_name, preferred_name)


class SessionManager:
    def __init__(self, decision_client: DecisionClient | None = None) -> None:
        self._store: dict[str, Session] = {}
        self._decision_client = decision_client

    async def warmup_decisions(self) -> None:
        warmup = getattr(self._decision_client, "warmup", None)
        if warmup:
            await warmup()

    def create(self) -> Session:
        sid = secrets.token_hex(16)
        session = Session(id=sid, decision_client=self._decision_client)
        self._store[sid] = session
        return session

    def get(self, sid: str) -> Session | None:
        return self._store.get(sid)

    def find_voice_audio(self, speaker_id: str, audio_ref: str) -> bytes | None:
        for session in self._store.values():
            audio = session.get_voice_audio(speaker_id, audio_ref)
            if audio is not None:
                return audio
        return None

    def update_speaker_profile(
        self,
        speaker_id: str,
        display_name: str | None,
        preferred_name: str | None,
    ) -> None:
        for session in self._store.values():
            session.update_speaker_profile(speaker_id, display_name, preferred_name)

    def remove_speaker(self, speaker_id: str) -> None:
        for session in self._store.values():
            session.observed_speakers.pop(speaker_id, None)
            session.voice_audio_refs = {
                token: ref
                for token, ref in session.voice_audio_refs.items()
                if ref.speaker_id != speaker_id
            }
            session.voice_audio_refs = {
                token: ref
                for token, ref in session.voice_audio_refs.items()
                if ref.speaker_id != speaker_id
            }

    def delete(self, sid: str) -> bool:
        return self._store.pop(sid, None) is not None


sessions = SessionManager(decision_client=build_decision_client())
