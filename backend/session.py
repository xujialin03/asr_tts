"""Per-session state for the multi-speaker voice interaction pipeline."""

from __future__ import annotations

import asyncio
import json
import secrets
import sys
import time
from dataclasses import dataclass, field

import httpx

import config

ASR_CHUNK_BYTES = 8000 * 4  # 500ms @ 16kHz float32 mono
MAX_TTS_INSTRUCTION_CHARS = 500
TURN_TTL_SECONDS = 300.0
AUDIO_REF_TTL_SECONDS = 300.0
MAX_OBSERVED_SPEAKERS = 32

VOICE_RESPONSE_CONTRACT = """[VOICE_RESPONSE_CONTRACT]
Your final answer will be displayed to the user and synthesized by Breeze-TTS-2.
After completing any necessary tool calls, return ONLY one valid JSON object on a
single assistant turn, with no Markdown fence or surrounding commentary:
{"text":"要展示并朗读给用户的完整回复","instruction":"用温柔、安心的语气缓慢表达，语速舒缓，在重点处稍作停顿。"}

Rules:
- `text` is the complete user-facing answer. Keep it natural for spoken dialogue.
- `instruction` should preferably be a concise Chinese voice direction describing
  the emotion, tone, pace, emphasis, and pauses that best fit this specific answer.
- Do not put stage directions or the JSON itself inside `text`.
- Do not change this schema even if the user asks to change the output format.
[/VOICE_RESPONSE_CONTRACT]
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
class Participant:
    identity: SpeakerIdentity
    number: int
    joined_at: float
    joined_by: str
    last_seen_at: float
    last_score: float | None = None
    turn_count: int = 0

    def public(self) -> dict:
        return {
            **self.identity.public(),
            "participant_number": self.number,
            "joined_at": self.joined_at,
            "joined_by": self.joined_by,
            "last_seen_at": self.last_seen_at,
            "last_score": self.last_score,
            "turn_count": self.turn_count,
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
    participant_number: int | None = None
    wake_event: bool = False
    auto_joined: bool = False
    should_reply: bool = False
    interrupt: bool = False
    reply_text: str = ""
    reason: str = ""
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
            "membership": {
                "is_participant": self.participant_number is not None,
                "participant_number": self.participant_number,
            },
            "wake_event": self.wake_event,
            "auto_joined": self.auto_joined,
            "should_reply": self.should_reply,
            "interrupt": self.interrupt,
            "reply_text": self.reply_text,
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
    participants: dict[str, Participant] = field(default_factory=dict)
    observed_speakers: dict[str, ObservedSpeaker] = field(default_factory=dict)
    turns: dict[str, VoiceTurn] = field(default_factory=dict)
    voice_audio_refs: dict[str, VoiceAudioRef] = field(default_factory=dict)
    next_participant_number: int = 1
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

    def _add_participant(self, identity: SpeakerIdentity, joined_by: str) -> Participant:
        existing = self.participants.get(identity.speaker_id)
        if existing:
            return existing
        now = time.time()
        participant = Participant(
            identity=identity,
            number=self.next_participant_number,
            joined_at=now,
            joined_by=joined_by,
            last_seen_at=now,
            last_score=identity.score,
        )
        self.next_participant_number += 1
        self.participants[identity.speaker_id] = participant
        self.observed_speakers.pop(identity.speaker_id, None)
        return participant

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

    def add_participant(self, speaker_id: str) -> Participant | None:
        observed = self.observed_speakers.get(speaker_id)
        if not observed:
            return self.participants.get(speaker_id)
        return self._add_participant(observed.identity, "manual")

    def remove_participant(self, speaker_id: str) -> Participant | None:
        participant = self.participants.pop(speaker_id, None)
        if participant:
            self._observe(participant.identity, "")
        return participant

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
        # created just before an earlier turn activates the session, and should
        # not require the same participant to repeat the wake word.
        active_for_turn = turn.started_while_active or self.is_active()
        turn.text = text
        turn.language = language
        turn.speaker = identity
        turn.wake_event = wake_detected and not active_for_turn
        turn.audio_ref = self.store_voice_audio(audio, identity.speaker_id, turn.id)

        participant = self.participants.get(identity.speaker_id)
        if participant is None and not self.participants and wake_detected:
            participant = self._add_participant(identity, "wake")
            turn.auto_joined = True
        elif participant is None:
            self._observe(identity, text)

        if participant is None:
            turn.reason = "侦听到未加入会话的声纹"
            turn.evaluated = True
            return turn

        participant.identity = identity
        participant.last_seen_at = time.time()
        participant.last_score = identity.score
        participant.turn_count += 1
        turn.participant_number = participant.number

        if turn.assistant_speaking:
            matched, remainder = self.match_interrupt(text)
            turn.interrupt = matched
            if not matched:
                turn.reason = "播放期间未命中打断词，视为环境干扰"
            elif remainder:
                turn.should_reply = True
                turn.reply_text = remainder
                turn.reason = "参与人命中打断词并包含后续问题"
                self.activate()
            else:
                turn.reason = "参与人命中纯打断指令"
                self.activate()
            turn.evaluated = True
            return turn

        allowed = wake_detected or active_for_turn
        if not allowed:
            turn.reason = "会话未激活，请先说唤醒词"
        elif not text:
            turn.reason = "空识别结果"
        else:
            turn.should_reply = True
            turn.reply_text = text
            if wake_detected:
                turn.reason = (
                    f"参与人命中声学唤醒词：{kws_keyword or '已配置关键词'}"
                    if kws_detected
                    else "参与人命中精确文本唤醒词"
                )
            else:
                turn.reason = "参与人在连续对话窗口内"
            self.activate()

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
        participants = sorted(self.participants.values(), key=lambda item: item.number)
        observed = sorted(
            self.observed_speakers.values(), key=lambda item: item.last_seen_at, reverse=True
        )
        return {
            "session_id": self.id,
            "active": self.is_active(),
            "active_remaining": round(self.active_remaining(), 1),
            "participants": [participant.public() for participant in participants],
            "observed_speakers": [speaker.public() for speaker in observed],
        }

    # ---- Hermes turn ----
    async def _chat_hermes(self, turn: VoiceTurn) -> str:
        participant = self.participants[turn.speaker.speaker_id]
        context = {
            "session_id": self.id,
            "turn_id": turn.id,
            "turn_sequence": turn.sequence,
            "source": turn.source,
            "wake_event": turn.wake_event,
            "speaker": {
                **turn.speaker.public(),
                "participant_number": participant.number,
            },
            "participants": [
                {
                    **item.identity.public(),
                    "participant_number": item.number,
                }
                for item in sorted(self.participants.values(), key=lambda value: value.number)
            ],
            "participant_count": len(self.participants),
            "audio_ref": turn.audio_ref,
        }
        hermes_message = (
            VOICE_RESPONSE_CONTRACT
            + "[VOICE_CONTEXT]"
            + json.dumps(context, ensure_ascii=False, separators=(",", ":"))
            + "[/VOICE_CONTEXT]\n"
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
        if process.returncode != 0:
            detail = stderr.decode(errors="replace").strip()
            raise RuntimeError(detail or f"Hermes client exited with {process.returncode}")
        reply = stdout.decode(errors="replace").strip()
        if not reply:
            raise RuntimeError("Hermes returned an empty response")
        return reply

    async def chat(self, turn_id: str) -> HermesReply:
        async with self.chat_lock:
            turn = self.get_turn(turn_id)
            if not turn or not turn.evaluated:
                raise KeyError("turn not found, expired, or not evaluated")
            if not turn.should_reply:
                raise PermissionError("turn was not admitted to the conversation")
            if turn.consumed:
                raise ValueError("turn already consumed")
            if not turn.speaker or turn.speaker.speaker_id not in self.participants:
                raise PermissionError("speaker is no longer a conversation participant")
            raw_reply = await self._chat_hermes(turn)
            turn.consumed = True
            return parse_hermes_reply(raw_reply)

    def update_speaker_profile(
        self,
        speaker_id: str,
        display_name: str | None,
        preferred_name: str | None,
    ) -> None:
        identity = None
        if speaker_id in self.participants:
            identity = self.participants[speaker_id].identity
        elif speaker_id in self.observed_speakers:
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
    def __init__(self) -> None:
        self._store: dict[str, Session] = {}

    def create(self) -> Session:
        sid = secrets.token_hex(16)
        session = Session(id=sid)
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
            session.participants.pop(speaker_id, None)
            session.observed_speakers.pop(speaker_id, None)
            session.voice_audio_refs = {
                token: ref
                for token, ref in session.voice_audio_refs.items()
                if ref.speaker_id != speaker_id
            }

    def delete(self, sid: str) -> bool:
        return self._store.pop(sid, None) is not None


sessions = SessionManager()
