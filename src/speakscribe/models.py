"""Stable data models returned by the public API."""

from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass(frozen=True)
class TranscriptionResult:
    text: str
    language: str | None = None
    confidence: float | None = None
    is_final: bool = True
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    utterance_id: int | None = None
    audio_duration: float | None = None
    queue_wait_seconds: float | None = None
    inference_seconds: float | None = None
    speech_to_result_seconds: float | None = None
