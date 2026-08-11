"""Replaceable transcription-engine contract."""

from abc import ABC, abstractmethod

from speakscribe.models import TranscriptionResult


class BaseTranscriptionEngine(ABC):
    @abstractmethod
    def transcribe(self, audio, sample_rate: int) -> TranscriptionResult:
        """Convert one mono audio chunk into a structured result."""

    def close(self) -> None:
        """Release optional engine resources."""
