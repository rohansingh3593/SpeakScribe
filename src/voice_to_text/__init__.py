"""Reusable voice-to-text public API with no GUI dependencies."""

from .config import SpeechConfig
from .exceptions import (
    AudioDeviceNotFoundError, MicrophoneError, ServiceStateError,
    TranscriptionError, VoiceToTextError,
)
from .models import TranscriptionResult
from .services import SpeechToText
from .transcription import BaseTranscriptionEngine, FasterWhisperEngine

__all__ = [
    "AudioDeviceNotFoundError", "BaseTranscriptionEngine", "FasterWhisperEngine",
    "MicrophoneError", "ServiceStateError", "SpeechConfig", "SpeechToText",
    "TranscriptionError", "TranscriptionResult", "VoiceToTextError",
]
