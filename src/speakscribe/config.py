"""Public configuration for the reusable speakscribe service."""

from dataclasses import dataclass


@dataclass(frozen=True)
class SpeechConfig:
    language: str | None = None
    capture_source: str = "microphone"
    sample_rate: int = 16_000
    channels: int = 1
    continuous: bool = False
    chunk_duration: float = 0.10
    partial_interval: float = 0.60
    silence_duration: float = 0.80
    minimum_speech_duration: float = 0.20
    pre_speech_duration: float = 0.20
    maximum_utterance_duration: float = 15.0
    minimum_rms: float = 0.001
    max_audio_queue: int = 32
    max_asr_queue: int = 2
    model_size: str = "small"
    device: str = "auto"
    compute_type: str = "auto"
    beam_size: int = 3

    def __post_init__(self) -> None:
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if self.channels <= 0:
            raise ValueError("channels must be positive")
        if self.chunk_duration <= 0:
            raise ValueError("chunk_duration must be positive")
        if self.minimum_rms < 0:
            raise ValueError("minimum_rms cannot be negative")
        if self.partial_interval <= 0 or self.silence_duration <= 0:
            raise ValueError("partial_interval and silence_duration must be positive")
        if self.minimum_speech_duration <= 0 or self.pre_speech_duration < 0:
            raise ValueError("speech durations are invalid")
        if self.max_audio_queue < 1 or self.max_asr_queue < 1:
            raise ValueError("queue sizes must be positive")
        if self.capture_source not in {"microphone", "loopback"}:
            raise ValueError("capture_source must be 'microphone' or 'loopback'")
