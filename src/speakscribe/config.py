"""Public configuration for the reusable speakscribe service."""

from dataclasses import dataclass


@dataclass(frozen=True)
class SpeechConfig:
    language: str | None = None
    sample_rate: int = 16_000
    channels: int = 1
    continuous: bool = False
    chunk_duration: float = 1.5
    minimum_rms: float = 0.001
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
