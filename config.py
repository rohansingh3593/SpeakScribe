"""Central configuration for SpeakScribe."""

from dataclasses import dataclass, field
from enum import Enum


class PerformanceMode(str, Enum):
    FAST = "fast"
    BALANCED = "balanced"
    ACCURATE = "accurate"


@dataclass(frozen=True)
class DecodeProfile:
    beam_size: int
    best_of: int
    temperature: float


PROFILES = {
    PerformanceMode.FAST: DecodeProfile(1, 1, 0.0),
    PerformanceMode.BALANCED: DecodeProfile(3, 3, 0.0),
    PerformanceMode.ACCURATE: DecodeProfile(5, 5, 0.0),
}


DEFAULT_VOCABULARY = (
    "Python", "PyQt", "PyQt6", "SQLAlchemy", "Alembic", "FastAPI",
    "Pydantic", "Jenkins", "Docker", "Kubernetes", "Git", "GitHub",
    "GitLab", "Jira", "API", "REST API", "pull request", "PR", "commit",
    "branch", "merge", "pipeline", "pytest", "database", "PostgreSQL",
    "MongoDB", "Kafka", "Redis",
)


@dataclass
class AppConfig:
    sample_rate: int = 16_000
    channels: int = 1
    frame_ms: int = 30
    # Laptop microphone levels are commonly well below 0.01 after SoundCard's
    # float conversion. These remain above a typical quiet-room noise floor but
    # do not discard softer Hindi phonemes before Whisper sees them.
    speech_threshold: float = 0.0025
    silence_threshold: float = 0.0008
    silence_duration: float = 1.10
    min_speech_duration: float = 0.50
    pre_speech_duration: float = 0.20
    partial_interval: float = 0.50
    min_partial_duration: float = 1.20
    max_utterance_seconds: float = 15.0
    rolling_window_seconds: float = 5.0
    overlap_seconds: float = 0.35
    max_audio_queue: int = 100
    max_asr_queue: int = 1
    context_sentences: int = 2
    model_size: str = "tiny"
    device: str = "auto"
    compute_type: str = "auto"
    performance_mode: PerformanceMode = PerformanceMode.BALANCED
    vad_filter: bool = False  # RMS VAD already runs before Whisper.
    no_speech_threshold: float = 0.85
    min_avg_logprob: float = -2.0
    script_mode: str = "original"
    language_mode: str = "auto"
    translation_enabled: bool = False
    translation_model: str = "Helsinki-NLP/opus-mt-hi-en"
    vocabulary: tuple[str, ...] = field(default_factory=lambda: DEFAULT_VOCABULARY)

    @property
    def frame_samples(self) -> int:
        return self.sample_rate * self.frame_ms // 1000

    @property
    def profile(self) -> DecodeProfile:
        return PROFILES[self.performance_mode]
