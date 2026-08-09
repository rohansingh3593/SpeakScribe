"""Central configuration for SpeakScribe."""

from dataclasses import dataclass, field
from enum import Enum
import os


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
    "MongoDB", "Kafka", "Redis", "update", "dependency", "run",
    "image", "verify", "endpoint", "response", "save", "service",
    "model", "result", "CPU", "RAM", "CI",
)


@dataclass
class AppConfig:
    sample_rate: int = 16_000
    # Realtek/WASAPI devices normally run natively at 48 kHz. Asking Media
    # Foundation to capture directly at 16 kHz caused discontinuities and garbled
    # Whisper input on the reported machine, so capture natively and downsample.
    capture_sample_rate: int = 48_000
    channels: int = 1
    frame_ms: int = 30
    # Read several VAD frames per backend call. Media Foundation can report
    # discontinuities when it is polled every 30 ms; 120 ms keeps enough
    # headroom while frames are still emitted to VAD at the original cadence.
    capture_chunk_ms: int = 120
    # Laptop microphone levels are commonly well below 0.01 after SoundCard's
    # float conversion. These remain above a typical quiet-room noise floor but
    # do not discard softer Hindi phonemes before Whisper sees them.
    speech_threshold: float = 0.003
    silence_threshold: float = 0.002
    speech_start_frames: int = 3
    adaptive_vad_enabled: bool = True
    adaptive_vad_floor: float = 0.0002
    adaptive_vad_multiplier: float = 2.5
    silence_duration: float = 1.50
    min_speech_duration: float = 0.50
    pre_speech_duration: float = 0.20
    partial_interval: float = 0.50
    min_partial_duration: float = 1.20
    min_partial_speech_duration: float = 0.50
    max_utterance_seconds: float = 15.0
    rolling_window_seconds: float = 5.0
    overlap_seconds: float = 0.35
    max_audio_queue: int = 100
    max_asr_queue: int = 1
    context_sentences: int = 2
    # `small` is the minimum production default with reliable multilingual
    # capacity; `base` caused broad Hindi/Hinglish degradation in real runs.
    model_size: str = "small"
    device: str = "auto"
    compute_type: str = "auto"
    performance_mode: PerformanceMode = PerformanceMode.BALANCED
    vad_filter: bool = False  # RMS VAD already runs before Whisper.
    no_speech_threshold: float = 0.85
    min_avg_logprob: float = -2.0
    max_compression_ratio: float = 2.4
    script_mode: str = "original"
    # Short Hindi/Hinglish chunks are frequently misdetected as English. Hindi
    # decoding still preserves embedded English technical terms.
    language_mode: str = "hi"
    translation_enabled: bool = False
    translation_model: str = "Helsinki-NLP/opus-mt-hi-en"
    debug_log_interval: float = 1.0
    debug_audio_enabled: bool = os.getenv("SPEAKSCRIBE_DEBUG_AUDIO", "0") == "1"
    debug_audio_directory: str = "data/debug_audio"
    vocabulary: tuple[str, ...] = field(default_factory=lambda: DEFAULT_VOCABULARY)

    @property
    def frame_samples(self) -> int:
        return self.sample_rate * self.frame_ms // 1000

    @property
    def capture_frame_samples(self) -> int:
        return self.capture_sample_rate * self.frame_ms // 1000

    @property
    def capture_chunk_samples(self) -> int:
        return self.capture_sample_rate * self.capture_chunk_ms // 1000

    @property
    def profile(self) -> DecodeProfile:
        return PROFILES[self.performance_mode]
