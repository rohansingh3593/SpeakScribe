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
    """All knobs which define a user-visible performance mode."""

    beam_size: int
    best_of: int
    temperature: float
    partial_interval: float
    min_partial_duration: float
    rolling_window_seconds: float
    overlap_seconds: float
    context_sentences: int
    silence_duration: float
    condition_on_previous_text: bool
    post_processing_level: str
    model_size: str


PERFORMANCE_PROFILES = {
    # FAST uses a dedicated lighter model; Balanced and Accurate share the
    # larger model when their device and compute settings match.
    PerformanceMode.FAST: DecodeProfile(
        beam_size=2, best_of=2, temperature=0.0, partial_interval=0.25,
        min_partial_duration=0.55, rolling_window_seconds=3.0,
        overlap_seconds=0.20, context_sentences=1, silence_duration=0.85,
        condition_on_previous_text=False, post_processing_level="light", model_size="small"),
    PerformanceMode.BALANCED: DecodeProfile(
        beam_size=3, best_of=3, temperature=0.0, partial_interval=0.40,
        min_partial_duration=0.80, rolling_window_seconds=5.0,
        overlap_seconds=0.35, context_sentences=2, silence_duration=1.25,
        condition_on_previous_text=True, post_processing_level="standard", model_size="small"),
    PerformanceMode.ACCURATE: DecodeProfile(
        beam_size=5, best_of=5, temperature=0.0, partial_interval=0.65,
        min_partial_duration=1.10, rolling_window_seconds=9.0,
        overlap_seconds=0.60, context_sentences=4, silence_duration=1.65,
        condition_on_previous_text=True, post_processing_level="full", model_size="small"),
}

# Backwards-compatible name for integrations which imported the old table.
PROFILES = PERFORMANCE_PROFILES


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
    # ASR pauses shorter than this remain in the same readable paragraph.
    paragraph_pause_threshold: float = 2.0
    sample_rate: int = 16_000
    # Spoken transcription should use the physical microphone by default.
    # Speaker loopback remains available for transcribing system playback.
    capture_source: str = "microphone"
    capture_sample_rate: int = 16_000
    channels: int = 1
    capture_warmup_blocks: int = 3
    capture_warmup_ms: int = 100
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
    # Publish a first rolling hypothesis quickly while the one-item ASR queue
    # continues to prevent CPU fallback from accumulating stale partial jobs.
    partial_interval: float = 0.40
    min_partial_duration: float = 0.80
    min_partial_speech_duration: float = 0.30
    max_utterance_seconds: float = 15.0
    rolling_window_seconds: float = 5.0
    overlap_seconds: float = 0.35
    max_audio_queue: int = 100
    max_asr_queue: int = 1
    # Compare All can take longer than real time on CPU. In that explicitly
    # selected mode, keep the newest completed segment rather than blocking VAD
    # and eventually overflowing the raw capture queue behind an obsolete final.
    asr_keep_latest_final: bool = False
    max_result_latency_seconds: float = 20.0
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
        return PERFORMANCE_PROFILES[self.performance_mode]

    def __post_init__(self) -> None:
        """Apply the selected centralized profile to the streaming pipeline."""
        profile = self.profile
        for name in ("partial_interval", "min_partial_duration",
                     "rolling_window_seconds", "overlap_seconds",
                     "context_sentences", "silence_duration"):
            setattr(self, name, getattr(profile, name))
