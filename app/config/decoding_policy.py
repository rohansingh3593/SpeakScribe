"""General prompt policy for multilingual Faster-Whisper decoding."""

from __future__ import annotations


def initial_prompt(*, final: bool, sample_count: int, sample_rate: int,
                   language_mode: str, vocabulary: tuple[str, ...],
                   context: str) -> str | None:
    """Build transcript-like context without injecting instructions into Whisper.

    Whisper's ``initial_prompt`` is previous transcript text, not a system prompt.
    Instructional English plus a large English vocabulary can dominate acoustically
    weak Hindi clips. All modes therefore receive only genuine prior transcript
    context; vocabulary uses Faster-Whisper's dedicated hotword channel.
    """
    if not final or sample_count < sample_rate:
        return None
    parts = []
    context = context.strip()
    if context:
        parts.append(context)
    return ". ".join(parts) or None


def hotwords(*, final: bool, language_mode: str,
             vocabulary: tuple[str, ...]) -> str | None:
    """Return safe vocabulary bias through Whisper's hotword channel.

    Latin hotwords can dominate both Hindi-pinned and acoustically ambiguous
    Hinglish audio. Restrict global vocabulary bias to English; mixed-language
    vocabulary needs confidence-aware biasing rather than an unconditional list.
    """
    if not final or language_mode != "en" or not vocabulary:
        return None
    return ", ".join(vocabulary)


def retry_thresholds(*, no_speech: float, log_probability: float,
                     compression_ratio: float) -> tuple[float, float, float]:
    """Relax rejection gates for one prompt-free final recovery attempt."""
    return max(no_speech, 0.95), min(log_probability, -3.0), max(compression_ratio, 4.0)
