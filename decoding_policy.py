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


def hotwords(*, final: bool, vocabulary: tuple[str, ...]) -> str | None:
    """Return vocabulary bias through Whisper's dedicated hotword channel."""
    if not final or not vocabulary:
        return None
    return ", ".join(vocabulary)
