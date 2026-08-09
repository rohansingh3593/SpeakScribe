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

    Latin hotwords can flip a Hindi-pinned decode into romanized output even when
    the acoustics are entirely Devanagari. Preserve them for English and automatic
    code-switching modes, but never bias a pinned Hindi decode with Latin tokens.
    """
    if not final or language_mode == "hi" or not vocabulary:
        return None
    return ", ".join(vocabulary)
