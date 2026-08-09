"""General prompt policy for multilingual Faster-Whisper decoding."""

from __future__ import annotations


def initial_prompt(*, final: bool, sample_count: int, sample_rate: int,
                   language_mode: str, vocabulary: tuple[str, ...],
                   context: str) -> str | None:
    """Build transcript-like context without injecting instructions into Whisper.

    Whisper's ``initial_prompt`` is previous transcript text, not a system prompt.
    Instructional English plus a large English vocabulary can dominate acoustically
    weak Hindi clips. Pinned Hindi therefore receives only genuine prior transcript
    context; auto/English modes may additionally receive vocabulary bias.
    """
    if not final or sample_count < sample_rate:
        return None
    parts = []
    context = context.strip()
    if context:
        parts.append(context)
    if language_mode != "hi" and vocabulary:
        parts.append(", ".join(vocabulary))
    return ". ".join(parts) or None
