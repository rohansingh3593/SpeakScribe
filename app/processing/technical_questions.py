"""Fast technical interview-question detection for finalized transcript text.

This deliberately avoids constructing a Transformers pipeline at import time.
ASR stays responsive and callers may explicitly provide an AI fallback when
they need semantic classification beyond the deterministic vocabulary matcher.
"""

from __future__ import annotations

import re
from collections.abc import Callable

from app.config.technical_vocabulary import detect_technical_topics


QUESTION_LEAD_IN = re.compile(
    r"\b(?:what|how|why|when|where|who|can you|could you|would you|should you|"
    r"explain|describe|define|tell me about|walk me through|discuss|outline|design)\b",
    re.IGNORECASE,
)
_SPLIT_BEFORE = re.compile(
    r"(?=\b(?:what|how|why|when|where|who|can you|could you|would you|"
    r"explain|describe|define|tell me about|walk me through|design)\b)",
    re.IGNORECASE,
)


def smart_sentence_split(text: str) -> tuple[str, ...]:
    """Recover questions even when ASR omitted question-mark punctuation."""
    normalized = re.sub(r"\s+", " ", text).strip()
    output = []
    for sentence in re.split(r"(?<=[.?!।])\s+", normalized):
        output.extend(part.strip(" .?!।") for part in _SPLIT_BEFORE.split(sentence)
                      if len(part.strip(" .?!।")) >= 15)
    return tuple(output)


def extract_technical_questions(
        text: str,
        semantic_fallback: Callable[[str], bool] | None = None,
) -> tuple[tuple[str, str], ...]:
    """Return ``(question, topic)`` pairs without blocking the ASR/UI threads.

    ``semantic_fallback`` is opt-in and called only for question-like text that
    has no vocabulary match. A heavyweight model can therefore live in a
    separate worker rather than being downloaded during application startup.
    """
    results = []
    for sentence in smart_sentence_split(text):
        if not QUESTION_LEAD_IN.search(sentence):
            continue
        topics = detect_technical_topics(sentence)
        if topics:
            results.append((sentence, topics[0]))
        elif semantic_fallback is not None and semantic_fallback(sentence):
            results.append((sentence, "General"))
    return tuple(results)

