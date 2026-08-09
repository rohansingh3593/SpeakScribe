"""Conservative language, cleanup, and optional script processing."""

from importlib import import_module
import re

TECHNICAL_CANONICAL = {
    "sql alchemy": "SQLAlchemy", "sqlalchemy": "SQLAlchemy", "fast api": "FastAPI",
    "py qt six": "PyQt6", "pyqt6": "PyQt6", "github": "GitHub",
    "gitlab": "GitLab", "rest api": "REST API", "pull request": "pull request",
}
HINGLISH_WORDS = {
    "aaj", "ab", "aur", "baad", "bad", "hai", "hain", "hoon", "hum", "kar",
    "karo", "karunga", "karungi", "kal", "ka", "ki", "main", "mein", "mujhe",
    "nahi", "pe", "raha", "rahi", "se", "task", "uske", "wala", "yeh",
}


def compose_live_transcript(final_texts, partial_text: str = "") -> str:
    """Render committed and in-progress recognition in one stable text stream."""
    lines = [str(text).strip() for text in final_texts if str(text).strip()]
    partial = partial_text.strip()
    # A queued partial may arrive immediately after its identical final. Never
    # show that transient callback twice in the combined view.
    if partial and (not lines or partial.casefold() != lines[-1].casefold()):
        lines.append(partial)
    return "\n".join(lines)


def detect_language(text: str, whisper_language: str | None = None) -> str:
    devanagari = len(re.findall(r"[\u0900-\u097f]", text))
    latin_words = re.findall(r"[A-Za-z']+", text.lower())
    hinglish = sum(word in HINGLISH_WORDS for word in latin_words)
    english = sum(word not in HINGLISH_WORDS for word in latin_words)
    if devanagari and latin_words:
        return "Hinglish"
    if devanagari:
        return "Hindi"
    if hinglish >= 2 and english >= 1:
        return "Hinglish"
    if whisper_language == "hi" and hinglish:
        return "Hinglish"
    return "Hindi" if whisper_language == "hi" and not latin_words else "English"


def clean_text(text: str, final: bool = False) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"([,.!?])\1+", r"\1", text)
    # Preserve a deliberate double word ("मैं मैं", "very very"). Collapse only
    # 3+ adjacent copies to two; severe repetition remains covered by the
    # low-quality detector instead of silently rewriting plausible speech.
    text = re.sub(r"\b([\w'\u0900-\u097f]+)(?:\s+\1\b){2,}", r"\1 \1", text,
                  flags=re.IGNORECASE)
    for raw, canonical in TECHNICAL_CANONICAL.items():
        text = re.sub(rf"(?<!\w){re.escape(raw)}(?!\w)", canonical, text,
                      flags=re.IGNORECASE)
    text = re.sub(r"\bpr\b", "PR", text, flags=re.IGNORECASE)
    if text:
        text = text[0].upper() + text[1:]
    if final and text and text[-1] not in ".!?।":
        text += "।" if re.search(r"[\u0900-\u097f]", text) else "."
    return text


def is_low_quality_text(text: str) -> bool:
    """Reject corruption/repetition without rewriting plausible speech."""
    compact = re.sub(r"\s+", "", text)
    lowered = text.casefold()
    if not compact:
        return False
    if "\ufffd" in text or "http://" in lowered or "https://" in lowered or "www." in lowered:
        return True
    if re.search(r"(.)\1{5,}", compact):
        return True
    if len(compact) >= 24 and len(set(compact)) / len(compact) < 0.18:
        return True
    for width in range(2, min(9, len(compact) // 4 + 1)):
        if any(compact[index:index + width] * 4 in compact
               for index in range(len(compact) - width + 1)):
            return True
    return False


def remove_history_overlap(history: str, text: str, limit: int = 12) -> str:
    old, new = history.split(), text.split()
    for count in range(min(limit, len(old), len(new)), 0, -1):
        if [w.casefold() for w in old[-count:]] == [w.casefold() for w in new[:count]]:
            return " ".join(new[count:])
    return text


def apply_script_mode(text: str, mode: str, technical_terms: tuple[str, ...]) -> str:
    if mode == "original":
        return text
    if mode not in {"latin", "devanagari"}:
        raise ValueError(f"Unsupported script mode: {mode}")

    # Transliteration is optional and is not part of language detection or text
    # cleanup. Load it only for an explicitly selected conversion mode so those
    # lightweight operations remain available without the optional package.
    sanscript = import_module("indic_transliteration.sanscript")
    protected: dict[str, str] = {}
    for index, term in enumerate(sorted(technical_terms, key=len, reverse=True)):
        # Private-use markers contain neither Latin nor Devanagari letters, so
        # the selective regex conversions below cannot mutate the placeholder.
        token = f"\ue000{index}\ue001"
        updated = re.sub(re.escape(term), token, text, flags=re.IGNORECASE)
        if updated != text:
            protected[token] = term
            text = updated
    if mode == "latin":
        # Preserve text that Whisper already emitted in Latin script and convert
        # only Devanagari runs.
        text = re.sub(
            r"[\u0900-\u097f]+",
            lambda match: sanscript.transliterate(
                match.group(0), sanscript.DEVANAGARI, sanscript.ITRANS),
            text,
        )
    else:
        # Most Hindi-mode Whisper output is already Devanagari. Never feed those
        # characters into the ITRANS parser; convert Latin words only.
        text = re.sub(
            r"[A-Za-z']+",
            lambda match: sanscript.transliterate(
                match.group(0), sanscript.ITRANS, sanscript.DEVANAGARI),
            text,
        )
    for token, term in protected.items():
        text = text.replace(token, term)
    return text
