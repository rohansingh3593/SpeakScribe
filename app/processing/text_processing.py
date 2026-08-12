"""Conservative language, cleanup, and optional script processing."""

from difflib import SequenceMatcher
from html import escape
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

SCRIPT_RANGES = {
    "devanagari": re.compile(r"[\u0900-\u097f\ua8e0-\ua8ff]"),
    "arabic": re.compile(r"[\u0600-\u06ff\u0750-\u077f\u08a0-\u08ff\ufb50-\ufdff\ufe70-\ufeff]"),
    "latin": re.compile(r"[A-Za-z]"),
}


def script_metadata(text: str, requested_script: str = "original",
                    language_mode: str = "auto") -> dict[str, str | float | bool]:
    """Classify writing systems and validate Hindi output without rewriting it."""
    counts = {name: len(pattern.findall(text)) for name, pattern in SCRIPT_RANGES.items()}
    total = sum(counts.values())
    ratios = {name: count / total if total else 0.0 for name, count in counts.items()}
    if counts["arabic"] and ratios["arabic"] >= 0.20:
        detected = "arabic"
    elif counts["devanagari"] and counts["latin"]:
        detected = "mixed-devanagari-latin"
    else:
        detected = max(counts, key=counts.get) if total else "none"
    hindi = language_mode == "hi"
    requested = requested_script.lower().replace(" / roman", "")
    valid = True
    if hindi and counts["arabic"]:
        # Arabic/Urdu output is evidence, not input to a lossy character map.
        # Keep it in metadata but never promote it as Hindi/Hinglish display text.
        valid = ratios["arabic"] < 0.20 and counts["arabic"] < counts["devanagari"]
    if hindi and requested == "devanagari" and counts["latin"] and not counts["devanagari"]:
        valid = False
    if hindi and requested == "latin" and counts["devanagari"]:
        valid = False
    expected = requested if requested in {"devanagari", "latin"} else (
        "devanagari-or-latin" if hindi else "original")
    return {
        "expected_script": expected,
        "detected_script": detected,
        "script_match": valid,
        "script_valid": valid,
        "requested_script": requested_script,
        "arabic_character_ratio": ratios["arabic"],
        "devanagari_character_ratio": ratios["devanagari"],
        "latin_character_ratio": ratios["latin"],
    }


def format_recording_time(seconds: float) -> str:
    """Format a monotonic recording duration for the compact UI timer."""
    total = max(0, int(seconds))
    minutes, remaining_seconds = divmod(total, 60)
    return f"{minutes:02d}:{remaining_seconds:02d}"


def format_processing_duration(seconds: float) -> str:
    """Show one measured duration in both seconds and milliseconds."""
    value = max(0.0, seconds)
    return f"{value:.2f}s ({value * 1000:.0f}ms)"


def compose_live_transcript(final_text: list[str], partial_text: str = "") -> str:
    """Render stable text plus one replaceable partial without duplicating it."""
    parts = [text.strip() for text in final_text if text.strip()]
    if partial_text.strip():
        parts.append(partial_text.strip())
    return "\n".join(parts)


def comparison_diff_html(transcripts: dict[str, str]) -> dict[str, str]:
    """Highlight words which are not aligned identically in every transcript."""
    tokenized = {mode: re.findall(r"\S+", text) for mode, text in transcripts.items()}

    def comparable(word: str) -> str:
        return re.sub(r"[^\w\u0900-\u097f]+", "", word, flags=re.UNICODE).casefold()

    keys = {mode: [comparable(word) for word in words]
            for mode, words in tokenized.items()}
    output = {}
    for mode, words in tokenized.items():
        matching = set(range(len(words)))
        for other_mode, other_keys in keys.items():
            if other_mode == mode:
                continue
            aligned = set()
            matcher = SequenceMatcher(None, keys[mode], other_keys, autojunk=False)
            for block in matcher.get_matching_blocks():
                aligned.update(range(block.a, block.a + block.size))
            matching.intersection_update(aligned)
        rendered = []
        for index, word in enumerate(words):
            safe = escape(word)
            rendered.append(safe if index in matching else
                            f'<span style="background-color:#8b2635;color:#fff;'
                            f'font-weight:600;padding:1px 2px">{safe}</span>')
        output[mode] = " ".join(rendered)
    return output


def comparison_agreement_percentages(transcripts: dict[str, str]) -> dict[str, float]:
    """Return reference-free word agreement; this is not ground-truth accuracy."""
    normalized = {
        mode: [re.sub(r"[^\w\u0900-\u097f]+", "", word, flags=re.UNICODE).casefold()
               for word in text.split()]
        for mode, text in transcripts.items()
    }
    scores = {}
    for mode, words in normalized.items():
        peers = [peer for other, peer in normalized.items() if other != mode]
        ratios = [SequenceMatcher(None, words, peer, autojunk=False).ratio()
                  for peer in peers]
        scores[mode] = round(100 * sum(ratios) / len(ratios), 1) if ratios else 100.0
    return scores


def descending_segment_row(existing_ids, segment_id: int) -> int:
    """Return the stable insertion row for newest-first segment ordering."""
    return sum(1 for existing_id in existing_ids if existing_id > segment_id)


def best_refinement_candidate(raw_results: dict[str, str | None],
                              partial_results: dict[str, str | None]) -> tuple[str | None, str]:
    """Choose Accurate > Balanced > Fast without promoting empty/corrupt text."""
    for candidates in (raw_results, partial_results):
        for mode in ("accurate", "balanced", "fast"):
            text = candidates.get(mode) or ""
            if text and not is_low_quality_text(text):
                return mode, text
    return None, ""


def incremental_transcript_delta(existing: str, candidate: str) -> str:
    """Return only words not already present at the end of the live stream."""
    old_words = existing.split()
    new_words = candidate.strip().split()
    if not new_words:
        return ""

    def comparable(word: str) -> str:
        word = word.strip(".,!?;:।")
        return re.sub(r"[^\w\u0900-\u097f]+", "", word).casefold()

    old_keys = [comparable(word) for word in old_words]
    new_keys = [comparable(word) for word in new_words]
    for count in range(min(len(old_keys), len(new_keys)), 0, -1):
        if old_keys[-count:] == new_keys[:count]:
            return " ".join(new_words[count:])
    # Whisper may revise the last partial instead of extending it. Since the UI
    # is intentionally append-only, preserve what is visible but append only
    # the revised suffix after the longest shared prefix, not the whole partial.
    common_prefix = 0
    for start in range(len(old_keys)):
        count = 0
        while (start + count < len(old_keys) and count < len(new_keys) and
               old_keys[start + count] == new_keys[count]):
            count += 1
        common_prefix = max(common_prefix, count)
    if common_prefix:
        return " ".join(new_words[common_prefix:])
    return " ".join(new_words)


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
    if final and text and text[-1] not in ".!?।؟":
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


def remove_history_overlap(history: str, text: str, limit: int = 12,
                           min_overlap: int = 1) -> str:
    old, new = history.split(), text.split()
    def comparable(word: str) -> str:
        return re.sub(r"[^\w\u0900-\u097f]+", "", word).casefold()
    for count in range(min(limit, len(old), len(new)), min_overlap - 1, -1):
        if ([comparable(w) for w in old[-count:]] ==
                [comparable(w) for w in new[:count]]):
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
