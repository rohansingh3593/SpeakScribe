import sys
from types import ModuleType

from app.processing.text_processing import (
    apply_script_mode, clean_text, detect_language, is_low_quality_text,
    remove_history_overlap,
)


def test_hinglish_detection():
    text = "Today main SQLAlchemy upgrade pe work kar raha hoon"
    assert detect_language(text, "hi") == "Hinglish"


def test_hindi_and_english_detection():
    assert detect_language("मुझे आज काम करना है", "hi") == "Hindi"
    assert detect_language("I will create the pull request", "en") == "English"


def test_conservative_cleanup():
    raw = "i am working on sql alchemy upgrade upgrade and pr create karunga"
    assert clean_text(raw, final=True) == (
        "I am working on SQLAlchemy upgrade upgrade and PR create karunga."
    )


def test_cleanup_preserves_intentional_hindi_and_english_repetition():
    assert clean_text("मैं मैं पहले रिपोर्ट देखूँगा", final=True) == (
        "मैं मैं पहले रिपोर्ट देखूँगा।")
    assert clean_text("This is very very important", final=True) == (
        "This is very very important.")


def test_cleanup_bounds_three_or_more_adjacent_copies_without_erasing_emphasis():
    assert clean_text("go go go go now", final=True) == "Go go now."


def test_history_overlap():
    assert remove_history_overlap("I updated SQLAlchemy", "SQLAlchemy and FastAPI") == "and FastAPI"


def test_original_script_mode_does_not_load_optional_transliteration():
    assert apply_script_mode("मुझे FastAPI चाहिए", "original", ("FastAPI",)) == (
        "मुझे FastAPI चाहिए"
    )
    assert "indic_transliteration" not in sys.modules


def test_corrupt_and_repetitive_whisper_output_is_rejected():
    assert is_low_quality_text("पर �")
    assert is_low_quality_text("https://www.cf.co.uk")
    assert is_low_quality_text("पुभुपुभुपुभुपुभुपुभु")
    assert not is_low_quality_text("आज हम SQLAlchemy upgrade पर काम करेंगे।")


def test_devanagari_mode_does_not_retransliterate_existing_hindi(monkeypatch):
    fake = ModuleType("indic_transliteration.sanscript")
    fake.ITRANS = "itrans"
    fake.DEVANAGARI = "devanagari"
    fake.transliterate = lambda value, _source, _target: f"<{value}>"
    monkeypatch.setitem(sys.modules, "indic_transliteration.sanscript", fake)

    converted = apply_script_mode(
        "आज main SQLAlchemy पर काम", "devanagari", ("SQLAlchemy",),
    )
    assert converted == "आज <main> SQLAlchemy पर काम"


def test_devanagari_mode_converts_romanized_hindi_but_protects_dev_vocabulary(monkeypatch):
    fake = ModuleType("indic_transliteration.sanscript")
    fake.ITRANS = "itrans"
    fake.DEVANAGARI = "devanagari"
    fake.transliterate = lambda value, _source, _target: {
        "Kaam": "काम", "ho": "हो", "gaya": "गया",
    }.get(value, f"<{value}>")
    monkeypatch.setitem(sys.modules, "indic_transliteration.sanscript", fake)

    converted = apply_script_mode(
        "Kaam ho gaya PostgreSQL update run",
        "devanagari", ("PostgreSQL", "update", "run"),
    )
    assert converted == "काम हो गया PostgreSQL update run"
