import sys

from text_processing import (
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
        "I am working on SQLAlchemy upgrade and PR create karunga."
    )


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
