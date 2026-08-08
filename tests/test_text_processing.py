from text_processing import clean_text, detect_language, remove_history_overlap


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

