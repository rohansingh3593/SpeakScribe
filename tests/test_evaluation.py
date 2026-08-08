from evaluation_runner import (
    compare_transcripts, normalize_transcript, status_for_similarity,
)


def test_normalization_ignores_case_punctuation_and_hindi_danda():
    assert normalize_transcript("  आज, काम पूरा है। ") == "आज काम पूरा है"
    assert normalize_transcript("REST API... Ready!") == "rest api ready"


def test_similarity_reports_word_errors_and_warning_status():
    expected = "Today main SQLAlchemy upgrade pe work kar raha hoon"
    actual = "Today main SQLAlchemy upgrade pe kaam kar raha hoon"
    similarity, wer, details = compare_transcripts(expected, actual)
    assert similarity >= 80
    assert wer == 0.1111
    assert details.substitutions == ["work -> kaam"]
    assert status_for_similarity(similarity) == "PASS"


def test_status_boundaries():
    assert status_for_similarity(80) == "PASS"
    assert status_for_similarity(60) == "PASS WITH WARNING"
    assert status_for_similarity(59.99) == "FAIL"
