from types import SimpleNamespace

from evaluation_runner import (
    compare_transcripts, evaluate_case_with_retries, format_duration,
    normalize_transcript, status_for_similarity,
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
    assert status_for_similarity(60) == "WARNING"
    assert status_for_similarity(59.99) == "FAIL"


def test_format_duration_is_readable_and_stable():
    assert format_duration(0) == "00:00:00"
    assert format_duration(3661) == "01:01:01"


def test_failed_case_is_retried_without_hiding_initial_failure():
    results = iter([
        SimpleNamespace(status="FAIL", similarity=40.0, quality_flags=[]),
        SimpleNamespace(status="PASS", similarity=85.0, quality_flags=[]),
    ])

    result = evaluate_case_with_retries(
        {}, None, None, retries=1, evaluator=lambda *_args: next(results),
    )

    assert result.status == "FAIL"
    assert result.similarity == 40.0
    assert result.attempts == 2
    assert result.initial_similarity == 40.0
    assert result.retry_improvement == 45.0
    assert result.best_retry_similarity == 85.0
    assert result.best_retry_status == "PASS"
    assert "UNSTABLE_RESULT" in result.quality_flags


def test_retry_limit_prevents_unbounded_failed_runs():
    calls = []

    def failed(*_args):
        calls.append(1)
        return SimpleNamespace(status="FAIL", similarity=20.0, quality_flags=[])

    result = evaluate_case_with_retries({}, None, None, retries=2, evaluator=failed)

    assert result.attempts == 3
    assert len(calls) == 3
