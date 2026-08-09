from types import SimpleNamespace

from evaluation_runner import (
    _root_cause, compare_transcripts, evaluate_case_with_retries, evaluation_error_result, format_duration,
    normalize_transcript, regression_metrics, status_for_similarity,
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


def test_accent_regression_remains_visible_instead_of_relaxing_accuracy():
    expected = "अलग क्षेत्रीय उच्चारण में भी यह पूरा वाक्य सही पहचाना जाना चाहिए।"
    actual = "अलक शेत्रिय उच्चारन में भी यह पुरा वाख्य सही पहचाना जाना जाहें।"
    similarity, wer, _ = compare_transcripts(expected, actual)
    assert wer == 0.5
    assert similarity == 50.0
    assert status_for_similarity(similarity) == "FAIL"


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


def test_accuracy_gain_does_not_hide_latency_regression():
    current = SimpleNamespace(
        similarity=90.0, wer=0.1, final_transcript_latency=3.0, memory_mb=500.0,
    )
    previous = {
        "similarity": 80.0, "wer": 0.2, "final_transcript_latency": 1.0,
        "memory_mb": 450.0,
    }

    metrics = regression_metrics(current, previous)

    assert metrics["accuracy_delta"] == 10.0
    assert metrics["latency_delta"] == 2.0
    assert metrics["regression"] is True


def test_case_crash_is_preserved_as_structured_result():
    case = {
        "id": "CRASH-01", "audio": "broken.wav", "language": "Hindi",
        "expected": "परीक्षण", "scenario": "reliability", "difficulty": "hard",
    }

    result = evaluation_error_result(case, "CRASH", RuntimeError("decoder stopped"))

    assert result.status == "CRASH"
    assert result.quality_flags == ["CRASH"]
    assert "decoder stopped" in result.actual


def test_pause_feature_is_not_misdiagnosed_as_vad_when_full_audio_was_decoded():
    result = SimpleNamespace(
        quality_flags=[], detected_language="Hindi", language="Hindi",
        technical_term_problems=[], dropped_chunks=0, duplicated_words=[],
        real_time_factor=0.5, status="FAIL", punctuation_difference=False,
        similarity=20,
    )
    cause, _ = _root_cause({"features": ["long_pause"]}, result)
    assert cause == "Whisper decoding"


def test_actual_dropped_chunks_are_diagnosed_as_vad_or_chunk_loss():
    result = SimpleNamespace(
        quality_flags=[], detected_language="English", language="English",
        technical_term_problems=[], dropped_chunks=2, duplicated_words=[],
        real_time_factor=0.5, status="FAIL", punctuation_difference=False,
        similarity=20,
    )
    cause, _ = _root_cause({"features": ["long_pause"]}, result)
    assert cause == "VAD/chunk loss"
