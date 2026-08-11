from types import SimpleNamespace

from app.config.settings import AppConfig, PERFORMANCE_PROFILES, PerformanceMode
from evaluation.mode_comparison import build_comparison, render_markdown


def result(mode, accuracy, latency, language="English"):
    return SimpleNamespace(
        case_id="same-audio", audio="same.wav", language=language,
        expected="SQLAlchemy works", actual=f"{mode} transcript",
        similarity=accuracy, wer=(100 - accuracy) / 100,
        first_partial_latency=latency / 2, final_transcript_latency=latency,
        inference_seconds=latency * .8, real_time_factor=latency * .4,
        cpu_percent=20 + latency * 10, memory_mb=500 + latency * 100,
        technical_term_accuracy=accuracy, partial_updates=2,
        duplicate_partials=0, dropped_chunks=0,
    )


def test_profiles_are_distinct_and_balanced_is_default():
    assert tuple(PERFORMANCE_PROFILES) == tuple(PerformanceMode)
    assert AppConfig().performance_mode is PerformanceMode.BALANCED
    fast, balanced, accurate = (AppConfig(performance_mode=mode)
                                for mode in PerformanceMode)
    assert fast.partial_interval < balanced.partial_interval < accurate.partial_interval
    assert fast.context_sentences < balanced.context_sentences < accurate.context_sentences
    assert fast.profile.beam_size < balanced.profile.beam_size < accurate.profile.beam_size


def test_report_uses_measured_winners_and_side_by_side_transcripts():
    report = build_comparison({
        "fast": [result("fast", 80, .3)],
        "balanced": [result("balanced", 92, .6)],
        "accurate": [result("accurate", 97, 1.1)],
    })
    assert report["same_audio_for_all_modes"] is True
    assert report["best_by_metric"]["first_partial_latency"] == "fast"
    assert report["best_by_metric"]["accuracy"] == "accurate"
    assert set(report["transcripts"][0]["transcripts"]) == {
        "fast", "balanced", "accurate"}
    markdown = render_markdown(report)
    assert "FAST: FAST" not in markdown
    assert "fast transcript" in markdown
