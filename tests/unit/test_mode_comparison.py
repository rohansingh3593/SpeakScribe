from types import SimpleNamespace
from queue import Queue
from threading import Event

import numpy as np

from app.asr.asr_engine import ComparisonASRWorker
from app.audio.audio_pipeline import ASRJob
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


class SignalRecorder:
    def __init__(self):
        self.values = []

    def emit(self, *values):
        self.values.append(values)


def test_comparison_worker_fans_same_audio_to_three_isolated_modes():
    calls = []

    class Provider:
        def get(self, config):
            class Engine:
                def transcribe(self, job, _context):
                    calls.append((config.performance_mode, job.audio))
                    if config.performance_mode is PerformanceMode.BALANCED:
                        raise RuntimeError("isolated failure")
                    return config.performance_mode.value, "English"
            return Engine()

    signals = SimpleNamespace(mode_text=SignalRecorder(), mode_status=SignalRecorder(),
                              mode_error=SignalRecorder())
    audio = np.ones(1600, dtype=np.float32)
    queue = Queue()
    queue.put(ASRJob(audio=audio, final=True, utterance_id=7, captured_at=0.0))
    stop = Event()
    stop.set()
    ComparisonASRWorker(AppConfig(), queue, stop, signals, Provider()).run()

    assert {mode for mode, _audio in calls} == set(PerformanceMode)
    assert all(shared is audio for _mode, shared in calls)
    assert {value[1] for value in signals.mode_text.values} == {"fast", "accurate"}
    assert signals.mode_error.values[0][1] == "balanced"


def test_comparison_worker_runs_each_mode_for_the_same_partial_segment():
    calls = []

    class Provider:
        def get(self, config):
            class Engine:
                def transcribe(self, job, _context):
                    calls.append((config.performance_mode, job.audio))
                    return "live words", "Hinglish"
            return Engine()

    signals = SimpleNamespace(mode_text=SignalRecorder(), mode_status=SignalRecorder(),
                              mode_error=SignalRecorder())
    audio = np.ones(3200, dtype=np.float32)
    queue = Queue()
    queue.put(ASRJob(audio=audio, final=False, utterance_id=8, captured_at=0.0))
    stop = Event()
    stop.set()
    ComparisonASRWorker(AppConfig(), queue, stop, signals, Provider()).run()

    assert {mode for mode, _audio in calls} == set(PerformanceMode)
    assert all(shared is audio for _mode, shared in calls)
    assert {(value[0], value[1], value[2], value[3])
            for value in signals.mode_text.values} == {
        (8, "fast", "live words", False),
        (8, "balanced", "live words", False),
        (8, "accurate", "live words", False),
    }
    assert all("result_latency" in value[4] and "queue_delay" in value[4]
               for value in signals.mode_text.values)


def test_mode_queue_final_evicts_partials_but_preserves_older_finals():
    queue = Queue(maxsize=8)
    audio = np.ones(10, dtype=np.float32)
    older_final = ASRJob(audio, True, 1, 0.0)
    queue.put(older_final)
    queue.put(ASRJob(audio, False, 2, 0.0))
    queue.put(ASRJob(audio, False, 2, 0.0))
    newest_final = ASRJob(audio, True, 2, 0.0)

    ComparisonASRWorker._enqueue_mode_job(
        PerformanceMode.FAST, queue, newest_final)

    assert queue.get_nowait() is older_final
    assert queue.get_nowait() is newest_final
    assert queue.empty()


def test_empty_partial_returns_to_listening_instead_of_blank_partial_cell():
    class Provider:
        def get(self, _config):
            return SimpleNamespace(transcribe=lambda _job, _context: ("", "English"))

    signals = SimpleNamespace(mode_text=SignalRecorder(), mode_status=SignalRecorder(),
                              mode_error=SignalRecorder())
    queue = Queue()
    queue.put(ASRJob(np.ones(10, dtype=np.float32), False, 9, 0.0))
    stop = Event(); stop.set()

    ComparisonASRWorker(AppConfig(), queue, stop, signals, Provider()).run()

    assert signals.mode_text.values == []
    assert {(item[0], item[1], item[2]) for item in signals.mode_status.values
            if item[2] == "Listening"} == {
        (9, "fast", "Listening"), (9, "balanced", "Listening"),
        (9, "accurate", "Listening"),
    }
