from types import SimpleNamespace
from queue import Queue
from threading import Event
import time

import numpy as np

from app.asr.asr_engine import ComparisonASRWorker, WhisperEngine
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
    assert fast.profile.model_size == balanced.profile.model_size == accurate.profile.model_size == (
        "small")


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


def test_live_worker_runs_final_audio_in_fast_mode_without_refinement_contention():
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

    assert {mode for mode, _audio in calls} == {PerformanceMode.FAST}
    assert all(shared is audio for _mode, shared in calls)
    assert {value[1] for value in signals.mode_text.values} == {"fast"}
    assert signals.mode_error.values == []


def test_comparison_worker_preserves_live_partial_refinement_for_all_modes():
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


def test_mode_queue_partial_replaces_obsolete_partials_and_preserves_finals():
    queue = Queue(maxsize=8)
    audio = np.ones(10, dtype=np.float32)
    older_final = ASRJob(audio, True, 1, 0.0)
    obsolete_same_segment = ASRJob(audio, False, 2, 0.0)
    obsolete_old_segment = ASRJob(audio, False, 1, 0.0)
    newest = ASRJob(audio, False, 2, 0.0)
    queue.put(older_final)
    queue.put(obsolete_same_segment)
    queue.put(obsolete_old_segment)

    ComparisonASRWorker._enqueue_mode_job(PerformanceMode.FAST, queue, newest)

    assert queue.get_nowait() is older_final
    assert queue.get_nowait() is newest
    assert queue.empty()


def test_latest_final_replaces_queued_final_when_live_asr_is_behind():
    from app.audio.audio_pipeline import SpeechBufferWorker

    config = AppConfig(asr_keep_latest_final=True, max_asr_queue=1)
    asr_queue = Queue(maxsize=1)
    audio_queue = Queue()
    stop = Event()
    worker = SpeechBufferWorker(config, audio_queue, asr_queue, stop)
    audio = np.ones(10, dtype=np.float32)
    stale = ASRJob(audio, True, 70, 0.0)
    newest = ASRJob(audio, True, 71, 0.0)
    asr_queue.put(stale)

    worker._submit(newest)

    assert asr_queue.get_nowait() is newest
    assert asr_queue.empty()

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
            if item[2] == "Listening"} == {(9, "fast", "Listening")}


def test_empty_final_promotes_latest_valid_partial_instead_of_no_speech():
    calls = 0

    class Provider:
        def get(self, _config):
            def transcribe(job, _context):
                nonlocal calls
                calls += 1
                return (("पहचाना हुआ हिंदी भाषण", "Hindi", {"script_valid": True})
                        if not job.final else ("", "Hindi", {"script_valid": True}))
            return SimpleNamespace(transcribe=transcribe)

    signals = SimpleNamespace(mode_text=SignalRecorder(), mode_status=SignalRecorder(),
                              mode_error=SignalRecorder())
    upstream = Queue()
    audio = np.ones(1600, dtype=np.float32)
    mode_queue = Queue()
    mode_queue.put(ASRJob(audio, False, 90, 0.0))
    mode_queue.put(ASRJob(audio, True, 90, 0.0))
    mode_queue.put(None)
    stop = Event(); stop.set()

    worker = ComparisonASRWorker(
        AppConfig(language_mode="hi"), upstream, stop, signals, Provider())
    worker._run_mode(PerformanceMode.FAST, mode_queue)

    finals = [item for item in signals.mode_text.values if item[3] is True]
    assert calls == 2
    assert finals[0][2] == "पहचाना हुआ हिंदी भाषण"
    assert finals[0][4]["recovered_from_partial"] is True
    assert any(item == (90, "fast", "Final") for item in signals.mode_status.values)


def test_empty_final_without_partial_is_terminal_but_not_displayed():
    class Provider:
        def get(self, _config):
            return SimpleNamespace(transcribe=lambda _job, _context: (
                "", "Hindi", {"script_valid": True}))

    signals = SimpleNamespace(mode_text=SignalRecorder(), mode_status=SignalRecorder(),
                              mode_error=SignalRecorder())
    queue = Queue()
    queue.put(ASRJob(np.ones(1600, dtype=np.float32), True, 91, 0.0))
    stop = Event(); stop.set()

    ComparisonASRWorker(AppConfig(language_mode="hi"), queue, stop, signals, Provider()).run()

    assert signals.mode_text.values == []
    assert any(item == (91, "fast", "No speech") for item in signals.mode_status.values)


def test_results_older_than_latency_target_are_still_decoded_and_displayed():
    calls = []

    class Provider:
        def get(self, _config):
            return SimpleNamespace(
                transcribe=lambda _job, _context: calls.append(True) or ("late", "English"))

    signals = SimpleNamespace(mode_text=SignalRecorder(), mode_status=SignalRecorder(),
                              mode_error=SignalRecorder())
    queue = Queue()
    queue.put(ASRJob(np.ones(10, dtype=np.float32), True, 10,
                     time.monotonic() - 21.0))
    stop = Event(); stop.set()

    ComparisonASRWorker(AppConfig(), queue, stop, signals, Provider()).run()

    assert len(calls) == 1
    assert {item[1] for item in signals.mode_text.values} == {"fast"}
    assert all(item[2] == "late" and item[3] is True
               for item in signals.mode_text.values)
    assert {item[2] for item in signals.mode_status.values} == {"Processing", "Final"}


def test_live_fast_hindi_result_is_emitted_with_script_metadata():
    class Provider:
        def get(self, config):
            def transcribe(_job, _context):
                if config.performance_mode is PerformanceMode.FAST:
                    return ("आपको क्या करना है?", "Hindi", {
                        "raw_text": "आपको क्या करना है?", "processed_text": "आपको क्या करना है?",
                        "detected_script": "devanagari", "requested_script": "original",
                        "script_valid": True,
                    })
                return ("آپ کو کیا کرنا ہے؟", "Hindi", {
                    "raw_text": "آپ کو کیا کرنا ہے؟", "processed_text": "آپ کو کیا کرنا ہے؟",
                    "detected_script": "arabic", "requested_script": "original",
                    "script_valid": False,
                })
            return SimpleNamespace(transcribe=transcribe)

    signals = SimpleNamespace(mode_text=SignalRecorder(), mode_status=SignalRecorder(),
                              mode_error=SignalRecorder())
    queue = Queue()
    queue.put(ASRJob(np.ones(1600, dtype=np.float32), True, 42, 0.0))
    stop = Event(); stop.set()
    ComparisonASRWorker(AppConfig(language_mode="hi"), queue, stop, signals, Provider()).run()

    displayed = {item[1]: item[2] for item in signals.mode_text.values}
    assert displayed["fast"] == "आपको क्या करना है?"
    assert set(displayed) == {"fast"}
    mismatch_modes = {item[1] for item in signals.mode_status.values
                      if item[2] == "Script mismatch"}
    assert mismatch_modes == set()


def test_invalid_final_is_not_added_to_future_asr_context():
    contexts = []

    class Provider:
        def get(self, _config):
            def transcribe(_job, context):
                contexts.append(context)
                return ("آپ کو کیا کرنا ہے؟", "Hindi", {
                    "detected_script": "arabic", "script_valid": False})
            return SimpleNamespace(transcribe=transcribe)

    signals = SimpleNamespace(mode_text=SignalRecorder(), mode_status=SignalRecorder(),
                              mode_error=SignalRecorder())
    queue = Queue()
    audio = np.ones(1600, dtype=np.float32)
    queue.put(ASRJob(audio, True, 50, 0.0))
    queue.put(ASRJob(audio, True, 51, 0.0))
    stop = Event(); stop.set()

    ComparisonASRWorker(AppConfig(language_mode="hi"), queue, stop, signals, Provider()).run()

    assert contexts and all(context == "" for context in contexts)
    assert {item[2] for item in signals.mode_status.values} == {
        "Processing", "Script mismatch"}


def test_consecutive_identical_fast_finals_are_marked_duplicate():
    class Provider:
        def get(self, _config):
            return SimpleNamespace(transcribe=lambda _job, _context: (
                "यह एक सही हिंदी वाक्य है।", "Hindi", {"script_valid": True}))

    signals = SimpleNamespace(mode_text=SignalRecorder(), mode_status=SignalRecorder(),
                              mode_error=SignalRecorder())
    queue = Queue()
    audio = np.ones(1600, dtype=np.float32)
    queue.put(ASRJob(audio, True, 80, 0.0))
    queue.put(ASRJob(audio, True, 81, 0.0))
    stop = Event(); stop.set()

    ComparisonASRWorker(AppConfig(language_mode="hi"), queue, stop, signals, Provider()).run()

    emitted = {item[0]: item for item in signals.mode_text.values}
    assert emitted[80][2] == "यह एक सही हिंदी वाक्य है।"
    assert 81 not in emitted
    assert any(item == (81, "fast", "Duplicate") for item in signals.mode_status.values)


def test_whisper_raw_arabic_evidence_is_preserved_and_flagged_before_display():
    segment = SimpleNamespace(
        start=0.0, end=1.0, no_speech_prob=0.0, avg_logprob=0.0,
        compression_ratio=1.0, text=" آپ کو کیا کرنا ہے؟ ")
    info = SimpleNamespace(language="hi", language_probability=0.98)
    model = SimpleNamespace(transcribe=lambda *_args, **_kwargs: ([segment], info))
    engine = WhisperEngine(AppConfig(language_mode="hi", script_mode="original"), model)

    text, language, metadata = engine.transcribe(
        ASRJob(np.ones(16000, dtype=np.float32), True, 43, 0.0), "")

    assert text == "آپ کو کیا کرنا ہے؟"
    assert language == "Hindi"
    assert metadata["raw_text"] == "آپ کو کیا کرنا ہے؟"
    assert metadata["processed_text"] == text
    assert metadata["detected_language"] == "hi"
    assert metadata["detected_script"] == "arabic"
    assert metadata["script_valid"] is False


def test_hindi_final_retries_wrong_script_and_uses_valid_devanagari_recovery():
    arabic = SimpleNamespace(
        start=0.0, end=1.0, no_speech_prob=0.0, avg_logprob=0.0,
        compression_ratio=1.0, text=" آپ کو کیا کرنا ہے؟ ")
    hindi = SimpleNamespace(
        start=0.0, end=1.0, no_speech_prob=0.0, avg_logprob=0.0,
        compression_ratio=1.0, text=" आपको क्या करना है? ")
    info = SimpleNamespace(language="hi", language_probability=0.98)
    calls = iter([([arabic], info), ([hindi], info)])
    model = SimpleNamespace(transcribe=lambda *_args, **_kwargs: next(calls))
    engine = WhisperEngine(AppConfig(language_mode="hi", script_mode="original"), model)

    text, language, metadata = engine.transcribe(
        ASRJob(np.ones(32000, dtype=np.float32), True, 44, 0.0), "")

    assert text == "आपको क्या करना है?"
    assert language == "Hindi"
    assert metadata["detected_script"] == "devanagari"
    assert metadata["script_valid"] is True


def test_hindi_final_retries_a_single_embedded_urdu_suffix():
    mixed = SimpleNamespace(
        start=0.0, end=2.0, no_speech_prob=0.0, avg_logprob=0.0,
        compression_ratio=1.0, text=" महिलाوں की भागीदारी 63 प्रतिशत रही है। ")
    recovered = SimpleNamespace(
        start=0.0, end=2.0, no_speech_prob=0.0, avg_logprob=0.0,
        compression_ratio=1.0, text=" महिलाओं की भागीदारी 63 प्रतिशत रही है। ")
    info = SimpleNamespace(language="hi", language_probability=0.99)
    calls = iter([([mixed], info), ([recovered], info)])
    engine = WhisperEngine(
        AppConfig(language_mode="hi", script_mode="original"),
        SimpleNamespace(transcribe=lambda *_args, **_kwargs: next(calls)))

    text, _language, metadata = engine.transcribe(
        ASRJob(np.ones(32000, dtype=np.float32), True, 45, 0.0), "")

    assert text == "महिलाओं की भागीदारी 63 प्रतिशत रही है।"
    assert metadata["detected_script"] == "devanagari"
    assert metadata["arabic_character_ratio"] == 0
    assert metadata["script_valid"] is True
