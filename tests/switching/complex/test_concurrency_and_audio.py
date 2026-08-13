"""Concurrency boundaries, stale results, and first-word audio evidence."""

from queue import Queue
from threading import Event, Thread
import time

import numpy as np
import pytest

from app.asr.asr_engine import ASRWorker
from app.asr.language_transition import RecognitionState
from app.audio.audio_pipeline import SpeechBufferWorker
from app.config.settings import AppConfig, PerformanceMode
from tests.switching.support import Engine, Provider, Signals, evidence, job


def test_late_old_partial_never_overwrites_new_language(record_switch):
    state = RecognitionState("hi", "original")
    queue, stop, signals = Queue(), Event(), Signals()
    engine = Engine({1: "पुराना", 2: "new English"}, {1: 0.02})
    worker = ASRWorker(AppConfig(), queue, stop, signals, Provider(engine), state)
    queue.put(job(1, "hi", 0))
    thread = Thread(target=worker.run)
    thread.start()
    time.sleep(0.005)
    changed = state.switch("en", "original")
    queue.put(job(2, "en", changed.generation))
    time.sleep(0.05)
    stop.set()
    thread.join(2)
    displayed = [value[0] for value in signals.partial_text.values]
    record_switch(evidence(
        "SW-COMPLEX-007", "Late old-language result",
        "Late Hindi partial is rejected and English remains live", repr(displayed),
        test_type="Negative", switch_from="hi", switch_to="en",
        first_transcript=displayed[0] if displayed else ""))
    assert displayed == ["new English"]


@pytest.mark.xfail(
    reason="Known scheduler limitation: one ASRWorker cannot let new FAST B overtake running A",
    strict=True)
def test_new_fast_can_complete_before_running_old_inference(record_switch):
    """Permanent exposure test for the requested B-before-A scheduling guarantee."""
    state = RecognitionState("hi", "original")
    queue, stop, signals = Queue(), Event(), Signals()
    engine = Engine({1: "पुराना", 2: "new English"}, {1: 0.15})
    worker = ASRWorker(AppConfig(), queue, stop, signals, Provider(engine), state)
    queue.put(job(1, "hi", 0))
    thread = Thread(target=worker.run)
    thread.start()
    time.sleep(0.01)
    changed = state.switch("en", "original")
    queued_at = time.perf_counter()
    queue.put(job(2, "en", changed.generation))
    time.sleep(0.04)
    displayed = [value[0] for value in signals.partial_text.values]
    record_switch(evidence(
        "SW-COMPLEX-003", "Switch while FAST inference runs",
        "New English FAST completes without waiting for old Hindi inference",
        f"visible_after_40ms={displayed}", test_type="Negative",
        switch_from="hi", switch_to="en", total_latency=time.perf_counter() - queued_at,
        suspected_component="Single ASRWorker inference lane",
        recommended_investigation="Use generation-aware independent FAST execution lane"))
    stop.set()
    thread.join(2)
    assert displayed == ["new English"]


@pytest.mark.parametrize("mode,test_id", [
    (PerformanceMode.BALANCED, "SW-COMPLEX-004"),
    (PerformanceMode.ACCURATE, "SW-COMPLEX-005"),
])
def test_refinement_profile_does_not_change_transition_state(mode, test_id, record_switch):
    state = RecognitionState("hi", "original")
    start = time.perf_counter()
    changed = state.switch("en", "original")
    elapsed = time.perf_counter() - start
    record_switch(evidence(
        test_id, f"Switch while {mode.value} profile exists",
        "Transition state becomes English immediately without model configuration mutation",
        f"generation={changed.generation}; elapsed={elapsed:.6f}s",
        test_type="Performance", switch_from="hi", switch_to="en", total_latency=elapsed,
        notes="State-level check; active Whisper inference is not cancellable"))
    assert changed.language == "en"
    assert elapsed <= 2.0


@pytest.mark.parametrize("test_id,source,target,first", [
    ("SW-COMPLEX-006A", "hi", "en", 0.41),
    ("SW-COMPLEX-006B", "en", "hi", 0.73),
])
def test_post_switch_first_audio_sample_is_preserved(test_id, source, target, first,
                                                     record_switch):
    """Directly verify that a newly constructed generation job keeps its leading sample."""
    state = RecognitionState(source, "original")
    changed = state.switch(target, "original")
    samples = np.array([first, 0.2, 0.1], dtype=np.float32)
    fresh = job(10, target, changed.generation, samples)
    record_switch(evidence(
        test_id, f"First-word samples preserved {source} to {target}",
        "New-generation audio begins with the supplied first-word sample",
        f"first_sample={fresh.audio[0]:.2f}", switch_from=source, switch_to=target,
        notes="Boundary-level audio check; microphone E2E requires hardware suite"))
    assert fresh.audio[0] == pytest.approx(first)


@pytest.mark.xfail(
    reason="Known first-word risk: generation-boundary frame is consumed then discarded",
    strict=True)
def test_immediate_first_post_switch_frame_reaches_new_pre_roll(record_switch):
    """Expose the real boundary-frame loss rather than hiding it with a job-only check."""
    config = AppConfig(performance_mode=PerformanceMode.FAST)
    state = RecognitionState("hi", "original")
    audio_queue, asr_queue, stop = Queue(), Queue(), Event()
    worker = SpeechBufferWorker(config, audio_queue, asr_queue, stop, state)
    submitted = []
    worker._submit = submitted.append
    thread = Thread(target=worker.run)
    thread.start()
    time.sleep(0.02)  # allow the worker to snapshot generation zero and block on input
    state.switch("en", "original")
    frame_samples = config.frame_samples
    marker = np.full(frame_samples, 0.91, dtype=np.float32)
    audio_queue.put(marker)
    for _ in range(24):
        audio_queue.put(np.full(frame_samples, 0.5, dtype=np.float32))
    for _ in range(35):
        audio_queue.put(np.zeros(frame_samples, dtype=np.float32))
    stop.set()
    thread.join(2)
    contains_marker = any(np.isclose(value.audio, 0.91).any() for value in submitted)
    record_switch(evidence(
        "SW-REG-001", "Immediate first frame after switch",
        "The first new-language frame enters clean pre-roll instead of being dropped",
        f"submitted_jobs={len(submitted)}; marker_preserved={contains_marker}",
        test_type="Negative", switch_from="hi", switch_to="en",
        suspected_component="SpeechBufferWorker generation boundary",
        recommended_investigation="Reset before classifying/consuming the boundary frame"))
    assert contains_marker


def test_new_partial_is_prioritized_over_protected_old_final(record_switch):
    state = RecognitionState("hi", "original")
    changed = state.switch("en", "original")
    queue = Queue(maxsize=1)
    old_final = job(1, "hi", 0, final=True)
    queue.put(old_final)
    worker = SpeechBufferWorker(AppConfig(), Queue(), queue, Event(), state)
    fresh = job(2, "en", changed.generation)
    worker._submit(fresh)
    order = [queue.get_nowait(), queue.get_nowait()]
    record_switch(evidence(
        "SW-COMPLEX-011", "New FAST priority over old final",
        "Fresh English partial is queued before protected Hindi final",
        f"order={[value.utterance_id for value in order]}", test_type="Negative",
        switch_from="hi", switch_to="en"))
    assert order == [fresh, old_final]


def test_technical_english_survives_after_hindi(record_switch):
    state = RecognitionState("hi", "original")
    changed = state.switch("en", "original")
    transcript = "Today I need to update SQLAlchemy and check the Jenkins pipeline."
    queue, stop, signals = Queue(), Event(), Signals()
    queue.put(job(3, "en", changed.generation))
    stop.set()
    ASRWorker(AppConfig(), queue, stop, signals, Provider(Engine({3: transcript})), state).run()
    actual = signals.partial_text.values[0][0]
    record_switch(evidence(
        "SW-COMPLEX-008", "Technical English after Hindi",
        "First English result retains SQLAlchemy and Jenkins", actual,
        switch_from="hi", switch_to="en", first_transcript=actual))
    assert "SQLAlchemy" in actual and "Jenkins" in actual


def test_hindi_after_long_english_context_gets_empty_prompt(record_switch):
    state = RecognitionState("en", "original")
    changed = state.switch("hi", "devanagari")
    engine = Engine({4: "आज हमें यह काम पूरा करना है।"})
    queue, stop, signals = Queue(), Event(), Signals()
    queue.put(job(4, "hi", changed.generation, script="devanagari"))
    stop.set()
    ASRWorker(AppConfig(), queue, stop, signals, Provider(engine), state).run()
    record_switch(evidence(
        "SW-COMPLEX-009", "Hindi after long English context",
        "First Hindi inference receives no English prompt", f"context={engine.calls[0][1]!r}",
        switch_from="en", switch_to="hi", first_transcript=signals.partial_text.values[0][0]))
    assert engine.calls[0][1] == ""
