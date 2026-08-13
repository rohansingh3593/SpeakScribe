"""Bad-result, saturation, rapid-switch, and shutdown behavior."""

from queue import Queue
from threading import Event
import time

import pytest

from app.asr.language_transition import RecognitionState
from app.audio.audio_pipeline import SpeechBufferWorker
from app.config.settings import AppConfig
from tests.switching.support import Engine, evidence, job, run_worker


def test_twenty_rapid_switches_have_monotonic_generations(record_switch):
    state = RecognitionState("hi", "original")
    sequence = ["en", "hi", "en", "auto"] * 5
    generations = [state.switch(language, "original").generation for language in sequence]
    record_switch(evidence(
        "SW-STRESS-001", "20 rapid mode switches",
        "No deadlock and each actual language change advances generation",
        f"final={state.snapshot().language}; generations={generations}",
        test_type="Stress", switch_from="hi", switch_to="auto"))
    assert generations == list(range(1, 21))
    assert state.snapshot().language == "auto"


def test_repeated_same_mode_does_not_create_generations(record_switch):
    state = RecognitionState("en", "original")
    generations = [state.switch("en", "original").generation for _ in range(20)]
    record_switch(evidence(
        "SW-NEG-001", "Repeated same-mode selection",
        "No unnecessary epochs or work are created", repr(generations), test_type="Negative",
        switch_from="en", switch_to="en"))
    assert generations == [0] * 20


def test_empty_first_result_does_not_emit_or_clear_text(record_switch):
    state = RecognitionState("hi", "original")
    changed = state.switch("en", "original")
    signals, _ = run_worker(state, [job(1, "en", changed.generation)], Engine({1: ""}))
    record_switch(evidence(
        "SW-NEG-002", "Empty ASR result after switch",
        "Empty candidate does not overwrite live display", repr(signals.partial_text.values),
        test_type="Negative", switch_from="hi", switch_to="en"))
    assert signals.partial_text.values == []


def test_worker_exception_is_reported_without_false_transcript(record_switch):
    state = RecognitionState("hi", "original")
    changed = state.switch("en", "original")
    signals, _ = run_worker(
        state, [job(1, "en", changed.generation)],
        Engine(failures={1: RuntimeError("temporary unavailable")}))
    record_switch(evidence(
        "SW-NEG-003", "Worker temporarily unavailable",
        "Exception reaches error signal and no transcript is displayed",
        f"errors={signals.error.values}; partials={signals.partial_text.values}",
        test_type="Negative", switch_from="hi", switch_to="en"))
    assert signals.error.values
    assert signals.partial_text.values == []


def test_old_job_finishing_after_switch_is_discarded(record_switch):
    state = RecognitionState("hi", "original")
    changed = state.switch("en", "original")
    signals, _ = run_worker(
        state, [job(1, "hi", 0), job(2, "en", changed.generation)],
        Engine({1: "राहुल गांधी", 2: "Today I need to update SQLAlchemy."}))
    displayed = [value[0] for value in signals.partial_text.values]
    record_switch(evidence(
        "SW-NEG-004", "Old job completes after switch",
        "Stale generation is rejected", repr(displayed), test_type="Negative",
        switch_from="hi", switch_to="en", first_transcript=displayed[0]))
    assert displayed == ["Today I need to update SQLAlchemy."]


def test_queue_pressure_retains_fresh_generation(record_switch):
    state = RecognitionState("hi", "original")
    changed = state.switch("en", "original")
    queue = Queue(maxsize=1)
    queue.put(job(1, "hi", 0))
    worker = SpeechBufferWorker(AppConfig(), Queue(), queue, Event(), state)
    fresh = job(2, "en", changed.generation)
    worker._submit(fresh)
    retained = queue.get_nowait()
    record_switch(evidence(
        "SW-NEG-005", "Queue pressure", "Fresh partial replaces obsolete partial",
        f"retained={retained.utterance_id}", test_type="Negative",
        switch_from="hi", switch_to="en"))
    assert retained is fresh


@pytest.mark.xfail(reason="RecognitionState currently accepts unknown language identifiers",
                   strict=True)
def test_invalid_language_is_rejected(record_switch):
    state = RecognitionState("hi", "original")
    record_switch(evidence(
        "SW-NEG-006", "Invalid language configuration",
        "Unknown explicit language is rejected", "No validation exception was raised",
        test_type="Negative", switch_from="hi", switch_to="invalid",
        suspected_component="RecognitionState.switch validation",
        recommended_investigation="Validate against hi/en/auto before updating generation"))
    with pytest.raises(ValueError):
        state.switch("invalid", "original")


def test_wrong_script_candidate_is_not_displayed(record_switch):
    class WrongScriptEngine(Engine):
        def transcribe(self, job, context):
            return "غلط", "hi", {"script_valid": False}

    state = RecognitionState("en", "original")
    changed = state.switch("hi", "devanagari")
    signals, _ = run_worker(state, [job(1, "hi", changed.generation,
                                      script="devanagari")], WrongScriptEngine())
    record_switch(evidence(
        "SW-NEG-007", "Script mismatch after switch",
        "Wrong-script candidate is rejected", repr(signals.partial_text.values),
        test_type="Negative", switch_from="en", switch_to="hi"))
    assert signals.partial_text.values == []


def test_first_inference_exception_does_not_reuse_old_output(record_switch):
    state = RecognitionState("hi", "original")
    changed = state.switch("en", "original")
    signals, _ = run_worker(
        state, [job(1, "en", changed.generation)],
        Engine(failures={1: ValueError("decode failed")}))
    record_switch(evidence(
        "SW-NEG-008", "First inference exception",
        "Failure is surfaced and no stale text is emitted",
        f"error={signals.error.values}; partial={signals.partial_text.values}",
        test_type="Negative", switch_from="hi", switch_to="en"))
    assert signals.error.values and not signals.partial_text.values


def test_switch_during_shutdown_is_a_safe_noop(record_switch):
    from pathlib import Path
    source = Path("app/main.py").read_text(encoding="utf-8")
    switch_body = source.split("def switch_recognition_language", 1)[1].split(
        "def start_stream", 1)[0]
    record_switch(evidence(
        "SW-STRESS-002", "Switch during shutdown",
        "Stopped controller returns before mutating recognition state",
        "running guard occurs before recognition_state.switch",
        test_type="Stress", switch_from="hi", switch_to="en"))
    assert switch_body.index("if not self.running") < switch_body.index(
        "self.recognition_state.switch")


def test_rapid_mixed_session_rejects_every_obsolete_generation(record_switch):
    state = RecognitionState("hi", "original")
    jobs = []
    responses = {}
    for index, language in enumerate(["en", "hi", "en", "auto", "en"], 1):
        changed = state.switch(language, "original")
        jobs.append(job(index, language, changed.generation))
        responses[index] = f"result-{language}-{index}"
    signals, _ = run_worker(state, jobs, Engine(responses))
    displayed = [value[0] for value in signals.partial_text.values]
    record_switch(evidence(
        "SW-COMPLEX-010", "Multiple consecutive switches",
        "Only final selected generation can update live output", repr(displayed),
        test_type="Stress", switch_from="hi", switch_to="en"))
    assert displayed == ["result-en-5"]
