"""Prompt, model, queue, and application-lifetime invariants."""

from pathlib import Path
from queue import Queue
from threading import Event

import pytest

from app.asr.asr_engine import WhisperEngine, WhisperModelProvider
from app.asr.language_transition import RecognitionState
from app.config.settings import AppConfig
from app.audio.audio_pipeline import SpeechBufferWorker
from tests.switching.support import Engine, evidence, job, run_worker


def test_context_is_reset_on_first_new_generation(record_switch):
    state = RecognitionState("hi", "original")
    changed = state.switch("en", "original")
    engine = Engine({1: "The deployment finished successfully."})
    signals, _ = run_worker(state, [job(1, "en", changed.generation)], engine)
    context = engine.calls[0][1]
    record_switch(evidence(
        "SW-STATE-001", "Context reset", "First English job receives no Hindi prompt",
        f"context={context!r}; result={signals.partial_text.values}",
        switch_from="hi", switch_to="en", first_transcript=signals.partial_text.values[0][0]))
    assert context == ""


def test_model_weights_are_reused(record_switch, monkeypatch):
    created = []

    def load(_engine):
        created.append(object())
        return created[-1]

    monkeypatch.setattr(WhisperEngine, "_load_model", load)
    provider = WhisperModelProvider()
    before = provider.get(AppConfig(language_mode="hi")).model
    after = provider.get(AppConfig(language_mode="en")).model
    record_switch(evidence(
        "SW-STATE-002", "Model not reloaded", "Same multilingual weights are reused",
        f"same_instance={before is after}; creations={len(created)}",
        switch_from="hi", switch_to="en"))
    assert before is after
    assert len(created) == 1


def test_switch_keeps_controller_workers_and_audio_capture(record_switch):
    source = Path("app/main.py").read_text(encoding="utf-8")
    switch_body = source.split("def switch_recognition_language", 1)[1].split(
        "def start_stream", 1)[0]
    record_switch(evidence(
        "SW-STATE-003", "Audio capture remains active",
        "Switch implementation does not create capture workers or stop the session",
        "No AudioCaptureWorker(), stop(), or thread replacement in switch method",
        switch_from="hi", switch_to="en"))
    assert "AudioCaptureWorker(" not in switch_body
    assert "self.stop(" not in switch_body
    assert "self.threads =" not in switch_body


def test_queued_replaceable_partials_are_cancelled(record_switch):
    state = RecognitionState("hi", "original")
    changed = state.switch("en", "original")
    queue = Queue(maxsize=1)
    queue.put(job(1, "hi", 0))
    worker = SpeechBufferWorker(AppConfig(), Queue(), queue, Event(), state)
    fresh = job(2, "en", changed.generation)
    worker._submit(fresh)
    record_switch(evidence(
        "SW-COMPLEX-002", "Switch with queued partials",
        "Replaceable old partial is superseded", f"queued={queue.queue[0].utterance_id}",
        test_type="Negative", switch_from="hi", switch_to="en"))
    assert queue.get_nowait() is fresh


@pytest.mark.parametrize("target,script", [("en", "original"), ("hi", "devanagari"),
                                            ("auto", "latin")])
def test_language_and_script_change_together(target, script, record_switch):
    state = RecognitionState("hi", "original")
    snapshot = state.switch(target, script)
    record_switch(evidence(
        f"SW-STATE-SCRIPT-{target.upper()}", "Script policy transition",
        "Language and requested script are updated atomically",
        f"language={snapshot.language}; script={snapshot.script}",
        switch_from="hi", switch_to=target))
    assert (snapshot.language, snapshot.script) == (target, script)
