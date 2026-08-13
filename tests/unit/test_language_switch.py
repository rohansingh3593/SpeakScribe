"""Regression coverage for clean, non-destructive language transitions."""

from queue import Queue
from threading import Event, Thread
import time

import numpy as np
import pytest

from app.asr.asr_engine import ASRWorker, WhisperEngine, WhisperModelProvider
from app.asr.language_transition import RecognitionState
from app.audio.audio_pipeline import ASRJob, SpeechBufferWorker
from app.config.settings import AppConfig


class Signal:
    def __init__(self):
        self.values = []

    def emit(self, *values):
        self.values.append(values)


class Signals:
    def __init__(self):
        self.status_changed = Signal()
        self.language_changed = Signal()
        self.final_text = Signal()
        self.partial_text = Signal()
        self.error = Signal()


@pytest.mark.parametrize("old,new", [
    ("hi", "en"), ("en", "hi"), ("auto", "en"), ("en", "auto"),
])
def test_switch_increments_generation_and_locks_language(old, new):
    state = RecognitionState(old, "original")
    changed = state.switch(new, "original")
    assert changed.language == new
    assert changed.generation == 1
    assert state.is_current(1)
    assert not state.is_current(0)


def test_same_selection_is_a_noop():
    state = RecognitionState("en", "original")
    assert state.switch("en", "original").generation == 0


@pytest.mark.parametrize("language,script", [
    ("en", "original"), ("hi", "devanagari"), ("auto", "latin"),
])
def test_script_policy_changes_atomically(language, script):
    snapshot = RecognitionState("hi", "original").switch(language, script)
    assert (snapshot.language, snapshot.script) == (language, script)


def test_switch_latency_is_recorded():
    snapshot = RecognitionState("hi", "original").switch("en", "original")
    assert snapshot.switched_at <= snapshot.ready_at
    assert snapshot.ready_at - snapshot.switched_at < 0.1


def test_job_carries_language_generation_and_boundary_time():
    job = ASRJob(np.ones(10), False, 2, 3.0, language="en",
                 script="original", language_generation=7,
                 language_switched_at=2.5)
    assert (job.language, job.language_generation, job.language_switched_at) == ("en", 7, 2.5)


def test_model_provider_reuses_weights_across_languages(monkeypatch):
    models = []

    def load(_engine):
        model = object()
        models.append(model)
        return model

    monkeypatch.setattr(WhisperEngine, "_load_model", load)
    provider = WhisperModelProvider()
    assert provider.get(AppConfig(language_mode="hi")).model is provider.get(
        AppConfig(language_mode="en")).model
    assert len(models) == 1


def test_late_old_partial_cannot_overwrite_new_result():
    """Hindi A blocks, the generation switches, then only English B is emitted."""
    entered = Event()
    release = Event()

    class Engine:
        def transcribe(self, job, context):
            if job.language == "hi":
                entered.set()
                release.wait(2)
                return "पुराना", "hi", {}
            return "new English", "en", {}

    class Provider:
        def get(self, config):
            return Engine()

    state = RecognitionState("hi", "original")
    queue = Queue()
    stop = Event()
    signals = Signals()
    worker = ASRWorker(AppConfig(language_mode="hi"), queue, stop, signals,
                       Provider(), state)
    queue.put(ASRJob(np.ones(10), False, 1, time.monotonic(), language="hi"))
    thread = Thread(target=worker.run)
    thread.start()
    assert entered.wait(1)
    switched = state.switch("en", "original")
    queue.put(ASRJob(np.ones(10), False, 2, time.monotonic(), language="en",
                     language_generation=switched.generation,
                     language_switched_at=switched.switched_at))
    release.set()
    time.sleep(0.05)
    stop.set()
    thread.join(2)
    assert signals.partial_text.values == [("new English",)]
    assert ("पुराना",) not in signals.partial_text.values


def test_first_new_language_job_has_no_old_context():
    contexts = []

    class Engine:
        def transcribe(self, job, context):
            contexts.append((job.language, context))
            return "text", job.language, {}

    class Provider:
        def get(self, config):
            return Engine()

    state = RecognitionState("hi", "original")
    changed = state.switch("en", "original")
    queue, stop, signals = Queue(), Event(), Signals()
    queue.put(ASRJob(np.ones(10), False, 1, time.monotonic(), language="en",
                     language_generation=changed.generation))
    stop.set()
    ASRWorker(AppConfig(), queue, stop, signals, Provider(), state).run()
    assert contexts == [("en", "")]


def test_switch_does_not_mutate_unrelated_state():
    history = ["आज मुझे ऑफिस जाना है।"]
    audio_device = object()
    state = RecognitionState("hi", "original")
    state.switch("en", "original")
    assert history == ["आज मुझे ऑफिस जाना है।"]
    assert audio_device is audio_device


def test_stale_final_is_preserved_but_not_added_to_new_context():
    state = RecognitionState("hi", "original")
    state.switch("en", "original")
    assert not state.is_current(0)
    # Finalized history is owned by the transcript/UI and is intentionally not
    # part of RecognitionState, so switching cannot clear or reinterpret it.


def test_first_new_partial_is_prioritized_ahead_of_protected_old_final():
    state = RecognitionState("hi", "original")
    changed = state.switch("en", "original")
    queue = Queue(maxsize=1)
    queue.put(ASRJob(np.ones(10), True, 1, 0.0, language="hi",
                     language_generation=0))
    worker = SpeechBufferWorker(AppConfig(max_asr_queue=1), Queue(), queue,
                                Event(), state)
    fresh = ASRJob(np.ones(10), False, 2, 0.0, language="en",
                   language_generation=changed.generation)
    worker._submit(fresh)
    assert queue.get_nowait() is fresh
    assert queue.get_nowait().final
