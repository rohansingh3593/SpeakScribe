"""In-flight results, queue saturation, and repeated session transitions."""

from queue import Queue
from threading import Event, Thread
import time

import pytest

from app.asr.asr_engine import ASRWorker
from app.asr.language_transition import RecognitionState
from app.asr.session_transition import LiveSessionBoundary
from app.config.settings import AppConfig, PerformanceMode
from tests.switching.support import Engine, Provider, Signals, job


@pytest.mark.parametrize("mode,test_id", [
    (PerformanceMode.FAST, "STOP-COMPLEX-001"),
    (PerformanceMode.BALANCED, "STOP-COMPLEX-002"),
    (PerformanceMode.ACCURATE, "STOP-COMPLEX-003"),
])
def test_stop_returns_while_inference_is_running(mode, test_id):
    state = RecognitionState("hi", "original")
    session = state.begin_session("hi", "original")
    queue, worker_stop, signals = Queue(), Event(), Signals()
    engine = Engine({1: "पुराना"}, {1: 0.2})
    queue.put(job(1, "hi", session.generation))
    worker = ASRWorker(AppConfig(performance_mode=mode), queue, worker_stop,
                       signals, Provider(engine), state)
    thread = Thread(target=worker.run)
    thread.start()
    time.sleep(0.02)
    started = time.perf_counter()
    LiveSessionBoundary(state, worker_stop, (queue,)).stop()
    elapsed = time.perf_counter() - started
    assert elapsed < 0.5, test_id
    assert signals.partial_text.values == []
    thread.join(2)
    assert signals.partial_text.values == []


def test_stop_complex_004_full_queue_is_cleared():
    queue = Queue(maxsize=1)
    queue.put(job(1, "hi", 0))
    metrics = LiveSessionBoundary(
        RecognitionState("hi", "original"), Event(), (queue,)).stop()
    assert metrics.jobs_cleared == 1
    assert queue.empty()


def test_stop_complex_005_immediate_language_switch():
    state = RecognitionState("hi", "original")
    stopped = LiveSessionBoundary(state, Event()).stop()
    started = time.perf_counter()
    english = state.begin_session("en", "original")
    assert time.perf_counter() - started < 0.05
    assert english.generation == stopped.generation + 1


def test_stop_complex_006_old_result_after_new_session_is_stale():
    state = RecognitionState("hi", "original")
    hindi = state.begin_session("hi", "original")
    queue, worker_stop, signals = Queue(), Event(), Signals()
    engine = Engine({1: "पुराना", 2: "new English"}, {1: 0.08})
    queue.put(job(1, "hi", hindi.generation))
    worker = ASRWorker(AppConfig(), queue, worker_stop, signals, Provider(engine), state)
    thread = Thread(target=worker.run)
    thread.start()
    time.sleep(0.01)
    LiveSessionBoundary(state, worker_stop).stop()
    english = state.begin_session("en", "original")
    # A reusable/new worker represents the next live lane while old C inference retires.
    next_signals = Signals()
    next_queue, next_stop = Queue(), Event()
    next_queue.put(job(2, "en", english.generation))
    next_stop.set()
    ASRWorker(AppConfig(), next_queue, next_stop, next_signals,
              Provider(engine), state).run()
    thread.join(2)
    assert next_signals.partial_text.values == [("new English",)]
    assert signals.partial_text.values == []


def test_stop_complex_007_rapid_stop_start_generations():
    state = RecognitionState("hi", "original")
    observed = []
    for index in range(20):
        session = state.begin_session("en" if index % 2 else "hi", "original")
        observed.append(session.generation)
        stopped = LiveSessionBoundary(state, Event()).stop()
        observed.append(stopped.generation)
    assert observed == sorted(observed)
    assert len(set(observed)) == 40
