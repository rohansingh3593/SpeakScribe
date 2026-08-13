"""Fast checks for immediate stop semantics and session reuse."""

from pathlib import Path
from queue import Queue
from threading import Event
import statistics

from app.asr.asr_engine import WhisperEngine, WhisperModelProvider
from app.asr.language_transition import RecognitionState
from app.asr.session_transition import LiveSessionBoundary
from app.config.settings import AppConfig
from tests.switching.support import job


def test_stop_001_disables_capture_immediately():
    stop = Event()
    boundary = LiveSessionBoundary(RecognitionState("hi", "original"), stop)
    metrics = boundary.stop()
    assert stop.is_set()
    assert metrics.capture_disabled_at >= metrics.stop_clicked_at


def test_stop_002_invalidates_old_generation():
    state = RecognitionState("hi", "original")
    old = state.snapshot().generation
    metrics = LiveSessionBoundary(state, Event()).stop()
    assert metrics.generation == old + 1
    assert not state.is_current(old)


def test_stop_003_clears_pending_jobs():
    queue = Queue()
    queue.put(job(1, "hi", 0))
    queue.put(job(2, "hi", 0))
    metrics = LiveSessionBoundary(
        RecognitionState("hi", "original"), Event(), (queue,)).stop()
    assert queue.empty()
    assert metrics.jobs_cleared == 2


def test_stop_005_completed_history_is_outside_live_boundary():
    history = ["आज मुझे ऑफिस जाना है।"]
    LiveSessionBoundary(RecognitionState("hi", "original"), Event()).stop()
    assert history == ["आज मुझे ऑफिस जाना है।"]


def test_stop_006_does_not_reload_model(monkeypatch):
    created = []
    monkeypatch.setattr(
        WhisperEngine, "_load_model",
        lambda _engine: created.append(object()) or created[-1])
    provider = WhisperModelProvider()
    before = provider.get(AppConfig(language_mode="hi")).model
    state = RecognitionState("hi", "original")
    LiveSessionBoundary(state, Event()).stop()
    state.begin_session("en", "original")
    after = provider.get(AppConfig(language_mode="en")).model
    assert before is after
    assert len(created) == 1


def test_stop_007_generation_state_is_reusable():
    state = RecognitionState("hi", "original")
    first = state.begin_session("hi", "original")
    LiveSessionBoundary(state, Event()).stop()
    second = state.begin_session("en", "original")
    assert second.generation > first.generation
    assert second.language == "en"


def test_stop_008_stop_to_english():
    state = RecognitionState("hi", "original")
    stopped = LiveSessionBoundary(state, Event()).stop()
    english = state.begin_session("en", "original")
    assert english.generation > stopped.generation
    assert english.language == "en"


def test_stop_009_stop_to_hindi():
    state = RecognitionState("en", "original")
    stopped = LiveSessionBoundary(state, Event()).stop()
    hindi = state.begin_session("hi", "original")
    assert hindi.generation > stopped.generation
    assert hindi.language == "hi"


def test_stop_010_p95_ready_latency_under_500ms():
    values = []
    for _ in range(20):
        metrics = LiveSessionBoundary(
            RecognitionState("hi", "original"), Event()).stop()
        values.append(metrics.stop_to_ready)
    p95 = sorted(values)[18]
    assert p95 < 0.5, (
        f"min={min(values):.6f} average={statistics.fmean(values):.6f} "
        f"median={statistics.median(values):.6f} p95={p95:.6f} max={max(values):.6f}")


def test_ui_stop_path_has_no_join_or_poll_wait():
    source = Path("app/main.py").read_text(encoding="utf-8")
    ui_stop = source.split("def stop_listening", 1)[1].split(
        "def _finish_stop_ui", 1)[0]
    controller_stop = source.split("def stop(self,", 1)[1].split(
        "class MainWindow", 1)[0]
    assert ".join(" not in ui_stop
    assert "_wait_for_workers_to_stop" not in ui_stop
    # join is permitted only inside the nested background reaper.
    assert controller_stop.index("def reap") < controller_stop.index("worker.join()")
