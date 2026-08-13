"""Reusable test doubles which exercise the production switching path."""

from dataclasses import dataclass, field
from queue import Queue
from threading import Event
import time

import numpy as np

from app.asr.asr_engine import ASRWorker
from app.asr.language_transition import RecognitionState
from app.audio.audio_pipeline import ASRJob
from app.config.settings import AppConfig


LANGUAGE_NAMES = {"hi": "Hindi", "en": "English", "auto": "Hinglish"}


class Signal:
    def __init__(self):
        self.values = []

    def emit(self, *values):
        self.values.append(values)


class Signals:
    def __init__(self):
        for name in (
            "status_changed", "language_changed", "final_text", "stale_final_text",
            "partial_text", "error", "mode_status", "mode_text", "mode_error",
        ):
            setattr(self, name, Signal())


class Engine:
    """Deterministic engine double; scheduling/state remain production code."""

    def __init__(self, responses=None, delays=None, failures=None):
        self.responses = responses or {}
        self.delays = delays or {}
        self.failures = failures or {}
        self.calls = []
        self.last_stage_timings = {}

    def transcribe(self, job, context):
        self.calls.append((job, context, time.perf_counter()))
        delay = self.delays.get(job.utterance_id, 0)
        if delay:
            time.sleep(delay)
        if job.utterance_id in self.failures:
            raise self.failures[job.utterance_id]
        text = self.responses.get(job.utterance_id, "")
        return text, job.language, {"script_valid": True, "requested_script": job.script}


class Provider:
    def __init__(self, engine):
        self.engine = engine
        self.calls = 0

    def get(self, _config):
        self.calls += 1
        return self.engine


def job(utterance_id, language, generation, text_audio=None, *, final=False,
        script="original", switched_at=0.0, ready_at=0.0):
    audio = (np.asarray(text_audio, dtype=np.float32) if text_audio is not None
             else np.ones(160, dtype=np.float32))
    return ASRJob(audio, final, utterance_id, time.monotonic(),
                  language=language, script=script,
                  language_generation=generation,
                  language_switched_at=switched_at,
                  language_ready_at=ready_at)


def run_worker(state, jobs, engine, config=None):
    queue, stop, signals = Queue(), Event(), Signals()
    for value in jobs:
        queue.put(value)
    stop.set()
    provider = Provider(engine)
    ASRWorker(config or AppConfig(), queue, stop, signals, provider, state).run()
    return signals, provider


@dataclass
class SwitchEvidence:
    test_id: str
    name: str
    expected_scenario: str
    reason: str
    test_type: str
    expected_outcome: str
    actual_outcome: str
    switch_from: str = ""
    switch_to: str = ""
    switch_time: float | None = None
    first_transcript_time: float | None = None
    total_latency: float | None = None
    first_transcript: str = ""
    final_transcript: str = ""
    notes: str = ""
    suspected_component: str = ""
    recommended_investigation: str = ""
    latency_runs: list[dict] = field(default_factory=list)


def evidence(test_id, name, expected, actual, *, test_type="Positive",
             switch_from="", switch_to="", reason="Requirement satisfied", **kwargs):
    return SwitchEvidence(
        test_id, name, expected, reason, test_type, expected, actual,
        LANGUAGE_NAMES.get(switch_from, switch_from),
        LANGUAGE_NAMES.get(switch_to, switch_to), **kwargs)
