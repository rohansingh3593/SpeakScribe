"""Non-blocking, utterance-conserving diagnostics for the live speech pipeline."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from queue import Empty, Full, Queue
from threading import Lock, Thread
import json
import time
import wave

import numpy as np


class RootCause(str, Enum):
    AUDIO_CAPTURE = "AUDIO_CAPTURE"
    VAD = "VAD"
    SEGMENTATION = "SEGMENTATION"
    QUEUE = "QUEUE"
    FAST_ASR = "FAST_ASR"
    LANGUAGE_DETECTION = "LANGUAGE_DETECTION"
    REFINEMENT = "REFINEMENT"
    DEDUPLICATION = "DEDUPLICATION"
    FINALIZATION = "FINALIZATION"
    GENERATION_FILTER = "GENERATION_FILTER"
    UI = "UI"
    UNKNOWN = "UNKNOWN"


@dataclass
class UtteranceTrace:
    utterance_id: int
    generation: int
    language: str
    stages: set[str] = field(default_factory=set)
    terminal: str | None = None
    reason: str | None = None
    timings: dict[str, float] = field(default_factory=dict)


class PipelineDiagnostics:
    """Enqueue tiny trace records; one low-priority writer performs all I/O."""

    STAGE_FILES = {
        "AUDIO": "audio", "VAD": "vad", "SEGMENT": "segmentation",
        "QUEUE": "queue", "FAST": "asr", "ASR": "asr",
        "REFINEMENT": "refinement", "DEDUP": "finalization",
        "FINALIZE": "finalization", "UI": "ui", "DISCARD": "finalization",
        "PERFORMANCE": "performance",
    }

    def __init__(self, directory: Path | None = None, enabled: bool = False,
                 sample_rate: int = 16_000):
        self.enabled = enabled
        self.directory = directory
        self.sample_rate = sample_rate
        self.started_at = time.monotonic()
        self.events: Queue = Queue(maxsize=4096)
        self.traces: dict[int, UtteranceTrace] = {}
        self.counts = defaultdict(int)
        self.fast_seconds: list[float] = []
        self.first_text_seconds: list[float] = []
        self.audio_seconds = 0.0
        self.speech_seconds = 0.0
        self._lock = Lock()
        self._writer: Thread | None = None
        if enabled and directory is not None:
            for path in (directory / "pipeline", directory / "languages",
                         directory / "debug_audio"):
                path.mkdir(parents=True, exist_ok=True)
            self._writer = Thread(target=self._write_loop, name="pipeline-log-writer",
                                  daemon=True)
            self._writer.start()

    @staticmethod
    def label(utterance_id: int | None) -> str:
        return "UTT-PENDING" if utterance_id is None else f"UTT-{utterance_id:06d}"

    def event(self, stage: str, utterance_id: int | None = None, **fields) -> None:
        if not self.enabled:
            return
        record = {"at": time.time(), "monotonic": time.monotonic(),
                  "stage": stage.upper(), "utterance": self.label(utterance_id), **fields}
        try:
            self.events.put_nowait(("event", record))
        except Full:
            with self._lock:
                self.counts["diagnostic_queue_dropped"] += 1

    def detected(self, utterance_id: int, generation: int, language: str, **fields) -> None:
        with self._lock:
            self.traces[utterance_id] = UtteranceTrace(
                utterance_id, generation, language, {"VAD"}, timings={"vad": time.monotonic()})
            self.counts["utterances_detected"] += 1
        self.event("VAD START", utterance_id, generation=generation, language=language, **fields)

    def stage(self, utterance_id: int, stage: str, **fields) -> None:
        with self._lock:
            trace = self.traces.get(utterance_id)
            if trace:
                trace.stages.add(stage.upper())
                trace.timings.setdefault(stage.casefold(), time.monotonic())
            self.counts[f"{stage.casefold()}_events"] += 1
        self.event(stage, utterance_id, **fields)

    def terminal(self, utterance_id: int, outcome: str, reason: str,
                 root_cause: RootCause | str, **fields) -> None:
        cause = root_cause.value if isinstance(root_cause, RootCause) else str(root_cause)
        with self._lock:
            trace = self.traces.get(utterance_id)
            if trace:
                trace.terminal, trace.reason = outcome, reason
            self.counts["finalized" if outcome == "FINAL" else "rejected"] += 1
            if outcome != "FINAL":
                self.counts[f"rejected_{cause}"] += 1
        self.event("FINALIZE" if outcome == "FINAL" else "DISCARD", utterance_id,
                   outcome=outcome, reason=reason, root_cause=cause, **fields)

    def save_audio(self, utterance_id: int, audio: np.ndarray) -> None:
        if not self.enabled:
            return
        try:
            self.events.put_nowait(("audio", utterance_id,
                                    np.asarray(audio, dtype=np.float32).copy()))
        except Full:
            self.event("DISCARD", utterance_id, stage_name="DEBUG_AUDIO",
                       reason="diagnostic_queue_full")

    def heartbeat(self, **fields) -> None:
        self.event("PIPELINE", None, **fields)

    def unresolved(self) -> list[UtteranceTrace]:
        with self._lock:
            return [trace for trace in self.traces.values() if trace.terminal is None]

    def classify_root_cause(self, utterance_id: int) -> RootCause:
        """Classify the first missing lifecycle checkpoint for one utterance."""
        trace = self.traces.get(utterance_id)
        if trace is None:
            return RootCause.AUDIO_CAPTURE
        reason = (trace.reason or "").casefold()
        explicit = {
            "stale": RootCause.GENERATION_FILTER, "generation": RootCause.GENERATION_FILTER,
            "duplicate": RootCause.DEDUPLICATION, "script": RootCause.LANGUAGE_DETECTION,
            "queue": RootCause.QUEUE, "minimum_speech": RootCause.VAD,
            "empty_asr": RootCause.FAST_ASR, "final_ui": RootCause.UI,
        }
        for marker, cause in explicit.items():
            if marker in reason:
                return cause
        for stage, cause in (("VAD", RootCause.VAD), ("SEGMENT", RootCause.SEGMENTATION),
                             ("QUEUE", RootCause.QUEUE), ("FAST", RootCause.FAST_ASR),
                             ("FINALIZE", RootCause.FINALIZATION), ("UI", RootCause.UI)):
            if not any(value.startswith(stage) for value in trace.stages):
                return cause
        return RootCause.UNKNOWN

    def close(self, reason: str = "session_stop") -> None:
        if not self.enabled:
            return
        # Conservation invariant: no detected utterance is allowed an implicit fate.
        for trace in self.unresolved():
            self.terminal(trace.utterance_id, "REJECTED", reason, RootCause.FINALIZATION,
                          stages=sorted(trace.stages))
        self.events.put(("close",))
        if self._writer:
            self._writer.join(timeout=2.0)
        self._write_summary()

    def _write_loop(self) -> None:
        handles = {}
        try:
            while True:
                item = self.events.get()
                if item[0] == "close":
                    return
                if item[0] == "audio":
                    _, utterance_id, audio = item
                    pcm = (np.clip(audio, -1, 1) * 32767).astype("<i2")
                    with wave.open(str(self.directory / "debug_audio" /
                                       f"{self.label(utterance_id)}_input.wav"), "wb") as stream:
                        stream.setnchannels(1); stream.setsampwidth(2)
                        stream.setframerate(self.sample_rate); stream.writeframes(pcm.tobytes())
                    continue
                record = item[1]
                stage = record["stage"].split()[0]
                name = self.STAGE_FILES.get(stage, "debug")
                path = (self.directory / "pipeline" / f"{name}.log"
                        if name not in {"debug", "performance"} else
                        self.directory / f"{name}.log")
                handle = handles.setdefault(path, path.open("a", encoding="utf-8"))
                handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
                language = record.get("language") or record.get("requested_language")
                if language in {"hi", "Hindi", "Hinglish", "en", "English"}:
                    lang_name = "hindi" if language in {"hi", "Hindi"} else (
                        "hinglish" if language == "Hinglish" else "english")
                    lang_path = self.directory / "languages" / f"{lang_name}.log"
                    lang = handles.setdefault(lang_path, lang_path.open("a", encoding="utf-8"))
                    lang.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        finally:
            for handle in handles.values():
                handle.close()

    def _write_summary(self) -> None:
        elapsed = time.monotonic() - self.started_at
        fast = sorted(self.fast_seconds)
        first = self.first_text_seconds
        p95 = fast[max(0, round(len(fast) * .95) - 1)] if fast else 0.0
        summary = (
            "==============================\nHINDI/HINGLISH SESSION SUMMARY\n"
            "==============================\n"
            f"Listening duration: {elapsed:.2f}s\nAudio captured: {self.audio_seconds:.2f}s\n"
            f"Speech detected: {self.speech_seconds:.2f}s\n"
            f"Utterances detected: {self.counts['utterances_detected']}\n"
            f"Utterances sent to FAST: {self.counts['queue_events']}\n"
            f"FAST results: {self.counts['fast_events']}\n"
            f"Finalized utterances: {self.counts['finalized']}\n"
            f"Dropped/rejected: {self.counts['rejected']}\n"
            f"VAD rejected: {self.counts['rejected_VAD']}\n"
            f"Generation stale: {self.counts['rejected_GENERATION_FILTER']}\n"
            f"Queue dropped: {self.counts['rejected_QUEUE']}\n"
            f"Dedup removed: {self.counts['rejected_DEDUPLICATION']}\n"
            f"UI missing: {self.counts['rejected_UI']}\n"
            f"Average FAST: {(sum(fast)/len(fast) if fast else 0):.3f}s\n"
            f"P95 FAST: {p95:.3f}s\nMaximum FAST: {(max(fast) if fast else 0):.3f}s\n"
            f"Average first text: {(sum(first)/len(first) if first else 0):.3f}s\n"
            f"Maximum first text: {(max(first) if first else 0):.3f}s\n"
            f"Diagnostic events dropped: {self.counts['diagnostic_queue_dropped']}\n"
            "==============================\n")
        (self.directory / "summary.log").write_text(summary, encoding="utf-8")


_ACTIVE = PipelineDiagnostics()


def configure_pipeline_diagnostics(directory: Path, enabled: bool) -> PipelineDiagnostics:
    global _ACTIVE
    _ACTIVE = PipelineDiagnostics(directory, enabled)
    return _ACTIVE


def pipeline_diagnostics() -> PipelineDiagnostics:
    return _ACTIVE
