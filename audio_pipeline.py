"""Continuous SoundCard capture and energy-based utterance segmentation."""

from collections import deque
from dataclasses import dataclass
from importlib import import_module
from queue import Empty, Full, Queue
from threading import Event
import time
import warnings

import numpy as np

from config import AppConfig
from logger import log_print


@dataclass(frozen=True)
class ASRJob:
    audio: np.ndarray
    final: bool
    utterance_id: int
    captured_at: float


def prepare_audio_for_asr(audio: np.ndarray) -> np.ndarray:
    """Remove DC and safely lift quiet, VAD-approved microphone speech.

    Whisper is tolerant of normal recording levels, so loud input is untouched.
    Quiet microphone input is capped at 10x gain to avoid turning tiny numerical
    noise into full-scale audio.
    """
    prepared = np.asarray(audio, dtype=np.float32)
    if prepared.size == 0:
        return prepared
    prepared = prepared - np.mean(prepared, dtype=np.float64)
    peak = float(np.max(np.abs(prepared)))
    if 1e-5 < peak < 0.25:
        prepared = prepared * min(10.0, 0.8 / peak)
    return np.clip(prepared, -1.0, 1.0).astype(np.float32, copy=False)


class EnergySpeechDetector:
    """Hysteretic RMS detector, isolated so another VAD can replace it later."""

    def __init__(self, config: AppConfig):
        self.config = config
        self.speaking = False

    def classify(self, frame: np.ndarray) -> tuple[bool, float]:
        rms = float(np.sqrt(np.mean(np.square(frame), dtype=np.float64)))
        threshold = (self.config.silence_threshold if self.speaking
                     else self.config.speech_threshold)
        active = rms >= threshold
        if active:
            self.speaking = True
        return active, rms

    def reset(self) -> None:
        self.speaking = False


class AudioCaptureWorker:
    def __init__(self, config: AppConfig, output: Queue, stop_event: Event,
                 on_error):
        self.config, self.output, self.stop_event = config, output, stop_event
        self.on_error = on_error

    def run(self) -> None:
        try:
            # SoundCard is a capture-only dependency. Importing it here keeps the
            # NumPy speech detector and buffer classes usable by tests, tooling,
            # and offline processing without initializing a platform audio
            # backend during module collection.
            sc = import_module("soundcard")
            mediafoundation = import_module("soundcard.mediafoundation")
            soundcard_warning = getattr(
                mediafoundation, "SoundcardRuntimeWarning", RuntimeWarning)
            microphone = sc.default_microphone()
            if microphone is None:
                raise RuntimeError("No default microphone is available")
            log_print(f"Audio device: {microphone.name}")
            with warnings.catch_warnings():
                warnings.filterwarnings("once", category=soundcard_warning)
                with microphone.recorder(samplerate=self.config.sample_rate,
                                         channels=self.config.channels) as recorder:
                    while not self.stop_event.is_set():
                        block = recorder.record(numframes=self.config.frame_samples)
                        frame = np.asarray(block[:, 0], dtype=np.float32)
                        try:
                            self.output.put(frame, timeout=0.05)
                        except Full:
                            try:
                                self.output.get_nowait()  # discard only oldest raw frame
                            except Empty:
                                pass
                            self.output.put_nowait(frame)
                            log_print("Audio queue full; dropped oldest raw frame")
        except Exception as exc:
            log_print(f"Capture error: {exc}")
            self.on_error(str(exc))


class SpeechBufferWorker:
    def __init__(self, config: AppConfig, audio_queue: Queue, asr_queue: Queue,
                 stop_event: Event):
        self.config, self.audio_queue, self.asr_queue = config, audio_queue, asr_queue
        self.stop_event = stop_event
        self.detector = EnergySpeechDetector(config)
        self.utterance_id = 0

    def _submit(self, job: ASRJob) -> None:
        if job.final:
            # A final supersedes queued partials for its utterance. Evict those
            # obsolete hypotheses so CPU fallback does not spend several seconds
            # decoding text that the final will immediately replace.
            try:
                pending = self.asr_queue.get_nowait()
            except Empty:
                pending = None
            if pending is not None and pending.final:
                self.asr_queue.put_nowait(pending)

            # Existing finals are never evicted; wait briefly for inference.
            deadline = time.monotonic() + 0.5 if self.stop_event.is_set() else None
            while deadline is None or time.monotonic() < deadline:
                try:
                    self.asr_queue.put(job, timeout=0.1)
                    return
                except Full:
                    continue
        else:
            try:
                self.asr_queue.put_nowait(job)
            except Full:
                # Replace an obsolete partial, but never evict a final.
                try:
                    pending = self.asr_queue.get_nowait()
                except Empty:
                    return
                if pending.final:
                    self.asr_queue.put_nowait(pending)
                    return
                try:
                    self.asr_queue.put_nowait(job)
                except Full:
                    pass

    def run(self) -> None:
        frame_seconds = self.config.frame_ms / 1000
        pre = deque(maxlen=max(1, round(self.config.pre_speech_duration / frame_seconds)))
        speech: list[np.ndarray] = []
        silence = 0.0
        last_partial = 0.0
        while not self.stop_event.is_set() or not self.audio_queue.empty():
            try:
                frame = self.audio_queue.get(timeout=0.1)
            except Empty:
                continue
            active, rms = self.detector.classify(frame)
            now = time.monotonic()
            if not speech:
                pre.append(frame)
                if active:
                    self.utterance_id += 1
                    speech.extend(pre)
                    pre.clear()
                    silence = 0.0
                    last_partial = now
                    log_print(f"Speech detected: utterance={self.utterance_id} rms={rms:.5f}")
                continue
            speech.append(frame)
            silence = 0.0 if active else silence + frame_seconds
            duration = len(speech) * frame_seconds
            if (duration >= self.config.min_partial_duration and
                    now - last_partial >= self.config.partial_interval):
                window_frames = round(self.config.rolling_window_seconds / frame_seconds)
                audio = np.concatenate(speech[-window_frames:])
                self._submit(ASRJob(audio, False, self.utterance_id, now))
                last_partial = now
            ended = silence >= self.config.silence_duration
            if ended or duration >= self.config.max_utterance_seconds:
                usable = duration - silence
                if usable >= self.config.min_speech_duration:
                    trim = round(silence * self.config.sample_rate)
                    audio = np.concatenate(speech)
                    if trim:
                        audio = audio[:-trim]
                    self._submit(ASRJob(audio, True, self.utterance_id, now))
                    log_print(f"Speech ended: utterance={self.utterance_id} duration={usable:.2f}s")
                speech.clear()
                silence = 0.0
                self.detector.reset()
        # A stop click may arrive mid-utterance. Preserve that speech as a final
        # job rather than silently throwing away the last words.
        duration = len(speech) * frame_seconds
        if duration >= self.config.min_speech_duration:
            self._submit(ASRJob(np.concatenate(speech), True, self.utterance_id,
                                time.monotonic()))
