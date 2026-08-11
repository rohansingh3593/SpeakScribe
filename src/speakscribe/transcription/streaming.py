"""Audio-time VAD/buffering that emits frequent partial and stable final jobs."""

from collections import deque
from dataclasses import dataclass
import time

import numpy as np

from speakscribe.audio.processor import rms
from speakscribe.config import SpeechConfig
from speakscribe.logging import get_logger

LOGGER = get_logger("transcription.streaming")


@dataclass(frozen=True)
class AudioFrame:
    audio: np.ndarray
    captured_at: float


@dataclass(frozen=True)
class TranscriptionJob:
    audio: np.ndarray
    is_final: bool
    utterance_id: int
    created_at: float
    speech_started_at: float


class StreamingBuffer:
    """Maintain one utterance and return partial jobs while speech is active."""

    def __init__(self, config: SpeechConfig):
        self.config = config
        self.frame_seconds = config.chunk_duration
        self.pre_frames = max(1, round(config.pre_speech_duration / self.frame_seconds))
        self.pre = deque(maxlen=self.pre_frames)
        self.speech = []
        self.voiced_seconds = 0.0
        self.silence_seconds = 0.0
        self.next_partial_seconds = config.partial_interval
        self.utterance_id = 0
        self.speech_started_at = 0.0

    def _job(self, is_final: bool, created_at: float) -> TranscriptionJob:
        audio = np.concatenate(self.speech)
        if is_final and self.silence_seconds:
            trim = round(self.silence_seconds * self.config.sample_rate)
            if 0 < trim < len(audio):
                audio = audio[:-trim]
        return TranscriptionJob(
            audio, is_final, self.utterance_id, created_at, self.speech_started_at)

    def _reset(self) -> None:
        self.speech.clear()
        self.voiced_seconds = 0.0
        self.silence_seconds = 0.0
        self.next_partial_seconds = self.config.partial_interval

    def push(self, frame: AudioFrame) -> list[TranscriptionJob]:
        jobs = []
        active = rms(frame.audio) >= self.config.minimum_rms
        if not self.speech:
            self.pre.append(frame.audio)
            if not active:
                return jobs
            self.utterance_id += 1
            self.speech = list(self.pre)
            self.pre.clear()
            self.voiced_seconds = self.frame_seconds
            self.speech_started_at = frame.captured_at - (
                max(0, len(self.speech) - 1) * self.frame_seconds)
            LOGGER.debug("[VAD] speech started utterance=%s rms=%.6f",
                         self.utterance_id, rms(frame.audio))
            return jobs

        self.speech.append(frame.audio)
        if active:
            self.voiced_seconds += self.frame_seconds
            self.silence_seconds = 0.0
        else:
            self.silence_seconds += self.frame_seconds
        duration = len(self.speech) * self.frame_seconds

        if (duration >= self.next_partial_seconds and
                self.voiced_seconds >= self.config.minimum_speech_duration):
            jobs.append(self._job(False, frame.captured_at))
            while self.next_partial_seconds <= duration:
                self.next_partial_seconds += self.config.partial_interval

        ended = (self.silence_seconds >= self.config.silence_duration or
                 duration >= self.config.maximum_utterance_duration)
        if ended:
            LOGGER.debug("[VAD] speech ended utterance=%s voiced=%.2fs silence=%.2fs",
                         self.utterance_id, self.voiced_seconds, self.silence_seconds)
            if self.voiced_seconds >= self.config.minimum_speech_duration:
                jobs.append(self._job(True, frame.captured_at))
            self._reset()
        return jobs

    def flush(self, created_at: float | None = None) -> list[TranscriptionJob]:
        if not self.speech or self.voiced_seconds < self.config.minimum_speech_duration:
            self._reset()
            return []
        job = self._job(True, created_at or time.monotonic())
        self._reset()
        return [job]
