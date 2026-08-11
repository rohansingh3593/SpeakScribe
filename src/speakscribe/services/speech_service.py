"""Concurrent, UI-independent, real-time speech-to-text service."""

from collections.abc import Callable, Iterator
from dataclasses import replace
from queue import Empty, Full, Queue
from threading import Event, RLock, Thread, current_thread
import time

from speakscribe.audio import BaseAudioRecorder, SoundCardRecorder
from speakscribe.config import SpeechConfig
from speakscribe.exceptions import ServiceStateError, SpeakScribeError, TranscriptionError
from speakscribe.logging import get_logger
from speakscribe.models import TranscriptionResult
from speakscribe.transcription import BaseTranscriptionEngine, FasterWhisperEngine
from speakscribe.transcription.streaming import AudioFrame, StreamingBuffer, TranscriptionJob

LOGGER = get_logger("service")


class SpeechToText:
    """Capture, buffer, and infer concurrently while streaming partial/final results."""

    def __init__(self, config: SpeechConfig | None = None, *,
                 recorder: BaseAudioRecorder | None = None,
                 engine: BaseTranscriptionEngine | None = None):
        self.config = config or SpeechConfig()
        self._recorder = recorder
        self._engine = engine
        self._stop_event = Event()
        self._capture_done = Event()
        self._buffer_done = Event()
        self._asr_done = Event()
        self._lock = RLock()
        self._workers = []
        self._callback_thread = None
        self._running = False
        self._audio_queue = Queue(maxsize=self.config.max_audio_queue)
        self._asr_queue = Queue(maxsize=self.config.max_asr_queue)
        self._result_queue = Queue()

    @property
    def is_running(self) -> bool:
        with self._lock:
            return self._running

    def _components(self):
        if self._recorder is None:
            self._recorder = SoundCardRecorder(self.config)
        if self._engine is None:
            self._engine = FasterWhisperEngine(self.config)
        return self._recorder, self._engine

    @staticmethod
    def _fresh_queue(maxsize=0):
        return Queue(maxsize=maxsize)

    def start(self) -> None:
        with self._lock:
            if self._running:
                return
            recorder, _ = self._components()
            self._stop_event.clear()
            self._capture_done.clear()
            self._buffer_done.clear()
            self._asr_done.clear()
            self._audio_queue = self._fresh_queue(self.config.max_audio_queue)
            self._asr_queue = self._fresh_queue(self.config.max_asr_queue)
            self._result_queue = self._fresh_queue()
            recorder.start()
            self._running = True
            self._workers = [
                Thread(target=self._capture_loop, name="speakscribe-capture", daemon=True),
                Thread(target=self._buffer_loop, name="speakscribe-buffer", daemon=True),
                Thread(target=self._asr_loop, name="speakscribe-asr", daemon=True),
            ]
            for worker in self._workers:
                worker.start()
            LOGGER.info("Real-time speech recognition started")

    def _capture_loop(self) -> None:
        recorder, _ = self._components()
        try:
            for audio in recorder.iter_audio(self._stop_event):
                if self._stop_event.is_set():
                    break
                frame = AudioFrame(audio, time.monotonic())
                try:
                    self._audio_queue.put(frame, timeout=.02)
                except Full:
                    try:
                        self._audio_queue.get_nowait()
                    except Empty:
                        pass
                    self._audio_queue.put_nowait(frame)
                    LOGGER.warning("[AUDIO] queue full; replaced oldest unprocessed frame")
                LOGGER.debug("[AUDIO] frame received samples=%s queue=%s",
                             len(audio), self._audio_queue.qsize())
        except Exception as exc:
            error = TranscriptionError("Audio capture failed")
            error.__cause__ = exc
            self._result_queue.put(error)
            LOGGER.exception("Audio capture failed: %s", exc)
        finally:
            self._capture_done.set()

    def _submit_job(self, job: TranscriptionJob) -> None:
        if not job.is_final:
            try:
                self._asr_queue.put_nowait(job)
                return
            except Full:
                try:
                    pending = self._asr_queue.get_nowait()
                except Empty:
                    return
                if pending.is_final:
                    self._asr_queue.put_nowait(pending)
                    LOGGER.debug("[QUEUE] final preserved; obsolete partial dropped")
                    return
                self._asr_queue.put_nowait(job)
                LOGGER.debug("[QUEUE] obsolete partial replaced with utterance=%s",
                             job.utterance_id)
                return
        while not self._asr_done.is_set():
            try:
                self._asr_queue.put(job, timeout=.05)
                return
            except Full:
                continue

    def _buffer_loop(self) -> None:
        buffer = StreamingBuffer(self.config)
        try:
            while not self._capture_done.is_set() or not self._audio_queue.empty():
                try:
                    frame = self._audio_queue.get(timeout=.05)
                except Empty:
                    continue
                jobs = buffer.push(frame)
                for job in jobs:
                    LOGGER.debug("[BUFFER] utterance=%s %.2fs available state=%s",
                                 job.utterance_id,
                                 len(job.audio) / self.config.sample_rate,
                                 "FINAL" if job.is_final else "PARTIAL")
                    self._submit_job(job)
            for job in buffer.flush():
                self._submit_job(job)
        except Exception as exc:
            error = TranscriptionError("Audio buffering failed")
            error.__cause__ = exc
            self._result_queue.put(error)
            LOGGER.exception("Audio buffering failed: %s", exc)
        finally:
            self._buffer_done.set()

    def _asr_loop(self) -> None:
        _, engine = self._components()
        try:
            while not self._buffer_done.is_set() or not self._asr_queue.empty():
                try:
                    job = self._asr_queue.get(timeout=.05)
                except Empty:
                    continue
                started = time.monotonic()
                queue_wait = started - job.created_at
                state = "FINAL" if job.is_final else "PARTIAL"
                LOGGER.debug("[ASR] %s started utterance=%s queue_wait=%.3fs",
                             state, job.utterance_id, queue_wait)
                try:
                    result = engine.transcribe(job.audio, self.config.sample_rate)
                except SpeakScribeError as exc:
                    self._result_queue.put(exc)
                    continue
                except Exception as exc:
                    error = TranscriptionError("Transcription engine failed")
                    error.__cause__ = exc
                    self._result_queue.put(error)
                    LOGGER.exception("Transcription engine failed: %s", exc)
                    self._stop_event.set()
                    break
                finished = time.monotonic()
                result = replace(
                    result,
                    is_final=job.is_final,
                    utterance_id=job.utterance_id,
                    audio_duration=len(job.audio) / self.config.sample_rate,
                    queue_wait_seconds=queue_wait,
                    inference_seconds=finished - started,
                    speech_to_result_seconds=finished - job.speech_started_at,
                )
                if result.text.strip():
                    self._result_queue.put(result)
                    LOGGER.debug(
                        "[TEXT] %s=%r inference=%.3fs speech_to_result=%.3fs",
                        state.lower(), result.text, result.inference_seconds,
                        result.speech_to_result_seconds)
        finally:
            self._asr_done.set()
            with self._lock:
                self._running = False

    def _iter_results(self) -> Iterator[TranscriptionResult]:
        while not self._asr_done.is_set() or not self._result_queue.empty():
            try:
                item = self._result_queue.get(timeout=.05)
            except Empty:
                continue
            if isinstance(item, SpeakScribeError):
                raise item
            yield item

    def listen_once(self) -> TranscriptionResult:
        started_here = not self.is_running
        if started_here:
            self.start()
        last_partial = None
        try:
            for result in self._iter_results():
                if result.is_final:
                    return result
                last_partial = result
            if last_partial is not None:
                return last_partial
            raise TranscriptionError("No speech was transcribed")
        finally:
            if started_here:
                self.stop()

    def listen_continuously(self) -> Iterator[TranscriptionResult]:
        started_here = not self.is_running
        if started_here:
            self.start()
        try:
            yield from self._iter_results()
        finally:
            if started_here:
                self.stop()

    def start_continuous(self, on_result: Callable[[TranscriptionResult], None],
                         on_error: Callable[[SpeakScribeError], None] | None = None) -> Thread:
        if not callable(on_result):
            raise TypeError("on_result must be callable")
        with self._lock:
            if self._callback_thread is not None and self._callback_thread.is_alive():
                raise ServiceStateError("Continuous callback worker is already running")
            self.start()

            def callback_loop():
                try:
                    for result in self._iter_results():
                        on_result(result)
                except SpeakScribeError as exc:
                    LOGGER.exception("Continuous transcription failed")
                    if on_error is not None:
                        on_error(exc)
                finally:
                    self.stop()

            self._callback_thread = Thread(
                target=callback_loop, name="speakscribe-callback", daemon=True)
            self._callback_thread.start()
            return self._callback_thread

    def set_engine(self, engine: BaseTranscriptionEngine) -> None:
        with self._lock:
            if self._running:
                raise ServiceStateError("Cannot switch engines while running")
            if self._engine is not None:
                self._engine.close()
            self._engine = engine

    def stop(self) -> None:
        self._stop_event.set()
        if self._recorder is not None:
            self._recorder.stop()
        for worker in self._workers:
            if worker is not current_thread() and worker.is_alive():
                worker.join(timeout=2)
        with self._lock:
            self._running = False
        LOGGER.info("Speech recognition stopped")

    def close(self) -> None:
        self.stop()
        if (self._callback_thread is not None and self._callback_thread.is_alive() and
                self._callback_thread is not current_thread()):
            self._callback_thread.join(timeout=2)
        if self._recorder is not None:
            self._recorder.close()
        if self._engine is not None:
            self._engine.close()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *_exc_info):
        self.close()
