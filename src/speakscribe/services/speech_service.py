"""Thread-safe, UI-independent high-level speech-to-text service."""

from collections.abc import Callable, Iterator
from threading import Event, RLock, Thread, current_thread

from speakscribe.audio import BaseAudioRecorder, SoundCardRecorder
from speakscribe.config import SpeechConfig
from speakscribe.exceptions import ServiceStateError, SpeakScribeError, TranscriptionError
from speakscribe.logging import get_logger
from speakscribe.models import TranscriptionResult
from speakscribe.transcription import BaseTranscriptionEngine, FasterWhisperEngine

LOGGER = get_logger("service")


class SpeechToText:
    """Reusable facade with generator, callback, lifecycle, and context-manager APIs."""

    def __init__(self, config: SpeechConfig | None = None, *,
                 recorder: BaseAudioRecorder | None = None,
                 engine: BaseTranscriptionEngine | None = None):
        self.config = config or SpeechConfig()
        self._recorder = recorder
        self._engine = engine
        self._stop_event = Event()
        self._lock = RLock()
        self._callback_thread: Thread | None = None
        self._running = False

    @property
    def is_running(self) -> bool:
        with self._lock:
            return self._running

    def _components(self) -> tuple[BaseAudioRecorder, BaseTranscriptionEngine]:
        if self._recorder is None:
            self._recorder = SoundCardRecorder(self.config)
        if self._engine is None:
            self._engine = FasterWhisperEngine(self.config)
        return self._recorder, self._engine

    def start(self) -> None:
        with self._lock:
            if self._running:
                return
            recorder, _ = self._components()
            self._stop_event.clear()
            recorder.start()
            self._running = True
            LOGGER.info("Speech recognition started")

    def _results(self) -> Iterator[TranscriptionResult]:
        recorder, engine = self._components()
        for audio in recorder.iter_audio(self._stop_event):
            if self._stop_event.is_set():
                break
            try:
                result = engine.transcribe(audio, self.config.sample_rate)
            except SpeakScribeError:
                raise
            except Exception as exc:
                raise TranscriptionError("Transcription engine failed") from exc
            if result.text.strip():
                LOGGER.debug("Transcript produced: %r", result.text)
                yield result

    def listen_once(self) -> TranscriptionResult:
        started_here = not self.is_running
        if started_here:
            self.start()
        try:
            return next(self._results())
        except StopIteration as exc:
            raise TranscriptionError("No speech was transcribed") from exc
        finally:
            if started_here:
                self.stop()

    def listen_continuously(self) -> Iterator[TranscriptionResult]:
        started_here = not self.is_running
        if started_here:
            self.start()
        try:
            yield from self._results()
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

            def worker() -> None:
                try:
                    for result in self._results():
                        on_result(result)
                except SpeakScribeError as exc:
                    LOGGER.exception("Continuous transcription failed")
                    if on_error is not None:
                        on_error(exc)
                finally:
                    self.stop()

            self._callback_thread = Thread(
                target=worker, name="speakscribe", daemon=True)
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
        with self._lock:
            self._stop_event.set()
            if self._recorder is not None:
                self._recorder.stop()
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

    def __exit__(self, *_exc_info) -> None:
        self.close()
