"""Audio recorder abstractions and the optional SoundCard implementation."""

from abc import ABC, abstractmethod
from threading import Event, Lock

from voice_to_text.audio.microphone import default_microphone
from voice_to_text.audio.processor import prepare_audio, rms
from voice_to_text.config import SpeechConfig
from voice_to_text.exceptions import MicrophoneError
from voice_to_text.logging import get_logger

LOGGER = get_logger("audio")


class BaseAudioRecorder(ABC):
    @abstractmethod
    def start(self) -> None: ...

    @abstractmethod
    def iter_audio(self, stop_event: Event): ...

    @abstractmethod
    def stop(self) -> None: ...

    def close(self) -> None:
        self.stop()


class SoundCardRecorder(BaseAudioRecorder):
    """Blocking chunk iterator; callers decide which thread consumes it."""

    def __init__(self, config: SpeechConfig):
        self.config = config
        self._running = False
        self._lock = Lock()
        self._device = None

    def start(self) -> None:
        with self._lock:
            if self._running:
                return
            self._device = default_microphone()
            self._running = True
            LOGGER.info("Microphone ready: %s", getattr(self._device, "name", "default"))

    def iter_audio(self, stop_event: Event):
        if not self._running or self._device is None:
            raise MicrophoneError("Recorder has not been started")
        frames = int(self.config.sample_rate * self.config.chunk_duration)
        try:
            with self._device.recorder(samplerate=self.config.sample_rate,
                                       channels=self.config.channels) as stream:
                while self._running and not stop_event.is_set():
                    audio = prepare_audio(stream.record(numframes=frames))
                    level = rms(audio)
                    LOGGER.debug("Captured samples=%s rms=%.6f", len(audio), level)
                    if level >= self.config.minimum_rms:
                        yield audio
                    else:
                        LOGGER.debug("No speech detected for current chunk")
        except MicrophoneError:
            raise
        except Exception as exc:
            raise MicrophoneError("Unable to record from the microphone") from exc

    def stop(self) -> None:
        with self._lock:
            self._running = False
            self._device = None
            LOGGER.info("Microphone stopped")
