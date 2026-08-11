"""Optional Faster-Whisper backend, imported only when it is actually used."""

from speakscribe.config import SpeechConfig
from speakscribe.exceptions import TranscriptionError
from speakscribe.logging import get_logger
from speakscribe.models import TranscriptionResult
from speakscribe.transcription.base import BaseTranscriptionEngine
from speakscribe.utils.helpers import language_code

LOGGER = get_logger("transcription.whisper")


class FasterWhisperEngine(BaseTranscriptionEngine):
    def __init__(self, config: SpeechConfig):
        self.config = config
        self._model = None

    def _load_model(self):
        if self._model is not None:
            return self._model
        try:
            from faster_whisper import WhisperModel
            candidates = ([('cuda', 'float16'), ('cpu', 'int8')]
                          if self.config.device == "auto"
                          else [(self.config.device, self.config.compute_type)])
            last_error = None
            for device, compute_type in candidates:
                try:
                    self._model = WhisperModel(self.config.model_size, device=device,
                                               compute_type=compute_type)
                    LOGGER.info("Whisper loaded: device=%s compute=%s", device, compute_type)
                    return self._model
                except Exception as exc:
                    last_error = exc
                    LOGGER.warning("Whisper initialization failed on %s: %s", device, exc)
            raise last_error or RuntimeError("No Whisper backend candidate was available")
        except Exception as exc:
            raise TranscriptionError("Unable to initialize Faster-Whisper") from exc

    def transcribe(self, audio, sample_rate: int) -> TranscriptionResult:
        del sample_rate  # Faster-Whisper expects the configured 16 kHz waveform.
        try:
            segments, info = self._load_model().transcribe(
                audio,
                language=language_code(self.config.language),
                task="transcribe",
                beam_size=self.config.beam_size,
                condition_on_previous_text=False,
                vad_filter=True,
            )
            text = " ".join(segment.text.strip() for segment in segments
                            if segment.text.strip()).strip()
            probability = getattr(info, "language_probability", None)
            return TranscriptionResult(
                text=text,
                language=getattr(info, "language", self.config.language),
                confidence=float(probability) if probability is not None else None,
            )
        except TranscriptionError:
            raise
        except Exception as exc:
            raise TranscriptionError("Faster-Whisper transcription failed") from exc

    def close(self) -> None:
        self._model = None
