"""Single-model Faster-Whisper inference worker."""

from collections import deque
from queue import Empty, Queue
from threading import Event
import time

from faster_whisper import WhisperModel
from psutil import Process

from audio_pipeline import ASRJob, prepare_audio_for_asr
from config import AppConfig
from logger import log_print
from text_processing import apply_script_mode, clean_text, detect_language


class WhisperEngine:
    def __init__(self, config: AppConfig):
        self.config = config
        self.model = self._load_model()

    def _load_model(self):
        started = time.monotonic()
        candidates = ([("cuda", "float16"), ("cpu", "int8")]
                      if self.config.device == "auto"
                      else [(self.config.device, self.config.compute_type)])
        last_error = None
        for device, compute in candidates:
            try:
                model = WhisperModel(self.config.model_size, device=device,
                                     compute_type=compute)
                log_print(f"Model loaded: device={device} compute={compute} "
                          f"seconds={time.monotonic() - started:.2f}")
                return model
            except Exception as exc:
                last_error = exc
                log_print(f"Model initialization failed on {device}: {exc}")
        raise RuntimeError(f"Could not load Whisper model: {last_error}")

    def transcribe(self, job: ASRJob, context: str) -> tuple[str, str]:
        profile = self.config.profile
        vocabulary = ", ".join(self.config.vocabulary)
        prompt = (
            "Speech may be Hindi (हिन्दी), English, or naturally mixed Hinglish. "
            "Transcribe exactly in the spoken language and original script; do not "
            f"translate. Technical vocabulary: {vocabulary}."
        )
        if context:
            prompt += f" Recent context: {context}"
        language = None if self.config.language_mode == "auto" else self.config.language_mode
        segments, info = self.model.transcribe(
            prepare_audio_for_asr(job.audio), language=language, task="transcribe",
            beam_size=profile.beam_size,
            best_of=profile.best_of, temperature=profile.temperature,
            initial_prompt=prompt, condition_on_previous_text=False,
            vad_filter=self.config.vad_filter, word_timestamps=False,
        )
        accepted = []
        for segment in segments:
            if (segment.no_speech_prob <= self.config.no_speech_threshold and
                    segment.avg_logprob >= self.config.min_avg_logprob):
                accepted.append(segment.text)
        text = clean_text(" ".join(accepted), final=job.final)
        text = apply_script_mode(text, self.config.script_mode, self.config.vocabulary)
        mode = detect_language(text, getattr(info, "language", None))
        return text, mode


class ASRWorker:
    def __init__(self, config: AppConfig, queue: Queue, stop_event: Event, signals):
        self.config, self.queue, self.stop_event, self.signals = config, queue, stop_event, signals
        self.history: deque[str] = deque(maxlen=config.context_sentences)

    def run(self) -> None:
        try:
            self.signals.status_changed.emit("Loading speech model…")
            engine = WhisperEngine(self.config)
            self.signals.status_changed.emit("🎤 Listening")
            process = Process()
            while not self.stop_event.is_set() or not self.queue.empty():
                try:
                    job = self.queue.get(timeout=0.1)
                except Empty:
                    continue
                started = time.monotonic()
                text, language = engine.transcribe(job, " ".join(self.history))
                elapsed = time.monotonic() - started
                duration = len(job.audio) / self.config.sample_rate
                latency = time.monotonic() - job.captured_at
                log_print(f"[ASR] final={job.final} audio={duration:.2f}s inference={elapsed:.2f}s "
                          f"rtf={elapsed/max(duration, .001):.2f} latency={latency:.2f}s "
                          f"queue={self.queue.qsize()} cpu={process.cpu_percent():.1f}% "
                          f"ram={process.memory_info().rss/1024**3:.2f}GB")
                if not text:
                    continue
                self.signals.language_changed.emit(language)
                if job.final:
                    self.history.append(text)
                    self.signals.final_text.emit(text)
                    self.signals.partial_text.emit("")
                    log_print(f"Final result: {text}")
                else:
                    self.signals.partial_text.emit(text)
                    log_print(f"Partial result: {text}")
        except Exception as exc:
            log_print(f"ASR worker error: {exc}")
            self.signals.error.emit(str(exc))
