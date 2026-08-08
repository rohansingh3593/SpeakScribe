"""Single-model Faster-Whisper inference worker."""

from collections import deque
from queue import Empty, Queue
from threading import Event, Lock
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
        # Repeated partials must be inexpensive. The selected profile is retained
        # for final correction, while live hypotheses use deterministic greedy
        # decoding to keep CPU fallback latency close to real time.
        beam_size = profile.beam_size if job.final else 1
        best_of = profile.best_of if job.final else 1
        vocabulary = ", ".join(self.config.vocabulary)
        final_prompt = (
            "Speech may be Hindi (हिन्दी), English, or naturally mixed Hinglish. "
            "Transcribe exactly in the spoken language and original script; do not "
            f"translate. Technical vocabulary: {vocabulary}."
        )
        if context:
            final_prompt += f" Recent context: {context}"
        # A long prompt on sub-second audio can itself seed hallucinations (for
        # example repeated URLs). Partials prioritize the actual audio; finals use
        # vocabulary and recent context for correction.
        prompt = final_prompt if job.final else None
        language = None if self.config.language_mode == "auto" else self.config.language_mode
        segments, info = self.model.transcribe(
            prepare_audio_for_asr(job.audio), language=language, task="transcribe",
            beam_size=beam_size,
            best_of=best_of, temperature=profile.temperature,
            initial_prompt=prompt, condition_on_previous_text=False,
            vad_filter=self.config.vad_filter, word_timestamps=False,
            no_speech_threshold=self.config.no_speech_threshold,
            log_prob_threshold=self.config.min_avg_logprob,
            compression_ratio_threshold=2.4,
        )
        accepted = []
        segment_count = 0
        for segment in segments:
            segment_count += 1
            if (segment.no_speech_prob <= self.config.no_speech_threshold and
                    segment.avg_logprob >= self.config.min_avg_logprob):
                accepted.append(segment.text)
            else:
                log_print(
                    "[ASR] rejected segment "
                    f"no_speech={segment.no_speech_prob:.2f} "
                    f"avg_logprob={segment.avg_logprob:.2f} "
                    f"text={segment.text.strip()!r}"
                )
        if not accepted:
            log_print(f"[ASR] no usable segments returned (segments={segment_count})")
        text = clean_text(" ".join(accepted), final=job.final)
        text = apply_script_mode(text, self.config.script_mode, self.config.vocabulary)
        mode = detect_language(text, getattr(info, "language", None))
        return text, mode


class WhisperModelProvider:
    """Thread-safe owner that loads exactly one Whisper engine per application."""

    def __init__(self):
        self._engine: WhisperEngine | None = None
        self._lock = Lock()

    def get(self, config: AppConfig) -> WhisperEngine:
        with self._lock:
            if self._engine is None:
                self._engine = WhisperEngine(config)
            else:
                # Model weights/device stay cached; lightweight decode and text
                # settings may change between listening sessions.
                self._engine.config = config
            return self._engine


class ASRWorker:
    def __init__(self, config: AppConfig, queue: Queue, stop_event: Event, signals,
                 model_provider: WhisperModelProvider):
        self.config, self.queue, self.stop_event, self.signals = config, queue, stop_event, signals
        self.model_provider = model_provider
        self.history: deque[str] = deque(maxlen=config.context_sentences)

    def run(self) -> None:
        try:
            self.signals.status_changed.emit("Loading speech model…")
            engine = self.model_provider.get(self.config)
            self.signals.status_changed.emit("🎤 Listening")
            process = Process()
            stop_empty_since = None
            while True:
                try:
                    job = self.queue.get(timeout=0.1)
                except Empty:
                    if not self.stop_event.is_set():
                        continue
                    if stop_empty_since is None:
                        stop_empty_since = time.monotonic()
                    # SpeechBufferWorker may still be draining raw audio and
                    # enqueueing the stop-time final. Give it a bounded handoff
                    # window rather than racing out on the first empty poll.
                    if time.monotonic() - stop_empty_since >= 0.75:
                        break
                    continue
                stop_empty_since = None
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
