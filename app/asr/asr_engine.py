"""Single-model Faster-Whisper inference worker."""

from collections import deque
from queue import Empty, Queue
from threading import Event, Lock
from pathlib import Path
import time
import wave

import numpy as np

from faster_whisper import WhisperModel
from psutil import Process

from app.audio.audio_pipeline import ASRJob, audio_statistics, prepare_audio_for_asr
from app.config.settings import AppConfig
from app.config.decoding_policy import hotwords, initial_prompt, retry_thresholds
from app.utils.logger import get_logger, log_exception
from app.processing.text_processing import (
    apply_script_mode, clean_text, detect_language, is_low_quality_text,
)


LOGGER = get_logger("asr")
KNOWN_SHORT_HALLUCINATIONS = {
    "thank you", "thank you.", "thanks", "thanks.", "thanks for watching",
    "thanks for watching.", "see you in the next video.",
    "thanks for watching, see you in the next video.",
}


def _write_debug_wav(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(sample_rate)
        stream.writeframes(pcm.tobytes())


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
                LOGGER.debug(f"Model loaded: device={device} compute={compute} "
                          f"seconds={time.monotonic() - started:.2f}")
                return model
            except Exception as exc:
                last_error = exc
                LOGGER.warning("Model initialization failed on %s; trying fallback: %s",
                               device, exc)
        raise RuntimeError(f"Could not load Whisper model: {last_error}")

    def transcribe(self, job: ASRJob, context: str) -> tuple[str, str]:
        profile = self.config.profile
        # Repeated partials must be inexpensive. The selected profile is retained
        # for final correction, while live hypotheses use deterministic greedy
        # decoding to keep CPU fallback latency close to real time.
        beam_size = profile.beam_size if job.final else 1
        best_of = profile.best_of if job.final else 1
        prompt = initial_prompt(
            final=job.final, sample_count=len(job.audio), sample_rate=self.config.sample_rate,
            language_mode=self.config.language_mode, vocabulary=self.config.vocabulary,
            context=context,
        )
        vocabulary_bias = hotwords(
            final=job.final, language_mode=self.config.language_mode,
            vocabulary=self.config.vocabulary,
        )
        language = None if self.config.language_mode == "auto" else self.config.language_mode
        prepared = prepare_audio_for_asr(job.audio)
        raw_stats = audio_statistics(job.audio)
        prepared_stats = audio_statistics(prepared)
        LOGGER.debug(
            "[ASR-INPUT] "
            f"utterance={job.utterance_id} final={job.final} "
            f"duration={len(job.audio) / self.config.sample_rate:.2f}s "
            f"voiced={job.speech_seconds if job.speech_seconds is not None else -1:.2f}s "
            f"raw_rms={raw_stats['rms']:.6f} raw_peak={raw_stats['peak']:.6f} "
            f"raw_mean={raw_stats['mean']:.6f} zeros={raw_stats['zero_ratio']:.3f} "
            f"prepared_rms={prepared_stats['rms']:.6f} "
            f"prepared_peak={prepared_stats['peak']:.6f} finite={prepared_stats['finite']} "
            f"language={language or 'auto'} script={self.config.script_mode} "
            f"beam={beam_size} prompt={'yes' if prompt else 'no'} "
            f"hotwords={'yes' if vocabulary_bias else 'no'}"
        )
        if self.config.debug_audio_enabled and job.final:
            stamp = time.strftime("%Y%m%d-%H%M%S")
            directory = Path(self.config.debug_audio_directory)
            raw_path = directory / f"{stamp}-u{job.utterance_id}-raw.wav"
            prepared_path = directory / f"{stamp}-u{job.utterance_id}-prepared.wav"
            _write_debug_wav(raw_path, job.audio, self.config.sample_rate)
            _write_debug_wav(prepared_path, prepared, self.config.sample_rate)
            LOGGER.debug(f"[ASR-INPUT] saved debug audio: {raw_path}, {prepared_path}")
        def decode(initial_prompt, word_bias, *, relaxed=False):
            no_speech_threshold = self.config.no_speech_threshold
            log_probability_threshold = self.config.min_avg_logprob
            compression_threshold = self.config.max_compression_ratio
            if relaxed:
                (no_speech_threshold, log_probability_threshold,
                 compression_threshold) = retry_thresholds(
                    no_speech=no_speech_threshold,
                    log_probability=log_probability_threshold,
                    compression_ratio=compression_threshold,
                )
            segments, decode_info = self.model.transcribe(
                prepared, language=language, task="transcribe",
                beam_size=beam_size, best_of=best_of,
                temperature=profile.temperature, initial_prompt=initial_prompt,
                hotwords=word_bias,
                condition_on_previous_text=False, vad_filter=self.config.vad_filter,
                word_timestamps=False, no_speech_threshold=no_speech_threshold,
                log_prob_threshold=log_probability_threshold,
                compression_ratio_threshold=compression_threshold,
            )
            accepted = []
            segment_count = 0
            for segment in segments:
                segment_count += 1
                LOGGER.debug(
                    "[ASR-SEGMENT] "
                    f"start={segment.start:.2f} end={segment.end:.2f} "
                    f"no_speech={segment.no_speech_prob:.3f} "
                    f"avg_logprob={segment.avg_logprob:.3f} "
                    f"compression={segment.compression_ratio:.3f} raw={segment.text!r}"
                )
                if (segment.no_speech_prob <= no_speech_threshold and
                        segment.avg_logprob >= log_probability_threshold and
                        segment.compression_ratio <= compression_threshold):
                    accepted.append(segment.text)
                else:
                    LOGGER.debug(
                        "[ASR] rejected segment "
                        f"no_speech={segment.no_speech_prob:.2f} "
                        f"avg_logprob={segment.avg_logprob:.2f} "
                        f"compression={segment.compression_ratio:.2f} "
                        f"text={segment.text.strip()!r}"
                    )
            if not accepted:
                LOGGER.debug(f"[ASR] no usable segments returned (segments={segment_count})")
            decoded = clean_text(" ".join(accepted), final=job.final)
            LOGGER.debug(f"[ASR-TEXT] accepted={len(accepted)}/{segment_count} cleaned={decoded!r}")
            return decoded, decode_info

        text, info = decode(prompt, vocabulary_bias)
        if (len(job.audio) / self.config.sample_rate < 2.0 and
                text.casefold() in KNOWN_SHORT_HALLUCINATIONS):
            LOGGER.debug(f"[ASR] rejected known short-audio hallucination: {text!r}")
            text = ""
        if is_low_quality_text(text):
            LOGGER.debug(f"[ASR] rejected corrupt/repetitive transcript: {text!r}")
            text = ""
        if not text and job.final:
            LOGGER.debug("[ASR] final was unusable; retrying prompt-free with recovery thresholds")
            text, info = decode(None, None, relaxed=True)
            if (len(job.audio) / self.config.sample_rate < 2.0 and
                    text.casefold() in KNOWN_SHORT_HALLUCINATIONS):
                LOGGER.debug(f"[ASR] rejected known fallback hallucination: {text!r}")
                text = ""
            if is_low_quality_text(text):
                LOGGER.debug(f"[ASR] rejected corrupt/repetitive fallback: {text!r}")
                text = ""
        text = apply_script_mode(text, self.config.script_mode, self.config.vocabulary)
        language_probability = float(getattr(info, "language_probability", 0.0) or 0.0)
        LOGGER.debug(f"[ASR-TEXT] post_script={text!r} detected={getattr(info, 'language', None)!r} "
                  f"probability={language_probability:.3f}")
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
            LOGGER.info("ASR worker ready | model=%s language=%s device=%s",
                        self.config.model_size, self.config.language_mode,
                        self.config.device)
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
                LOGGER.debug(f"[ASR] final={job.final} audio={duration:.2f}s inference={elapsed:.2f}s "
                          f"rtf={elapsed/max(duration, .001):.2f} latency={latency:.2f}s "
                          f"queue={self.queue.qsize()} cpu={process.cpu_percent():.1f}% "
                          f"ram={process.memory_info().rss/1024**3:.2f}GB")
                if not text:
                    if job.final:
                        LOGGER.warning(
                            "ASR returned no text | utterance=%s audio=%.2fs inference=%.2fs",
                            job.utterance_id, duration, elapsed)
                    continue
                self.signals.language_changed.emit(language)
                if job.final:
                    self.history.append(text)
                    self.signals.final_text.emit(text)
                    self.signals.partial_text.emit("")
                    LOGGER.info("Transcription ready | utterance=%s language=%s text=%r",
                                job.utterance_id, language, text)
                else:
                    self.signals.partial_text.emit(text)
                    LOGGER.info("Live transcription | utterance=%s language=%s text=%r",
                                job.utterance_id, language, text)
        except Exception as exc:
            log_exception("ASR-WORKER", exc)
            self.signals.error.emit(str(exc))
