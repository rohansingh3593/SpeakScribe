"""Single-model Faster-Whisper inference worker."""

from collections import deque
from queue import Empty, Full, Queue
from threading import Event, Lock, Thread
from pathlib import Path
import time
import wave

import numpy as np

from faster_whisper import WhisperModel
from psutil import Process

from app.audio.audio_pipeline import (
    ASRJob, audio_normalization_gain, audio_statistics, prepare_audio_for_asr,
)
from app.config.settings import AppConfig
from app.config.settings import PerformanceMode
from app.config.decoding_policy import hotwords, initial_prompt, retry_thresholds
from app.utils.logger import get_logger, log_exception
from app.processing.text_processing import (
    apply_script_mode, clean_text, detect_language, is_low_quality_text, script_metadata,
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
    def __init__(self, config: AppConfig, model=None):
        self.config = config
        self.model = model if model is not None else self._load_model()

    def _load_model(self):
        started = time.monotonic()
        candidates = ([("cuda", "float16"), ("cpu", "int8")]
                      if self.config.device == "auto"
                      else [(self.config.device, self.config.compute_type)])
        last_error = None
        for device, compute in candidates:
            try:
                # Live ASR runs one inference at a time. A single CTranslate2
                # worker avoids internal oversubscription on the CPU fallback.
                model_workers = 1
                model = WhisperModel(self.config.model_size, device=device,
                                     compute_type=compute, num_workers=model_workers)
                LOGGER.debug(f"Model loaded: device={device} compute={compute} "
                          f"seconds={time.monotonic() - started:.2f}")
                return model
            except Exception as exc:
                last_error = exc
                LOGGER.warning("Model initialization failed on %s; trying fallback: %s",
                               device, exc)
        raise RuntimeError(f"Could not load Whisper model: {last_error}")

    def transcribe(self, job: ASRJob, context: str) -> tuple[str, str, dict]:
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
        normalization_gain = audio_normalization_gain(
            job.audio - np.mean(job.audio, dtype=np.float64))
        LOGGER.debug(
            "[ASR-INPUT] "
            f"utterance={job.utterance_id} final={job.final} "
            f"duration={len(job.audio) / self.config.sample_rate:.2f}s "
            f"voiced={job.speech_seconds if job.speech_seconds is not None else -1:.2f}s "
            f"raw_rms={raw_stats['rms']:.6f} raw_peak={raw_stats['peak']:.6f} "
            f"raw_mean={raw_stats['mean']:.6f} zeros={raw_stats['zero_ratio']:.3f} "
            f"prepared_rms={prepared_stats['rms']:.6f} "
            f"prepared_peak={prepared_stats['peak']:.6f} gain={normalization_gain:.2f} "
            f"finite={prepared_stats['finite']} "
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
                condition_on_previous_text=(profile.condition_on_previous_text and job.final),
                vad_filter=self.config.vad_filter,
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
            raw_text = " ".join(accepted).strip()
            decoded = clean_text(raw_text, final=job.final)
            LOGGER.debug(f"[ASR-TEXT] accepted={len(accepted)}/{segment_count} cleaned={decoded!r}")
            return decoded, decode_info, raw_text

        text, info, raw_text = decode(prompt, vocabulary_bias)
        if (len(job.audio) / self.config.sample_rate < 2.0 and
                text.casefold() in KNOWN_SHORT_HALLUCINATIONS):
            LOGGER.debug(f"[ASR] rejected known short-audio hallucination: {text!r}")
            text = ""
        if is_low_quality_text(text):
            LOGGER.debug(f"[ASR] rejected corrupt/repetitive transcript: {text!r}")
            text = ""
        if not text and job.final:
            LOGGER.debug("[ASR] final was unusable; retrying prompt-free with recovery thresholds")
            text, info, raw_text = decode(None, None, relaxed=True)
            if (len(job.audio) / self.config.sample_rate < 2.0 and
                    text.casefold() in KNOWN_SHORT_HALLUCINATIONS):
                LOGGER.debug(f"[ASR] rejected known fallback hallucination: {text!r}")
                text = ""
            if is_low_quality_text(text):
                LOGGER.debug(f"[ASR] rejected corrupt/repetitive fallback: {text!r}")
                text = ""
        after_language = text
        after_transliteration = apply_script_mode(
            after_language, self.config.script_mode, self.config.vocabulary)
        text = clean_text(after_transliteration, final=job.final)
        language_probability = float(getattr(info, "language_probability", 0.0) or 0.0)
        metadata = script_metadata(text, self.config.script_mode, self.config.language_mode)
        # A non-empty decode can still be unusable for Hindi (Arabic script or
        # an English hallucination). Give finalized audio one prompt-free,
        # relaxed recovery pass before declaring that no safe speech was found.
        if (job.final and self.config.language_mode == "hi" and text and
                not metadata["script_valid"]):
            first_raw = raw_text
            LOGGER.info("Retrying Hindi final after script/language mismatch | utterance=%s",
                        job.utterance_id)
            retry_text, retry_info, retry_raw = decode(None, None, relaxed=True)
            retry_after_script = apply_script_mode(
                retry_text, self.config.script_mode, self.config.vocabulary)
            retry_text = clean_text(retry_after_script, final=True)
            retry_metadata = script_metadata(
                retry_text, self.config.script_mode, self.config.language_mode)
            if retry_text and retry_metadata["script_valid"]:
                text, info, raw_text, metadata = (
                    retry_text, retry_info, retry_raw, retry_metadata)
                after_language = retry_text
                after_transliteration = retry_after_script
                LOGGER.info("Hindi recovery produced a valid transcript | utterance=%s",
                            job.utterance_id)
            else:
                metadata["initial_raw_text"] = first_raw
                metadata["retry_raw_text"] = retry_raw
        metadata.update({
            "raw_text": raw_text, "processed_text": text,
            "detected_language": getattr(info, "language", None) or self.config.language_mode,
        })
        LOGGER.debug(
            "[ASR-PIPELINE] raw=%r detected_language=%r recognition=%s script_mode=%s "
            "after_language=%r after_transliteration=%r after_normalization=%r "
            "translation=%s final_candidate=%r detected_script=%s script_valid=%s",
            raw_text, getattr(info, "language", None), self.config.language_mode,
            self.config.script_mode, after_language, after_transliteration, text,
            "enabled-downstream" if self.config.translation_enabled else "off",
            text, metadata["detected_script"], metadata["script_valid"])
        if not metadata["script_valid"]:
            LOGGER.warning(
                "Unexpected Arabic/Urdu or mismatched script detected for Hindi/Hinglish "
                "segment | utterance=%s raw=%r detected=%s requested=%s",
                job.utterance_id, raw_text, metadata["detected_script"],
                self.config.script_mode)
        LOGGER.debug(f"[ASR-TEXT] post_script={text!r} detected={getattr(info, 'language', None)!r} "
                  f"probability={language_probability:.3f}")
        mode = detect_language(text, getattr(info, "language", None))
        return text, mode, metadata


def _unpack_transcription(result) -> tuple[str, str, dict]:
    """Accept legacy two-item test/integration engines while carrying metadata."""
    if len(result) == 3:
        return result
    text, language = result
    return text, language, {}


class WhisperModelProvider:
    """Thread-safe lazy cache keyed by the settings which select model weights."""

    def __init__(self):
        self._engines: dict[tuple, WhisperEngine] = {}
        self._models: dict[tuple[str, str, str], object] = {}
        self._lock = Lock()

    def get(self, config: AppConfig) -> WhisperEngine:
        with self._lock:
            model_key = (config.model_size, config.device, config.compute_type)
            engine_key = (*model_key, config.performance_mode, config.language_mode,
                          config.script_mode)
            engine = self._engines.get(engine_key)
            if engine is None:
                model = self._models.get(model_key)
                engine = WhisperEngine(config, model=model)
                self._models.setdefault(model_key, engine.model)
                self._engines[engine_key] = engine
            else:
                # Mode/language changes never reload identical model weights.
                engine.config = config
            LOGGER.info("Performance profile active | mode=%s model=%s beam=%s "
                        "partial_interval=%.2fs context=%s window=%.1fs",
                        config.performance_mode.value, config.model_size,
                        config.profile.beam_size, config.partial_interval,
                        config.context_sentences, config.rolling_window_seconds)
            return engine


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
            source_label = ("Microphone" if self.config.capture_source == "microphone"
                            else "System audio")
            self.signals.status_changed.emit(f"🎤 Listening via {source_label}")
            LOGGER.info("ASR worker ready | model=%s language=%s device=%s capture=%s",
                        self.config.model_size, self.config.language_mode,
                        self.config.device, self.config.capture_source)
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
                text, language, metadata = _unpack_transcription(
                    engine.transcribe(job, " ".join(self.history)))
                if not metadata.get("script_valid", True):
                    text = ""
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


class ComparisonASRWorker:
    """Keep live capture real-time by decoding the active stream with FAST."""

    def __init__(self, config: AppConfig, queue: Queue, stop_event: Event, signals,
                 model_provider: WhisperModelProvider):
        self.config, self.queue, self.stop_event = config, queue, stop_event
        self.signals, self.model_provider = signals, model_provider
        self.histories = {mode: deque(maxlen=AppConfig(
            performance_mode=mode).context_sentences) for mode in PerformanceMode}
        self.first_seen: dict[tuple[int, PerformanceMode], float] = {}

    def run(self) -> None:
        # Do not even construct/load idle refinement engines in a live session.
        # This makes the runtime architecture unambiguous in logs and avoids a
        # second CTranslate2 model reserving CPU threads and memory.
        queues = {PerformanceMode.FAST: Queue(maxsize=2)}
        workers = [Thread(target=self._run_mode,
                          args=(PerformanceMode.FAST, queues[PerformanceMode.FAST]),
                          name="asr-fast", daemon=True)]
        LOGGER.info("Live ASR scheduler active | pipeline=fast-live-v2 modes=fast "
                    "partial_queue=latest final_queue=latest")
        for worker in workers:
            worker.start()
        stop_empty_since = None
        while True:
            try:
                job = self.queue.get(timeout=0.1)
            except Empty:
                if not self.stop_event.is_set():
                    continue
                stop_empty_since = stop_empty_since or time.monotonic()
                if time.monotonic() - stop_empty_since >= 0.75:
                    break
                continue
            stop_empty_since = None
            # Live snapshots are latency-sensitive and supersede one another.
            # Running the two refinement profiles for every snapshot starves
            # FAST on CPU-only systems; they receive the finalized audio below.
            # Refinement models cannot run concurrently with FAST on the common
            # CPU fallback without starving the live stream. A 4-second final
            # was taking 30-100 seconds in production and making capture drift
            # minutes behind. Keep the recording path exclusively FAST; offline
            # evaluation remains available for side-by-side profile comparison.
            self._enqueue_mode_job(PerformanceMode.FAST, queues[PerformanceMode.FAST], job)
        for mode_queue in queues.values():
            mode_queue.put(None)
        for worker in workers:
            worker.join()

    @staticmethod
    def _enqueue_mode_job(mode: PerformanceMode, queue: Queue, job: ASRJob) -> None:
        if job.final:
            retained_finals = []
            while True:
                try:
                    pending = queue.get_nowait()
                except Empty:
                    break
                if pending.final:
                    retained_finals.append(pending)
            for pending in retained_finals:
                queue.put(pending)
            queue.put(job)
            return
        # A partial is a snapshot, not an event which must be replayed. Keep
        # finalized jobs in FIFO order and replace every queued partial with the
        # newest snapshot. This bounds live work to one running inference plus
        # one pending inference instead of displaying progressively stale text.
        retained_finals = []
        coalesced = 0
        while True:
            try:
                pending = queue.get_nowait()
            except Empty:
                break
            if pending.final:
                retained_finals.append(pending)
            else:
                coalesced += 1
        for pending in retained_finals:
            queue.put(pending)
        try:
            queue.put_nowait(job)
        except Full:
            # A queue containing only required finals has priority over a live
            # hypothesis; the next snapshot can try again after it drains.
            LOGGER.debug("Dropped partial behind final backlog | mode=%s segment=%s",
                         mode.value, job.utterance_id)
            return
        if coalesced:
            LOGGER.debug("Coalesced stale partials | mode=%s segment=%s count=%s",
                         mode.value, job.utterance_id, coalesced)

    def _run_mode(self, mode: PerformanceMode, queue: Queue) -> None:
        config = AppConfig(**{**self.config.__dict__, "performance_mode": mode})
        config.model_size = config.profile.model_size
        engine = self.model_provider.get(config)
        process = Process()
        while True:
            job = queue.get()
            if job is None:
                return
            self.signals.mode_status.emit(job.utterance_id, mode.value, "Processing")
            started = time.monotonic()
            try:
                text, language, script = _unpack_transcription(
                    engine.transcribe(job, " ".join(self.histories[mode])))
                elapsed = time.monotonic() - started
                result_latency = (time.monotonic() - job.captured_at
                                  if job.captured_at > 0 else elapsed)
                duration = len(job.audio) / config.sample_rate
                late = result_latency >= config.max_result_latency_seconds
                script_valid = script.get("script_valid", True)
                duplicate = bool(
                    text and job.final and script_valid and self.histories[mode] and
                    clean_text(self.histories[mode][-1]).casefold() ==
                    clean_text(text).casefold())
                if text and job.final and script_valid and not duplicate:
                    self.histories[mode].append(text)
                metrics = {
                    "segment_id": job.utterance_id,
                    "mode": mode.value,
                    "asr_time": elapsed,
                    "queue_delay": (max(0.0, started - job.captured_at)
                                    if job.captured_at > 0 else 0.0),
                    "result_latency": result_latency,
                    "final_latency": result_latency if job.final else None,
                    "first_partial_latency": result_latency if not job.final and text else None,
                    "real_time_factor": elapsed / max(duration, .001),
                    "cpu_percent": process.cpu_percent(None),
                    "memory_mb": process.memory_info().rss / 1024 ** 2,
                    "language": language, "start_time": job.audio_start_time,
                    "end_time": job.audio_end_time,
                    "duplicate": duplicate,
                    **script,
                }
                # The latency target is diagnostic, not a destructive timeout.
                # Slow CPU inference cannot be cancelled safely, so throwing its
                # eventual transcript away leaves a permanent blank row after
                # the user has already waited for it.
                display_text = text if script_valid and not duplicate else ""
                if display_text or job.final:
                    self.signals.mode_text.emit(
                        job.utterance_id, mode.value, display_text, job.final, metrics)
                if late:
                    LOGGER.warning(
                        "ASR result exceeded latency target but was retained | "
                        "segment=%s mode=%s latency=%.2fs target=%.2fs",
                        job.utterance_id, mode.value, result_latency,
                        config.max_result_latency_seconds)
                if job.final:
                    LOGGER.info(
                        "Comparison final | segment=%s mode=%s text=%r inference=%.2fs",
                        job.utterance_id, mode.value, text, elapsed)
                self.signals.mode_status.emit(
                    job.utterance_id, mode.value,
                    "Script mismatch" if not script_valid else
                    "Duplicate" if duplicate else
                    "Final" if job.final else "Partial" if text else "Listening")
            except Exception as exc:
                log_exception(f"ASR-{mode.value.upper()}", exc)
                self.signals.mode_error.emit(job.utterance_id, mode.value, str(exc))
