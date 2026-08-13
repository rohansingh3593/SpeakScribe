"""Single-model Faster-Whisper inference worker."""

from collections import deque
from difflib import SequenceMatcher
from queue import Empty, Full, Queue
from threading import Event, Lock, Thread
from pathlib import Path
import time
import wave

import numpy as np

from faster_whisper import WhisperModel
from psutil import Process

from app.audio.audio_pipeline import (
    ASRJob, audio_statistics, prepare_audio_for_asr,
)
from app.config.settings import AppConfig
from app.config.settings import PerformanceMode
from app.config.decoding_policy import hotwords, initial_prompt, retry_thresholds
from app.utils.logger import get_logger, log_exception
from app.utils.pipeline_diagnostics import RootCause, pipeline_diagnostics
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
        self.last_stage_timings: dict[str, float] = {}

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
        transcription_started = time.monotonic()
        profile = self.config.profile
        # Repeated partials must be inexpensive. The selected profile is retained
        # for final correction, while live hypotheses use deterministic greedy
        # decoding to keep CPU fallback latency close to real time.
        beam_size = profile.beam_size if job.final else 1
        best_of = profile.best_of if job.final else 1
        prompt = initial_prompt(
            final=job.final, sample_count=len(job.audio), sample_rate=self.config.sample_rate,
            language_mode=job.language, vocabulary=self.config.vocabulary,
            context=context,
        )
        vocabulary_bias = hotwords(
            final=job.final, language_mode=job.language,
            vocabulary=self.config.vocabulary,
        )
        language = None if job.language == "auto" else job.language
        preprocessing_started = time.monotonic()
        # Preparation already computes the centered signal and normalization
        # gain. Return that gain rather than creating a second full-size
        # centered array solely for diagnostics on every live snapshot.
        prepared, normalization_gain = prepare_audio_for_asr(
            job.audio, return_gain=True)
        raw_stats = audio_statistics(job.audio)
        prepared_stats = audio_statistics(prepared)
        preprocessing_seconds = time.monotonic() - preprocessing_started
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
            f"language={language or 'auto'} script={job.script} "
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
        inference_seconds = 0.0
        text_processing_seconds = 0.0

        def decode(initial_prompt, word_bias, *, relaxed=False):
            nonlocal inference_seconds, text_processing_seconds
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
            inference_started = time.monotonic()
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
            inference_seconds += time.monotonic() - inference_started
            text_started = time.monotonic()
            if not accepted:
                LOGGER.debug(f"[ASR] no usable segments returned (segments={segment_count})")
            raw_text = " ".join(accepted).strip()
            decoded = clean_text(raw_text, final=job.final)
            LOGGER.debug(f"[ASR-TEXT] accepted={len(accepted)}/{segment_count} cleaned={decoded!r}")
            text_processing_seconds += time.monotonic() - text_started
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
            after_language, job.script, self.config.vocabulary)
        text = clean_text(after_transliteration, final=job.final)
        language_script_started = time.monotonic()
        language_probability = float(getattr(info, "language_probability", 0.0) or 0.0)
        metadata = script_metadata(text, job.script, job.language)
        # A non-empty decode can still be unusable for Hindi (Arabic script or
        # an English hallucination). Give finalized audio one prompt-free,
        # relaxed recovery pass before declaring that no safe speech was found.
        if (job.final and job.language == "hi" and text and
                not metadata["script_valid"]):
            first_raw = raw_text
            LOGGER.info("Retrying Hindi final after script/language mismatch | utterance=%s",
                        job.utterance_id)
            retry_text, retry_info, retry_raw = decode(None, None, relaxed=True)
            retry_after_script = apply_script_mode(
                retry_text, job.script, self.config.vocabulary)
            retry_text = clean_text(retry_after_script, final=True)
            retry_metadata = script_metadata(
                retry_text, job.script, job.language)
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
            "detected_language": getattr(info, "language", None) or job.language,
            "language_probability": language_probability,
        })
        LOGGER.debug(
            "[ASR-PIPELINE] raw=%r detected_language=%r recognition=%s script_mode=%s "
            "after_language=%r after_transliteration=%r after_normalization=%r "
            "translation=%s final_candidate=%r detected_script=%s script_valid=%s",
            raw_text, getattr(info, "language", None), job.language,
            job.script, after_language, after_transliteration, text,
            "enabled-downstream" if self.config.translation_enabled else "off",
            text, metadata["detected_script"], metadata["script_valid"])
        if not metadata["script_valid"]:
            LOGGER.warning(
                "Unexpected Arabic/Urdu or mismatched script detected for Hindi/Hinglish "
                "segment | utterance=%s raw=%r detected=%s requested=%s",
                job.utterance_id, raw_text, metadata["detected_script"],
                job.script)
        LOGGER.debug(f"[ASR-TEXT] post_script={text!r} detected={getattr(info, 'language', None)!r} "
                  f"probability={language_probability:.3f}")
        mode = detect_language(text, getattr(info, "language", None))
        language_script_seconds = time.monotonic() - language_script_started
        self.last_stage_timings = {
            "asr_preprocessing": preprocessing_seconds,
            "whisper_inference": inference_seconds,
            "text_processing": text_processing_seconds,
            "language_script_processing": language_script_seconds,
            "asr_total": time.monotonic() - transcription_started,
        }
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
                 model_provider: WhisperModelProvider, recognition_state=None):
        self.config, self.queue, self.stop_event, self.signals = config, queue, stop_event, signals
        self.model_provider = model_provider
        self.recognition_state = recognition_state
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
            while True:
                try:
                    job = self.queue.get(timeout=0.1)
                except Empty:
                    if not self.stop_event.is_set():
                        continue
                    break
                if (self.recognition_state is not None and
                        not self.recognition_state.is_current(job.language_generation)):
                    LOGGER.info("[GENERATION] cancelled queued stale ASR job generation=%s "
                                "utterance=%s", job.language_generation, job.utterance_id)
                    pipeline_diagnostics().terminal(
                        job.utterance_id, "REJECTED", "stale_generation_before_asr",
                        RootCause.GENERATION_FILTER, job_generation=job.language_generation,
                        current_generation=self.recognition_state.snapshot().generation)
                    continue
                started = time.monotonic()
                pipeline_diagnostics().stage(
                    job.utterance_id, "FAST START", final=job.final,
                    audio_duration=len(job.audio) / self.config.sample_rate,
                    queue_wait_ms=max(0, (started - job.captured_at) * 1000),
                    queue_depth=self.queue.qsize())
                text, language, metadata = _unpack_transcription(
                    engine.transcribe(job, " ".join(self.history)))
                stale = (self.recognition_state is not None and
                         not self.recognition_state.is_current(job.language_generation))
                if stale:
                    LOGGER.info("[GENERATION] ignored stale ASR result generation=%s "
                                "current=%s utterance=%s", job.language_generation,
                                self.recognition_state.snapshot().generation,
                                job.utterance_id)
                    pipeline_diagnostics().terminal(
                        job.utterance_id, "REJECTED", "stale_generation_after_asr",
                        RootCause.GENERATION_FILTER, job_generation=job.language_generation,
                        current_generation=self.recognition_state.snapshot().generation)
                    continue
                if not metadata.get("script_valid", True):
                    pipeline_diagnostics().terminal(
                        job.utterance_id, "REJECTED", "script_language_mismatch",
                        RootCause.LANGUAGE_DETECTION,
                        detected_language=metadata.get("detected_language"),
                        detected_script=metadata.get("detected_script"))
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
                    if job.final and metadata.get("script_valid", True):
                        pipeline_diagnostics().terminal(
                            job.utterance_id, "REJECTED", "empty_asr_result",
                            RootCause.FAST_ASR, audio_duration=duration,
                            inference_ms=elapsed * 1000)
                    continue
                diagnostics = pipeline_diagnostics()
                diagnostics.stage(
                    job.utterance_id, "FAST RESULT", final=job.final,
                    requested_language=job.language,
                    detected_language=metadata.get("detected_language", language),
                    language_probability=metadata.get("language_probability"),
                    audio_duration=duration,
                    queue_wait_ms=max(0, (started - job.captured_at) * 1000),
                    inference_ms=elapsed * 1000,
                    rtf=elapsed / max(duration, .001), text=text)
                diagnostics.fast_seconds.append(elapsed)
                if not stale:
                    self.signals.language_changed.emit(language)
                if job.final:
                    if not stale:
                        self.history.append(text)
                        self.signals.final_text.emit(text)
                        self.signals.partial_text.emit("")
                        diagnostics.stage(job.utterance_id, "FINALIZE",
                                          reason="speech_end", source="FAST")
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
                 model_provider: WhisperModelProvider, recognition_state=None):
        self.config, self.queue, self.stop_event = config, queue, stop_event
        self.signals, self.model_provider = signals, model_provider
        self.recognition_state = recognition_state
        self.histories = {mode: deque(maxlen=AppConfig(
            performance_mode=mode).context_sentences) for mode in PerformanceMode}
        self.first_seen: dict[tuple[int, PerformanceMode], float] = {}
        self.latest_partials: dict[tuple[int, PerformanceMode], tuple[str, str, dict]] = {}

    def run(self) -> None:
        # Keep the live user-facing path stable while still allowing partial
        # comparison jobs to fan out for diagnostics. Finalized audio remains
        # routed through FAST to avoid starving the active capture session on
        # CPU-only systems, while partial snapshots can still be compared across
        # modes when the comparison worker is active.
        active_modes = (tuple(PerformanceMode) if self.config.compare_live_partials
                        else (PerformanceMode.FAST,))
        queues = {mode: Queue(maxsize=2) for mode in active_modes}
        workers = [Thread(target=self._run_mode,
                          args=(mode, queues[mode]),
                          name=f"asr-{mode.value}", daemon=True)
                   for mode in active_modes]
        LOGGER.info("Live ASR scheduler active | pipeline=fast-live-v2 modes=%s "
                    "partial_queue=latest final_queue=latest",
                    ",".join(mode.value for mode in active_modes))
        for worker in workers:
            worker.start()
        while True:
            try:
                job = self.queue.get(timeout=0.1)
            except Empty:
                if not self.stop_event.is_set():
                    continue
                break
            if (self.recognition_state is not None and
                    not self.recognition_state.is_current(job.language_generation)):
                LOGGER.info("[GENERATION] cancelled stale scheduler job generation=%s "
                            "utterance=%s", job.language_generation, job.utterance_id)
                pipeline_diagnostics().terminal(
                    job.utterance_id, "REJECTED", "stale_scheduler_generation",
                    RootCause.GENERATION_FILTER,
                    job_generation=job.language_generation,
                    current_generation=self.recognition_state.snapshot().generation)
                continue
            if job.final:
                self._enqueue_mode_job(PerformanceMode.FAST, queues[PerformanceMode.FAST], job)
            else:
                for mode in active_modes:
                    self._enqueue_mode_job(mode, queues[mode], job)
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

        # A partial is a replaceable snapshot, not an ordered event. Retain
        # finals, discard every obsolete pending partial, and enqueue only the
        # freshest snapshot so a late decoder cannot resurrect stale text.
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
            LOGGER.debug("Dropped partial behind final backlog | mode=%s segment=%s",
                         mode.value, job.utterance_id)
            pipeline_diagnostics().event(
                "DISCARD", job.utterance_id, stage_name="QUEUE", mode=mode.value,
                reason="partial_behind_final_backlog")
            return
        if coalesced:
            LOGGER.debug("Coalesced stale partials | mode=%s segment=%s count=%s",
                         mode.value, job.utterance_id, coalesced)
            pipeline_diagnostics().event(
                "DISCARD", job.utterance_id, stage_name="QUEUE", mode=mode.value,
                reason="obsolete_partial_coalesced", count=coalesced)

    def _run_mode(self, mode: PerformanceMode, queue: Queue) -> None:
        config = AppConfig(**{**self.config.__dict__, "performance_mode": mode})
        config.model_size = config.profile.model_size
        engine = self.model_provider.get(config)
        process = Process()
        while True:
            job = queue.get()
            if job is None:
                return
            if (self.recognition_state is not None and
                    not self.recognition_state.is_current(job.language_generation)):
                self.signals.mode_status.emit(job.utterance_id, mode.value, "Expired")
                LOGGER.info("[GENERATION] cancelled queued %s refinement generation=%s "
                            "utterance=%s", mode.value, job.language_generation,
                            job.utterance_id)
                continue
            self.signals.mode_status.emit(job.utterance_id, mode.value, "Processing")
            started = time.monotonic()
            pipeline_diagnostics().stage(
                job.utterance_id, f"{mode.value.upper()} START", final=job.final,
                audio_duration=len(job.audio) / config.sample_rate,
                queue_wait_ms=max(0, (started - job.captured_at) * 1000),
                queue_depth=queue.qsize())
            try:
                current = (self.recognition_state is None or
                           self.recognition_state.is_current(job.language_generation))
                context = " ".join(self.histories[mode]) if current else ""
                # Context is generation-local: the first decode after a switch is
                # always prompt-free with respect to recognized prior speech.
                if getattr(self, "_history_generation", {}).get(mode) != job.language_generation:
                    context = ""
                text, language, script = _unpack_transcription(
                    engine.transcribe(job, context))
                current = (self.recognition_state is None or
                           self.recognition_state.is_current(job.language_generation))
                if not current:
                    LOGGER.info("[GENERATION] ignored stale comparison result generation=%s "
                                "utterance=%s mode=%s", job.language_generation,
                                job.utterance_id, mode.value)
                    self.signals.mode_status.emit(
                        job.utterance_id, mode.value, "Expired")
                    pipeline_diagnostics().terminal(
                        job.utterance_id, "REJECTED", "stale_refinement_generation",
                        RootCause.GENERATION_FILTER, mode=mode.value,
                        job_generation=job.language_generation)
                    continue
                if current:
                    if not hasattr(self, "_history_generation"):
                        self._history_generation = {}
                    if self._history_generation.get(mode) != job.language_generation:
                        self.histories[mode].clear()
                        self._history_generation[mode] = job.language_generation
                        self.latest_partials.clear()
                partial_key = (job.utterance_id, mode)
                recovered_from_partial = False
                if not job.final and text and script.get("script_valid", True):
                    self.latest_partials[partial_key] = (text, language, script)
                elif (job.final and partial_key in self.latest_partials and
                      (not text or not script.get("script_valid", True))):
                    # A safe partial is better evidence than an empty or
                    # wrong-script final decode of the exact same audio. This
                    # is the lifecycle guarantee behind Processing → Final:
                    # final confidence/script failure must not strand already
                    # accepted Hindi solely in the temporary panel.
                    text, language, script = self.latest_partials[partial_key]
                    recovered_from_partial = True
                    LOGGER.info("Promoted latest valid partial after unusable final | "
                                "segment=%s mode=%s", job.utterance_id, mode.value)
                if job.final:
                    self.latest_partials.pop(partial_key, None)
                elapsed = time.monotonic() - started
                result_latency = (time.monotonic() - job.captured_at
                                  if job.captured_at > 0 else elapsed)
                duration = len(job.audio) / config.sample_rate
                late = result_latency >= config.max_result_latency_seconds
                script_valid = script.get("script_valid", True)
                # Different utterance IDs are different speech events. Do not
                # discard a legitimate repeated Hindi sentence merely because
                # its normalized text matches the preceding utterance.
                duplicate = False
                if text and job.final:
                    existing = self.histories[mode][-1] if self.histories[mode] else ""
                    similarity = SequenceMatcher(
                        None, clean_text(existing).casefold(),
                        clean_text(text).casefold(), autojunk=False).ratio() if existing else 0.0
                    pipeline_diagnostics().stage(
                        job.utterance_id, "DEDUP", candidate=text, existing=existing,
                        similarity=similarity,
                        decision="KEEP", reason="distinct_utterance_id")
                if text and job.final and script_valid and not duplicate and current:
                    self.histories[mode].append(text)
                stage_timings = getattr(engine, "last_stage_timings", {})
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
                    "recovered_from_partial": recovered_from_partial,
                    "candidate_speech_at": job.candidate_speech_at,
                    "vad_wait": (max(0.0, job.vad_activated_at - job.candidate_speech_at)
                                 if job.candidate_speech_at > 0 and
                                 job.vad_activated_at > 0 else None),
                    "audio_buffer": (max(0.0, job.captured_at - job.vad_activated_at)
                                     if job.vad_activated_at > 0 else None),
                    **stage_timings,
                    **script,
                }
                if job.language_switched_at:
                    metrics.update({
                        "language_generation": job.language_generation,
                        "language_switch_requested_at": job.language_switched_at,
                        "language_ready_at": job.language_ready_at,
                        "first_new_speech_at": job.candidate_speech_at or None,
                        "first_fast_result_at": (time.monotonic()
                                                 if mode is PerformanceMode.FAST else None),
                    })
                # The latency target is diagnostic, not a destructive timeout.
                # Slow CPU inference cannot be cancelled safely, so throwing its
                # eventual transcript away leaves a permanent blank row after
                # the user has already waited for it.
                display_text = text if script_valid and not duplicate else ""
                diagnostics = pipeline_diagnostics()
                diagnostics.stage(
                    job.utterance_id,
                    "FAST RESULT" if mode is PerformanceMode.FAST else "REFINEMENT",
                    mode=mode.value, final=job.final, requested_language=job.language,
                    detected_language=script.get("detected_language", language),
                    language_probability=script.get("language_probability"),
                    audio_duration=duration,
                    queue_wait_ms=max(0, (started - job.captured_at) * 1000),
                    inference_ms=elapsed * 1000,
                    result_latency_ms=result_latency * 1000,
                    rtf=elapsed / max(duration, .001), text=text,
                    accepted=bool(display_text), script_valid=script_valid,
                    recovered_from_partial=recovered_from_partial)
                if mode is PerformanceMode.FAST:
                    diagnostics.fast_seconds.append(elapsed)
                if display_text:
                    metrics["signal_emitted_at"] = time.monotonic()
                    self.signals.mode_text.emit(
                        job.utterance_id, mode.value, display_text, job.final, metrics)
                    if job.final:
                        diagnostics.stage(job.utterance_id, "FINALIZE",
                                          reason="speech_end", source=mode.value.upper())
                elif job.final:
                    diagnostics.terminal(
                        job.utterance_id, "REJECTED",
                        "duplicate" if duplicate else
                        "script_language_mismatch" if not script_valid else "empty_asr_result",
                        RootCause.DEDUPLICATION if duplicate else
                        RootCause.LANGUAGE_DETECTION if not script_valid else RootCause.FAST_ASR,
                        mode=mode.value)
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
                    "Final" if job.final and text else
                    "No speech" if job.final else
                    "Partial" if text else
                    "Listening" if mode is PerformanceMode.FAST else "Processing")
            except Exception as exc:
                log_exception(f"ASR-{mode.value.upper()}", exc)
                self.signals.mode_error.emit(job.utterance_id, mode.value, str(exc))
