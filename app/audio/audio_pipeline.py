"""Continuous SoundCard capture and energy-based utterance segmentation."""

from collections import deque
from dataclasses import dataclass
from importlib import import_module
from queue import Empty, Full, Queue
from threading import Event
import time
import warnings

import numpy as np

from speakscribe.audio.processor import audio_normalization_gain, prepare_audio_for_asr

from app.config.settings import AppConfig
from app.utils.logger import get_logger, log_exception
from app.utils.pipeline_diagnostics import RootCause, pipeline_diagnostics


LOGGER = get_logger("audio")

@dataclass(frozen=True)
class ASRJob:
    audio: np.ndarray
    final: bool
    utterance_id: int
    captured_at: float
    speech_seconds: float | None = None
    audio_start_time: float = 0.0
    audio_end_time: float = 0.0
    candidate_speech_at: float = 0.0
    vad_activated_at: float = 0.0
    language: str = "hi"
    script: str = "original"
    language_generation: int = 0
    language_switched_at: float = 0.0
    language_ready_at: float = 0.0


def select_capture_device(soundcard, source: str):
    """Resolve a physical microphone or the legacy speaker-loopback source."""
    if source not in {"loopback", "microphone"}:
        raise ValueError(f"Unsupported capture source: {source}")
    if source == "loopback":
        speaker = soundcard.default_speaker()
        if speaker is None:
            raise RuntimeError("No default speaker is available for loopback capture")
        device = soundcard.get_microphone(id=str(speaker.name), include_loopback=True)
        if device is None:
            raise RuntimeError(f"No loopback capture device is available for {speaker.name}")
        return device, f"speaker-loopback:{speaker.name}", None
    device = soundcard.default_microphone()
    if device is None:
        raise RuntimeError("No default microphone is available")
    return device, f"microphone:{device.name}", 1


def resample_audio_block(audio: np.ndarray, source_rate: int,
                         target_rate: int) -> np.ndarray:
    """Downsample one capture block while preserving its exact duration."""
    audio = np.asarray(audio, dtype=np.float32)
    if source_rate == target_rate or audio.size == 0:
        return audio
    if source_rate % target_rate == 0:
        factor = source_rate // target_rate
        usable = audio.size - (audio.size % factor)
        # Averaging provides a cheap anti-alias filter for the native 48k -> 16k
        # path and is substantially safer than asking WASAPI to convert live.
        return audio[:usable].reshape(-1, factor).mean(axis=1, dtype=np.float32)
    output_size = max(1, round(audio.size * target_rate / source_rate))
    positions = np.linspace(0, audio.size - 1, output_size)
    return np.interp(positions, np.arange(audio.size), audio).astype(np.float32)


def audio_statistics(audio: np.ndarray) -> dict[str, float | int | bool]:
    """Return compact diagnostics without retaining or copying capture blocks."""
    values = np.asarray(audio, dtype=np.float32)
    if values.size == 0:
        return {"samples": 0, "rms": 0.0, "peak": 0.0, "mean": 0.0,
                "zero_ratio": 1.0, "finite": True}
    finite = bool(np.isfinite(values).all())
    safe = values if finite else np.nan_to_num(values)
    return {
        "samples": int(safe.size),
        "rms": float(np.sqrt(np.mean(np.square(safe), dtype=np.float64))),
        "peak": float(np.max(np.abs(safe))),
        "mean": float(np.mean(safe, dtype=np.float64)),
        "zero_ratio": float(np.count_nonzero(safe == 0) / safe.size),
        "finite": finite,
    }


class EnergySpeechDetector:
    """Hysteretic RMS detector, isolated so another VAD can replace it later."""

    def __init__(self, config: AppConfig):
        self.config = config
        self.speaking = False
        self.start_frames = 0
        self.noise_floor = config.adaptive_vad_floor
        self.effective_start_threshold = config.speech_threshold
        self.effective_silence_threshold = config.silence_threshold

    def classify(self, frame: np.ndarray) -> tuple[bool, float]:
        rms = float(np.sqrt(np.mean(np.square(frame), dtype=np.float64)))
        if self.speaking:
            active = rms >= self.effective_silence_threshold
        else:
            # A fixed threshold makes quiet microphones appear completely
            # silent. Track only sub-threshold frames and cap the adaptive
            # threshold at the configured value so normal/loud capture keeps
            # its historical behaviour.
            if self.config.adaptive_vad_enabled:
                if rms < self.config.speech_threshold and self.start_frames == 0:
                    self.noise_floor = 0.95 * self.noise_floor + 0.05 * rms
                self.effective_start_threshold = min(
                    self.config.speech_threshold,
                    max(self.config.adaptive_vad_floor,
                        self.noise_floor * self.config.adaptive_vad_multiplier),
                )
                self.effective_silence_threshold = min(
                    self.config.silence_threshold,
                    self.effective_start_threshold * 0.7,
                )
            else:
                self.effective_start_threshold = self.config.speech_threshold
                self.effective_silence_threshold = self.config.silence_threshold
            self.start_frames = (self.start_frames + 1
                                 if rms >= self.effective_start_threshold else 0)
            active = self.start_frames >= self.config.speech_start_frames
            if active:
                self.speaking = True
        return active, rms

    def reset(self) -> None:
        self.speaking = False
        self.start_frames = 0


class AudioCaptureWorker:
    def __init__(self, config: AppConfig, output: Queue, stop_event: Event,
                 on_error):
        self.config, self.output, self.stop_event = config, output, stop_event
        self.on_error = on_error

    def run(self) -> None:
        try:
            # SoundCard is a capture-only dependency. Importing it here keeps the
            # NumPy speech detector and buffer classes usable by tests, tooling,
            # and offline processing without initializing a platform audio
            # backend during module collection.
            sc = import_module("soundcard")
            mediafoundation = import_module("soundcard.mediafoundation")
            soundcard_warning = getattr(
                mediafoundation, "SoundcardRuntimeWarning", RuntimeWarning)
            microphone, source_description, recorder_channels = select_capture_device(
                sc, self.config.capture_source)
            LOGGER.info(
                "Audio capture opening | source=%s device=%s rate=%s channels=%s chunk_ms=%s",
                source_description, microphone.name, self.config.capture_sample_rate,
                recorder_channels or "all",
                self.config.capture_chunk_ms,
            )
            with microphone.recorder(samplerate=self.config.capture_sample_rate,
                                     channels=recorder_channels) as recorder:
                LOGGER.info("Audio stream warming up | blocks=%s block_ms=%s",
                            self.config.capture_warmup_blocks,
                            self.config.capture_warmup_ms)
                warmup_discontinuities = 0
                for _ in range(self.config.capture_warmup_blocks):
                    with warnings.catch_warnings(record=True) as warmup_warnings:
                        warnings.simplefilter("always", soundcard_warning)
                        recorder.record(numframes=(self.config.capture_sample_rate *
                                                   self.config.capture_warmup_ms // 1000))
                    warmup_discontinuities += len(warmup_warnings)
                if warmup_discontinuities:
                    LOGGER.warning(
                        "Audio backend reported %s warm-up discontinuities; discarded warm-up "
                        "samples as expected", warmup_discontinuities)
                LOGGER.info("Audio stream warm-up complete | source=%s", source_description)
                next_diagnostic = time.monotonic()
                next_warning_log = 0.0
                next_queue_warning = 0.0
                dropped_audio_frames = 0
                first_block = True
                while not self.stop_event.is_set():
                    with warnings.catch_warnings(record=True) as captured_warnings:
                        warnings.simplefilter("always", soundcard_warning)
                        block = recorder.record(numframes=self.config.capture_chunk_samples)
                    now = time.monotonic()
                    if captured_warnings and now >= next_warning_log:
                        LOGGER.warning(
                            "Audio backend reported a data discontinuity; capture continues "
                            "with the newest samples | device=%s chunk_ms=%s",
                            microphone.name, self.config.capture_chunk_ms,
                        )
                        next_warning_log = now + 5.0
                    samples = np.asarray(block, dtype=np.float32)
                    raw_frame = (samples if samples.ndim == 1 else
                                 np.mean(samples, axis=1, dtype=np.float32))
                    resampled = resample_audio_block(
                        raw_frame, self.config.capture_sample_rate,
                        self.config.sample_rate)
                    frame_samples = self.config.frame_samples
                    frames = [resampled[index:index + frame_samples]
                              for index in range(0, len(resampled), frame_samples)
                              if len(resampled[index:index + frame_samples]) == frame_samples]
                    if not frames:
                        LOGGER.warning("Audio backend returned an incomplete capture block")
                        continue
                    if first_block:
                        stats = audio_statistics(resampled)
                        LOGGER.info(
                            "Audio capture active | source=%s device=%s samples=%s "
                            "rms=%.6f peak=%.6f",
                            source_description, microphone.name, stats["samples"],
                            stats["rms"], stats["peak"],
                        )
                        first_block = False
                    if now >= next_diagnostic:
                        raw_stats = audio_statistics(raw_frame)
                        asr_stats = audio_statistics(resampled)
                        LOGGER.debug(
                            "[AUDIO] capture "
                            f"raw_shape={tuple(block.shape)} raw_rate={self.config.capture_sample_rate} "
                            f"raw_rms={raw_stats['rms']:.6f} raw_peak={raw_stats['peak']:.6f} "
                            f"asr_samples={asr_stats['samples']} asr_rms={asr_stats['rms']:.6f} "
                            f"asr_peak={asr_stats['peak']:.6f} mean={asr_stats['mean']:.6f} "
                            f"zeros={asr_stats['zero_ratio']:.3f} finite={asr_stats['finite']} "
                            f"audio_queue={self.output.qsize()}"
                        )
                        diagnostics = pipeline_diagnostics()
                        diagnostics.audio_seconds += len(resampled) / self.config.sample_rate
                        diagnostics.event(
                            "AUDIO", None, device=source_description,
                            sample_rate=self.config.sample_rate, channels=1,
                            frames_received=asr_stats["samples"],
                            duration_ms=round(1000 * len(resampled) / self.config.sample_rate),
                            rms=asr_stats["rms"], peak=asr_stats["peak"],
                            audio_received_sec=round(diagnostics.audio_seconds, 3),
                            audio_dropped_frames=dropped_audio_frames)
                        next_diagnostic = now + self.config.debug_log_interval
                    for frame in frames:
                        try:
                            self.output.put(frame, timeout=0.05)
                        except Full:
                            try:
                                self.output.get_nowait()  # discard only oldest raw frame
                            except Empty:
                                pass
                            self.output.put_nowait(frame)
                            dropped_audio_frames += 1
                            pipeline_diagnostics().event(
                                "DISCARD", None, stage_name="AUDIO_CAPTURE",
                                reason="audio_queue_full_oldest_frame",
                                dropped_frames=dropped_audio_frames)
                            if now >= next_queue_warning:
                                LOGGER.warning(
                                    "Audio queue full; dropped %s oldest raw frame(s) since "
                                    "the previous warning", dropped_audio_frames)
                                dropped_audio_frames = 0
                                next_queue_warning = now + 5.0
        except Exception as exc:
            log_exception("CAPTURE", exc)
            self.on_error(str(exc))


class SpeechBufferWorker:
    def __init__(self, config: AppConfig, audio_queue: Queue, asr_queue: Queue,
                 stop_event: Event, recognition_state=None):
        self.config, self.audio_queue, self.asr_queue = config, audio_queue, asr_queue
        self.stop_event = stop_event
        self.recognition_state = recognition_state
        self.detector = EnergySpeechDetector(config)
        self.utterance_id = 0

    def _submit(self, job: ASRJob) -> None:
        diagnostics = pipeline_diagnostics()
        diagnostics.stage(
            job.utterance_id, "QUEUE", action="SUBMIT", mode="FAST",
            final=job.final, depth=self.asr_queue.qsize(),
            audio_duration=len(job.audio) / self.config.sample_rate,
            generation=job.language_generation, language=job.language)
        LOGGER.debug(
            f"[QUEUE] submit final={job.final} utterance={job.utterance_id} "
            f"audio={len(job.audio) / self.config.sample_rate:.2f}s "
            f"voiced={job.speech_seconds if job.speech_seconds is not None else -1:.2f}s "
            f"asr_queue_before={self.asr_queue.qsize()}"
        )
        if job.final:
            # A final supersedes queued partials for its utterance. Evict those
            # obsolete hypotheses so CPU fallback does not spend several seconds
            # decoding text that the final will immediately replace.
            try:
                pending = self.asr_queue.get_nowait()
            except Empty:
                pending = None
            if pending is not None and pending.final:
                if self.config.asr_keep_latest_final:
                    LOGGER.warning(
                        "ASR is behind; replaced stale final utterance=%s with newest "
                        "utterance=%s", pending.utterance_id, job.utterance_id)
                    diagnostics.terminal(
                        pending.utterance_id, "REJECTED", "replaced_by_newest_final",
                        RootCause.QUEUE, replacement=self.utterance_id)
                    self.asr_queue.put_nowait(job)
                    return
                self.asr_queue.put_nowait(pending)
            elif pending is not None:
                LOGGER.debug(f"[QUEUE] evicted obsolete partial utterance={pending.utterance_id}")
                diagnostics.event("DISCARD", pending.utterance_id,
                                  stage_name="QUEUE", reason="final_supersedes_partial")

            # Existing finals are never evicted; wait briefly for inference.
            deadline = time.monotonic() + 0.5 if self.stop_event.is_set() else None
            while deadline is None or time.monotonic() < deadline:
                try:
                    self.asr_queue.put(job, timeout=0.1)
                    return
                except Full:
                    continue
        else:
            try:
                self.asr_queue.put_nowait(job)
            except Full:
                # Replace an obsolete partial, but never evict a final.
                try:
                    pending = self.asr_queue.get_nowait()
                except Empty:
                    diagnostics.event("DISCARD", job.utterance_id,
                                      stage_name="QUEUE", reason="queue_race_empty_partial")
                    return
                if pending.final:
                    self.asr_queue.put_nowait(pending)
                    if (self.recognition_state is not None and
                            pending.language_generation != job.language_generation):
                        # Preserve the required old final, but put the first new
                        # generation snapshot ahead of it. Queue.Queue has no
                        # public front-insert operation, so perform the same
                        # mutation as put() under its own condition lock. The
                        # temporary one-item overflow is intentional and bounded.
                        with self.asr_queue.not_full:
                            self.asr_queue.queue.appendleft(job)
                            self.asr_queue.unfinished_tasks += 1
                            self.asr_queue.not_empty.notify()
                        LOGGER.info("[LANG-SWITCH] prioritized generation=%s partial ahead "
                                    "of protected old final", job.language_generation)
                        return
                    LOGGER.debug("[QUEUE] kept queued final; dropped new partial")
                    diagnostics.event("DISCARD", job.utterance_id,
                                      stage_name="QUEUE", reason="final_backlog_partial_replaceable")
                    return
                try:
                    self.asr_queue.put_nowait(job)
                except Full:
                    diagnostics.event("DISCARD", job.utterance_id,
                                      stage_name="QUEUE", reason="queue_still_full")

    def run(self) -> None:
        frame_seconds = self.config.frame_ms / 1000
        pre = deque(maxlen=max(1, round(self.config.pre_speech_duration / frame_seconds)))
        speech: list[np.ndarray] = []
        voiced_duration = 0.0
        silence = 0.0
        last_partial = 0.0
        diagnostic_rms: list[float] = []
        next_diagnostic = time.monotonic() + self.config.debug_log_interval
        stream_seconds = 0.0
        utterance_start = 0.0
        candidate_speech_at = 0.0
        vad_activated_at = 0.0
        recognition = (self.recognition_state.snapshot()
                       if self.recognition_state is not None else None)
        # Stop is a live-session boundary, not a request to drain captured audio.
        # Frames remaining in this session's private queue are intentionally stale.
        while not self.stop_event.is_set():
            try:
                frame = self.audio_queue.get(timeout=0.1)
            except Empty:
                continue
            current = (self.recognition_state.snapshot()
                       if self.recognition_state is not None else recognition)
            if (recognition is not None and
                    current.generation != recognition.generation):
                # A switch is a hard utterance/audio boundary.  Captured old-language
                # audio is deliberately not reused as new-language pre-roll.
                speech.clear()
                pre.clear()
                voiced_duration = silence = 0.0
                candidate_speech_at = vad_activated_at = 0.0
                self.detector.reset()
                recognition = current
                self.utterance_id += 1
                LOGGER.info("[LANG-SWITCH] old utterance closed; rolling audio and VAD reset "
                            "generation=%s", current.generation)
            # Classify the boundary frame only after reset and retain it as the
            # first clean pre-roll frame for the new generation.
            active, rms = self.detector.classify(frame)
            stream_seconds += frame_seconds
            now = time.monotonic()
            if not speech and self.detector.start_frames == 1:
                candidate_speech_at = now
            diagnostic_rms.append(rms)
            if now >= next_diagnostic:
                LOGGER.debug(
                    "[VAD] "
                    f"rms_min={min(diagnostic_rms):.6f} "
                    f"rms_avg={sum(diagnostic_rms) / len(diagnostic_rms):.6f} "
                    f"rms_max={max(diagnostic_rms):.6f} "
                    f"start_threshold={self.detector.effective_start_threshold:.6f} "
                    f"configured_threshold={self.config.speech_threshold:.6f} "
                    f"noise_floor={self.detector.noise_floor:.6f} "
                    f"continue_threshold={self.detector.effective_silence_threshold:.6f} "
                    f"start_frames={self.detector.start_frames}/{self.config.speech_start_frames} "
                    f"speaking={self.detector.speaking} active={active} "
                    f"buffer={len(speech) * frame_seconds:.2f}s voiced={voiced_duration:.2f}s "
                    f"silence={silence:.2f}s "
                    f"audio_queue={self.audio_queue.qsize()} asr_queue={self.asr_queue.qsize()}"
                )
                pipeline_diagnostics().heartbeat(
                    capture="OK", audio_rms=round(rms, 6),
                    vad="SPEECH" if self.detector.speaking else "SILENCE",
                    buffer_sec=round(len(speech) * frame_seconds, 3),
                    fast_queue=self.asr_queue.qsize(), balanced_queue=0,
                    accurate_queue=0, fast_worker="ALIVE",
                    generation=current.generation if current else 0,
                    language=current.language if current else self.config.language_mode,
                    dropped_segments=pipeline_diagnostics().counts["rejected"])
                diagnostic_rms.clear()
                next_diagnostic = now + self.config.debug_log_interval
            if not speech:
                pre.append(frame)
                if active:
                    self.utterance_id += 1
                    utterance_start = max(0.0, stream_seconds - len(pre) * frame_seconds)
                    speech.extend(pre)
                    voiced_duration = self.config.speech_start_frames * frame_seconds
                    pre.clear()
                    silence = 0.0
                    last_partial = now
                    vad_activated_at = now
                    LOGGER.info(
                        "Voice detected | utterance=%s rms=%.6f threshold=%.6f",
                        self.utterance_id, rms, self.detector.effective_start_threshold,
                    )
                    pipeline_diagnostics().detected(
                        self.utterance_id, recognition.generation if recognition else 0,
                        recognition.language if recognition else self.config.language_mode,
                        timestamp=round(utterance_start, 3), rms=rms,
                        threshold=self.detector.effective_start_threshold)
                continue
            speech.append(frame)
            if active:
                voiced_duration += frame_seconds
            silence = 0.0 if active else silence + frame_seconds
            duration = len(speech) * frame_seconds
            if (duration >= self.config.min_partial_duration and
                    voiced_duration >= self.config.min_partial_speech_duration and
                    now - last_partial >= self.config.partial_interval):
                window_frames = round(self.config.rolling_window_seconds / frame_seconds)
                audio = np.concatenate(speech[-window_frames:])
                partial_end = utterance_start + duration
                partial_start = max(
                    utterance_start, partial_end - len(audio) / self.config.sample_rate)
                self._submit(ASRJob(
                    audio, False, self.utterance_id, now, voiced_duration,
                    partial_start, partial_end, candidate_speech_at,
                    vad_activated_at,
                    recognition.language if recognition else self.config.language_mode,
                    recognition.script if recognition else self.config.script_mode,
                    recognition.generation if recognition else 0,
                    recognition.switched_at if recognition else 0.0,
                    recognition.ready_at if recognition else 0.0))
                last_partial = now
            ended = silence >= self.config.silence_duration
            if ended or duration >= self.config.max_utterance_seconds:
                usable = voiced_duration
                if usable >= self.config.min_speech_duration:
                    trim = round(silence * self.config.sample_rate)
                    audio = np.concatenate(speech)
                    if trim:
                        audio = audio[:-trim]
                    self._submit(ASRJob(
                        audio, True, self.utterance_id, now, voiced_duration,
                        utterance_start,
                        utterance_start + len(audio) / self.config.sample_rate,
                        candidate_speech_at, vad_activated_at,
                        recognition.language if recognition else self.config.language_mode,
                        recognition.script if recognition else self.config.script_mode,
                        recognition.generation if recognition else 0,
                        recognition.switched_at if recognition else 0.0,
                        recognition.ready_at if recognition else 0.0))
                    diagnostics = pipeline_diagnostics()
                    diagnostics.speech_seconds += usable
                    diagnostics.stage(
                        self.utterance_id, "SEGMENT",
                        captured_start=utterance_start,
                        captured_end=utterance_start + duration,
                        captured_duration=duration, asr_start=utterance_start,
                        asr_end=utterance_start + len(audio) / self.config.sample_rate,
                        asr_duration=len(audio) / self.config.sample_rate,
                        trimmed_start_ms=0, trimmed_end_ms=round(silence * 1000),
                        reason="silence" if ended else "maximum_duration")
                    diagnostics.stage(
                        self.utterance_id, "VAD END", speech_duration=usable,
                        silence_duration=silence)
                    diagnostics.save_audio(self.utterance_id, audio)
                    LOGGER.info(
                        "Voice captured | utterance=%s voiced=%.2fs audio=%.2fs; queued for ASR",
                        self.utterance_id, usable, len(audio) / self.config.sample_rate,
                    )
                else:
                    LOGGER.debug(
                        f"[VAD] discarded short utterance={self.utterance_id} "
                        f"usable={usable:.2f}s minimum={self.config.min_speech_duration:.2f}s"
                    )
                    pipeline_diagnostics().terminal(
                        self.utterance_id, "REJECTED", "below_minimum_speech_duration",
                        RootCause.VAD, duration_ms=round(usable * 1000),
                        minimum_ms=round(self.config.min_speech_duration * 1000))
                speech.clear()
                voiced_duration = 0.0
                silence = 0.0
                self.detector.reset()
                candidate_speech_at = 0.0
                vad_activated_at = 0.0
        LOGGER.info("[STOP] speech buffer closed without stop-time refinement | "
                    "utterance=%s buffered=%.2fs", self.utterance_id,
                    len(speech) * frame_seconds)
