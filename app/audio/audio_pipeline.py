"""Continuous SoundCard capture and energy-based utterance segmentation."""

from collections import deque
from dataclasses import dataclass
from importlib import import_module
from queue import Empty, Full, Queue
from threading import Event
import time
import warnings

import numpy as np

from voice_to_text.audio.processor import audio_normalization_gain, prepare_audio_for_asr

from app.config.settings import AppConfig
from app.utils.logger import get_logger, log_exception


LOGGER = get_logger("audio")

@dataclass(frozen=True)
class ASRJob:
    audio: np.ndarray
    final: bool
    utterance_id: int
    captured_at: float
    speech_seconds: float | None = None


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
                            LOGGER.warning("Audio queue full; dropped oldest raw frame")
        except Exception as exc:
            log_exception("CAPTURE", exc)
            self.on_error(str(exc))


class SpeechBufferWorker:
    def __init__(self, config: AppConfig, audio_queue: Queue, asr_queue: Queue,
                 stop_event: Event):
        self.config, self.audio_queue, self.asr_queue = config, audio_queue, asr_queue
        self.stop_event = stop_event
        self.detector = EnergySpeechDetector(config)
        self.utterance_id = 0

    def _submit(self, job: ASRJob) -> None:
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
                self.asr_queue.put_nowait(pending)
            elif pending is not None:
                LOGGER.debug(f"[QUEUE] evicted obsolete partial utterance={pending.utterance_id}")

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
                    return
                if pending.final:
                    self.asr_queue.put_nowait(pending)
                    LOGGER.debug("[QUEUE] kept queued final; dropped new partial")
                    return
                try:
                    self.asr_queue.put_nowait(job)
                except Full:
                    pass

    def run(self) -> None:
        frame_seconds = self.config.frame_ms / 1000
        pre = deque(maxlen=max(1, round(self.config.pre_speech_duration / frame_seconds)))
        speech: list[np.ndarray] = []
        voiced_duration = 0.0
        silence = 0.0
        last_partial = 0.0
        diagnostic_rms: list[float] = []
        next_diagnostic = time.monotonic() + self.config.debug_log_interval
        while not self.stop_event.is_set() or not self.audio_queue.empty():
            try:
                frame = self.audio_queue.get(timeout=0.1)
            except Empty:
                continue
            active, rms = self.detector.classify(frame)
            now = time.monotonic()
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
                diagnostic_rms.clear()
                next_diagnostic = now + self.config.debug_log_interval
            if not speech:
                pre.append(frame)
                if active:
                    self.utterance_id += 1
                    speech.extend(pre)
                    voiced_duration = self.config.speech_start_frames * frame_seconds
                    pre.clear()
                    silence = 0.0
                    last_partial = now
                    LOGGER.info(
                        "Voice detected | utterance=%s rms=%.6f threshold=%.6f",
                        self.utterance_id, rms, self.detector.effective_start_threshold,
                    )
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
                self._submit(ASRJob(audio, False, self.utterance_id, now,
                                    voiced_duration))
                last_partial = now
            ended = silence >= self.config.silence_duration
            if ended or duration >= self.config.max_utterance_seconds:
                usable = voiced_duration
                if usable >= self.config.min_speech_duration:
                    trim = round(silence * self.config.sample_rate)
                    audio = np.concatenate(speech)
                    if trim:
                        audio = audio[:-trim]
                    self._submit(ASRJob(audio, True, self.utterance_id, now,
                                        voiced_duration))
                    LOGGER.info(
                        "Voice captured | utterance=%s voiced=%.2fs audio=%.2fs; queued for ASR",
                        self.utterance_id, usable, len(audio) / self.config.sample_rate,
                    )
                else:
                    LOGGER.debug(
                        f"[VAD] discarded short utterance={self.utterance_id} "
                        f"usable={usable:.2f}s minimum={self.config.min_speech_duration:.2f}s"
                    )
                speech.clear()
                voiced_duration = 0.0
                silence = 0.0
                self.detector.reset()
        # A stop click may arrive mid-utterance. Preserve that speech as a final
        # job rather than silently throwing away the last words.
        duration = len(speech) * frame_seconds
        if voiced_duration >= self.config.min_speech_duration:
            self._submit(ASRJob(np.concatenate(speech), True, self.utterance_id,
                                time.monotonic(), voiced_duration))
