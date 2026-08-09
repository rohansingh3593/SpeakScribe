"""Backend-neutral mono conversion and conservative level preparation."""

import numpy as np


def to_mono(audio) -> np.ndarray:
    values = np.asarray(audio, dtype=np.float32)
    if values.ndim == 1:
        return values
    if values.ndim != 2:
        raise ValueError("audio must be a one- or two-dimensional array")
    return np.mean(values, axis=1, dtype=np.float32)


def rms(audio) -> float:
    values = np.asarray(audio, dtype=np.float32)
    if values.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(values), dtype=np.float64)))


def audio_normalization_gain(audio, *, minimum_peak: float = 0.002,
                             target_peak: float = 0.8,
                             maximum_gain: float = 15.0) -> float:
    values = np.asarray(audio, dtype=np.float32)
    if values.size == 0:
        return 1.0
    peak = float(np.max(np.abs(values)))
    if not np.isfinite(peak) or peak < minimum_peak or peak >= target_peak:
        return 1.0
    return min(maximum_gain, target_peak / peak)


def prepare_audio_for_asr(audio, *, maximum_gain: float = 15.0,
                          minimum_peak: float = 0.002) -> np.ndarray:
    values = to_mono(audio).astype(np.float32, copy=True)
    if values.size == 0:
        return values
    values -= np.mean(values, dtype=np.float64)
    values *= audio_normalization_gain(
        values, minimum_peak=minimum_peak, maximum_gain=maximum_gain)
    return np.clip(values, -1.0, 1.0).astype(np.float32, copy=False)


prepare_audio = prepare_audio_for_asr
