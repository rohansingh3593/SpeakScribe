import numpy as np

from audio_pipeline import (
    EnergySpeechDetector, audio_statistics, prepare_audio_for_asr,
    resample_audio_block,
)
from config import AppConfig


def test_energy_detector_hysteresis_and_reset():
    detector = EnergySpeechDetector(AppConfig(
        speech_threshold=0.012, silence_threshold=0.008, speech_start_frames=1,
    ))
    assert not detector.classify(np.zeros(480, dtype=np.float32))[0]
    assert detector.classify(np.full(480, 0.02, dtype=np.float32))[0]
    assert detector.classify(np.full(480, 0.009, dtype=np.float32))[0]
    detector.reset()
    assert not detector.classify(np.full(480, 0.009, dtype=np.float32))[0]


def test_default_detector_accepts_quiet_laptop_microphone_speech():
    detector = EnergySpeechDetector(AppConfig())
    frame = np.full(480, 0.0035, dtype=np.float32)
    assert not detector.classify(frame)[0]
    assert not detector.classify(frame)[0]
    assert detector.classify(frame)[0]


def test_quiet_speech_is_centered_and_amplified_for_whisper():
    time_axis = np.linspace(0, 4 * np.pi, 1600, dtype=np.float32)
    quiet_speech = 0.02 + 0.01 * np.sin(time_axis)
    prepared = prepare_audio_for_asr(quiet_speech)
    assert abs(float(np.mean(prepared))) < 1e-5
    assert float(np.max(np.abs(prepared))) > 0.09
    assert float(np.max(np.abs(prepared))) <= 1.0


def test_native_48khz_capture_is_downsampled_to_16khz():
    source = np.arange(1440, dtype=np.float32)
    result = resample_audio_block(source, 48_000, 16_000)
    assert result.shape == (480,)
    assert np.isclose(result[0], 1.0)


def test_audio_statistics_expose_silence_and_invalid_samples():
    silence = audio_statistics(np.zeros(480, dtype=np.float32))
    assert silence == {
        "samples": 480, "rms": 0.0, "peak": 0.0, "mean": 0.0,
        "zero_ratio": 1.0, "finite": True,
    }
    invalid = audio_statistics(np.array([0.0, np.nan], dtype=np.float32))
    assert invalid["finite"] is False
