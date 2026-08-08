import numpy as np

from audio_pipeline import EnergySpeechDetector, prepare_audio_for_asr
from config import AppConfig


def test_energy_detector_hysteresis_and_reset():
    detector = EnergySpeechDetector(AppConfig(
        speech_threshold=0.012, silence_threshold=0.008,
    ))
    assert not detector.classify(np.zeros(480, dtype=np.float32))[0]
    assert detector.classify(np.full(480, 0.02, dtype=np.float32))[0]
    assert detector.classify(np.full(480, 0.009, dtype=np.float32))[0]
    detector.reset()
    assert not detector.classify(np.full(480, 0.009, dtype=np.float32))[0]


def test_default_detector_accepts_quiet_laptop_microphone_speech():
    detector = EnergySpeechDetector(AppConfig())
    assert detector.classify(np.full(480, 0.003, dtype=np.float32))[0]


def test_quiet_speech_is_centered_and_amplified_for_whisper():
    time_axis = np.linspace(0, 4 * np.pi, 1600, dtype=np.float32)
    quiet_speech = 0.02 + 0.01 * np.sin(time_axis)
    prepared = prepare_audio_for_asr(quiet_speech)
    assert abs(float(np.mean(prepared))) < 1e-5
    assert float(np.max(np.abs(prepared))) > 0.09
    assert float(np.max(np.abs(prepared))) <= 1.0
