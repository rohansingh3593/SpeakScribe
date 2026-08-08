import numpy as np

from audio_pipeline import EnergySpeechDetector
from config import AppConfig


def test_energy_detector_hysteresis_and_reset():
    detector = EnergySpeechDetector(AppConfig())
    assert not detector.classify(np.zeros(480, dtype=np.float32))[0]
    assert detector.classify(np.full(480, 0.02, dtype=np.float32))[0]
    assert detector.classify(np.full(480, 0.009, dtype=np.float32))[0]
    detector.reset()
    assert not detector.classify(np.full(480, 0.009, dtype=np.float32))[0]

