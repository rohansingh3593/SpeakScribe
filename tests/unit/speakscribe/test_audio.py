import numpy as np
import pytest

from speakscribe.audio.processor import prepare_audio, rms, to_mono
from speakscribe.exceptions import AudioDeviceNotFoundError, MicrophoneError
import speakscribe.audio.microphone as microphone


def test_audio_processor_downmixes_centers_and_bounds_audio():
    stereo = np.array([[.01, -.01], [.02, 0], [-.02, 0]], dtype=np.float32)
    mono = to_mono(stereo)
    prepared = prepare_audio(mono)
    assert mono.shape == (3,)
    assert abs(float(np.mean(prepared))) < 1e-6
    assert np.max(np.abs(prepared)) <= 1
    assert rms(prepared) > 0


def test_audio_processor_rejects_invalid_dimensions():
    with pytest.raises(ValueError):
        to_mono(np.zeros((2, 2, 2), dtype=np.float32))


def test_missing_microphone_uses_domain_exception(monkeypatch):
    monkeypatch.setattr(microphone, "import_module", lambda _: type(
        "SoundCard", (), {"default_microphone": staticmethod(lambda: None)})())
    with pytest.raises(AudioDeviceNotFoundError):
        microphone.default_microphone()


def test_backend_lookup_failure_preserves_cause(monkeypatch):
    monkeypatch.setattr(microphone, "import_module",
                        lambda _: (_ for _ in ()).throw(RuntimeError("driver")))
    with pytest.raises(MicrophoneError) as error:
        microphone.default_microphone()
    assert isinstance(error.value.__cause__, RuntimeError)
