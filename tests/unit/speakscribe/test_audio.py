from threading import Event

import numpy as np
import pytest

from speakscribe import SpeechConfig
import speakscribe.audio.microphone as microphone
import speakscribe.audio.recorder as recorder_module
from speakscribe.audio.processor import prepare_audio, rms, to_mono
from speakscribe.audio.recorder import SoundCardRecorder
from speakscribe.exceptions import AudioDeviceNotFoundError, MicrophoneError


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


def test_loopback_capture_uses_default_speaker_mapping(monkeypatch):
    speaker = type("Speaker", (), {"name": "Speakers"})()
    loopback = object()
    calls = []
    backend = type("SoundCard", (), {
        "default_speaker": staticmethod(lambda: speaker),
        "get_microphone": staticmethod(lambda **kwargs: calls.append(kwargs) or loopback),
    })()
    monkeypatch.setattr(microphone, "import_module", lambda _: backend)
    assert microphone.capture_device("loopback") is loopback
    assert calls == [{"id": "Speakers", "include_loopback": True}]


def test_soundcard_recorder_uses_all_loopback_channels_and_yields_mono(monkeypatch):
    recorder_arguments = []

    class Stream:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def record(self, numframes):
            channel_a = np.linspace(-.02, .02, numframes, dtype=np.float32)
            channel_b = np.linspace(-.01, .03, numframes, dtype=np.float32)
            return np.column_stack((channel_a, channel_b))

    device = type("Device", (), {
        "name": "Loopback",
        "recorder": lambda self, **kwargs: recorder_arguments.append(kwargs) or Stream(),
    })()
    monkeypatch.setattr(recorder_module, "capture_device", lambda _: device)
    recorder = SoundCardRecorder(SpeechConfig(
        capture_source="loopback", chunk_duration=.01, minimum_rms=0))
    recorder.start()
    chunks = recorder.iter_audio(Event())
    audio = next(chunks)
    chunks.close()
    recorder.stop()

    assert audio.ndim == 1
    assert recorder_arguments == [{"samplerate": 16_000, "channels": None}]
