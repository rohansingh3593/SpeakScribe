"""Lazy SoundCard device lookup kept separate for easy replacement and mocking."""

from importlib import import_module

from voice_to_text.exceptions import AudioDeviceNotFoundError, MicrophoneError


def default_microphone():
    try:
        soundcard = import_module("soundcard")
        device = soundcard.default_microphone()
    except Exception as exc:
        raise MicrophoneError("Unable to query the default microphone") from exc
    if device is None:
        raise AudioDeviceNotFoundError("No default microphone is available")
    return device
