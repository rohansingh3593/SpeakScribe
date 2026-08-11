"""Lazy SoundCard device lookup kept separate for easy replacement and mocking."""

from importlib import import_module

from speakscribe.exceptions import AudioDeviceNotFoundError, MicrophoneError


def capture_device(source: str = "microphone"):
    try:
        soundcard = import_module("soundcard")
        if source == "loopback":
            speaker = soundcard.default_speaker()
            if speaker is None:
                raise AudioDeviceNotFoundError(
                    "No default speaker is available for loopback capture")
            device = soundcard.get_microphone(id=str(speaker.name), include_loopback=True)
        elif source == "microphone":
            device = soundcard.default_microphone()
        else:
            raise ValueError(f"Unsupported capture source: {source}")
    except AudioDeviceNotFoundError:
        raise
    except Exception as exc:
        raise MicrophoneError(f"Unable to query the {source} capture device") from exc
    if device is None:
        raise AudioDeviceNotFoundError(f"No {source} capture device is available")
    return device


def default_microphone():
    return capture_device("microphone")
