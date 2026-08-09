"""Domain exceptions that hide confusing backend-specific failures."""


class VoiceToTextError(Exception):
    """Base class for public library failures."""


class MicrophoneError(VoiceToTextError):
    """The configured input stream could not be opened or read."""


class AudioDeviceNotFoundError(MicrophoneError):
    """No suitable capture device is available."""


class TranscriptionError(VoiceToTextError):
    """The selected transcription engine failed."""


class ServiceStateError(VoiceToTextError):
    """An operation is invalid for the current service state."""
