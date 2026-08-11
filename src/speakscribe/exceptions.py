"""Domain exceptions that hide confusing backend-specific failures."""


class SpeakScribeError(Exception):
    """Base class for public SpeakScribe library failures."""


# Compatibility alias for the generic name documented during initial extraction.
VoiceToTextError = SpeakScribeError


class MicrophoneError(SpeakScribeError):
    """The configured input stream could not be opened or read."""


class AudioDeviceNotFoundError(MicrophoneError):
    """No suitable capture device is available."""


class TranscriptionError(SpeakScribeError):
    """The selected transcription engine failed."""


class ServiceStateError(SpeakScribeError):
    """An operation is invalid for the current service state."""
