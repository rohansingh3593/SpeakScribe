"""One-shot library usage without any GUI dependency."""

from speakscribe import SpeechToText


with SpeechToText() as speech:
    result = speech.listen_once()
    print(result.text)
