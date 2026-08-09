"""One-shot library usage without any GUI dependency."""

from voice_to_text import SpeechToText


with SpeechToText() as speech:
    result = speech.listen_once()
    print(result.text)
