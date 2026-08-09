"""Continuous terminal transcription. Ctrl+C closes the microphone safely."""

from speakscribe import SpeechToText


def main() -> None:
    speech = SpeechToText()
    print("Listening... Press Ctrl+C to stop.")
    try:
        for result in speech.listen_continuously():
            print(result.text)
    except KeyboardInterrupt:
        pass
    finally:
        speech.close()


if __name__ == "__main__":
    main()
