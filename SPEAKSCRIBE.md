# SpeakScribe Library

`speakscribe` is the reusable, UI-independent speech layer extracted from
SpeakScribe. The package contains no Tkinter, PyQt, Flask, FastAPI, or display
dependencies. Applications receive structured results and decide how to display them.

## Installation

```bash
pip install .
pip install -e ".[dev]"
pip install ".[audio,whisper]"
```

The base installation is intentionally small. Install the `audio` extra for live
SoundCard capture and `whisper` for the Faster-Whisper backend; `all` installs both.

## Public API

```python
from speakscribe import SpeakScribeError, SpeechConfig, SpeechToText

config = SpeechConfig(language="en-IN", sample_rate=16_000)
with SpeechToText(config) as speech:
    result = speech.listen_once()
    print(result.text, result.language, result.confidence)
```

Continuous generator usage exposes each available result immediately:

```python
from speakscribe import SpeechToText

speech = SpeechToText()
for result in speech.listen_continuously():
    print(result.text)
```

Callback consumers can use `start_continuous(on_result, on_error)` and stop safely
with `stop()` or `close()`. Inject a `BaseAudioRecorder` and
`BaseTranscriptionEngine` to use a different microphone stack or ASR provider.

The library installs a `NullHandler` only. Parent applications retain full control of
logging through the `speakscribe` logger or `speakscribe.logging.configure_logging`.

See `examples/cli_example.py`, `examples/simple_example.py`, and
`examples/tkinter_example.py`.
