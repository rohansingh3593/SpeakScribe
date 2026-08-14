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

config = SpeechConfig(
    language="en-IN",
    sample_rate=16_000,
    partial_interval=0.6,
    silence_duration=0.8,
)
with SpeechToText(config) as speech:
    result = speech.listen_once()
    print(result.text, result.language, result.confidence)
```

Continuous generator usage exposes each available result immediately:

```python
from speakscribe import SpeechToText

speech = SpeechToText()
for result in speech.listen_continuously():
    state = "FINAL" if result.is_final else "PARTIAL"
    print(state, result.text)
```

Capture, VAD/buffering, and inference run on separate bounded worker queues. Partial
jobs are emitted while speech is active and stale partial jobs are coalesced if the
ASR backend is slower than capture; final jobs are retained. Consumers should replace
their displayed partial for the same `utterance_id`, then commit it when `is_final` is
true instead of appending every partial. Timing fields on `TranscriptionResult` expose
audio duration, queue wait, inference, and speech-to-result latency for diagnostics.

Callback consumers can use `start_continuous(on_result, on_error)` and stop safely
with `stop()` or `close()`. Inject a `BaseAudioRecorder` and
`BaseTranscriptionEngine` to use a different microphone stack or ASR provider.

The library installs a `NullHandler` only. Parent applications retain full control of
logging through the `speakscribe` logger or `speakscribe.logging.configure_logging`.

See `examples/cli_example.py`, `examples/simple_example.py`,
`examples/tkinter_example.py`, and `examples/pyqt_recording_panel.py`. The PyQt example
recreates the compact timer/button/transcript/move-bar panel while keeping every widget
outside the reusable package.

`examples/save_transcript.py` demonstrates persistent transcription without a GUI. It
prints partial results as temporary processing output, but appends only non-empty final
results to a timestamped UTF-8 file under `examples/transcripts/`. Each final is flushed
and synchronized immediately, duplicate segment callbacks are ignored, and the
recognizer and file are closed on normal exit or Ctrl+C.

For an existing PyQt project, use `examples/pyqt_library_template.py` as the migration
template. It replaces application-owned SoundCard, queue, thread, VAD, and Whisper
classes with `SpeechToText`; a Qt signal safely forwards the library's background
callbacks to the GUI thread. Its language buttons rebuild the recognizer with `en`,
`hi`, or automatic/Hinglish configuration, reject stale callbacks from older sessions,
and save only final results.

In a separate repository, install the library from a local SpeakScribe checkout (or
from its built wheel) into that repository's virtual environment:

```bash
python -m pip install "/path/to/SpeakScribe[audio,whisper]"
python -m pip install PyQt6
```

Then copy `examples/pyqt_library_template.py` into the consumer repository. The template
intentionally has no direct imports of SoundCard, NumPy, Faster-Whisper, Transformers,
or Indic transliteration. Delete the consumer's `AudioRecorder`, `WhisperTranscriber`,
and `LiveTranscriptionEngine` classes; `SpeechToText` replaces those responsibilities.
Keep PyQt signals because library callbacks run on a worker thread and widgets must be
updated on Qt's GUI thread. Set `capture_source="loopback"` for computer playback or
`capture_source="microphone"` for a physical microphone.
