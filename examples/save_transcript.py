"""Save only finalized live recognition results to a UTF-8 transcript file.

Run this example after installing SpeakScribe with the audio and Whisper extras:

    python -m pip install -e ".[audio,whisper]"
    python examples/save_transcript.py
"""

from __future__ import annotations

from datetime import datetime
import os
from pathlib import Path
import sys
from typing import TextIO

from speakscribe import SpeechConfig, SpeechToText, TranscriptionResult


OUTPUT_DIR = Path(__file__).parent / "transcripts"


class TranscriptWriter:
    """Append each finalized utterance once and durably flush it to disk."""

    def __init__(self, output_dir: Path = OUTPUT_DIR, *,
                 timestamp: datetime | None = None) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        started_at = timestamp or datetime.now().astimezone()
        filename = started_at.strftime("transcript_%Y-%m-%d_%H-%M-%S_%f.txt")
        self.path = output_dir / filename
        self._stream = self.path.open("a", encoding="utf-8", buffering=1)
        self._saved_segments: set[tuple[str, int | str]] = set()

    @property
    def closed(self) -> bool:
        return self._stream.closed

    def append_final(self, text: str, utterance_id: int | None = None) -> bool:
        """Write non-empty final text unless this finalized segment was already saved."""
        clean_text = text.strip()
        if not clean_text:
            return False

        # Recognition results normally carry an utterance ID. Text is the safe
        # fallback for manually constructed results that do not have one.
        segment_key: tuple[str, int | str] = (
            ("utterance", utterance_id) if utterance_id is not None
            else ("text", clean_text)
        )
        if segment_key in self._saved_segments:
            return False

        self._stream.write(f"{clean_text}\n")
        self._stream.flush()
        os.fsync(self._stream.fileno())
        self._saved_segments.add(segment_key)
        return True

    def close(self) -> None:
        if not self._stream.closed:
            self._stream.flush()
            self._stream.close()

    def __enter__(self) -> TranscriptWriter:
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self.close()


def process_result(result: TranscriptionResult, writer: TranscriptWriter, *,
                   output: TextIO = sys.stdout) -> None:
    """Display partials temporarily and persist final results only."""
    if result.is_final:
        if writer.append_final(result.text, result.utterance_id):
            print(f"\nFinal: {result.text.strip()}", file=output, flush=True)
        return

    partial = result.text.strip()
    if partial:
        print(f"\rProcessing: {partial}", end="", file=output, flush=True)


def main() -> None:
    config = SpeechConfig(language="hi")
    try:
        with TranscriptWriter() as writer, SpeechToText(config) as recognizer:
            print(f"Saving finalized speech to: {writer.path}")
            print("Listening... Press Ctrl+C to stop.")
            for result in recognizer.listen_continuously():
                process_result(result, writer)
    except KeyboardInterrupt:
        print("\nStopped. Finalized transcript text has been saved.")


if __name__ == "__main__":
    main()
