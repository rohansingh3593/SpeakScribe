"""PyQt template consuming the public SpeakScribe API from another repository."""

from __future__ import annotations

from datetime import datetime
import os
from pathlib import Path
import signal
import sys
from threading import RLock

from PyQt6.QtCore import QObject, Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QApplication, QLabel, QPushButton, QTextEdit, QVBoxLayout, QWidget,
)

from speakscribe import SpeakScribeError, SpeechConfig, SpeechToText, TranscriptionResult


LANGUAGES = {
    "english": "en",
    "hindi": "hi",
    "hinglish": None,  # Automatic detection supports natural code switching.
}


class TranscriptionSignals(QObject):
    result_ready = pyqtSignal(object, int)
    recognition_error = pyqtSignal(str, int)


class TranscriptionApp(QWidget):
    """A thin Qt consumer; recognition behavior stays in SpeakScribe."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("SpeakScribe — Hindi, English & Hinglish")
        self.setGeometry(200, 200, 560, 420)
        self.setWindowFlags(Qt.WindowType.WindowStaysOnTopHint | Qt.WindowType.Tool)

        self.status_label = QLabel("Status: Idle")
        self.processing_label = QLabel("Processing: —")
        self.text_area = QTextEdit()
        self.text_area.setReadOnly(True)
        self.start_button = QPushButton("Start Recording")
        self.stop_button = QPushButton("Stop Recording")
        self.stop_button.setEnabled(False)
        self.language_buttons = {mode: QPushButton(mode.title()) for mode in LANGUAGES}

        layout = QVBoxLayout(self)
        layout.addWidget(self.status_label)
        layout.addWidget(self.processing_label)
        layout.addWidget(self.text_area)
        for button in self.language_buttons.values():
            layout.addWidget(button)
        layout.addWidget(self.start_button)
        layout.addWidget(self.stop_button)

        self.signals = TranscriptionSignals()
        self.signals.result_ready.connect(self.display_result)
        self.signals.recognition_error.connect(self.display_error)
        self.start_button.clicked.connect(self.start_recording)
        self.stop_button.clicked.connect(self.stop_recording)
        for mode, button in self.language_buttons.items():
            button.clicked.connect(lambda _checked=False, value=mode: self.change_mode(value))

        self.current_mode = "english"
        self.recognizer: SpeechToText | None = None
        self.generation = 0
        self._lock = RLock()
        self._saved_utterances: set[tuple[int, int | str]] = set()

        output_dir = Path(__file__).parent / "transcripts"
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().astimezone().strftime("%Y-%m-%d_%H-%M-%S_%f")
        self.transcript_path = output_dir / f"transcript_{timestamp}.txt"
        self.transcript_file = self.transcript_path.open(
            "a", encoding="utf-8", buffering=1
        )
        self.update_mode_buttons()

    def make_config(self) -> SpeechConfig:
        return SpeechConfig(
            language=LANGUAGES[self.current_mode],
            capture_source="loopback",  # Change to "microphone" for spoken input.
            model_size="small",
            device="auto",  # CUDA first, then CPU/int8 fallback.
            compute_type="auto",
        )

    def start_recording(self) -> None:
        with self._lock:
            if self.recognizer is not None and self.recognizer.is_running:
                return
            previous, self.recognizer = self.recognizer, None
        if previous is not None:
            previous.close()

        with self._lock:
            self.generation += 1
            generation = self.generation
            recognizer = SpeechToText(self.make_config())
            self.recognizer = recognizer

        # These callbacks run outside Qt's GUI thread, so forward them via signals.
        try:
            recognizer.start_continuous(
                lambda result: self.signals.result_ready.emit(result, generation),
                lambda error: self.signals.recognition_error.emit(str(error), generation),
            )
        except SpeakScribeError as error:
            with self._lock:
                self.generation += 1
                self.recognizer = None
            recognizer.close()
            self.status_label.setText(f"Error: {error}")
            return
        self.status_label.setText(f"Status: Recording ({self.current_mode.title()})")
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)

    def stop_recording(self) -> None:
        with self._lock:
            self.generation += 1  # Reject callbacks arriving from the old session.
            recognizer, self.recognizer = self.recognizer, None
        if recognizer is not None:
            recognizer.close()
        self.processing_label.setText("Processing: —")
        self.status_label.setText("Status: Stopped")
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)

    def change_mode(self, mode: str) -> None:
        if mode == self.current_mode:
            return
        was_recording = self.recognizer is not None and self.recognizer.is_running
        self.stop_recording()
        self.current_mode = mode
        self.update_mode_buttons()
        self.status_label.setText(f"Mode: {mode.title()}")
        if was_recording:
            self.start_recording()

    def update_mode_buttons(self) -> None:
        for mode, button in self.language_buttons.items():
            button.setEnabled(mode != self.current_mode)

    def display_result(self, result: TranscriptionResult, generation: int) -> None:
        if generation != self.generation:
            return
        text = result.text.strip()
        if not text:
            return
        if not result.is_final:
            self.processing_label.setText(f"Processing: {text}")
            return

        self.processing_label.setText("Processing: —")
        segment = result.utterance_id if result.utterance_id is not None else text
        segment_key = (generation, segment)
        if segment_key in self._saved_utterances:
            return
        self._saved_utterances.add(segment_key)

        cursor = self.text_area.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        cursor.insertText(f"{text}\n")
        self.text_area.setTextCursor(cursor)
        self.text_area.ensureCursorVisible()
        self.transcript_file.write(f"{text}\n")
        self.transcript_file.flush()
        os.fsync(self.transcript_file.fileno())

    def display_error(self, message: str, generation: int) -> None:
        if generation == self.generation:
            self.status_label.setText(f"Error: {message}")

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt API name
        self.stop_recording()
        self.transcript_file.close()
        event.accept()


def main() -> None:
    app = QApplication(sys.argv)
    window = TranscriptionApp()
    window.show()
    signal.signal(signal.SIGINT, lambda *_args: window.close())
    sys.exit(app.exec())


if __name__ == "__main__":
    try:
        main()
    except SpeakScribeError as error:
        print(f"SpeakScribe could not start: {error}", file=sys.stderr)
        raise SystemExit(1) from error
