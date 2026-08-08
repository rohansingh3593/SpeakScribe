"""SpeakScribe application entry point and thread-safe PyQt interface."""

from faulthandler import enable
from queue import Empty, Full, Queue
from threading import Event, Thread
import signal
import sys
import time

from PyQt6.QtCore import QObject, pyqtSignal
from PyQt6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QHBoxLayout, QLabel, QPushButton,
    QTextEdit, QVBoxLayout, QWidget,
)

from asr_engine import ASRWorker
from audio_pipeline import AudioCaptureWorker, SpeechBufferWorker
from config import AppConfig, PerformanceMode
from logger import get_output_path, log_print
from translation import TranslationWorker


class SpeechSignals(QObject):
    partial_text = pyqtSignal(str)
    final_text = pyqtSignal(str)
    language_changed = pyqtSignal(str)
    status_changed = pyqtSignal(str)
    error = pyqtSignal(str)
    translation_ready = pyqtSignal(str)


class SpeechController:
    """Own queues/workers; widgets never cross this boundary into worker threads."""

    def __init__(self, signals: SpeechSignals):
        self.signals = signals
        self.stop_event = Event()
        self.threads: list[Thread] = []
        self.translation_queue: Queue | None = None
        self.running = False

    def start(self, config: AppConfig) -> None:
        if self.running:
            return
        self.running = True
        self.stop_event = Event()
        audio_queue = Queue(maxsize=config.max_audio_queue)
        asr_queue = Queue(maxsize=config.max_asr_queue)
        capture = AudioCaptureWorker(config, audio_queue, self.stop_event,
                                     self.signals.error.emit)
        buffer = SpeechBufferWorker(config, audio_queue, asr_queue, self.stop_event)
        asr = ASRWorker(config, asr_queue, self.stop_event, self.signals)
        self.threads = [
            Thread(target=capture.run, name="audio-capture", daemon=True),
            Thread(target=buffer.run, name="speech-buffer", daemon=True),
            Thread(target=asr.run, name="whisper-asr", daemon=True),
        ]
        if config.translation_enabled:
            self.translation_queue = Queue(maxsize=4)
            translator = TranslationWorker(config.translation_model,
                                           self.translation_queue,
                                           self.stop_event, self.signals)
            self.threads.append(Thread(target=translator.run, name="translation",
                                       daemon=True))
        for thread in self.threads:
            thread.start()
        log_print(f"Listening started; log={get_output_path()}")

    def translate(self, text: str) -> None:
        if self.translation_queue is None:
            return
        try:
            self.translation_queue.put_nowait(text)
        except Full:
            try:
                self.translation_queue.get_nowait()
                self.translation_queue.put_nowait(text)
            except (Empty, Full):
                pass

    def stop(self, timeout: float = 3.0) -> None:
        if not self.running:
            return
        self.stop_event.set()
        deadline = time.monotonic() + timeout
        for thread in self.threads:
            thread.join(timeout=max(0.0, deadline - time.monotonic()))
            if thread.is_alive():
                log_print(f"Worker still winding down: {thread.name}")
        self.threads.clear()
        self.translation_queue = None
        self.running = False
        self.signals.status_changed.emit("Stopped")
        log_print("Listening stopped")


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SpeakScribe")
        self.resize(760, 600)
        self.signals = SpeechSignals()
        self.controller = SpeechController(self.signals)
        self.final_history: list[str] = []
        self._build_ui()
        self._connect_signals()

    def _build_ui(self) -> None:
        self.status = QLabel("Ready")
        self.language = QLabel("Language: —")
        self.performance_label = QLabel("Performance: Balanced")
        self.live = QTextEdit()
        self.live.setReadOnly(True)
        self.live.setMaximumHeight(110)
        self.transcription = QTextEdit()
        self.transcription.setReadOnly(True)
        self.translation = QTextEdit()
        self.translation.setReadOnly(True)
        self.translation.setMaximumHeight(80)
        self.translation.hide()

        self.performance = QComboBox()
        self.performance.addItems(["Fast", "Balanced", "Accurate"])
        self.performance.setCurrentText("Balanced")
        self.script = QComboBox()
        self.script.addItems(["Original", "Latin", "Devanagari"])
        self.translation_toggle = QCheckBox("Translation")
        controls = QHBoxLayout()
        controls.addWidget(QLabel("Performance"))
        controls.addWidget(self.performance)
        controls.addWidget(QLabel("Script"))
        controls.addWidget(self.script)
        controls.addWidget(self.translation_toggle)

        self.start_button = QPushButton("Start Listening")
        self.stop_button = QPushButton("Stop Listening")
        self.stop_button.setEnabled(False)
        clear = QPushButton("Clear")
        copy = QPushButton("Copy")
        buttons = QHBoxLayout()
        for button in (self.start_button, self.stop_button, clear, copy):
            buttons.addWidget(button)
        self.start_button.clicked.connect(self.start_listening)
        self.stop_button.clicked.connect(self.stop_listening)
        clear.clicked.connect(self.clear_text)
        copy.clicked.connect(lambda: QApplication.clipboard().setText(
            self.transcription.toPlainText()))

        layout = QVBoxLayout(self)
        layout.addWidget(self.status)
        layout.addWidget(self.language)
        layout.addWidget(self.performance_label)
        layout.addLayout(controls)
        layout.addWidget(QLabel("Live:"))
        layout.addWidget(self.live)
        layout.addWidget(QLabel("Transcription:"))
        layout.addWidget(self.transcription)
        layout.addWidget(self.translation)
        layout.addLayout(buttons)

    def _connect_signals(self) -> None:
        self.signals.partial_text.connect(self.live.setPlainText)
        self.signals.final_text.connect(self.add_final)
        self.signals.language_changed.connect(
            lambda value: self.language.setText(f"Language: {value}"))
        self.signals.status_changed.connect(self.status.setText)
        self.signals.error.connect(self.show_error)
        self.signals.translation_ready.connect(self.show_translation)

    def start_listening(self) -> None:
        mode = PerformanceMode(self.performance.currentText().lower())
        config = AppConfig(performance_mode=mode,
                           script_mode=self.script.currentText().lower(),
                           translation_enabled=self.translation_toggle.isChecked())
        self.performance_label.setText(f"Performance: {self.performance.currentText()}")
        self.status.setText("Starting…")
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.performance.setEnabled(False)
        self.script.setEnabled(False)
        self.translation_toggle.setEnabled(False)
        self.translation.setVisible(config.translation_enabled)
        self.controller.start(config)

    def stop_listening(self) -> None:
        self.status.setText("Stopping…")
        # Joining is bounded; normal workers exit quickly without forceful termination.
        self.controller.stop()
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.performance.setEnabled(True)
        self.script.setEnabled(True)
        self.translation_toggle.setEnabled(True)

    def add_final(self, text: str) -> None:
        self.final_history.append(text)
        self.transcription.setPlainText("\n".join(self.final_history))
        self.live.clear()
        self.controller.translate(text)  # display has already happened

    def show_translation(self, text: str) -> None:
        self.translation.setPlainText(f"Translation: {text}")

    def show_error(self, message: str) -> None:
        self.status.setText(f"Error: {message}")
        log_print(f"GUI error: {message}")

    def clear_text(self) -> None:
        self.final_history.clear()
        self.live.clear()
        self.transcription.clear()
        self.translation.clear()

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt API
        self.controller.stop()
        log_print("Application shutdown")
        event.accept()


def main() -> int:
    enable()
    log_print("Application startup")
    app = QApplication(sys.argv)
    signal.signal(signal.SIGINT, lambda *_: app.quit())
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
