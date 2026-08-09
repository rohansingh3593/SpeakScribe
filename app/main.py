"""SpeakScribe application entry point and thread-safe PyQt interface."""

from faulthandler import enable
import argparse
import logging
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

from app.asr.asr_engine import ASRWorker, WhisperModelProvider
from app.audio.audio_pipeline import AudioCaptureWorker, SpeechBufferWorker
from app.config.settings import AppConfig, PerformanceMode
from app.utils.logger import (
    configure_logging, emit_status, get_logger, get_output_path, log_exception, log_print,
)
from app.processing.translation import TranslationWorker


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
        self.model_provider = WhisperModelProvider()
        self.preload_thread: Thread | None = None
        self.running = False

    def preload_model(self) -> None:
        """Warm Whisper after the window opens instead of after speech begins."""
        if self.preload_thread is not None:
            return

        def load() -> None:
            try:
                self.signals.status_changed.emit("Loading speech model…")
                self.model_provider.get(AppConfig())
                self.signals.status_changed.emit("Ready")
            except Exception as exc:
                log_exception("MODEL-PRELOAD", exc)
                self.signals.error.emit(str(exc))

        self.preload_thread = Thread(target=load, name="whisper-preload", daemon=True)
        self.preload_thread.start()

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
        asr = ASRWorker(config, asr_queue, self.stop_event, self.signals,
                        self.model_provider)
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
        log_print(
            f"Listening started; log={get_output_path()} threads="
            f"{[thread.name for thread in self.threads]} "
            f"capture_rate={config.capture_sample_rate} asr_rate={config.sample_rate} "
            f"frame_ms={config.frame_ms} speech_threshold={config.speech_threshold} "
            f"silence_threshold={config.silence_threshold} "
            f"language={config.language_mode} script={config.script_mode} "
            f"model={config.model_size} mode={config.performance_mode.value} "
            f"debug_audio={config.debug_audio_enabled}"
        )

    def start_stream(self, config: AppConfig):
        """Preserve ``start`` while yielding live, already-logged status updates."""
        if self.running:
            yield emit_status("Listening session is already running", level=logging.WARNING,
                              component="controller")
            return
        yield emit_status("Preparing audio and ASR workers", component="controller")
        self.start(config)
        yield emit_status("Audio capture and transcription workers started",
                          component="controller")

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
        self.ever_started = False
        self._build_ui()
        self._connect_signals()
        self.controller.preload_model()

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
        self.language_mode = QComboBox()
        self.language_mode.addItems(["Hindi / Hinglish", "Auto", "English"])
        self.translation_toggle = QCheckBox("Translation")
        controls = QHBoxLayout()
        controls.addWidget(QLabel("Performance"))
        controls.addWidget(self.performance)
        controls.addWidget(QLabel("Script"))
        controls.addWidget(self.script)
        controls.addWidget(QLabel("Recognition"))
        controls.addWidget(self.language_mode)
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
        self.signals.partial_text.connect(self.show_partial)
        self.signals.final_text.connect(self.add_final)
        self.signals.language_changed.connect(
            lambda value: self.language.setText(f"Language: {value}"))
        self.signals.status_changed.connect(self.status.setText)
        self.signals.error.connect(self.show_error)
        self.signals.translation_ready.connect(self.show_translation)

    def start_listening(self) -> None:
        self.ever_started = True
        mode = PerformanceMode(self.performance.currentText().lower())
        recognition_modes = {
            "Hindi / Hinglish": "hi", "Auto": "auto", "English": "en",
        }
        config = AppConfig(performance_mode=mode,
                           script_mode=self.script.currentText().lower(),
                           language_mode=recognition_modes[self.language_mode.currentText()],
                           translation_enabled=self.translation_toggle.isChecked())
        self.performance_label.setText(f"Performance: {self.performance.currentText()}")
        self.status.setText("Starting…")
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.performance.setEnabled(False)
        self.script.setEnabled(False)
        self.language_mode.setEnabled(False)
        self.translation_toggle.setEnabled(False)
        self.translation.setVisible(config.translation_enabled)
        for update in self.controller.start_stream(config):
            self.status.setText(update.message)

    def stop_listening(self) -> None:
        self.status.setText("Stopping…")
        # Joining is bounded; normal workers exit quickly without forceful termination.
        self.controller.stop()
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.performance.setEnabled(True)
        self.script.setEnabled(True)
        self.language_mode.setEnabled(True)
        self.translation_toggle.setEnabled(True)

    def add_final(self, text: str) -> None:
        log_print(f"[GUI] final signal received chars={len(text)} text={text!r}")
        self.final_history.append(text)
        self.transcription.setPlainText("\n".join(self.final_history))
        self.live.clear()
        self.controller.translate(text)  # display has already happened

    def show_partial(self, text: str) -> None:
        log_print(f"[GUI] partial signal received chars={len(text)} text={text!r}")
        self.live.setPlainText(text)

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
        if not self.ever_started:
            log_print(
                "Application closed before Start Listening was clicked; "
                "no microphone capture or transcription session ran"
            )
        self.controller.stop()
        log_print("Application shutdown")
        event.accept()


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true",
                        help="Show DEBUG logs in the terminal; files always retain them")
    return parser.parse_known_args(argv)


def startup_status():
    """Yield application startup milestones immediately after logging them."""
    yield emit_status("Starting SpeakScribe", component="application")
    yield emit_status("Initializing Qt application", component="application")
    yield emit_status("Creating main window", component="application")


def main(argv=None) -> int:
    enable()
    args, qt_args = parse_args(argv)
    configure_logging(debug=args.debug)
    statuses = startup_status()
    try:
        next(statuses)
        next(statuses)
        app = QApplication([sys.argv[0], *qt_args])
        signal.signal(signal.SIGINT, lambda *_: app.quit())
        next(statuses)
        window = MainWindow()
        window.show()
        emit_status("SpeakScribe is ready", component="application")
        return app.exec()
    except Exception:
        get_logger("application").critical("Application startup failed", exc_info=True)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
