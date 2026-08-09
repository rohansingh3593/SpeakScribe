"""SpeakScribe application entry point and thread-safe PyQt interface."""

from faulthandler import enable
import argparse
import logging
from queue import Empty, Full, Queue
from threading import Event, Thread
import signal
import sys
import time

from PyQt6.QtCore import QObject, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QHBoxLayout, QLabel, QPushButton,
    QPlainTextEdit, QSizePolicy, QTextEdit, QVBoxLayout, QWidget,
)

from app.asr.asr_engine import ASRWorker, WhisperModelProvider
from app.audio.audio_pipeline import AudioCaptureWorker, SpeechBufferWorker
from app.config.settings import AppConfig, PerformanceMode
from app.utils.logger import (
    configure_logging, emit_status, get_logger, get_output_path, log_exception, log_print,
)
from app.processing.translation import TranslationWorker
from app.processing.text_processing import format_recording_time, incremental_transcript_delta


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
        self.resize(1080, 540)
        self.signals = SpeechSignals()
        self.controller = SpeechController(self.signals)
        self.final_history: list[str] = []
        self.ever_started = False
        self.record_started_at: float | None = None
        self.record_timer = QTimer(self)
        self.record_timer.setInterval(250)
        self.record_timer.timeout.connect(self._update_record_timer)
        self._build_ui()
        self._connect_signals()
        self.controller.preload_model()

    def _build_ui(self) -> None:
        self.status = QLabel("Ready")
        self.language = QLabel("Language: —")
        self.performance_label = QLabel("Performance: Balanced")
        self.translation = QTextEdit()
        self.translation.setReadOnly(True)
        self.translation.setMaximumHeight(80)
        self.translation.hide()

        self.performance = QComboBox()
        self.performance.addItems(["Fast", "Balanced", "Accurate"])
        self.performance.setCurrentText("Balanced")
        self.performance.setMinimumWidth(115)
        self.script = QComboBox()
        self.script.addItems(["Original", "Latin", "Devanagari"])
        self.script.setMinimumWidth(125)
        self.language_mode = QComboBox()
        self.language_mode.addItems(["Hindi / Hinglish", "Auto", "English"])
        self.language_mode.setMinimumWidth(155)
        self.capture_source = QComboBox()
        self.capture_source.addItems(["System audio (legacy)", "Microphone"])
        self.capture_source.setMinimumWidth(200)
        self.translation_toggle = QCheckBox("Translation")
        self.settings_bar = QWidget()
        self.settings_bar.setObjectName("recordSettingsBar")
        self.settings_bar.setStyleSheet(
            "#recordSettingsBar { background: #f6f6f6; border-radius: 6px; } "
            "#recordSettingsBar QLabel, #recordSettingsBar QCheckBox { color: #111; } "
            "#recordSettingsBar QComboBox { min-height: 26px; padding: 0 8px; }")
        controls = QHBoxLayout(self.settings_bar)
        controls.setContentsMargins(10, 5, 10, 5)
        controls.setSpacing(8)
        for label, control in (
                ("Performance", self.performance),
                ("Script", self.script),
                ("Recognition", self.language_mode),
                ("Capture", self.capture_source)):
            controls.addWidget(QLabel(label))
            controls.addWidget(control)
        controls.addWidget(self.translation_toggle)
        controls.addStretch()

        self._build_recording_bar()

        layout = QVBoxLayout(self)
        layout.addWidget(self.status)
        layout.addWidget(self.language)
        layout.addWidget(self.performance_label)
        layout.addWidget(self.record_output_container, alignment=Qt.AlignmentFlag.AlignHCenter)
        layout.addWidget(self.translation)

    def _build_recording_bar(self) -> None:
        """Build the compact timer/button/output panel used for live recording."""
        log_print("[SpeakScribeUI] Creating recording panel")
        self.record_timer_label = QLabel("00:00")
        self.record_timer_label.setStyleSheet(
            "color: white; font-weight: bold; font-size: 14px;")

        self.record_output_container = QWidget()
        self.record_output_container.setObjectName("recordOutputPanel")
        self.record_output_container.setMinimumWidth(900)
        self.record_output_container.setMaximumWidth(1080)
        self.record_output_container.setFixedHeight(340)
        self.record_output_container.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.record_output_container.setStyleSheet(
            "#recordOutputPanel { background: #20242b; border: 1px solid #3b414b; "
            "border-radius: 8px; } QPushButton { min-height: 25px; }")

        outer_layout = QVBoxLayout(self.record_output_container)
        outer_layout.setContentsMargins(10, 8, 10, 8)
        outer_layout.setSpacing(6)
        outer_layout.addWidget(self.settings_bar)
        top_row = QHBoxLayout()
        top_row.setSpacing(12)
        button_layout = QVBoxLayout()
        button_layout.setSpacing(3)
        button_layout.addWidget(self.record_timer_label)
        button_layout.addSpacing(5)

        self.btn_record_start = QPushButton("🎤")
        self.btn_record_stop = QPushButton("⛔")
        self.btn_record_clear = QPushButton("🧹")
        self.btn_lang_eng = QPushButton("Eng")
        self.btn_lang_hin = QPushButton("Hin")
        self.btn_lang_hing = QPushButton("Hing")
        self.btn_copy_transcript = QPushButton("📋 Copy")
        self.btn_record_start.setToolTip("Start live transcription")
        self.btn_record_stop.setToolTip("Stop live transcription")
        self.btn_record_clear.setToolTip("Clear transcription")
        self.btn_copy_transcript.setToolTip("Copy all transcription")

        for action, language_button in (
                (self.btn_record_start, self.btn_lang_eng),
                (self.btn_record_stop, self.btn_lang_hin),
                (self.btn_record_clear, self.btn_lang_hing)):
            row = QHBoxLayout()
            row.setSpacing(4)
            row.addWidget(action)
            row.addWidget(language_button)
            button_layout.addLayout(row)
        button_layout.addSpacing(6)
        button_layout.addWidget(self.btn_copy_transcript)
        button_layout.addStretch()

        self.record_output = QPlainTextEdit()
        self.transcription = self.record_output  # compatibility for existing callbacks
        font = QFont()
        font.setPointSize(12)
        font.setBold(True)
        self.record_output.setFont(font)
        self.record_output.setPlaceholderText("Live transcription will appear here...")
        self.record_output.setStyleSheet(
            "QPlainTextEdit { color: #f5f7fa; background: transparent; "
            "selection-background-color: #426a9b; }")
        self.record_output.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse |
            Qt.TextInteractionFlag.TextSelectableByKeyboard)
        self.record_output.setFrameShape(QPlainTextEdit.Shape.NoFrame)
        self.record_output.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.record_output.viewport().setAttribute(
            Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.record_output.setCursor(Qt.CursorShape.ArrowCursor)
        self.record_output.viewport().setCursor(Qt.CursorShape.ArrowCursor)
        self.record_output.setCursorWidth(0)
        self.record_output.setReadOnly(True)
        self.record_output.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.record_output.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        top_row.addLayout(button_layout, stretch=1)
        top_row.addWidget(self.record_output, stretch=5)
        outer_layout.addLayout(top_row)

        self.record_move_bar = QWidget()
        move_layout = QHBoxLayout(self.record_move_bar)
        move_layout.setContentsMargins(0, 4, 0, 0)
        self.btn_record_move = QPushButton("Move")
        self.btn_record_move.setFixedSize(325, 25)
        self.btn_record_move.pressed.connect(self._start_system_move)
        move_layout.addStretch()
        move_layout.addWidget(self.btn_record_move)
        outer_layout.addWidget(self.record_move_bar)

        self.start_button = self.btn_record_start
        self.stop_button = self.btn_record_stop
        self.stop_button.setEnabled(False)
        self.start_button.clicked.connect(self.start_listening)
        self.stop_button.clicked.connect(self.stop_listening)
        self.btn_record_clear.clicked.connect(self.clear_text)
        self.btn_copy_transcript.clicked.connect(lambda: QApplication.clipboard().setText(
            self.transcription.toPlainText()))
        self.btn_lang_eng.clicked.connect(lambda: self._select_language("English"))
        self.btn_lang_hin.clicked.connect(lambda: self._select_language("Hindi / Hinglish"))
        self.btn_lang_hing.clicked.connect(lambda: self._select_language("Auto"))
        self.language_mode.currentTextChanged.connect(self._sync_language_buttons)
        self._sync_language_buttons()

    def _select_language(self, label: str) -> None:
        self.language_mode.setCurrentText(label)
        self._sync_language_buttons()

    def _sync_language_buttons(self, *_args) -> None:
        selected = self.language_mode.currentText()
        self.btn_lang_eng.setEnabled(selected != "English")
        self.btn_lang_hin.setEnabled(selected != "Hindi / Hinglish")
        self.btn_lang_hing.setEnabled(selected != "Auto")

    def _start_system_move(self) -> None:
        handle = self.windowHandle()
        if handle is not None and hasattr(handle, "startSystemMove"):
            handle.startSystemMove()

    def _update_record_timer(self) -> None:
        if self.record_started_at is None:
            return
        self.record_timer_label.setText(format_recording_time(
            time.monotonic() - self.record_started_at))

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
                           capture_source=("loopback" if self.capture_source.currentIndex() == 0
                                           else "microphone"),
                           translation_enabled=self.translation_toggle.isChecked())
        self.performance_label.setText(f"Performance: {self.performance.currentText()}")
        self.status.setText("Starting…")
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.performance.setEnabled(False)
        self.script.setEnabled(False)
        self.language_mode.setEnabled(False)
        for button in (self.btn_lang_eng, self.btn_lang_hin, self.btn_lang_hing):
            button.setEnabled(False)
        self.capture_source.setEnabled(False)
        self.translation_toggle.setEnabled(False)
        self.translation.setVisible(config.translation_enabled)
        for update in self.controller.start_stream(config):
            self.status.setText(update.message)
        self.record_started_at = time.monotonic()
        self.record_timer_label.setText("00:00")
        self.record_timer.start()

    def stop_listening(self) -> None:
        self.status.setText("Stopping…")
        # Joining is bounded; normal workers exit quickly without forceful termination.
        self.controller.stop()
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.performance.setEnabled(True)
        self.script.setEnabled(True)
        self.language_mode.setEnabled(True)
        self._sync_language_buttons()
        self.capture_source.setEnabled(True)
        self.translation_toggle.setEnabled(True)
        self.record_timer.stop()
        self._update_record_timer()

    def add_final(self, text: str) -> None:
        log_print(f"[GUI] final signal received chars={len(text)} text={text!r}")
        self.final_history.append(text)
        self._append_live_text(text, final=True)
        self.controller.translate(text)  # display has already happened

    def show_partial(self, text: str) -> None:
        log_print(f"[GUI] partial signal received chars={len(text)} text={text!r}")
        self._append_live_text(text)

    def _append_live_text(self, text: str, *, final: bool = False) -> None:
        """Append only the new suffix; never clear or replace visible live text."""
        existing = self.transcription.toPlainText()
        delta = incremental_transcript_delta(existing, text)
        cursor = self.transcription.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        if delta:
            if existing and not existing[-1].isspace():
                cursor.insertText(" ")
            cursor.insertText(delta)
        if final and self.transcription.toPlainText().strip():
            cursor.insertText("\n")
        self.transcription.setTextCursor(cursor)
        self.transcription.ensureCursorVisible()

    def show_translation(self, text: str) -> None:
        self.translation.setPlainText(f"Translation: {text}")

    def show_error(self, message: str) -> None:
        self.status.setText(f"Error: {message}")
        log_print(f"GUI error: {message}")

    def clear_text(self) -> None:
        self.final_history.clear()
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
