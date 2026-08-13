"""Complete PyQt recording-panel example consuming the GUI-free SpeakScribe API."""

import queue
import sys
import time

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QHBoxLayout, QLabel, QPlainTextEdit,
    QPushButton, QSizePolicy, QVBoxLayout, QWidget,
)

from speakscribe import SpeakScribeError, SpeechConfig, SpeechToText


class RecordingPanel(QWidget):
    """The UI owns widgets; SpeakScribe owns capture, threads, and transcription."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("SpeakScribe Recording Panel Example")
        self.resize(1080, 480)
        self.speech = None
        self.events = queue.Queue()
        self.started_at = None

        self.ui_timer = QTimer(self)
        self.ui_timer.setInterval(100)
        self.ui_timer.timeout.connect(self.process_events)
        self.ui_timer.start()
        self.record_timer = QTimer(self)
        self.record_timer.setInterval(250)
        self.record_timer.timeout.connect(self.update_record_timer)

        self.right_layout = QVBoxLayout(self)
        self._build_recording_bar()

    def _build_recording_bar(self):
        self.record_timer_label = QLabel("00:00")
        self.record_timer_label.setStyleSheet(
            "color: white; font-weight: bold; font-size: 14px;")

        self.record_output_container = QWidget()
        self.record_output_container.setObjectName("recordOutputPanel")
        self.record_output_container.setMinimumWidth(900)
        self.record_output_container.setMaximumWidth(1080)
        self.record_output_container.setFixedHeight(420)
        self.record_output_container.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.record_output_container.setStyleSheet(
            "#recordOutputPanel { background: #20242b; border: 1px solid #3b414b; "
            "border-radius: 8px; } QPushButton { min-height: 25px; }")
        outer_layout = QVBoxLayout(self.record_output_container)
        outer_layout.setContentsMargins(10, 8, 10, 8)
        outer_layout.setSpacing(6)

        settings = QWidget()
        settings.setObjectName("settingsBar")
        settings.setStyleSheet(
            "#settingsBar { background: #f6f6f6; border-radius: 6px; } "
            "#settingsBar QLabel, #settingsBar QCheckBox { color: #111; }")
        settings_layout = QHBoxLayout(settings)
        settings_layout.setContentsMargins(10, 5, 10, 5)
        self.performance = QComboBox()
        self.performance.addItems(["Fast", "Balanced", "Accurate"])
        self.performance.setCurrentText("Balanced")
        self.script = QComboBox()
        self.script.addItems(["Original", "Latin", "Devanagari"])
        self.script.setCurrentText("Original")
        self.recognition = QComboBox()
        self.recognition.addItems(["Hindi / Hinglish", "Auto", "English"])
        self.recognition.setCurrentText("English")
        self.capture = QComboBox()
        self.capture.addItems(["Microphone", "System audio (loopback)"])
        self.capture.setCurrentText("System audio (loopback)")
        self.translation = QCheckBox("Translation")
        for label, control in (
                ("Performance", self.performance), ("Script", self.script),
                ("Recognition", self.recognition), ("Capture", self.capture)):
            settings_layout.addWidget(QLabel(label))
            settings_layout.addWidget(control)
        settings_layout.addWidget(self.translation)
        settings_layout.addStretch()
        outer_layout.addWidget(settings)

        top_row_layout = QHBoxLayout()
        top_row_layout.setSpacing(12)
        button_layout = QVBoxLayout()
        button_layout.setSpacing(2)
        button_layout.addWidget(self.record_timer_label)
        button_layout.addSpacing(5)

        self.btn_record_start = QPushButton("🎤")
        self.btn_record_stop = QPushButton("⛔")
        self.btn_record_clear = QPushButton("🧹")
        self.btn_lang_eng = QPushButton("Eng")
        self.btn_lang_hin = QPushButton("Hin")
        self.btn_lang_hing = QPushButton("Auto")
        for action, language in (
                (self.btn_record_start, self.btn_lang_eng),
                (self.btn_record_stop, self.btn_lang_hin),
                (self.btn_record_clear, self.btn_lang_hing)):
            row = QHBoxLayout()
            row.setSpacing(4)
            row.addWidget(action)
            row.addWidget(language)
            button_layout.addLayout(row)

        button_layout.addSpacing(5)
        self.btn_record_mark = QPushButton("⏸ Finalize segment")
        self.btn_record_restore = QPushButton("🔄 Restore")
        button_layout.addWidget(self.btn_record_mark)
        button_layout.addWidget(self.btn_record_restore)
        button_layout.addSpacing(5)

        self.btn_copy_categories = QPushButton("📋 Copy")
        self.btn_send_all_questions = QPushButton("0 Qs")
        self.btn_show_all = QPushButton("📂 Show all")
        self.btn_clear_words = QPushButton("🧹 Clear")
        self.btn_auto = QPushButton("Auto")
        for button in (self.btn_copy_categories, self.btn_send_all_questions,
                       self.btn_show_all, self.btn_clear_words, self.btn_auto):
            button_layout.addWidget(button)
        button_layout.addStretch()

        self.record_output = QPlainTextEdit()
        font = QFont()
        font.setPointSize(12)
        font.setBold(True)
        self.record_output.setFont(font)
        self.record_output.setPlaceholderText("Live transcription will appear here...")
        self.record_output.setStyleSheet(
            "color: #f5f7fa; background: transparent; selection-background-color: #426a9b;")
        self.record_output.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse |
            Qt.TextInteractionFlag.TextSelectableByKeyboard)
        self.record_output.setFrameShape(QPlainTextEdit.Shape.NoFrame)
        self.record_output.setReadOnly(True)
        self.record_output.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.record_output.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        top_row_layout.addLayout(button_layout, stretch=1)
        top_row_layout.addWidget(self.record_output, stretch=5)
        outer_layout.addLayout(top_row_layout)

        self.btn_record_move = QPushButton("Move")
        self.btn_record_move.setFixedSize(325, 25)
        self.btn_record_move.pressed.connect(self.start_system_move)
        move_layout = QHBoxLayout()
        move_layout.addStretch()
        move_layout.addWidget(self.btn_record_move)
        outer_layout.addLayout(move_layout)
        self.right_layout.addWidget(self.record_output_container)

        self.btn_record_start.clicked.connect(self.start_recording)
        self.btn_record_stop.clicked.connect(self.stop_recording)
        self.btn_record_clear.clicked.connect(self.record_output.clear)
        self.btn_clear_words.clicked.connect(self.record_output.clear)
        self.btn_copy_categories.clicked.connect(
            lambda: QApplication.clipboard().setText(self.record_output.toPlainText()))
        self.btn_record_mark.clicked.connect(lambda: self.record_output.appendPlainText(""))
        self.btn_lang_eng.clicked.connect(lambda: self.recognition.setCurrentText("English"))
        self.btn_lang_hin.clicked.connect(
            lambda: self.recognition.setCurrentText("Hindi / Hinglish"))
        self.btn_lang_hing.clicked.connect(lambda: self.recognition.setCurrentText("Auto"))
        self.btn_auto.clicked.connect(lambda: self.recognition.setCurrentText("Auto"))
        self.btn_record_stop.setEnabled(False)

    def make_config(self) -> SpeechConfig:
        languages = {"English": "en", "Hindi / Hinglish": "hi", "Auto": None}
        beams = {"Fast": 1, "Balanced": 3, "Accurate": 5}
        return SpeechConfig(
            language=languages[self.recognition.currentText()],
            capture_source=("loopback" if "System audio" in self.capture.currentText()
                            else "microphone"),
            beam_size=beams[self.performance.currentText()],
            continuous=True,
        )

    def start_recording(self):
        if self.speech is not None and self.speech.is_running:
            return
        self.speech = SpeechToText(self.make_config())
        self.speech.start_continuous(self.events.put, self.events.put)
        self.started_at = time.monotonic()
        self.record_timer_label.setText("00:00")
        self.record_timer.start()
        self.btn_record_start.setEnabled(False)
        self.btn_record_stop.setEnabled(True)

    def stop_recording(self):
        if self.speech is not None:
            self.speech.stop()
        self.record_timer.stop()
        self.btn_record_start.setEnabled(True)
        self.btn_record_stop.setEnabled(False)

    def process_events(self):
        try:
            while True:
                event = self.events.get_nowait()
                if isinstance(event, SpeakScribeError):
                    self.record_output.appendPlainText(f"Error: {event}")
                else:
                    self.record_output.appendPlainText(event.text)
        except queue.Empty:
            pass

    def update_record_timer(self):
        if self.started_at is None:
            return
        elapsed = int(time.monotonic() - self.started_at)
        minutes, seconds = divmod(max(0, elapsed), 60)
        self.record_timer_label.setText(f"{minutes:02d}:{seconds:02d}")

    def start_system_move(self):
        if self.windowHandle() is not None:
            self.windowHandle().startSystemMove()

    def closeEvent(self, event):
        if self.speech is not None:
            self.speech.close()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    panel = RecordingPanel()
    panel.show()
    raise SystemExit(app.exec())
