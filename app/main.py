"""SpeakScribe application entry point and thread-safe PyQt interface."""

from faulthandler import enable
import argparse
from html import escape
import json
import logging
from pathlib import Path
from queue import Empty, Full, Queue
from threading import Event, Thread
import signal
import sys
import time

from PyQt6.QtCore import QObject, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QAbstractItemView, QApplication, QCheckBox, QComboBox, QHBoxLayout, QLabel, QPushButton,
    QPlainTextEdit, QSizePolicy, QTableWidget, QTableWidgetItem, QTextEdit,
    QVBoxLayout, QWidget,
)

from app.asr.asr_engine import ASRWorker, ComparisonASRWorker, WhisperModelProvider
from app.asr.language_transition import RecognitionState
from app.asr.session_transition import LiveSessionBoundary
from app.audio.audio_pipeline import AudioCaptureWorker, SpeechBufferWorker
from app.config.settings import AppConfig, PerformanceMode
from app.utils.logger import (
    configure_logging, emit_status, get_log_session, get_logger, get_output_path,
    log_exception, log_print,
)
from app.utils.pipeline_diagnostics import (
    configure_pipeline_diagnostics, pipeline_diagnostics,
)
from app.processing.translation import TranslationWorker
from app.processing.live_transcript import LiveTranscriptModel
from app.processing.text_processing import (
    best_refinement_candidate, comparison_agreement_percentages, comparison_diff_html,
    compose_live_transcript, descending_segment_row, format_processing_duration,
    format_recording_time,
    remove_history_overlap,
)

LOGGER = get_logger("ui")
DEBUG_DIAGNOSTICS = False


class SpeechSignals(QObject):
    partial_text = pyqtSignal(str)
    final_text = pyqtSignal(str)
    stale_final_text = pyqtSignal(str)
    language_changed = pyqtSignal(str)
    status_changed = pyqtSignal(str)
    error = pyqtSignal(str)
    translation_ready = pyqtSignal(str)
    mode_text = pyqtSignal(int, str, str, bool, object)
    mode_status = pyqtSignal(int, str, str)
    mode_error = pyqtSignal(int, str, str)


class SpeechController:
    """Own queues/workers; widgets never cross this boundary into worker threads."""

    def __init__(self, signals: SpeechSignals):
        self.signals = signals
        self.stop_event = Event()
        self.threads: list[Thread] = []
        self.translation_queue: Queue | None = None
        self.model_provider = WhisperModelProvider()
        self.preload_thread: Thread | None = None
        self.shutdown_thread: Thread | None = None
        self.running = False
        self.state = "READY"
        self.recognition_state: RecognitionState | None = None
        self.asr_queue: Queue | None = None
        self.retired_threads: list[Thread] = []
        self.last_stop_metrics: dict[str, float | int] = {}

    def preload_model(self) -> None:
        """Warm Whisper after the window opens instead of after speech begins."""
        if self.preload_thread is not None:
            return

        def load() -> None:
            try:
                self.signals.status_changed.emit("Loading speech model…")
                self.model_provider.get(AppConfig())
                fast_config = AppConfig(performance_mode=PerformanceMode.FAST)
                fast_config.model_size = fast_config.profile.model_size
                self.model_provider.get(fast_config)
                self.signals.status_changed.emit("Ready")
            except Exception as exc:
                log_exception("MODEL-PRELOAD", exc)
                self.signals.error.emit(str(exc))

        self.preload_thread = Thread(target=load, name="whisper-preload", daemon=True)
        self.preload_thread.start()

    def start(self, config: AppConfig, compare_all: bool = False) -> None:
        if self.running:
            return
        self.running = True
        log_session = get_log_session()
        diagnostic_root = ((log_session.directory /
                            time.strftime("listening_%Y-%m-%d_%H-%M-%S"))
                           if log_session else Path("logs") / "pipeline")
        configure_pipeline_diagnostics(diagnostic_root, config.diagnostics_enabled)
        self.state = "LISTENING"
        self.stop_event = Event()
        audio_queue = Queue(maxsize=config.max_audio_queue)
        asr_queue = Queue(maxsize=config.max_asr_queue)
        self.asr_queue = asr_queue
        if self.recognition_state is None:
            self.recognition_state = RecognitionState(
                config.language_mode, config.script_mode)
        session = self.recognition_state.begin_session(
            config.language_mode, config.script_mode)
        capture = AudioCaptureWorker(config, audio_queue, self.stop_event,
                                     self.signals.error.emit)
        buffer = SpeechBufferWorker(config, audio_queue, asr_queue, self.stop_event,
                                    self.recognition_state)
        asr = (ComparisonASRWorker(config, asr_queue, self.stop_event, self.signals,
                                   self.model_provider, self.recognition_state) if compare_all else
               ASRWorker(config, asr_queue, self.stop_event, self.signals,
                         self.model_provider, self.recognition_state))
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
            f"Listening started; pipeline=fast-live-v2 log={get_output_path()} threads="
            f"{[thread.name for thread in self.threads]} "
            f"capture_rate={config.capture_sample_rate} asr_rate={config.sample_rate} "
            f"frame_ms={config.frame_ms} speech_threshold={config.speech_threshold} "
            f"silence_threshold={config.silence_threshold} "
            f"language={config.language_mode} script={config.script_mode} "
            f"model={config.model_size} mode={config.performance_mode.value} "
            f"debug_audio={config.debug_audio_enabled}"
        )
        LOGGER.info("[SESSION] listening generation=%s language=%s",
                    session.generation, session.language)

    def switch_recognition_language(self, language: str, script: str) -> None:
        """Atomically switch future ASR work while retaining capture/model/history."""
        if not self.running or self.recognition_state is None:
            return
        previous = self.recognition_state.snapshot()
        if previous.language == language and previous.script == script:
            return
        LOGGER.info("[LANG-SWITCH] requested %s → %s", previous.language, language)
        self.signals.status_changed.emit("Switching language…")
        current = self.recognition_state.switch(language, script)
        cancelled = 0
        retained = []
        if self.asr_queue is not None:
            while True:
                try:
                    pending = self.asr_queue.get_nowait()
                except Empty:
                    break
                if pending.final:
                    retained.append(pending)
                else:
                    cancelled += 1
            for pending in retained:
                try:
                    self.asr_queue.put_nowait(pending)
                except Full:
                    break
        LOGGER.info("[LANG-SWITCH] old partial jobs cancelled count=%s", cancelled)
        LOGGER.info("[LANG-SWITCH] context reset; active language=%s script=%s generation=%s",
                    current.language, current.script, current.generation)
        elapsed_ms = (current.ready_at - current.switched_at) * 1000
        LOGGER.info("[LANG-SWITCH] ready in %.1fms (model reused; audio capture retained)",
                    elapsed_ms)
        self.signals.status_changed.emit("Ready")

    def start_stream(self, config: AppConfig, compare_all: bool = False):
        """Preserve ``start`` while yielding live, already-logged status updates."""
        if self.running:
            yield emit_status("Listening session is already running", level=logging.WARNING,
                              component="controller")
            return
        yield emit_status("Preparing audio and ASR workers", component="controller")
        self.start(config, compare_all)
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

    def reset_live_session(self) -> dict[str, float | int]:
        """Invalidate and detach the current live session without waiting for ASR."""
        self.state = "STOPPING"
        if self.recognition_state is None:
            raise RuntimeError("Cannot reset a session before recognition state exists")
        queues = (self.asr_queue,) if self.asr_queue is not None else ()
        stopped = LiveSessionBoundary(
            self.recognition_state, self.stop_event, queues).stop()
        self.signals.partial_text.emit("")
        self.running = False
        self.state = "READY"
        metrics = {
            **stopped.__dict__, "state_ready_at": time.monotonic(),
        }
        self.last_stop_metrics = metrics
        pipeline_diagnostics().close("session_stop")
        LOGGER.info(
            "[STOP] capture-disabled=%.1fms generation-invalid=%.1fms "
            "queues-cleared=%.1fms ready=%.1fms jobs-cleared=%s generation=%s",
            (stopped.capture_disabled_at - stopped.stop_clicked_at) * 1000,
            (stopped.generation_invalidated_at - stopped.stop_clicked_at) * 1000,
            (stopped.queues_cleared_at - stopped.stop_clicked_at) * 1000,
            (metrics["state_ready_at"] - stopped.stop_clicked_at) * 1000,
            stopped.jobs_cleared, stopped.generation)
        return metrics

    def stop(self, timeout: float = 0.0) -> bool:
        if not self.running:
            return True
        old_threads = list(self.threads)
        self.reset_live_session()
        self.retired_threads.extend(old_threads)
        self.threads = []
        self.translation_queue = None
        def reap() -> None:
            for worker in old_threads:
                worker.join()
                try:
                    self.retired_threads.remove(worker)
                except ValueError:
                    pass
            log_print("Retired listening workers cleaned up in background")
        Thread(target=reap, name="worker-cleanup", daemon=True).start()
        self.signals.status_changed.emit("Ready")
        return True


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SpeakScribe")
        self.resize(1080, 780)
        self.signals = SpeechSignals()
        self.controller = SpeechController(self.signals)
        self.final_history: list[str] = []
        self.current_partial = ""
        self.live_transcript = LiveTranscriptModel(AppConfig().paragraph_pause_threshold)
        self._legacy_utterance_id = 0
        self.auto_scroll_enabled = True
        self._programmatic_scroll = False
        self.mode_states = {mode.value: {"finals": [], "partial": "", "metrics": {}}
                            for mode in PerformanceMode}
        self.selected_mode = PerformanceMode.BALANCED
        self.recommended_mode = PerformanceMode.BALANCED
        self.ever_started = False
        self.record_started_at: float | None = None
        self.last_processing_refresh = 0.0
        self._meter_phase = 0
        self.live_result_deadline = AppConfig().max_result_latency_seconds
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
        self.performance.currentTextChanged.connect(
            lambda value: self.performance_label.setText(f"Performance: {value}"))
        self.display_mode = QComboBox()
        self.display_mode.addItems(["Progressive", "Compare All", "Single Mode"])
        self.display_mode.setToolTip(
            "Progressive replaces one live output as Fast, Balanced, then Accurate complete")
        self.script = QComboBox()
        self.script.addItems(["Original", "Devanagari", "Latin / Roman"])
        self.script.setMinimumWidth(125)
        self.language_mode = QComboBox()
        self.language_mode.addItems(["Hindi / Hinglish", "Auto", "English"])
        self.language_mode.setMinimumWidth(155)
        self.capture_source = QComboBox()
        self.capture_source.addItems(["Microphone", "System audio (legacy)"])
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
        self.record_output_container.setFixedHeight(680)
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
        self.listening_card = QWidget()
        self.listening_card.setObjectName("listeningCard")
        self.listening_card.setStyleSheet(
            "#listeningCard { background:#142b24;border:1px solid #24523d;"
            "border-radius:6px; } #listeningCard QLabel { color:#54d66b;"
            "font-weight:bold; }")
        listening_layout = QHBoxLayout(self.listening_card)
        listening_layout.setContentsMargins(10, 7, 10, 7)
        self.listening_card_label = QLabel("Ready")
        self.audio_level_label = QLabel("▁▁▁▁▁▁▁")
        self.audio_level_label.setAccessibleName("Audio level indicator")
        listening_layout.addWidget(self.listening_card_label)
        listening_layout.addStretch()
        listening_layout.addWidget(self.audio_level_label)
        button_layout.addWidget(self.listening_card)

        transcript_panel = QWidget()
        transcript_layout = QVBoxLayout(transcript_panel)
        transcript_layout.setContentsMargins(0, 0, 0, 0)
        transcript_layout.setSpacing(4)
        processing_title = QLabel("PROCESSING — LIVE UPDATE")
        processing_title.setStyleSheet("color:#5aa9ff;font-weight:bold;padding:5px")
        self.processing_output = QPlainTextEdit()
        self.processing_output.setReadOnly(True)
        self.processing_output.setMaximumHeight(125)
        self.processing_output.setPlaceholderText("Listening for live speech…")
        self.processing_output.setStyleSheet(
            "color:#e8f2ff;background:#172231;border:1px solid #29466d;padding:7px")
        final_header = QHBoxLayout()
        final_title = QLabel("FINAL TRANSCRIPT")
        final_title.setStyleSheet("color:#54d66b;font-weight:bold;padding:5px")
        self.jump_to_latest = QPushButton("↓ Jump to latest")
        self.jump_to_latest.hide()
        self.jump_to_latest.clicked.connect(self._jump_to_latest)
        final_header.addWidget(final_title)
        final_header.addStretch()
        final_header.addWidget(self.jump_to_latest)
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

        transcript_layout.addWidget(processing_title)
        transcript_layout.addWidget(self.processing_output)
        transcript_layout.addLayout(final_header)
        transcript_layout.addWidget(self.record_output, stretch=1)
        self.record_output.verticalScrollBar().valueChanged.connect(
            self._final_scroll_changed)

        top_row.addLayout(button_layout, stretch=1)
        top_row.addWidget(transcript_panel, stretch=5)
        outer_layout.addLayout(top_row)

        self.live_status_bar = QWidget()
        self.live_status_bar.setObjectName("liveStatusBar")
        self.live_status_bar.setStyleSheet(
            "#liveStatusBar { background:#171c23;border-top:1px solid #343b46; } "
            "#liveStatusBar QLabel { color:#cbd2dc;padding:2px 10px; }")
        status_layout = QHBoxLayout(self.live_status_bar)
        status_layout.setContentsMargins(5, 4, 5, 4)
        status_layout.setSpacing(0)
        self.live_state_label = QLabel("● Status: Ready")
        self.live_mode_label = QLabel("▰ Mode: Balanced")
        self.live_words_label = QLabel("Words: 0")
        self.live_characters_label = QLabel("Characters: 0")
        for index, label in enumerate((self.live_state_label, self.live_mode_label,
                                       self.live_words_label,
                                       self.live_characters_label)):
            if index:
                label.setStyleSheet("border-left:1px solid #343b46")
            status_layout.addWidget(label, stretch=1)
        outer_layout.addWidget(self.live_status_bar)

        self.comparison_panel = QWidget()
        comparison_layout = QVBoxLayout(self.comparison_panel)
        comparison_layout.setContentsMargins(0, 0, 0, 0)
        comparison_layout.setSpacing(5)
        self.mode_outputs, self.mode_statuses = {}, {}
        self.mode_metrics, self.mode_buttons, self.mode_rows = {}, {}, {}
        for mode in PerformanceMode:
            row = QWidget()
            row.setObjectName(f"comparisonRow_{mode.value}")
            row_layout = QVBoxLayout(row)
            row_layout.setContentsMargins(8, 5, 8, 5)
            header = QHBoxLayout()
            title = QLabel(mode.value.upper())
            title.setStyleSheet("font-weight: bold; color: #f5f7fa")
            status = QLabel("● Waiting")
            status.setStyleSheet("color: #aeb7c4")
            header.addWidget(title)
            header.addStretch()
            header.addWidget(status)
            output = QTextEdit()
            output.setReadOnly(True)
            output.setMaximumHeight(92)
            output.setPlaceholderText(f"{mode.value.title()} transcript…")
            output.setStyleSheet("color: #f5f7fa; background: #171a1f; border: 0")
            footer = QHBoxLayout()
            metrics = QLabel("Relative accuracy n/a | First n/a | Final n/a")
            metrics.setStyleSheet("color: #cbd2dc")
            button = QPushButton(f"Select {mode.value.title()}")
            button.clicked.connect(lambda _checked=False, selected=mode: self.select_mode(selected))
            footer.addWidget(metrics)
            footer.addStretch()
            footer.addWidget(button)
            row_layout.addLayout(header)
            row_layout.addWidget(output)
            row_layout.addLayout(footer)
            comparison_layout.addWidget(row)
            self.mode_rows[mode] = row
            self.mode_outputs[mode] = output
            self.mode_statuses[mode] = status
            self.mode_metrics[mode] = metrics
            self.mode_buttons[mode] = button
        decision = QHBoxLayout()
        self.recommended_label = QLabel("Recommended: BALANCED")
        self.selected_label = QLabel("Selected: BALANCED")
        for label in (self.recommended_label, self.selected_label):
            label.setStyleSheet("color: white; font-weight: bold")
        self.use_selected_button = QPushButton("Use Selected Output")
        self.use_selected_button.clicked.connect(self.use_selected_output)
        decision.addWidget(self.recommended_label)
        decision.addWidget(self.selected_label)
        decision.addStretch()
        decision.addWidget(self.use_selected_button)
        comparison_layout.addLayout(decision)
        top_row.addWidget(self.comparison_panel, stretch=5)
        self.comparison_panel.hide()
        self.select_mode(PerformanceMode.BALANCED, persist=False)

        self.segment_table = QTableWidget(0, 1)
        self.segment_table.setHorizontalHeaderLabels(["LIVE TRANSCRIPT"])
        self.segment_table.setStyleSheet(
            "QTableWidget { background:#171a1f; color:#f5f7fa; gridline-color:#3b414b; } "
            "QHeaderView::section { background:#252a32; color:white; padding:6px; }")
        self.segment_table.verticalHeader().hide()
        self.segment_table.horizontalHeader().setStretchLastSection(True)
        self.segment_table.setVerticalScrollMode(
            QAbstractItemView.ScrollMode.ScrollPerPixel)
        self.segment_rows: dict[int, int] = {}
        self.segment_states: dict[int, dict] = {}
        top_row.addWidget(self.segment_table, stretch=5)
        self.segment_table.hide()  # retained as a diagnostic model, not transcript UI

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
        self.btn_copy_transcript.clicked.connect(self.copy_selected)
        self.btn_lang_eng.clicked.connect(lambda: self._select_language("English"))
        self.btn_lang_hin.clicked.connect(lambda: self._select_language("Hindi / Hinglish"))
        self.btn_lang_hing.clicked.connect(lambda: self._select_language("Auto"))
        self.language_mode.currentTextChanged.connect(self._sync_language_buttons)
        self.language_mode.currentTextChanged.connect(self._switch_language_if_running)
        self._sync_language_buttons()

    def _select_language(self, label: str) -> None:
        self.language_mode.setCurrentText(label)
        self._sync_language_buttons()

    def _sync_language_buttons(self, *_args) -> None:
        selected = self.language_mode.currentText()
        self.btn_lang_eng.setEnabled(selected != "English")
        self.btn_lang_hin.setEnabled(selected != "Hindi / Hinglish")
        self.btn_lang_hing.setEnabled(selected != "Auto")

    def _switch_language_if_running(self, *_args) -> None:
        modes = {"Hindi / Hinglish": "hi", "Auto": "auto", "English": "en"}
        script = self.script.currentText().lower()
        script = "latin" if script == "latin / roman" else script
        self.current_partial = ""
        self.live_transcript.clear_processing()
        self._render_processing()
        self.controller.switch_recognition_language(
            modes[self.language_mode.currentText()], script)

    def _start_system_move(self) -> None:
        handle = self.windowHandle()
        if handle is not None and hasattr(handle, "startSystemMove"):
            handle.startSystemMove()

    def _update_record_timer(self) -> None:
        if self.record_started_at is None:
            return
        self.record_timer_label.setText(format_recording_time(
            time.monotonic() - self.record_started_at))
        if self.controller.running:
            # Reuse the existing 250 ms timer: no extra animation thread or
            # high-frequency paint work competes with capture/ASR.
            frames = ("▁▃▆▄▂▅▃", "▂▅▃▇▄▂▆", "▃▆▂▅▇▃▁", "▁▄▇▃▅▂▄")
            self._meter_phase = (self._meter_phase + 1) % len(frames)
            self.audio_level_label.setText(frames[self._meter_phase])
        now = time.monotonic()
        if now - self.last_processing_refresh >= 1.0:
            self.last_processing_refresh = now
            for segment_id, state in tuple(self.segment_states.items()):
                if any(item["status"] == "PROCESSING" for item in state["modes"].values()):
                    self._render_segment(segment_id, refresh_successor=False)

    def _connect_signals(self) -> None:
        self.signals.partial_text.connect(self.show_partial)
        self.signals.final_text.connect(self.add_final)
        self.signals.stale_final_text.connect(self.add_stale_final)
        self.signals.language_changed.connect(
            lambda value: self.language.setText(f"Language: {value}"))
        self.signals.status_changed.connect(self._set_status)
        self.signals.error.connect(self.show_error)
        self.signals.translation_ready.connect(self.show_translation)
        self.signals.mode_text.connect(self.show_mode_text)
        self.signals.mode_status.connect(self.show_mode_status)
        self.signals.mode_error.connect(self.show_mode_error)

    def _set_status(self, value: str) -> None:
        """Keep the window status, sidebar card, and footer in sync."""
        self.status.setText(value)
        normalized = value.rstrip("…. ") or "Ready"
        lowered = normalized.casefold()
        if "error" in lowered:
            state, color = "Error", "#ff6174"
        elif "stop" in lowered:
            state, color = "Stopped", "#aeb7c4"
        elif "process" in lowered or "switch" in lowered or "start" in lowered:
            state, color = "Processing", "#5aa9ff"
        elif "listen" in lowered or self.controller.running:
            state, color = "Listening", "#54d66b"
        else:
            state, color = "Ready", "#aeb7c4"
        self.listening_card_label.setText(f"{state}…" if state == "Listening" else state)
        self.live_state_label.setText(f"● Status: {state}")
        self.live_state_label.setStyleSheet(f"color:{color};font-weight:bold")
        if state != "Listening":
            self.audio_level_label.setText("▁▁▁▁▁▁▁")

    def _update_transcript_statistics(self) -> None:
        text = self.live_transcript.clean_text()
        self.live_words_label.setText(f"Words: {len(text.split())}")
        self.live_characters_label.setText(f"Characters: {len(text)}")

    def _sync_display_mode(self, *_args) -> None:
        self.comparison_panel.hide()
        self.performance_label.setText("FAST live transcription — refinements deferred")

    @property
    def selected_transcript(self) -> str:
        state = self.mode_states[self.selected_mode.value]
        return compose_live_transcript(state["finals"], state["partial"])

    def select_mode(self, mode: PerformanceMode, persist: bool = True) -> None:
        """Change the user's choice only; recommendations never call this method."""
        self.selected_mode = mode
        self.selected_label.setText(f"Selected: {mode.value.upper()}")
        for candidate in PerformanceMode:
            selected = candidate is mode
            self.mode_buttons[candidate].setText(
                "✓ Selected" if selected else f"Select {candidate.value.title()}")
            self.mode_rows[candidate].setStyleSheet(
                "background: #29466d; border: 2px solid #65a8ff; border-radius: 5px;"
                if selected else
                "background: #252a32; border: 1px solid #3b414b; border-radius: 5px;")
        if persist:
            self._save_selection()

    def _save_selection(self) -> None:
        segment_id = max((len(state["finals"]) for state in self.mode_states.values()),
                         default=0)
        record = {
            "segment_id": segment_id,
            **{mode.value: compose_live_transcript(
                self.mode_states[mode.value]["finals"],
                self.mode_states[mode.value]["partial"])
               for mode in PerformanceMode},
            "selected_mode": self.selected_mode.value,
            "selected_text": self.selected_transcript,
            "recommended_mode": self.recommended_mode.value,
        }
        path = Path("evaluation/mode_comparison/user_selections.jsonl")
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")

    def copy_selected(self) -> None:
        QApplication.clipboard().setText(self.full_script())

    def full_script(self) -> str:
        """Return clean finalized paragraphs, excluding live/diagnostic text."""
        return self.live_transcript.clean_text()

    def use_selected_output(self) -> None:
        text = self.selected_transcript
        if text:
            QApplication.clipboard().setText(text)
            self.controller.translate(text)
            self._save_selection()
            self._set_status(f"Using {self.selected_mode.value.upper()} output")

    def _ensure_segment(self, segment_id: int, metrics: dict | None = None) -> dict:
        if segment_id in self.segment_states:
            state = self.segment_states[segment_id]
        else:
            row = descending_segment_row(self.segment_rows, segment_id)
            scrollbar = self.segment_table.verticalScrollBar()
            previous_scroll = scrollbar.value()
            was_near_top = previous_scroll <= 8
            previous_top_row = self.segment_table.rowAt(0)
            for existing_id, existing_row in tuple(self.segment_rows.items()):
                if existing_row >= row:
                    self.segment_rows[existing_id] = existing_row + 1
            self.segment_table.insertRow(row)
            self.segment_table.setRowHeight(row, 90)
            self.segment_rows[segment_id] = row
            state = {"start": 0.0, "end": 0.0, "display_text": "",
                     "display_source": None, "modes": {
                mode.value: {"raw": None, "partial": "",
                             "status": ("WAITING" if mode is PerformanceMode.FAST
                                        else "DEFERRED"),
                             "latency": None, "processing_started": None}
                for mode in PerformanceMode}}
            self.segment_states[segment_id] = state
            cell = QTextEdit()
            cell.setReadOnly(True)
            cell.setStyleSheet("background:#171a1f;color:#f5f7fa;border:0;padding:5px")
            cell.setHtml("<i>Processing speech…</i>")
            self.segment_table.setCellWidget(row, 0, cell)
            inserted_above_view = previous_top_row >= 0 and row <= previous_top_row
            # Follow a newest live segment only when already near the top.
            # Otherwise compensate for insertion above the viewport so the
            # older text being inspected stays at the same visual position.
            QTimer.singleShot(
                0, lambda old=previous_scroll, follow=(row == 0 and was_near_top),
                compensate=inserted_above_view:
                scrollbar.setValue(0 if follow else old + 90 if compensate else old))
        if metrics:
            state["start"] = metrics.get("start_time", state["start"])
            state["end"] = metrics.get("end_time", state["end"])
        return state

    def show_mode_text(self, segment_id: int, mode_name: str, text: str,
                       final: bool, metrics: dict) -> None:
        slot_entered = time.monotonic()
        emitted_at = metrics.get("signal_emitted_at")
        if emitted_at is not None:
            metrics["result_coordination"] = max(0.0, slot_entered - emitted_at)
        state = self._ensure_segment(segment_id, metrics)
        if metrics.get("duplicate"):
            self._remove_segment(segment_id)
            LOGGER.info("Suppressed duplicate live segment | segment=%s mode=%s",
                        segment_id, mode_name)
            return
        mode_state = state["modes"][mode_name]
        mode_state["metadata"] = metrics
        script_valid = metrics.get("script_valid", True)
        if final:
            if script_valid:
                mode_state["raw"] = text
                mode_state["partial"] = ""
                mode_state["status"] = "FINAL"
            else:
                # Do not erase a valid live hypothesis merely because the final
                # refinement came back in an invalid script. Keep displaying
                # that hypothesis while the other profiles get their chance.
                mode_state["status"] = "SCRIPT MISMATCH"
            mode_state["latency"] = metrics.get("final_latency")
            mode_state["processing_started"] = None
        else:
            if script_valid:
                mode_state["partial"] = text
                mode_state["status"] = "PARTIAL"
            else:
                mode_state["status"] = "SCRIPT MISMATCH"
            mode_state["latency"] = metrics.get("first_partial_latency")
            mode_state["processing_started"] = None
        self._render_segment(segment_id)
        if final:
            existed = segment_id in self.live_transcript._finals
            changed = self.live_transcript.commit(
                segment_id, state.get("display_text") or text,
                language=metrics.get("language", ""),
                start_time=metrics.get("start_time", 0.0),
                end_time=metrics.get("end_time", 0.0),
                refinement_level=mode_name)
            self._render_processing()
            if changed:
                self._render_final()
                diagnostics = pipeline_diagnostics()
                diagnostics.stage(
                    segment_id, "UI", view="FINAL",
                    action="REPLACE" if existed else "APPEND",
                    paragraph=len(self.live_transcript.paragraphs()), accepted=True,
                    source=mode_name.upper())
                diagnostics.terminal(
                    segment_id, "FINAL", "delivered_to_final_ui", "UI",
                    action="REPLACE" if existed else "APPEND")
        else:
            self.live_transcript.update_partial(text, segment_id)
            self._render_processing(mode_name, metrics)
            diagnostics = pipeline_diagnostics()
            ui_latency = max(0.0, time.monotonic() -
                             metrics.get("candidate_speech_at", time.monotonic()))
            if text and mode_name == "fast":
                diagnostics.first_text_seconds.append(ui_latency)
            diagnostics.stage(
                segment_id, "UI", view="PROCESSING", source=mode_name.upper(),
                text_length=len(text), accepted=True,
                signal_to_ui_ms=max(0.0, (time.monotonic() -
                    metrics.get("signal_emitted_at", time.monotonic())) * 1000))
        render_finished = time.monotonic()
        metrics["ui_render"] = render_finished - slot_entered
        if metrics.get("language_switch_requested_at"):
            metrics["first_ui_display_at"] = render_finished
            switch_ms = max(0.0, (
                metrics.get("language_ready_at") or render_finished) -
                metrics["language_switch_requested_at"]) * 1000
            speech_to_fast_ms = max(0.0, (
                (metrics.get("first_fast_result_at") or render_finished) -
                (metrics.get("first_new_speech_at") or render_finished))) * 1000
            fast_to_ui_ms = max(0.0, render_finished - (
                metrics.get("first_fast_result_at") or render_finished)) * 1000
            LOGGER.info("[LANG-SWITCH] latency generation=%s config=%.1fms "
                        "speech-to-fast=%.1fms fast-to-ui=%.1fms",
                        metrics.get("language_generation"), switch_ms,
                        speech_to_fast_ms, fast_to_ui_ms)
        candidate_at = metrics.get("candidate_speech_at")
        if candidate_at:
            metrics["total_visible_latency"] = render_finished - candidate_at
        LOGGER.debug(
            "[FINAL DISPLAY] segment=%s mode=%s raw=%r processed=%r displayed=%r "
            "detected_language=%s detected_script=%s requested_script=%s script_valid=%s",
            segment_id, mode_name, metrics.get("raw_text"), metrics.get("processed_text"),
            state.get("display_text"), metrics.get("detected_language"),
            metrics.get("detected_script"), metrics.get("requested_script"), script_valid)
        if text and state.get("display_text") != text and not final:
            LOGGER.warning(
                "[UI CANDIDATE MISMATCH] segment=%s incoming_mode=%s incoming=%r "
                "selected_source=%s selected=%r; preserving newest partial in Processing",
                segment_id, mode_name, text, state.get("display_source"),
                state.get("display_text"))
        LOGGER.debug(
            "[LATENCY] segment=%s mode=%s final=%s vad=%.3fs buffer=%.3fs "
            "queue=%.3fs preprocess=%.3fs inference=%.3fs text=%.3fs "
            "language_script=%.3fs coordination=%.3fs ui=%.3fs total=%.3fs",
            segment_id, mode_name, final,
            metrics.get("vad_wait") or 0.0, metrics.get("audio_buffer") or 0.0,
            metrics.get("queue_delay") or 0.0,
            metrics.get("asr_preprocessing") or 0.0,
            metrics.get("whisper_inference") or metrics.get("asr_time") or 0.0,
            metrics.get("text_processing") or 0.0,
            metrics.get("language_script_processing") or 0.0,
            metrics.get("result_coordination") or 0.0,
            metrics.get("ui_render") or 0.0,
            metrics.get("total_visible_latency") or metrics.get("result_latency") or 0.0)

    def _remove_segment(self, segment_id: int) -> None:
        """Remove a rejected live row and keep the stable row index map valid."""
        row = self.segment_rows.pop(segment_id, None)
        self.segment_states.pop(segment_id, None)
        if row is None:
            return
        self.segment_table.removeRow(row)
        for existing_id, existing_row in tuple(self.segment_rows.items()):
            if existing_row > row:
                self.segment_rows[existing_id] = existing_row - 1

    def _render_segment(self, segment_id: int, refresh_successor: bool = True) -> None:
        state = self.segment_states[segment_id]
        source_name, text = best_refinement_candidate(
            {mode: data["raw"] for mode, data in state["modes"].items()},
            {mode: data["partial"] for mode, data in state["modes"].items()})
        source = PerformanceMode(source_name) if source_name else None

        display_text = text
        previous_ids = [value for value in self.segment_states if value < segment_id]
        if display_text and previous_ids:
            previous = self.segment_states[max(previous_ids)]
            # Remove boundary overlap only for adjacent audio and at least two
            # matching words, preserving intentional single-word repetition.
            if state["start"] <= previous["end"] + 0.5:
                display_text = remove_history_overlap(
                    previous.get("display_text", ""), display_text, min_overlap=2)
        state["display_text"] = display_text
        state["display_source"] = source.value if source else None

        row = self.segment_rows[segment_id]
        cell = self.segment_table.cellWidget(row, 0)
        timing = []
        now = time.monotonic()
        for mode in PerformanceMode:
            data = state["modes"][mode.value]
            if data["status"] == "PROCESSING" and data["processing_started"] is not None:
                elapsed = now - data["processing_started"]
                if elapsed >= self.live_result_deadline:
                    timing.append(
                        f'{mode.value.upper()} is still processing after '
                        f'{format_processing_duration(self.live_result_deadline)}; '
                        'result will appear when ready')
                else:
                    timing.append(
                        f'{mode.value.upper()} processing '
                        f'{format_processing_duration(elapsed)} elapsed')
            elif data["latency"] is not None:
                timing.append(
                    f'{mode.value.upper()} took {format_processing_duration(data["latency"])}')
        timing_text = " · ".join(timing) or "waiting for first result"
        if not display_text:
            statuses = {item["status"] for item in state["modes"].values()}
            terminal = {"FINAL", "ERROR", "SCRIPT MISMATCH", "DEFERRED"}
            if statuses <= terminal:
                body = "<i>No speech was recognized in this segment.</i>"
            else:
                body = f"<i>Processing speech…</i><br><small>{timing_text}</small>"
        else:
            badge = source.value.upper()
            body = (f'<div style="font-size:15px">{escape(display_text)}</div>'
                    f'<div style="color:#8fa3bb;font-size:10px;margin-top:4px">'
                    f'{state["start"]:07.3f}s → {state["end"]:07.3f}s '
                    f'· refined by {badge}<br>{timing_text}</div>')
        cell.setHtml(body)

        # If an older segment refinement changes its boundary, conservatively
        # refresh only its immediate chronological successor, never the document.
        next_ids = [value for value in self.segment_states if value > segment_id]
        if refresh_successor and next_ids:
            next_id = min(next_ids)
            if self.segment_states[next_id].get("display_source"):
                self._render_segment(next_id, refresh_successor=False)

    def _update_mode_metrics(self, mode: PerformanceMode) -> None:
        state = self.mode_states[mode.value]
        metrics = state["metrics"]
        def value(key, suffix="", digits=2):
            item = metrics.get(key)
            return "n/a" if item is None else f"{item:.{digits}f}{suffix}"
        agreement = state.get("agreement")
        agreement_text = "n/a" if agreement is None else f"{agreement:.1f}% (agreement)"
        self.mode_metrics[mode].setText(
            f"Relative accuracy {agreement_text} | "
            f"First {value('first_partial_latency', 's')} | "
            f"Final {value('final_latency', 's')} | ASR {value('asr_time', 's')} | "
            f"RTF {value('real_time_factor')} | {metrics.get('language', '—')}")

    def _render_mode_comparison(self) -> None:
        """Render finalized pauses line-by-line and highlight every differing word."""
        states = self.mode_states
        max_finals = max((len(state["finals"]) for state in states.values()), default=0)
        rendered = {mode.value: [] for mode in PerformanceMode}
        # Keep the comparison focused: only the previous and latest pause are
        # visible. Full untouched history remains in mode_states for Copy/Save.
        visible_start = max(0, max_finals - 2)
        for display_index, segment_index in enumerate(range(visible_start, max_finals), 1):
            present = {
                mode.value: segment_index < len(states[mode.value]["finals"])
                for mode in PerformanceMode
            }
            segment = {
                mode.value: (states[mode.value]["finals"][segment_index]
                             if segment_index < len(states[mode.value]["finals"]) else "")
                for mode in PerformanceMode
            }
            complete = all(present.values())
            highlighted = comparison_diff_html(segment) if complete else {
                mode: escape(text) for mode, text in segment.items()}
            agreements = (comparison_agreement_percentages(segment)
                          if complete else {mode.value: None for mode in PerformanceMode})
            for mode in PerformanceMode:
                states[mode.value]["agreement"] = agreements[mode.value]
                content = highlighted[mode.value]
                if not present[mode.value]:
                    content = "<i>Waiting for this mode…</i>"
                elif not content:
                    content = "<i>No speech recognized for this line.</i>"
                rendered[mode.value].append(
                    f'<div style="margin:3px 0 7px 0"><span style="color:#8fa3bb;'
                    f'font-size:10px">LINE {display_index}</span><br>{content}</div>')
        for mode in PerformanceMode:
            partial = states[mode.value]["partial"]
            parts = list(rendered[mode.value])
            if partial:
                parts.append(
                    '<div style="color:#d9e6f5;border-left:3px solid #5aa9ff;'
                    f'padding-left:6px"><i>LIVE</i><br>{escape(partial)}</div>')
            self.mode_outputs[mode].setHtml("".join(parts))

    def show_mode_status(self, segment_id: int, mode_name: str, status: str) -> None:
        if status.upper() in {"DUPLICATE", "NO SPEECH"}:
            self._remove_segment(segment_id)
            return
        state = self._ensure_segment(segment_id)
        mode_state = state["modes"][mode_name]
        mode_state["status"] = status.upper()
        if status.upper() == "PROCESSING" and mode_state["processing_started"] is None:
            mode_state["processing_started"] = time.monotonic()
        elif status.upper() in {
                "FINAL", "ERROR", "EXPIRED", "LISTENING", "PARTIAL", "SCRIPT MISMATCH",
                "DUPLICATE"}:
            mode_state["processing_started"] = None
        if status.upper() == "LISTENING":
            mode_state["partial"] = ""
        self._render_segment(segment_id)

    def show_mode_error(self, segment_id: int, mode_name: str, message: str) -> None:
        state = self._ensure_segment(segment_id)
        state["modes"][mode_name].update(
            error=message, status="ERROR", processing_started=None)
        self._render_segment(segment_id)

    def start_listening(self) -> None:
        self.ever_started = True
        # Retain the ID-aware scheduler, but interactive snapshots use its
        # dedicated FAST lane. Multi-profile fan-out is evaluation-only.
        compare_all = True
        # Live capture must not queue minutes of immutable audio while Whisper
        # runs slower than real time. FAST uses newest-value scheduling; profile
        # comparisons belong to the prerecorded evaluation path.
        mode = PerformanceMode.FAST
        recognition_modes = {
            "Hindi / Hinglish": "hi", "Auto": "auto", "English": "en",
        }
        selected_script = self.script.currentText().lower()
        config = AppConfig(performance_mode=mode,
                           script_mode=("latin" if selected_script == "latin / roman"
                                        else selected_script),
                           language_mode=recognition_modes[self.language_mode.currentText()],
                           capture_source=("loopback" if self.capture_source.currentText() ==
                                           "System audio (legacy)" else "microphone"),
                           translation_enabled=self.translation_toggle.isChecked(),
                           # CPU comparison can finalize slower than capture.
                           # Never let ASR backpressure stop VAD from draining
                           # the one shared capture stream.
                           asr_keep_latest_final=True,
                           compare_live_partials=False,
                           max_audio_queue=100,
                           max_asr_queue=1,
                           # Supplied Hindi traces showed uninterrupted audio
                           # being stopped at 13.59s, before the old 15s forced
                           # boundary, so no final could exist. Bound continuous
                           # live chunks while paragraphing keeps prose natural.
                           max_utterance_seconds=8.0)
        config.diagnostics_enabled = DEBUG_DIAGNOSTICS
        self.performance_label.setText("FAST live transcription — refinements deferred")
        self._set_status("Starting…")
        self.live_mode_label.setText(f"▰ Mode: {mode.value.title()}")
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.performance.setEnabled(False)
        self.script.setEnabled(False)
        self.language_mode.setEnabled(True)
        self._sync_language_buttons()
        self.capture_source.setEnabled(False)
        self.translation_toggle.setEnabled(False)
        self.translation.setVisible(config.translation_enabled)
        self.display_mode.setEnabled(False)
        for update in self.controller.start_stream(config, compare_all):
            self._set_status(update.message)
        self._set_status("Listening")
        self.record_started_at = time.monotonic()
        self.record_timer_label.setText("00:00")
        self.record_timer.start()

    def stop_listening(self) -> None:
        self._set_status("Stopping…")
        # Stop invalidates the live generation and returns without joining ASR.
        self.controller.stop()
        self._finish_stop_ui()

    def _finish_stop_ui(self) -> None:
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.performance.setEnabled(True)
        self.script.setEnabled(True)
        self.language_mode.setEnabled(True)
        self._sync_language_buttons()
        self.capture_source.setEnabled(True)
        self.translation_toggle.setEnabled(True)
        self.display_mode.setEnabled(True)
        self._sync_display_mode()
        self.record_timer.stop()
        self._update_record_timer()
        self.live_transcript.clear_processing()
        self.current_partial = ""
        self._render_processing()
        self._set_status("Ready")

    def add_final(self, text: str) -> None:
        started = time.monotonic()
        log_print(f"[GUI] final signal received chars={len(text)} text={text!r}")
        self.final_history.append(text)
        self.current_partial = ""
        self._legacy_utterance_id += 1
        self.live_transcript.commit(self._legacy_utterance_id, text)
        self.live_transcript.clear_processing()
        self._render_processing()
        self._render_final()
        log_print(f"[GUI] final rendered in {(time.monotonic() - started) * 1000:.1f}ms")
        self.controller.translate(text)  # display has already happened

    def add_stale_final(self, text: str) -> None:
        """Preserve required old speech without clearing the new live partial."""
        self.final_history.append(text)
        self._legacy_utterance_id += 1
        self.live_transcript.commit(self._legacy_utterance_id, text)
        self._render_final()
        self.controller.translate(text)

    def show_partial(self, text: str) -> None:
        started = time.monotonic()
        log_print(f"[GUI] partial signal received chars={len(text)} text={text!r}")
        self.current_partial = text
        self.live_transcript.update_partial(text)
        self._render_processing()
        log_print(f"[GUI] partial rendered in {(time.monotonic() - started) * 1000:.1f}ms")

    def _render_live_text(self) -> None:
        """Compatibility renderer for callers predating the split display."""
        self._render_processing()
        self._render_final()

    def _render_processing(self, mode: str = "fast", metrics: dict | None = None) -> None:
        """Update only the inexpensive, single live candidate widget."""
        text = self.live_transcript.processing_text
        self.processing_output.setPlainText(f"● {text}" if text else "")
        confidence = (metrics or {}).get("confidence")
        if text:
            suffix = f" • confidence {confidence:.2f}" if confidence is not None else ""
            self.processing_output.setToolTip(f"{mode.upper()}{suffix} • updating…")
        else:
            self.processing_output.setToolTip("")

    def _render_final(self) -> None:
        """Render only after a final is added/refined and respect reader position."""
        scrollbar = self.record_output.verticalScrollBar()
        old_value = scrollbar.value()
        self._programmatic_scroll = True
        self.record_output.setPlainText(self.live_transcript.clean_text())
        if self.auto_scroll_enabled:
            scrollbar.setValue(scrollbar.maximum())
        else:
            scrollbar.setValue(min(old_value, scrollbar.maximum()))
        self._programmatic_scroll = False
        self._update_transcript_statistics()

    def _final_scroll_changed(self, value: int) -> None:
        if self._programmatic_scroll:
            return
        scrollbar = self.record_output.verticalScrollBar()
        near_bottom = scrollbar.maximum() - value <= 24
        self.auto_scroll_enabled = near_bottom
        self.jump_to_latest.setVisible(not near_bottom)

    def _jump_to_latest(self) -> None:
        self.auto_scroll_enabled = True
        self._programmatic_scroll = True
        self.record_output.verticalScrollBar().setValue(
            self.record_output.verticalScrollBar().maximum())
        self._programmatic_scroll = False
        self.jump_to_latest.hide()

    def show_translation(self, text: str) -> None:
        self.translation.setPlainText(f"Translation: {text}")

    def show_error(self, message: str) -> None:
        self._set_status(f"Error: {message}")
        log_print(f"GUI error: {message}")

    def clear_text(self) -> None:
        self.final_history.clear()
        self.current_partial = ""
        self.live_transcript.clear()
        self.processing_output.clear()
        self.transcription.clear()
        self.translation.clear()
        self._update_transcript_statistics()
        self.segment_table.setRowCount(0)
        self.segment_rows.clear()
        self.segment_states.clear()
        for mode in PerformanceMode:
            self.mode_states[mode.value] = {"finals": [], "partial": "", "metrics": {}}
            self.mode_outputs[mode].clear()
            self.mode_metrics[mode].setText(
                "Relative accuracy n/a | First n/a | Final n/a")
            self.mode_statuses[mode].setText("● Waiting")

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
    global DEBUG_DIAGNOSTICS
    enable()
    args, qt_args = parse_args(argv)
    DEBUG_DIAGNOSTICS = args.debug
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
