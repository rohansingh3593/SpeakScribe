from pathlib import Path
import subprocess
import sys


def test_public_api_is_lightweight_and_does_not_import_optional_backends():
    completed = subprocess.run(
        [sys.executable, "-c", (
            "import sys; sys.path.insert(0, 'src'); import speakscribe; "
            "assert speakscribe.SpeechToText; "
            "assert speakscribe.SpeakScribeError; "
            "assert 'faster_whisper' not in sys.modules; "
            "assert 'soundcard' not in sys.modules")],
        check=False, capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr


def test_core_library_contains_no_gui_or_tkinter_imports():
    root = Path(__file__).resolve().parents[3]
    sources = "\n".join(path.read_text(encoding="utf-8")
                        for path in (root / "src/speakscribe").rglob("*.py"))
    assert "import tkinter" not in sources
    assert "from tkinter" not in sources
    assert "PyQt" not in sources


def test_tkinter_is_confined_to_external_example():
    root = Path(__file__).resolve().parents[3]
    example = (root / "examples/tkinter_example.py").read_text(encoding="utf-8")
    assert "import tkinter as tk" in example
    assert "queue.Queue()" in example
    assert "threading.Thread" in example


def test_pyqt_recording_panel_is_external_and_consumes_public_api():
    root = Path(__file__).resolve().parents[3]
    example = (root / "examples/pyqt_recording_panel.py").read_text(encoding="utf-8")
    assert "from speakscribe import" in example
    assert "SpeechToText(self.make_config())" in example
    assert "start_continuous(self.events.put" in example
    assert "def _build_recording_bar" in example


def test_existing_desktop_pipeline_consumes_shared_library_preprocessing():
    from app.audio.audio_pipeline import prepare_audio_for_asr as desktop_prepare
    from speakscribe.audio.processor import prepare_audio_for_asr as library_prepare
    assert desktop_prepare is library_prepare
