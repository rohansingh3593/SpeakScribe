from pathlib import Path
import subprocess
import sys


def test_public_api_is_lightweight_and_does_not_import_optional_backends():
    completed = subprocess.run(
        [sys.executable, "-c", (
            "import sys; sys.path.insert(0, 'src'); import voice_to_text; "
            "assert voice_to_text.SpeechToText; "
            "assert 'faster_whisper' not in sys.modules; "
            "assert 'soundcard' not in sys.modules")],
        check=False, capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr


def test_core_library_contains_no_gui_or_tkinter_imports():
    root = Path(__file__).resolve().parents[3]
    sources = "\n".join(path.read_text(encoding="utf-8")
                        for path in (root / "src/voice_to_text").rglob("*.py"))
    assert "import tkinter" not in sources
    assert "from tkinter" not in sources
    assert "PyQt" not in sources


def test_tkinter_is_confined_to_external_example():
    root = Path(__file__).resolve().parents[3]
    example = (root / "examples/tkinter_example.py").read_text(encoding="utf-8")
    assert "import tkinter as tk" in example
    assert "queue.Queue()" in example
    assert "threading.Thread" in example


def test_existing_desktop_pipeline_consumes_shared_library_preprocessing():
    from app.audio.audio_pipeline import prepare_audio_for_asr as desktop_prepare
    from voice_to_text.audio.processor import prepare_audio_for_asr as library_prepare
    assert desktop_prepare is library_prepare
