from pathlib import Path

from app.audio.audio_pipeline import ASRJob
from app.config.decoding_policy import retry_thresholds
from app.config.settings import AppConfig
from app.processing.text_processing import clean_text


def test_production_modules_are_importable_from_responsibility_packages():
    assert AppConfig().sample_rate == 16_000
    assert ASRJob.__module__ == "app.audio.audio_pipeline"
    assert retry_thresholds(no_speech=.85, log_probability=-2,
                            compression_ratio=2.4) == (.95, -3.0, 4.0)
    assert clean_text("hello", final=True) == "Hello."


def test_root_evaluation_facade_preserves_supported_imports():
    from evaluation.evaluation_runner import compare_transcripts as packaged
    from evaluation_runner import compare_transcripts as compatible
    assert compatible is packaged


def test_implementations_are_not_duplicated_in_repository_root():
    root = Path(__file__).resolve().parents[2]
    moved = {
        "asr_engine.py", "audio_pipeline.py", "config.py", "decoding_policy.py",
        "logger.py", "text_processing.py", "translation.py",
    }
    assert not any((root / name).exists() for name in moved)
    assert (root / "main.py").read_text(encoding="utf-8").count(
        "from app.final_only_main import main") == 1


def test_evaluation_fixture_paths_remain_repository_relative():
    from evaluation.audio_generation import GENERATED_MANIFEST, MANIFEST_PATH
    root = Path(__file__).resolve().parents[2]
    assert MANIFEST_PATH == root / "tests/expected/transcripts.json"
    assert GENERATED_MANIFEST == root / "tests/generated_audio_manifest.json"


def test_recording_settings_strip_is_built_inside_the_compact_panel():
    root = Path(__file__).resolve().parents[2]
    source = (root / "app/main.py").read_text(encoding="utf-8")
    recording_bar = source.split("def _build_recording_bar", 1)[1].split(
        "def _select_language", 1)[0]
    assert "outer_layout.addWidget(self.settings_bar)" in recording_bar
    assert recording_bar.index("outer_layout.addWidget(self.settings_bar)") < (
        recording_bar.index("top_row = QHBoxLayout()"))
    for control in ("self.performance", "self.script", "self.language_mode",
                    "self.capture_source", "self.translation_toggle"):
        assert control in source
