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
    assert (root / "main.py").read_text(encoding="utf-8").count("from app.main import main") == 1


def test_evaluation_fixture_paths_remain_repository_relative():
    from evaluation.audio_generation import GENERATED_MANIFEST, MANIFEST_PATH
    root = Path(__file__).resolve().parents[2]
    assert MANIFEST_PATH == root / "tests/expected/transcripts.json"
    assert GENERATED_MANIFEST == root / "tests/generated_audio_manifest.json"
