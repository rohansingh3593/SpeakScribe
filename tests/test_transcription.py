"""Mandatory, manifest-driven production ASR regression suite."""

import importlib.util
from pathlib import Path

import pytest

from evaluation_runner import evaluate_case_with_retries
from tests.audio_generation import ensure_audio
from tests.manifest_policy import load_manifest


if importlib.util.find_spec("faster_whisper") is None:
    raise pytest.UsageError(
        "ASR_DEPENDENCY_ERROR: faster-whisper is not installed in the active Python. "
        "Run: python -m pip install -r requirements.txt; then verify with: "
        'python -c "from faster_whisper import WhisperModel; print(\'ready\')"'
    )


ROOT = Path(__file__).resolve().parents[1]
_, CASES = load_manifest(ROOT / "tests/expected/transcripts.json")
_MODEL_PROVIDER = None


def model_provider():
    """Load one model only after the current case's recording is verified."""
    global _MODEL_PROVIDER
    if _MODEL_PROVIDER is not None:
        return _MODEL_PROVIDER
    from asr_engine import WhisperModelProvider
    _MODEL_PROVIDER = WhisperModelProvider()
    return _MODEL_PROVIDER


@pytest.mark.parametrize("case", CASES, ids=[case["id"] for case in CASES])
def test_prerecorded_transcription(case, record_test_observation):
    audio = ROOT / case["audio"]
    if not audio.is_file():
        generation = ensure_audio(case, ROOT)
        if generation.status == "TTS_GENERATION_ERROR":
            pytest.fail(
                f"TTS_GENERATION_ERROR {case['id']}: {generation.error}",
                pytrace=False,
            )
        case = {**case, "audio_source": generation.audio_source}
    else:
        generation = ensure_audio(case, ROOT)
        case = {**case, "audio_source": generation.audio_source}
    if not audio.is_file():
        pytest.fail(
            f"INVALID_AUDIO: generation reported success but WAV is missing: {audio}",
            pytrace=False,
        )
    result = record_test_observation(
        evaluate_case_with_retries(case, ROOT, model_provider(), retries=1))
    assert result.status != "FAIL", (
        f"{result.case_id}: {result.similarity}% WER={result.wer}; "
        f"root_cause={result.root_cause}; expected={result.expected!r}; "
        f"actual={result.actual!r}"
    )
