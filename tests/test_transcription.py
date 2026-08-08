"""Mandatory 120-case production ASR regression suite."""

import json
from pathlib import Path

import pytest

from evaluation_runner import evaluate_case


ROOT = Path(__file__).resolve().parents[1]
CASES = json.loads(
    (ROOT / "tests/expected/transcripts.json").read_text(encoding="utf-8")
)["cases"]
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
def test_prerecorded_transcription(case):
    audio = ROOT / case["audio"]
    if not audio.is_file():
        pytest.fail(
            f"Required target recording is missing: {audio}. "
            "All 120 WAV files are mandatory; missing cases are failures, never skips.",
            pytrace=False,
        )
    result = evaluate_case(case, ROOT, model_provider())
    assert result.status != "FAIL", (
        f"{result.case_id}: {result.similarity}% WER={result.wer}; "
        f"root_cause={result.root_cause}; expected={result.expected!r}; "
        f"actual={result.actual!r}"
    )
