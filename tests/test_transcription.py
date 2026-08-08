"""Opt-in, 120-case production ASR regression suite."""

import json
import os
from pathlib import Path

import pytest

from evaluation_runner import evaluate_case


ROOT = Path(__file__).resolve().parents[1]
CASES = json.loads(
    (ROOT / "tests/expected/transcripts.json").read_text(encoding="utf-8")
)["cases"]


@pytest.fixture(scope="session")
def model_provider():
    if os.getenv("SPEAKSCRIBE_RUN_AUDIO_TESTS") != "1":
        pytest.skip("Set SPEAKSCRIBE_RUN_AUDIO_TESTS=1 to run prerecorded ASR")
    from asr_engine import WhisperModelProvider
    return WhisperModelProvider()


@pytest.mark.parametrize("case", CASES, ids=[case["id"] for case in CASES])
def test_prerecorded_transcription(case, model_provider):
    audio = ROOT / case["audio"]
    if not audio.is_file():
        pytest.skip(f"Missing target recording: {audio}")
    result = evaluate_case(case, ROOT, model_provider)
    assert result.status != "FAIL", (
        f"{result.case_id}: {result.similarity}% WER={result.wer}; "
        f"root_cause={result.root_cause}; expected={result.expected!r}; "
        f"actual={result.actual!r}"
    )

