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
RUN_AUDIO_TESTS = os.getenv("SPEAKSCRIBE_RUN_AUDIO_TESTS") == "1"
PARAMETERS = CASES if RUN_AUDIO_TESTS else [None]
PARAMETER_IDS = ([case["id"] for case in CASES]
                 if RUN_AUDIO_TESTS else ["audio-suite-disabled"])


@pytest.fixture(scope="session")
def model_provider():
    if not RUN_AUDIO_TESTS:
        pytest.skip(
            "120-case ASR suite disabled; set SPEAKSCRIBE_RUN_AUDIO_TESTS=1 "
            "after installing all target WAV recordings"
        )
    missing = [case["audio"] for case in CASES if not (ROOT / case["audio"]).is_file()]
    if missing:
        pytest.fail(
            f"Prerecorded corpus incomplete: {len(missing)}/120 WAV files missing; "
            f"first missing file: {missing[0]}",
            pytrace=False,
        )
    from asr_engine import WhisperModelProvider
    return WhisperModelProvider()


@pytest.mark.parametrize("case", PARAMETERS, ids=PARAMETER_IDS)
def test_prerecorded_transcription(case, model_provider):
    if case is None:
        pytest.skip(
            "120-case ASR suite disabled; set SPEAKSCRIBE_RUN_AUDIO_TESTS=1 "
            "after installing all target WAV recordings"
        )
    audio = ROOT / case["audio"]
    if not audio.is_file():
        pytest.fail(f"Required target recording is missing: {audio}")
    result = evaluate_case(case, ROOT, model_provider)
    assert result.status != "FAIL", (
        f"{result.case_id}: {result.similarity}% WER={result.wer}; "
        f"root_cause={result.root_cause}; expected={result.expected!r}; "
        f"actual={result.actual!r}"
    )
