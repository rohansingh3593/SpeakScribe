import json
from pathlib import Path


MANIFEST = Path(__file__).parent / "expected" / "transcripts.json"


def test_manifest_contains_exactly_four_distinct_cases_for_all_thirty_scenarios():
    cases = json.loads(MANIFEST.read_text(encoding="utf-8"))["cases"]
    assert len(cases) == 120
    scenarios = {case["scenario"] for case in cases}
    assert len(scenarios) == 30
    for scenario in scenarios:
        selected = [case for case in cases if case["scenario"] == scenario]
        assert len(selected) == 4
        assert len({case["expected"] for case in selected}) == 4


def test_manifest_metadata_and_paths_are_complete_and_unique():
    cases = json.loads(MANIFEST.read_text(encoding="utf-8"))["cases"]
    assert len({case["id"] for case in cases}) == 120
    assert len({case["audio"] for case in cases}) == 120
    for case in cases:
        assert case["language"] in {"English", "Hindi", "Hinglish"}
        assert case["difficulty"] in {"easy", "medium", "hard", "extreme"}
        assert case["features"]
        assert case["audio_profile"]["speaking_rate"] in {
            "normal", "fast", "very_fast", "slow", "extreme_slow",
        }
        assert "gain_db" in case["audio_profile"]
        assert "noise" in case["audio_profile"]
        assert case["audio"].endswith(".wav")
        assert case["expected"].strip()
