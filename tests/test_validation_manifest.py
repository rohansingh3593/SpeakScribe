import json
from pathlib import Path


MANIFEST = Path(__file__).parent / "expected" / "transcripts.json"


BASELINE_COUNT = 120
TEST_TYPES = {
    "baseline", "feature", "integration", "edge", "stress", "regression",
    "performance", "stability",
}


def load_manifest():
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def test_manifest_preserves_four_distinct_baseline_cases_for_thirty_scenarios():
    manifest = load_manifest()
    cases = manifest["cases"]
    assert manifest["baseline_count"] == BASELINE_COUNT
    assert len(cases) >= BASELINE_COUNT
    baseline = cases[:BASELINE_COUNT]
    scenarios = {case["scenario"] for case in baseline}
    assert len(scenarios) == 30
    for scenario in scenarios:
        selected = [case for case in baseline if case["scenario"] == scenario]
        assert len(selected) == 4
        assert len({case["expected"] for case in selected}) == 4


def test_manifest_metadata_and_paths_are_complete_and_unique():
    cases = load_manifest()["cases"]
    assert len({case["id"] for case in cases}) == len(cases)
    assert len({case["audio"] for case in cases}) == len(cases)
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


def test_cases_added_after_the_baseline_explain_the_risk_they_cover():
    manifest = load_manifest()
    policy = manifest["growth_policy"]
    assert policy["minimum_cases"] == BASELINE_COUNT
    assert set(policy["extension_required_fields"]) == {"reason", "feature", "type"}
    assert set(policy["test_types"]) == TEST_TYPES
    cases = manifest["cases"]
    for case in cases[BASELINE_COUNT:]:
        assert case["reason"].strip(), f"{case['id']} must explain why it exists"
        assert case["feature"].strip(), f"{case['id']} must identify its feature"
        assert case["type"] in TEST_TYPES, f"{case['id']} has an unsupported test type"
