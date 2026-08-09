import pytest

from evaluation.manifest_policy import ManifestPolicyError, validate_manifest


def manifest_with(cases, baseline_count=1):
    return {
        "baseline_count": baseline_count,
        "growth_policy": {
            "extension_required_fields": ["reason", "feature", "type"],
            "test_types": ["regression", "stress"],
        },
        "cases": cases,
    }


def case(case_id, audio, **metadata):
    return {"id": case_id, "audio": audio, **metadata}


def test_basic_baseline_manifest_is_valid_without_extension_metadata():
    baseline = case("BASE-01", "baseline.wav")
    assert validate_manifest(manifest_with([baseline])) == [baseline]


def test_realistic_regression_extension_records_why_it_exists():
    cases = [
        case("BASE-01", "baseline.wav"),
        case("HI-ACCENT-121", "regression/accent.wav",
             reason="Regional pronunciation produced destructive substitutions",
             feature="Hindi accent recognition", type="regression"),
    ]
    assert validate_manifest(manifest_with(cases)) == cases


@pytest.mark.parametrize("missing", ("reason", "feature", "type"))
def test_extension_missing_required_metadata_fails_before_audio_work(missing):
    metadata = {"reason": "Boundary coverage", "feature": "VAD", "type": "stress"}
    metadata.pop(missing)
    cases = [case("BASE-01", "baseline.wav"),
             case("EDGE-121", "stress/edge.wav", **metadata)]
    with pytest.raises(ManifestPolicyError, match=missing):
        validate_manifest(manifest_with(cases))


def test_complex_extension_with_unknown_category_is_rejected():
    cases = [case("BASE-01", "baseline.wav"),
             case("COMBO-121", "integration/combo.wav", reason="Interaction coverage",
                  feature="noise plus language switching", type="miscellaneous")]
    with pytest.raises(ManifestPolicyError, match="unsupported type"):
        validate_manifest(manifest_with(cases))


@pytest.mark.parametrize("field", ("id", "audio"))
def test_duplicate_identity_or_audio_is_rejected(field):
    first = case("BASE-01", "baseline.wav")
    second = case("BASE-02", "second.wav", reason="Regression", feature="ASR",
                  type="regression")
    second[field] = first[field]
    with pytest.raises(ManifestPolicyError, match="duplicate"):
        validate_manifest(manifest_with([first, second]))


def test_removing_a_baseline_case_is_rejected():
    with pytest.raises(ManifestPolicyError, match="removed baseline"):
        validate_manifest(manifest_with([case("BASE-01", "baseline.wav")], baseline_count=2))
