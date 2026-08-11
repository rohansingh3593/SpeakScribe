from tests.fixtures.pytest_observability import (
    PytestObserver, TestRecord as ObservationRecord, _artifact_stem, _safe,
)


class Config:
    def __init__(self, debug=False, name=None):
        self.values = {"debug": debug, "--session-name": name}

    def getoption(self, name):
        return self.values[name]


def test_session_creates_complete_unique_layout(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    first = PytestObserver(Config(name="nightly"))
    second = PytestObserver(Config(name="nightly"))
    assert first.path.name == "nightly"
    assert second.path.name == "nightly_1"
    for relative in (
        "session.log", "summary.log", "debug.log", "failures.log", "session.json",
        "success/modules", "success/classes", "success/tests", "failed/modules",
        "failed/classes", "failed/tests", "artifacts/actual_transcripts",
        "artifacts/expected_transcripts", "artifacts/metrics",
    ):
        assert (first.path / relative).exists()


def test_failure_category_respects_pytest_phase(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    observer = PytestObserver(Config())
    record = ObservationRecord("node", 1, "test_x.py", "-", "test_x", "now", 0)
    record.phases["setup"] = {"status": "FAILED"}
    assert observer._category(record, "FAIL") == "SETUP_FAILURE"
    record.phases = {"teardown": {"status": "FAILED"}}
    assert observer._category(record, "FAIL") == "TEARDOWN_FAILURE"


def test_warning_is_recorded_with_node_and_phase_context(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    observer = PytestObserver(Config())
    observer.warning("accuracy near threshold", "runtest", "tests/test_x.py::test_x",
                     ("tests/test_x.py", 10, "test_x"))
    assert "accuracy near threshold" in observer.warnings["tests/test_x.py::test_x"][0]
    assert "WARNING | tests/test_x.py::test_x" in (
        observer.path / "session.log").read_text(encoding="utf-8")


def test_recorded_asr_failure_writes_evidence_without_changing_result(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    observer = PytestObserver(Config())
    observer.collected = [object()]
    record = ObservationRecord(
        "tests/test_speech.py::test_case", 1, "tests/test_speech.py", "-",
        "test_case", "now", 0, {"id": "HI-EDGE-121", "language": "Hindi"})
    record.phases = {
        "setup": {"status": "PASSED", "duration_seconds": .1, "exception": ""},
        "call": {"status": "FAILED", "duration_seconds": 1.2,
                 "exception": "AssertionError: transcript", "captured_stdout": "",
                 "captured_stderr": "", "captured_logs": ""},
        "teardown": {"status": "PASSED", "duration_seconds": .1, "exception": ""},
    }
    record.evaluation = {
        "status": "FAIL", "expected": "काम", "actual": "kaam", "similarity": 20.0,
        "wer": 1.0, "language": "Hindi", "detected_language": "English",
        "quality_flags": ["WRONG_LANGUAGE"], "root_cause": "Language detection",
        "possible_problem": "Script mismatch", "recommended_fix": "Inspect decoder",
    }
    observer.records[record.nodeid] = record
    observer.finish_test(record)
    detail = next((observer.path / "failed/tests").glob("*.log")).read_text(encoding="utf-8")
    assert "FAILURE CATEGORY: LANGUAGE_DETECTION" in detail
    assert "EXPECTED:\nकाम" in detail
    assert "ACTUAL:\nkaam" in detail
    assert (observer.path / "artifacts/metrics/HI-EDGE-121__main.json").is_file()


def test_long_unicode_parameter_ids_produce_short_unique_windows_safe_names(tmp_path):
    escaped_hindi = "test_live_partial_" + "_u0906_u091c_u092e_u0948_u0902" * 12
    other_case = escaped_hindi + "_different"
    directory = tmp_path / "session_2026-08-11_16-47-51" / "success" / "tests"
    first = _artifact_stem(escaped_hindi, directory, "__main.log")
    second = _artifact_stem(other_case, directory, "__main.log")

    assert len(_safe(escaped_hindi)) <= 72
    assert first != second
    assert len(str(directory / f"{first}__main.log")) <= 235


def test_artifact_io_failure_does_not_raise_pytest_internal_error(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    observer = PytestObserver(Config())
    record = ObservationRecord("node", 1, "test_x.py", "-", "test_x", "now", 0)
    observer.records["node"] = record
    monkeypatch.setattr(observer, "finish_test", lambda _record: (_ for _ in ()).throw(
        OSError("path too long")))
    report = type("Report", (), {
        "sections": [], "when": "teardown", "outcome": "passed",
        "duration": 0.01, "failed": False, "longrepr": "",
    })()
    item = type("Item", (), {
        "nodeid": "node", "path": "test_x.py", "cls": None, "name": "test_x",
    })()

    observer.phase_report(item, report)
    assert "OBSERVABILITY ERROR | node" in (
        observer.path / "session.log").read_text(encoding="utf-8")
