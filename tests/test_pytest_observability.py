from tests.pytest_observability import PytestObserver, TestRecord as ObservationRecord


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
