"""Switch-suite evidence collection and automatic Excel reporting."""

from pathlib import Path
import pytest

from tests.switching.reporting import write_report
from tests.switching.support import SwitchEvidence


def pytest_configure(config):
    config._switch_records = []


@pytest.fixture
def record_switch(request):
    def record(evidence):
        assert isinstance(evidence, SwitchEvidence)
        request.node._switch_evidence = evidence
        return evidence
    return record


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    if report.when != "call" or not str(item.path).replace("\\", "/").find("/switching/") >= 0:
        return
    evidence = getattr(item, "_switch_evidence", None)
    if evidence is None:
        # Even setup/call errors must be visible rather than disappearing from Excel.
        test_id = item.name.split("__", 1)[0].replace("test_", "SW-UNMAPPED-")
        evidence = SwitchEvidence(test_id, item.name, "Test completes", "No evidence recorded",
                                  "Infrastructure", "Test completes", "Test aborted")
    if hasattr(report, "wasxfail"):
        status = "XFAIL" if report.skipped else "XPASS"
    elif report.failed:
        status = "FAIL"
    elif report.skipped:
        status = "ERROR"
    else:
        status = "PASS"
    observer = getattr(item.config, "_speakscribe_observer", None)
    log_path = ""
    detail = str(getattr(report, "wasxfail", "")) or (
        str(report.longrepr) if report.failed else "")
    if observer is not None:
        bucket = "failed" if status in {"FAIL", "ERROR", "XFAIL", "XPASS"} else "success"
        artifact = observer.path / bucket / "tests" / f"{evidence.test_id}__main.log"
        log_path = str(artifact.relative_to(Path.cwd()))
        if status in {"FAIL", "ERROR", "XFAIL", "XPASS"}:
            artifact.write_text(
                f"TEST ID: {evidence.test_id}\nSTATUS: {status}\n"
                f"EXPECTED: {evidence.expected_outcome}\nACTUAL: {evidence.actual_outcome}\n"
                f"REASON: {detail or evidence.reason}\n",
                encoding="utf-8")
    item.config._switch_records.append({
        "evidence": evidence, "status": status,
        "exception": detail, "log_path": log_path,
    })


@pytest.hookimpl(trylast=True)
def pytest_sessionfinish(session, exitstatus):
    records = getattr(session.config, "_switch_records", [])
    if records:
        path = write_report(records)
        session.config._switch_report_path = path
        print(f"\nSwitching Excel report: {path}")
