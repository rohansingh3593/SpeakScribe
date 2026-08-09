import logging

from evaluation.suite_logging import ProgressTracker, configure_logging, log


def test_eta_waits_for_multiple_samples(monkeypatch):
    tracker = ProgressTracker(10)
    tracker.record("PASS", 1.0)
    tracker.record("WARNING", 2.0)
    assert tracker.eta is None
    tracker.record("FAIL", 3.0)
    assert tracker.eta == 14.0
    message = tracker.progress_message()
    assert "3/10 completed" in message
    assert "PASS 1 WARN 1 FAIL 1" in message


def test_logging_configuration_is_idempotent_and_writes_failure_log(tmp_path):
    logger, _ = configure_logging(log_dir=tmp_path)
    logger, paths = configure_logging(log_dir=tmp_path)
    assert len(logger.handlers) == 3
    log(logger, logging.DEBUG, "engineering detail", component="ASR", test_id="EN-1")
    log(logger, logging.WARNING, "low accuracy", component="TEST", test_id="EN-1",
        test_status="WARNING")
    for handler in logger.handlers:
        handler.flush()
    assert "engineering detail" in paths.main.read_text(encoding="utf-8")
    failure_text = paths.failures.read_text(encoding="utf-8")
    assert "low accuracy" in failure_text
    assert "engineering detail" not in failure_text
