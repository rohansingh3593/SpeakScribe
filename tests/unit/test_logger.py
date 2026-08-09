import logging

from app.utils.logger import (
    cleanup_sessions, configure_logging, configure_runtime_logging, emit_status,
    get_logger, log_print, stream_status,
)


def test_runtime_debug_is_file_only_in_normal_mode(tmp_path, capsys):
    output = tmp_path / "runtime.log"
    logger = configure_runtime_logging(console_level=logging.INFO, output_path=output)
    log_print("internal chunk detail")
    log_print("fallback selected", logging.WARNING)
    for handler in logger.handlers:
        handler.flush()

    captured = capsys.readouterr()
    assert "internal chunk detail" not in captured.err
    assert "fallback selected" in captured.err
    assert "internal chunk detail" in output.read_text(encoding="utf-8")


def test_runtime_debug_is_visible_at_debug_level(tmp_path, capsys):
    configure_runtime_logging(console_level=logging.DEBUG,
                              output_path=tmp_path / "runtime.log")
    log_print("raw transcription")
    assert "raw transcription" in capsys.readouterr().err


def test_central_session_writes_main_debug_error_and_module_logs(tmp_path):
    session = configure_logging(logs_root=tmp_path, session_name="integration")
    logger = get_logger("validation")
    logger.debug("internal decision")
    logger.info("validation started")
    logger.warning("recoverable issue")
    for handler in logging.getLogger("speakscribe").handlers + logger.logger.handlers:
        handler.flush()
    assert "validation started" in session.session_log.read_text(encoding="utf-8")
    assert "internal decision" in session.debug_log.read_text(encoding="utf-8")
    assert "recoverable issue" in session.errors_log.read_text(encoding="utf-8")
    assert "validation started" in (
        session.directory / "modules/validation.log").read_text(encoding="utf-8")


def test_status_stream_logs_and_yields_each_message_immediately(tmp_path):
    session = configure_logging(logs_root=tmp_path)
    updates = stream_status(("Loading repository", "Checking dependencies"),
                            component="repository", repository="service-a")
    first = next(updates)
    assert first.message == "Loading repository"
    assert "Loading repository" in session.session_log.read_text(encoding="utf-8")
    assert next(updates).message == "Checking dependencies"
    assert "Checking dependencies" in (
        session.directory / "repos/service-a.log").read_text(encoding="utf-8")


def test_emit_status_logs_exception_details(tmp_path):
    session = configure_logging(logs_root=tmp_path)
    try:
        raise RuntimeError("broken repository")
    except RuntimeError:
        update = emit_status("Repository processing failed", level=logging.ERROR,
                             component="repository", exc_info=True)
    assert update.level_name == "ERROR"
    error_log = session.errors_log.read_text(encoding="utf-8")
    assert "Repository processing failed" in error_log
    assert "RuntimeError: broken repository" in error_log


def test_session_retention_never_deletes_current_session(tmp_path):
    sessions = [configure_logging(logs_root=tmp_path, session_name=f"session_old_{index}",
                                  retention=2) for index in range(3)]
    cleanup_sessions(tmp_path, keep=2, current=sessions[-1].directory)
    assert sessions[-1].directory.exists()
    assert len(list(tmp_path.glob("session_*"))) == 2
