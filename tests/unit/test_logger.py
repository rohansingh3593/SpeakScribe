import logging

from app.utils.logger import configure_runtime_logging, log_print


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
