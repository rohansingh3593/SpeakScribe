"""Thread-safe, level-aware logging used by the application and ASR pipeline."""

from __future__ import annotations

import logging
from pathlib import Path
import sys
import traceback

_OUTPUT = Path(__file__).resolve().parent / "speakscribe.log"
_LOGGER_NAME = "speakscribe.runtime"


def get_output_path() -> str:
    return str(_OUTPUT)


def configure_runtime_logging(*, console_level: int = logging.INFO,
                              output_path: Path | str | None = None) -> logging.Logger:
    """Configure runtime diagnostics once without leaking DEBUG to normal output."""
    logger = logging.getLogger(_LOGGER_NAME)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    for handler in logger.handlers[:]:
        if getattr(handler, "speakscribe_observability", False):
            continue
        handler.close()
        logger.removeHandler(handler)

    console = logging.StreamHandler(sys.stderr)
    console.setLevel(console_level)
    console.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(console)

    destination = Path(output_path) if output_path is not None else _OUTPUT
    destination.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(destination, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)-8s | RUNTIME | - | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"))
    logger.addHandler(file_handler)
    return logger


def _logger() -> logging.Logger:
    logger = logging.getLogger(_LOGGER_NAME)
    if not logger.handlers:
        configure_runtime_logging()
    return logger


def log_print(message: object, level: int = logging.DEBUG) -> None:
    """Log legacy call sites as technical diagnostics unless a level is supplied."""
    _logger().log(level, "%s", message)


def log_exception(context: str, exception: BaseException) -> None:
    """Log a worker failure at ERROR with its complete cross-thread traceback."""
    log_print(f"[{context}] {type(exception).__name__}: {exception}\n{traceback.format_exc()}",
              logging.ERROR)
