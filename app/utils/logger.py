"""Central session logging and live status streaming for SpeakScribe."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import logging
from pathlib import Path
import re
import shutil
import sys
from typing import Iterable, Iterator

_LOGGER_NAME = "speakscribe"
_SESSION = None


class _Context(logging.Filter):
    def filter(self, record):
        record.component = getattr(record, "component", record.name.rsplit(".", 1)[-1])
        record.repository = getattr(record, "repository", "-")
        return True


@dataclass(frozen=True)
class LogSession:
    session_id: str
    directory: Path
    session_log: Path
    debug_log: Path
    errors_log: Path


@dataclass(frozen=True)
class StatusUpdate:
    message: str
    level: int
    component: str
    repository: str | None = None

    @property
    def level_name(self) -> str:
        return logging.getLevelName(self.level)


def _safe(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "general"


def _new_session_directory(root: Path, name: str | None) -> Path:
    stem = _safe(name) if name else datetime.now().astimezone().strftime(
        "session_%Y-%m-%d_%H-%M-%S")
    candidate, suffix = root / stem, 1
    while candidate.exists():
        candidate = root / f"{stem}_{suffix}"
        suffix += 1
    candidate.mkdir(parents=True)
    (candidate / "modules").mkdir()
    (candidate / "repos").mkdir()
    return candidate


def cleanup_sessions(logs_root: Path | str = "logs", keep: int = 10,
                     current: Path | None = None) -> None:
    """Delete oldest completed session directories, never the active session."""
    root = Path(logs_root)
    if keep < 1 or not root.exists():
        return
    sessions = sorted((path for path in root.iterdir() if path.is_dir() and
                       path.name.startswith("session_")), key=lambda path: path.stat().st_mtime)
    removable = [path for path in sessions if current is None or path.resolve() != current.resolve()]
    for path in removable[:max(0, len(sessions) - keep)]:
        shutil.rmtree(path, ignore_errors=True)


def _handler(path, level, formatter, filter_=None):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setLevel(level)
    handler.setFormatter(formatter)
    handler.addFilter(_Context())
    if filter_:
        handler.addFilter(filter_)
    return handler


def configure_logging(*, debug: bool = False, logs_root: Path | str = "logs",
                      session_name: str | None = None, retention: int = 10) -> LogSession:
    """Create one structured application log session and configure all loggers."""
    global _SESSION
    directory = _new_session_directory(Path(logs_root), session_name)
    session = LogSession(directory.name, directory, directory / "session.log",
                         directory / "debug.log", directory / "errors.log")
    logger = logging.getLogger(_LOGGER_NAME)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    for existing in logger.handlers[:]:
        if getattr(existing, "speakscribe_observability", False):
            continue
        existing.close()
        logger.removeHandler(existing)
    for candidate in logging.Logger.manager.loggerDict.values():
        if isinstance(candidate, logging.Logger) and candidate.name.startswith(_LOGGER_NAME + "."):
            for existing in candidate.handlers[:]:
                if getattr(existing, "speakscribe_observability", False):
                    continue
                existing.close()
                candidate.removeHandler(existing)
    detailed = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(component)s | %(repository)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S")
    console = logging.StreamHandler(sys.stderr)
    console.setLevel(logging.DEBUG if debug else logging.INFO)
    console.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-8s | %(component)s | %(message)s",
                                           datefmt="%H:%M:%S"))
    console.addFilter(_Context())
    logger.addHandler(console)
    logger.addHandler(_handler(session.session_log, logging.INFO, detailed))
    logger.addHandler(_handler(session.debug_log, logging.DEBUG, detailed))
    logger.addHandler(_handler(session.errors_log, logging.WARNING, detailed))
    _SESSION = session
    for candidate in logging.Logger.manager.loggerDict.values():
        if isinstance(candidate, logging.Logger) and candidate.name.startswith(_LOGGER_NAME + "."):
            component = candidate.name[len(_LOGGER_NAME) + 1:]
            target = session.directory / "modules" / f"{_safe(component)}.log"
            handler = _handler(target, logging.DEBUG, detailed)
            handler.speakscribe_target = str(target)
            candidate.addHandler(handler)
    cleanup_sessions(logs_root, retention, directory)
    get_logger("application").info("Logging session started | path=%s", directory)
    return session


def configure_runtime_logging(*, console_level: int = logging.INFO,
                              output_path: Path | str | None = None) -> logging.Logger:
    """Compatibility configuration for callers needing one explicit output file."""
    if output_path is None:
        configure_logging(debug=console_level <= logging.DEBUG,
                          logs_root=Path(__file__).resolve().parents[2] / "logs")
        return logging.getLogger(_LOGGER_NAME)
    logger = logging.getLogger(_LOGGER_NAME)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    for existing in logger.handlers[:]:
        if getattr(existing, "speakscribe_observability", False):
            continue
        existing.close()
        logger.removeHandler(existing)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(component)s | %(repository)s | %(message)s")
    console = logging.StreamHandler(sys.stderr)
    console.setLevel(console_level)
    console.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    console.addFilter(_Context())
    logger.addHandler(console)
    logger.addHandler(_handler(Path(output_path), logging.DEBUG, formatter))
    return logger


def get_logger(component: str, *, repository: str | None = None) -> logging.LoggerAdapter:
    """Return a contextual logger and lazily add module/repository files."""
    logger = logging.getLogger(f"{_LOGGER_NAME}.{_safe(component)}")
    if _SESSION is not None:
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(component)s | %(repository)s | %(message)s")
        targets = [(_SESSION.directory / "modules" / f"{_safe(component)}.log", "module")]
        if repository:
            targets.append((_SESSION.directory / "repos" / f"{_safe(repository)}.log", "repo"))
        for target, marker in targets:
            if not any(getattr(handler, "speakscribe_target", None) == str(target)
                       for handler in logger.handlers):
                handler = _handler(target, logging.DEBUG, formatter)
                handler.speakscribe_target = str(target)
                logger.addHandler(handler)
    return logging.LoggerAdapter(logger, {"component": component,
                                          "repository": repository or "-"})


def emit_status(message: str, *, level: int = logging.INFO, component: str = "application",
                repository: str | None = None, exc_info=False) -> StatusUpdate:
    """Log once and return the same event for generator-based status streaming."""
    if not logging.getLogger(_LOGGER_NAME).handlers:
        configure_logging(logs_root=Path(__file__).resolve().parents[2] / "logs")
    get_logger(component, repository=repository).log(level, message, exc_info=exc_info)
    return StatusUpdate(message, level, component, repository)


def stream_status(messages: Iterable[str], *, component: str = "application",
                  repository: str | None = None) -> Iterator[StatusUpdate]:
    """Yield each status immediately after persisting it to centralized logs."""
    for message in messages:
        yield emit_status(message, component=component, repository=repository)


def get_output_path() -> str:
    if _SESSION is None:
        configure_logging(logs_root=Path(__file__).resolve().parents[2] / "logs")
    return str(_SESSION.session_log)


def log_print(message: object, level: int = logging.DEBUG) -> None:
    """Compatibility API for existing call sites; new code should use get_logger."""
    if not logging.getLogger(_LOGGER_NAME).handlers:
        configure_logging(logs_root=Path(__file__).resolve().parents[2] / "logs")
    get_logger("runtime").log(level, "%s", message)


def log_exception(context: str, exception: BaseException) -> None:
    """Persist complete exception context and traceback at ERROR."""
    if not logging.getLogger(_LOGGER_NAME).handlers:
        configure_logging(logs_root=Path(__file__).resolve().parents[2] / "logs")
    get_logger(context).exception("%s: %s", type(exception).__name__, exception)
