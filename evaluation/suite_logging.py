"""Central logging and progress reporting for the speech validation suite."""

from __future__ import annotations

from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime
import logging
from pathlib import Path
import shutil
import statistics
import time

SLOW_TEST_WARNING_SECONDS = 5.0
LOGGER_NAME = "speakscribe.speech_suite"


class _FailuresOnly(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return record.levelno >= logging.WARNING or getattr(record, "test_status", "") in {
            "WARNING", "FAIL", "ERROR", "TIMEOUT", "CRASH"
        }


@dataclass(frozen=True)
class LogPaths:
    run_id: str
    main: Path
    latest: Path
    failures: Path


def configure_logging(*, debug: bool = False, quiet: bool = False,
                      log_level: str | None = None, log_dir: Path | str = "logs"):
    """Configure idempotent console, complete-run, and failure-only handlers."""
    directory = Path(log_dir)
    (directory / "errors").mkdir(parents=True, exist_ok=True)
    run_id = datetime.now().astimezone().strftime("%Y-%m-%d_%H%M%S_%f")
    paths = LogPaths(run_id, directory / f"test_run_{run_id}.log",
                     directory / "latest.log",
                     directory / "errors" / f"failed_tests_{run_id}.log")
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)

    explicit = getattr(logging, log_level.upper()) if log_level else None
    console_level = explicit if explicit is not None else (
        logging.DEBUG if debug else logging.ERROR if quiet else logging.INFO)
    console = logging.StreamHandler()
    console.setLevel(console_level)
    console.setFormatter(logging.Formatter("[%(levelname)s] %(message)s") if debug
                         else logging.Formatter("%(message)s"))
    logger.addHandler(console)

    structured = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(component)s | %(test_id)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S")
    for path, failure_only in ((paths.main, False), (paths.failures, True)):
        handler = logging.FileHandler(path, encoding="utf-8")
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(structured)
        if failure_only:
            handler.addFilter(_FailuresOnly())
        logger.addHandler(handler)
    # Route the existing ASR/audio-pipeline diagnostics to the same complete log.
    # Their legacy log_print calls default to DEBUG and therefore stay off a normal
    # console while becoming visible with --debug.
    from app.utils.logger import configure_runtime_logging
    configure_runtime_logging(console_level=console_level, output_path=paths.main)
    return logger, paths


def log(logger: logging.Logger, level: int, message: str, *, component="SUITE",
        test_id="-", test_status="") -> None:
    logger.log(level, message, extra={"component": component, "test_id": test_id,
                                      "test_status": test_status})


def finalize_latest(paths: LogPaths) -> None:
    """Publish latest.log only after handlers have flushed the completed run."""
    for handler in logging.getLogger(LOGGER_NAME).handlers:
        handler.flush()
    for handler in logging.getLogger("speakscribe.runtime").handlers:
        handler.flush()
    shutil.copyfile(paths.main, paths.latest)


def format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "calculating..."
    seconds = max(0, round(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


@dataclass
class ProgressTracker:
    total: int
    started: float = field(default_factory=time.perf_counter)
    durations: deque = field(default_factory=lambda: deque(maxlen=10))
    counts: Counter = field(default_factory=Counter)
    completed: int = 0

    def record(self, status: str, duration: float) -> None:
        self.completed += 1
        self.counts[status] += 1
        self.durations.append(duration)

    @property
    def remaining(self) -> int:
        return max(0, self.total - self.completed)

    @property
    def percentage(self) -> float:
        return 100 * self.completed / self.total if self.total else 100.0

    @property
    def elapsed(self) -> float:
        return time.perf_counter() - self.started

    @property
    def eta(self) -> float | None:
        # One outlier must not establish an apparently authoritative estimate.
        return (statistics.fmean(self.durations) * self.remaining
                if len(self.durations) >= 3 else None)

    def progress_message(self) -> str:
        width = 24
        filled = round(width * self.percentage / 100)
        bar = "█" * filled + "-" * (width - filled)
        passed = self.counts["EXCELLENT"] + self.counts["PASS"]
        failed = self.counts["FAIL"] + self.counts["CRASH"] + self.counts["TIMEOUT"]
        return (f"Progress: [{bar}] {self.percentage:.1f}% | "
                f"{self.completed}/{self.total} completed | {self.remaining} remaining | "
                f"PASS {passed} WARN {self.counts['WARNING']} FAIL {failed} | "
                f"Elapsed {format_duration(self.elapsed)} | ETA {format_duration(self.eta)}")


def aggregate_results(results) -> tuple[dict, list[tuple[str, float]]]:
    languages = defaultdict(list)
    scenarios = defaultdict(list)
    for result in results:
        languages[result.language].append(result)
        scenarios[result.scenario].append(result.total_processing_seconds)
    language_summary = {
        language: {
            "tests": len(items),
            "average_time": statistics.fmean(x.total_processing_seconds for x in items),
            "accuracy": statistics.fmean(x.similarity for x in items),
        } for language, items in languages.items()
    }
    slow_scenarios = sorted(
        ((name, statistics.fmean(values)) for name, values in scenarios.items()),
        key=lambda item: item[1], reverse=True)[:5]
    return language_summary, slow_scenarios
