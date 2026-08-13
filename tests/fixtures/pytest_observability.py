"""Artifact writer used by the centralized hooks in ``tests/conftest.py``."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime
import hashlib
import json
import logging
import os
from pathlib import Path
import platform
import re
import sys
import time


def _safe(value: str, max_length: int = 48) -> str:
    """Return a Windows-safe, bounded, collision-resistant path component."""
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")
    value = value or "unnamed"
    if len(value) <= max_length:
        return value
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]
    return f"{value[:max_length - len(digest) - 2]}__{digest}"


def _artifact_stem(value: str, _directory: Path, _suffix: str = "") -> str:
    """Create a compact deterministic filename independent of checkout depth."""
    return _safe(value, 48)


def _artifact_error(observer, context: str, exc: BaseException) -> None:
    """Report an observer failure without ever aborting the pytest run."""
    logging.getLogger(__name__).error(
        "Pytest observability disabled for %s: %s", context, exc, exc_info=True)
    try:
        observer.info(f"OBSERVABILITY ERROR | {context} | {exc}")
    except Exception:
        pass


@dataclass
class TestRecord:
    nodeid: str
    index: int
    module: str
    class_name: str
    function: str
    started: str
    started_clock: float
    metadata: dict = field(default_factory=dict)
    phases: dict = field(default_factory=dict)
    repository_logs: list[str] = field(default_factory=list)
    evaluation: dict = field(default_factory=dict)


class _RepositoryCapture(logging.Handler):
    def __init__(self, observer):
        super().__init__(logging.DEBUG)
        self.speakscribe_observability = True
        self.observer = observer
        self.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))

    def emit(self, record):
        line = self.format(record)
        self.observer.debug(line)
        if self.observer.current_node in self.observer.records:
            self.observer.records[self.observer.current_node].repository_logs.append(line)


class PytestObserver:
    """Own one self-contained filesystem session for a pytest invocation."""

    def __init__(self, config):
        now = datetime.now().astimezone()
        requested = config.getoption("--session-name")
        stem = _safe(requested) if requested else now.strftime("session_%Y-%m-%d_%H-%M-%S")
        root = (Path.cwd() / "test_logs").resolve()
        path, suffix = root / stem, 1
        while path.exists():
            path = root / f"{stem}_{suffix}"
            suffix += 1
        self.path, self.session_id = path, path.name
        for relative in (
            "success/modules", "success/classes", "success/tests",
            "failed/modules", "failed/classes", "failed/tests",
            "artifacts/actual_transcripts", "artifacts/expected_transcripts",
            "artifacts/metrics",
        ):
            (path / relative).mkdir(parents=True, exist_ok=True)
        for name in ("session.log", "summary.log", "debug.log", "failures.log"):
            (path / name).touch()
        self.config = config
        self.started_at, self.started_clock = now, time.perf_counter()
        self.records: dict[str, TestRecord] = {}
        self.warnings = defaultdict(list)
        self.collected = []
        self.current_node = ""
        self.current_phase = "SESSION"
        self.last_module = self.last_class = None
        self.interrupted = False
        self._capture = _RepositoryCapture(self)
        logging.getLogger("speakscribe.runtime").addHandler(self._capture)
        self.info("SESSION START")
        self._write_json("session.json", self._session_data(status="RUNNING"))

    def _append(self, name: str, message: str):
        with (self.path / name).open("a", encoding="utf-8") as stream:
            stream.write(message.rstrip() + "\n")

    def info(self, message: str):
        self._append("session.log", f"{datetime.now().astimezone().isoformat()} | {message}")

    def debug(self, message: str):
        context = self.records.get(self.current_node)
        fields = [self.session_id, context.module if context else "-",
                  context.class_name if context else "-",
                  context.function if context else "-", self.current_phase]
        self._append("debug.log", f"{datetime.now().astimezone().isoformat()} | DEBUG | " +
                     " | ".join(fields) + f" | {message}")
        if bool(self.config.getoption("debug")):
            print(f"[DEBUG] {message}")

    def _write_json(self, relative: str, value):
        (self.path / relative).write_text(
            json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")

    def _session_data(self, **extra):
        data = {
            "session_id": self.session_id, "started": self.started_at.isoformat(),
            "python": sys.version, "pytest": __import__("pytest").__version__,
            "platform": platform.platform(), "command_line": sys.argv,
            "debug": bool(self.config.getoption("debug")),
            "asr_model": os.getenv("SPEAKSCRIBE_EVAL_MODEL", "medium"),
            "device": os.getenv("SPEAKSCRIBE_DEVICE", "auto"),
            "compute_type": os.getenv("SPEAKSCRIBE_COMPUTE_TYPE", "auto"),
            "worker": os.getenv("PYTEST_XDIST_WORKER", "main"),
        }
        data.update(extra)
        return data

    def collected_items(self, items):
        self.collected = list(items)
        languages, modules, classes, scenarios = Counter(), Counter(), Counter(), Counter()
        for item in items:
            case = getattr(getattr(item, "callspec", None), "params", {}).get("case", {})
            languages[case.get("language", "Other")] += 1
            modules[item.path.name] += 1
            classes[item.cls.__name__ if item.cls else "-"] += 1
            scenarios[case.get("scenario", "Other")] += 1
        self.info(f"COLLECTED {len(items)} tests | languages={dict(languages)} | "
                  f"modules={dict(modules)} | classes={dict(classes)} | scenarios={dict(scenarios)}")
        if len(items) < 120:
            self.info(f"WARNING | Expected baseline >= 120 tests; collected only {len(items)}")

    def start_test(self, item):
        module = str(item.path)
        class_name = item.cls.__name__ if item.cls else "-"
        if module != self.last_module:
            self.info(f"MODULE START | {module}")
            self.last_module = module
        if class_name != "-" and (module, class_name) != self.last_class:
            self.info(f"CLASS START | {module}::{class_name}")
            self.last_class = (module, class_name)
        params = getattr(getattr(item, "callspec", None), "params", {})
        case = params.get("case", {})
        metadata = {key: case.get(key) for key in
                    ("id", "language", "scenario", "difficulty", "audio") if case.get(key) is not None}
        record = TestRecord(item.nodeid, len(self.records) + 1, module, class_name,
                            item.name, datetime.now().astimezone().isoformat(),
                            time.perf_counter(), metadata)
        self.records[item.nodeid] = record
        self.current_node, self.current_phase = item.nodeid, "SETUP"
        self.info(f"[{record.index:03d}/{len(self.collected):03d}] {metadata.get('id', item.name)} START")

    def set_phase(self, item, phase):
        self.current_node, self.current_phase = item.nodeid, phase.upper()

    def warning(self, warning_message, when, nodeid, location):
        key = nodeid or self.current_node or "SESSION"
        detail = f"{warning_message} | when={when} | location={location}"
        self.warnings[key].append(detail)
        self.info(f"WARNING | {key} | {detail}")

    def record_evaluation(self, item, result):
        value = asdict(result) if is_dataclass(result) else dict(result)
        if "expected" in value and "actual" in value:
            from evaluation.evaluation_runner import normalize_transcript
            value["expected_normalized"] = normalize_transcript(value["expected"])
            value["actual_normalized"] = normalize_transcript(value["actual"])
        self.records[item.nodeid].evaluation = value

    def phase_report(self, item, report, call=None):
        record = self.records.setdefault(
            item.nodeid, TestRecord(item.nodeid, len(self.records) + 1, str(item.path),
                                    item.cls.__name__ if item.cls else "-", item.name,
                                    datetime.now().astimezone().isoformat(), time.perf_counter()))
        sections = {name: content for name, content in report.sections}
        record.phases[report.when] = {
            "status": report.outcome.upper(), "duration_seconds": report.duration,
            "exception": str(report.longrepr) if report.failed else "",
            "captured_stdout": sections.get("Captured stdout call", ""),
            "captured_stderr": sections.get("Captured stderr call", ""),
            "captured_logs": sections.get("Captured log call", ""),
            "exception_type": (call.excinfo.type.__name__
                               if call is not None and call.excinfo is not None else ""),
            "exception_message": (str(call.excinfo.value)
                                  if call is not None and call.excinfo is not None else ""),
        }
        self.info(f"PHASE | {record.nodeid} | {report.when.upper()} | "
                  f"{report.outcome.upper()} | {report.duration:.3f}s")
        if report.when == "teardown":
            try:
                self.finish_test(record)
            except Exception as exc:
                # Artifact collection must never turn a valid pytest run into an
                # INTERNALERROR. Preserve the failure in the session log when possible.
                _artifact_error(self, record.nodeid, exc)

    def _category(self, record, status):
        failed_phase = next((name for name, value in record.phases.items()
                             if value["status"] == "FAILED"), "")
        if failed_phase == "setup": return "SETUP_FAILURE"
        if failed_phase == "teardown": return "TEARDOWN_FAILURE"
        evaluation = record.evaluation
        flags = set(evaluation.get("quality_flags", []))
        if "TIMEOUT" in flags: return "TIMEOUT"
        if "SCRIPT_MISMATCH" in flags: return "SCRIPT_MISMATCH"
        if "WRONG_LANGUAGE" in flags: return "LANGUAGE_DETECTION"
        if evaluation.get("technical_term_problems"): return "TECHNICAL_TERMS"
        if "NO_TRANSCRIPTION" in flags or evaluation.get("status") == "FAIL": return "ASR_ACCURACY"
        if status == "ERROR": return "EXCEPTION"
        return "ASSERTION_FAILURE" if status == "FAIL" else "UNKNOWN"

    def finish_test(self, record):
        evaluation = record.evaluation
        failed_phases = [(name, value) for name, value in record.phases.items()
                         if value["status"] == "FAILED"]
        phase_failed = bool(failed_phases)
        phase_skipped = any(value["status"] == "SKIPPED" for value in record.phases.values())
        if phase_failed:
            phase, detail = failed_phases[0]
            status = ("FAIL" if phase == "call" and
                      detail.get("exception_type") in {"AssertionError", "Failed"} else "ERROR")
        else:
            status = "SKIPPED" if phase_skipped else evaluation.get("status", "PASS")
            if status == "PASS" and self.warnings.get(record.nodeid):
                status = "WARNING"
        if status == "EXCELLENT": status = "PASS"
        hard = status in {"FAIL", "ERROR", "CRASH", "TIMEOUT"}
        duration = time.perf_counter() - record.started_clock
        category = self._category(record, status)
        test_id = record.metadata.get("id", record.function)
        worker = os.getenv("PYTEST_XDIST_WORKER", "main")
        test_directory = self.path / ("failed" if hard else "success") / "tests"
        filename_stem = _artifact_stem(
            str(test_id), test_directory, f"__{worker}.log")
        filename = f"{filename_stem}__{worker}.log"
        expected, actual = evaluation.get("expected", ""), evaluation.get("actual", "")
        details = [
            f"SESSION ID: {self.session_id}", f"NODE ID: {record.nodeid}",
            f"MODULE: {record.module}", f"CLASS: {record.class_name}",
            f"FUNCTION: {record.function}", f"TEST ID: {test_id}",
            f"LANGUAGE: {record.metadata.get('language', evaluation.get('language', '-'))}",
            f"SCENARIO: {record.metadata.get('scenario', evaluation.get('scenario', '-'))}",
            f"DIFFICULTY: {record.metadata.get('difficulty', evaluation.get('difficulty', '-'))}",
            f"STARTED: {record.started}", f"FINISHED: {datetime.now().astimezone().isoformat()}",
            f"STATUS: {status}", f"DURATION: {duration:.3f}s", "", "PYTEST PHASES:",
            json.dumps(record.phases, ensure_ascii=False, indent=2),
            "", f"EXPECTED:\n{expected}", f"ACTUAL:\n{actual}",
            f"NORMALIZED EXPECTED: {evaluation.get('expected_normalized', '')}",
            f"NORMALIZED ACTUAL: {evaluation.get('actual_normalized', '')}",
            f"ACCURACY: {evaluation.get('similarity', 'n/a')}", f"WER: {evaluation.get('wer', 'n/a')}",
            f"MISSING WORDS: {evaluation.get('missing_words', [])}",
            f"EXTRA WORDS: {evaluation.get('extra_words', [])}",
            f"SUBSTITUTIONS: {evaluation.get('substitutions', [])}",
            f"EXPECTED LANGUAGE: {evaluation.get('language', record.metadata.get('language', '-'))}",
            f"DETECTED LANGUAGE: {evaluation.get('detected_language', '-')}",
            f"AUDIO PATH: {evaluation.get('audio', record.metadata.get('audio', '-'))}",
            f"AUDIO SOURCE: {evaluation.get('audio_source', '-')}",
            f"AUDIO DURATION: {evaluation.get('duration_seconds', 'n/a')}",
            f"ASR DURATION: {evaluation.get('inference_seconds', 'n/a')}",
            f"FAILURE CATEGORY: {category}",
            f"OBSERVED FAILURE: {next((v['exception'] for v in record.phases.values() if v['exception']), '-')}",
            f"SUSPECTED COMPONENT: {evaluation.get('root_cause', 'Unknown')}",
            f"SUSPECTED ROOT CAUSE: {evaluation.get('possible_problem', 'Unknown')}",
            "ROOT CAUSE STATUS: Suspected" if hard else "ROOT CAUSE STATUS: Not applicable",
            f"RECOMMENDED INVESTIGATION: {evaluation.get('recommended_fix', '-')}",
            "", "CAPTURED REPOSITORY LOGS:", *record.repository_logs,
            "", "PYTEST WARNINGS:", *self.warnings.get(record.nodeid, []),
        ]
        destination = test_directory / filename
        destination.write_text("\n".join(map(str, details)) + "\n", encoding="utf-8")
        metrics_directory = self.path / "artifacts/metrics"
        longest_artifact_directory = self.path / "artifacts/expected_transcripts"
        artifact = _artifact_stem(
            str(test_id), longest_artifact_directory, f"__{worker}.json"
        ) + f"__{worker}"
        (self.path / "artifacts/expected_transcripts" / f"{artifact}.txt").write_text(str(expected), encoding="utf-8")
        (self.path / "artifacts/actual_transcripts" / f"{artifact}.txt").write_text(str(actual), encoding="utf-8")
        metrics = {"test_id": test_id, "nodeid": record.nodeid, "status": status,
                   "failure_category": category, "total_duration": duration, **record.metadata,
                   **{key: evaluation.get(key) for key in ("similarity", "wer", "inference_seconds",
                      "duration_seconds", "detected_language", "missing_words", "extra_words",
                      "substitutions")}}
        metrics.update({
            "insertions": len(evaluation.get("extra_words", [])),
            "deletions": len(evaluation.get("missing_words", [])),
            "substitution_count": len(evaluation.get("substitutions", [])),
        })
        self._write_json(f"artifacts/metrics/{artifact}.json", metrics)
        record.evaluation["_observability"] = metrics
        self.info(f"[{record.index:03d}/{len(self.collected):03d}] {test_id} {status} | "
                  f"accuracy={evaluation.get('similarity', 'n/a')} | WER={evaluation.get('wer', 'n/a')} | {duration:.2f}s")
        if hard:
            self._append("failures.log", "\n".join([
                "=" * 60, f"FAILURE: {test_id}", "=" * 60,
                f"Module: {record.module}", f"Class: {record.class_name}",
                f"Function: {record.function}",
                f"Phase: {next((p.upper() for p,v in record.phases.items() if v['status']=='FAILED'), 'UNKNOWN')}",
                f"Expected: {expected}", f"Actual: {actual}",
                f"Accuracy: {evaluation.get('similarity', 'n/a')}", f"WER: {evaluation.get('wer', 'n/a')}",
                f"Failure Category: {category}",
                f"Suspected Root Cause: {evaluation.get('possible_problem', 'Unknown')}",
                f"Detailed Test Log: {destination.relative_to(self.path)}", ""]))

    def finish(self, exitstatus):
        duration = time.perf_counter() - self.started_clock
        metrics = [r.evaluation.get("_observability", {}) for r in self.records.values()
                   if r.evaluation.get("_observability")]
        counts = Counter(item.get("status") for item in metrics)
        failed = counts["FAIL"]
        errors = counts["ERROR"] + counts["CRASH"] + counts["TIMEOUT"]
        by_language = defaultdict(list)
        for item in metrics: by_language[item.get("language", "Other")].append(item)
        categories = Counter(item.get("failure_category") for item in metrics
                             if item.get("status") in {"FAIL", "ERROR", "CRASH", "TIMEOUT"})
        summary = ["=" * 60, "TEST SESSION SUMMARY", "=" * 60,
                   f"Session: {self.session_id}", f"Total Tests : {len(self.collected)}",
                   f"Passed      : {counts['PASS']}", f"Warning     : {counts['WARNING']}",
                   f"Failed      : {failed}", f"Errors      : {errors}",
                   f"Skipped     : {counts['SKIPPED']}",
                   f"Total Duration: {duration:.2f}s"]
        for language, values in sorted(by_language.items()):
            accuracies = [x["similarity"] for x in values if x.get("similarity") is not None]
            summary += ["", language, "-" * len(language), f"Tests: {len(values)}",
                        f"Passed: {sum(x['status']=='PASS' for x in values)}",
                        f"Failed: {sum(x['status']=='FAIL' for x in values)}",
                        f"Accuracy: {sum(accuracies)/len(accuracies):.1f}%" if accuracies else "Accuracy: n/a"]
        wers = [x["wer"] for x in metrics if x.get("wer") is not None]
        slowest = max(metrics, key=lambda x: x.get("total_duration", 0), default={})
        summary += ["", f"Average WER: {sum(wers)/len(wers):.4f}" if wers else "Average WER: n/a",
                    f"Average Test Duration: {sum(x.get('total_duration',0) for x in metrics)/len(metrics):.3f}s" if metrics else "Average Test Duration: n/a",
                    f"Slowest Test: {slowest.get('test_id', 'n/a')} ({slowest.get('total_duration', 0):.3f}s)",
                    f"Most Common Failure Category: {categories.most_common(1)[0][0] if categories else 'n/a'}"]
        (self.path / "summary.log").write_text("\n".join(summary) + "\n", encoding="utf-8")
        session_metrics = {"counts": dict(counts), "failure_categories": dict(categories),
                           "tests": metrics, "duration_seconds": duration}
        self._write_json("artifacts/metrics/session_metrics.json", session_metrics)
        status = "SESSION_INTERRUPTED" if exitstatus == 2 else "COMPLETE"
        self.interrupted = exitstatus == 2
        self._write_json("session.json", self._session_data(
            status=status, finished=datetime.now().astimezone().isoformat(),
            total_tests=len(self.collected), passed=counts["PASS"], warning=counts["WARNING"],
            failed=failed, errors=errors, skipped=counts["SKIPPED"], duration_seconds=duration,
            exitstatus=exitstatus))
        self._write_group_summaries(metrics, "module", "modules")
        self._write_group_summaries(metrics, "class_name", "classes", skip="-")
        self.info(f"SESSION END | status={status} exitstatus={exitstatus} duration={duration:.3f}s")
        logging.getLogger("speakscribe.runtime").removeHandler(self._capture)

    def _write_group_summaries(self, metrics, key, directory, skip=None):
        groups = defaultdict(list)
        for record in self.records.values():
            value = record.module if key == "module" else record.class_name
            if value != skip and record.evaluation.get("_observability"):
                groups[value].append(record.evaluation["_observability"])
        for value, items in groups.items():
            hard = any(x["status"] in {"FAIL", "ERROR", "CRASH", "TIMEOUT"} for x in items)
            text = "\n".join([f"{key.upper()}: {value}", f"Tests: {len(items)}",
                              f"Passed: {sum(x['status']=='PASS' for x in items)}",
                              f"Warning: {sum(x['status']=='WARNING' for x in items)}",
                              f"Failed: {sum(x['status']=='FAIL' for x in items)}",
                              f"Errors: {sum(x['status'] in {'ERROR','CRASH','TIMEOUT'} for x in items)}",
                              f"Duration: {sum(x.get('total_duration',0) for x in items):.3f}s"]) + "\n"
            target_directory = self.path / ("failed" if hard else "success") / directory
            target_stem = _artifact_stem(str(value), target_directory, ".log")
            target = target_directory / f"{target_stem}.log"
            target.write_text(text, encoding="utf-8")
            self.info(f"{key.upper()} END | {value} | passed={sum(x['status']=='PASS' for x in items)} "
                      f"failed={sum(x['status']=='FAIL' for x in items)} "
                      f"errors={sum(x['status'] in {'ERROR','CRASH','TIMEOUT'} for x in items)} | "
                      f"duration={sum(x.get('total_duration',0) for x in items):.3f}s")
