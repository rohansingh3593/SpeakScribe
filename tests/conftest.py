"""Test bootstrap that is independent of how the pytest executable is launched.

On Windows, the ``pytest.exe`` entry point can put its Scripts directory (rather
than the current project) at ``sys.path[0]``.  Some IDEs and wrapper scripts also
override the ``pythonpath`` value from pytest.ini.  Add the repository root
before test modules are imported so the application's existing top-level module
layout remains importable in all of those cases.
"""

from pathlib import Path
import sys

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
project_root = str(PROJECT_ROOT)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from tests.pytest_observability import PytestObserver


def pytest_addoption(parser):
    group = parser.getgroup("SpeakScribe observability")
    group.addoption("--session-name", help="Human-readable pytest artifact session name")


def pytest_sessionstart(session):
    global _ACTIVE_OBSERVER
    session.config._speakscribe_observer = PytestObserver(session.config)
    _ACTIVE_OBSERVER = session.config._speakscribe_observer


def pytest_collection_modifyitems(session, config, items):
    config._speakscribe_observer.collected_items(items)


def pytest_runtest_logstart(nodeid, location):
    # Item metadata is only available in pytest_runtest_setup; logstart is kept
    # intentionally free of duplicate lifecycle output.
    return None


def pytest_runtest_setup(item):
    observer = item.config._speakscribe_observer
    observer.start_test(item)
    observer.set_phase(item, "setup")


def pytest_runtest_call(item):
    item.config._speakscribe_observer.set_phase(item, "call")


def pytest_runtest_teardown(item):
    item.config._speakscribe_observer.set_phase(item, "teardown")


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    item.config._speakscribe_observer.phase_report(item, outcome.get_result(), call)


def pytest_sessionfinish(session, exitstatus):
    global _ACTIVE_OBSERVER
    observer = getattr(session.config, "_speakscribe_observer", None)
    if observer is not None:
        observer.finish(exitstatus)
    _ACTIVE_OBSERVER = None


def pytest_warning_recorded(warning_message, when, nodeid, location):
    # pytest does not pass Config to this hook, so use the active observer set at
    # session start. This module owns exactly one observer per process.
    observer = globals().get("_ACTIVE_OBSERVER")
    if observer is not None:
        observer.warning(warning_message, when, nodeid, location)


@pytest.fixture
def record_test_observation(request):
    """Attach a real repository result to centralized pytest artifacts."""
    def record(result):
        request.config._speakscribe_observer.record_evaluation(request.node, result)
        return result
    return record
