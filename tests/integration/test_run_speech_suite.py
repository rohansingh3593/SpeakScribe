from types import SimpleNamespace
import subprocess
import sys

from tests import run_speech_suite


def test_script_help_works_without_pythonpath_configuration():
    completed = subprocess.run(
        [sys.executable, "tests/run_speech_suite.py", "--help"],
        cwd=run_speech_suite.ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0
    assert "--cleanup-after" in completed.stdout
    assert "--debug" in completed.stdout
    assert "--quiet" in completed.stdout
    assert "--log-level" in completed.stdout


def test_logging_mode_is_forwarded_to_evaluator(monkeypatch):
    commands = []

    def run(command, **_kwargs):
        commands.append(command)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(run_speech_suite.subprocess, "run", run)
    monkeypatch.setattr(run_speech_suite, "missing_asr_dependencies", lambda: [])
    assert run_speech_suite.main(["--debug"]) == 0
    evaluator = next(command for command in commands if "evaluation.evaluation_runner" in command)
    assert "--debug" in evaluator


def test_cleanup_generated_runs_after_success(monkeypatch):
    commands = []

    def run(command, **_kwargs):
        commands.append(command)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(run_speech_suite.subprocess, "run", run)
    monkeypatch.setattr(run_speech_suite, "missing_asr_dependencies", lambda: [])
    assert run_speech_suite.main(["--cleanup-after", "generated"]) == 0
    assert commands[-1][-1] == "--remove-generated"
    assert any("evaluation.evaluation_runner" in command for command in commands)


def test_cleanup_all_runs_even_when_generation_fails(monkeypatch):
    commands = []

    def run(command, **_kwargs):
        commands.append(command)
        return SimpleNamespace(returncode=1 if len(commands) == 1 else 0)

    monkeypatch.setattr(run_speech_suite.subprocess, "run", run)
    monkeypatch.setattr(run_speech_suite, "missing_asr_dependencies", lambda: [])
    assert run_speech_suite.main(["--cleanup-after", "all"]) == 3
    assert commands[-1][-1] == "--remove-all"
    assert not any("evaluation.evaluation_runner" in command for command in commands)


def test_missing_asr_dependency_stops_before_generation(monkeypatch, capsys):
    monkeypatch.setattr(run_speech_suite, "missing_asr_dependencies",
                        lambda: ["faster_whisper"])
    monkeypatch.setattr(
        run_speech_suite.subprocess, "run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("must not run")),
    )

    assert run_speech_suite.main([]) == 5
    error = capsys.readouterr().err
    assert "ASR_DEPENDENCY_ERROR" in error
    assert "pip install -r requirements.txt" in error
