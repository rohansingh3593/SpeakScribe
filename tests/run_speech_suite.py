"""Generate missing audio, then run all ASR cases and reports."""

import argparse
from datetime import datetime
import importlib.util
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation_runner import format_duration


def missing_asr_dependencies() -> list[str]:
    return [name for name in ("numpy", "faster_whisper")
            if importlib.util.find_spec(name) is None]


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cleanup-after", choices=("none", "generated", "all"), default="none",
        help=("Remove audio after the suite: 'generated' preserves human recordings; "
              "'all' also removes human recordings"),
    )
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--debug", action="store_true", help="Show detailed technical logs")
    modes.add_argument("--quiet", action="store_true", help="Show errors and the final summary only")
    parser.add_argument("--log-level", choices=("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"),
                        help="Explicitly override the console log level")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    root = ROOT
    started_at = datetime.now().astimezone()
    started = time.perf_counter()
    exit_code = 0
    cleanup_failed = False
    show_info = not args.quiet and (args.log_level not in {"WARNING", "ERROR", "CRITICAL"})
    def info(message):
        if show_info:
            print(message, flush=True)

    info("=" * 56)
    info(" SPEECH RECOGNITION VALIDATION SUITE")
    info("=" * 56)
    info(f"Started: {started_at.isoformat(timespec='seconds')}")
    info("Estimated execution time: calculating...")
    try:
        missing = missing_asr_dependencies()
        if missing:
            print(
                "ASR_DEPENDENCY_ERROR: missing " + ", ".join(missing) + ".\n"
                f"Active Python: {sys.executable}\n"
                "Install into this environment with:\n"
                f'  "{sys.executable}" -m pip install -r requirements.txt\n'
                "Then verify with:\n"
                f'  "{sys.executable}" -c "from faster_whisper import WhisperModel; '
                'print(\'faster-whisper ready\')"',
                file=sys.stderr,
            )
            exit_code = 5
        else:
            stage_started = time.perf_counter()
            info("[Stage 1/3] Generating and validating test audio...")
            generation = subprocess.run(
                [sys.executable, "tests/generate_test_audio.py"], cwd=root, check=False,
                capture_output=True, text=True)
            generation_output = (getattr(generation, "stdout", "") or "") + (
                getattr(generation, "stderr", "") or "")
            if args.debug and generation_output:
                print("[DEBUG] Audio generation details:\n" + generation_output)
            info(f"[Stage 1/3] Finished in {format_duration(time.perf_counter() - stage_started)}")
            if generation.returncode:
                print("Audio generation did not complete; ASR was not run.", file=sys.stderr)
                exit_code = 3
            else:
                stage_started = time.perf_counter()
                info("[Stage 2/3] Running ASR evaluations and writing reports...")
                command = [sys.executable, "evaluation_runner.py", "--no-generate"]
                if args.debug:
                    command.append("--debug")
                if args.quiet:
                    command.append("--quiet")
                if args.log_level:
                    command.extend(("--log-level", args.log_level))
                evaluation = subprocess.run(
                    command, cwd=root, check=False,
                )
                exit_code = evaluation.returncode
                info(f"[Stage 2/3] Finished in {format_duration(time.perf_counter() - stage_started)}")
    finally:
        if args.cleanup_after != "none":
            stage_started = time.perf_counter()
            info(f"[Stage 3/3] Removing {args.cleanup_after} test audio...")
            cleanup_flag = "--remove-all" if args.cleanup_after == "all" else "--remove-generated"
            cleanup = subprocess.run(
                [sys.executable, "tests/generate_test_audio.py", cleanup_flag],
                cwd=root, check=False,
            )
            if cleanup.returncode:
                print("Test-audio cleanup failed.", file=sys.stderr)
                cleanup_failed = True
            info(f"[Stage 3/3] Finished in {format_duration(time.perf_counter() - stage_started)}")
        else:
            info("[Stage 3/3] Cleanup disabled; test audio retained.")

    if cleanup_failed and exit_code == 0:
        exit_code = 4
    status = "COMPLETE" if exit_code == 0 else f"INCOMPLETE (exit code {exit_code})"
    # The final summary is intentionally visible in every mode, including --quiet.
    print("=" * 56)
    print(f"Suite status: {status}")
    print(f"Completed: {datetime.now().astimezone().isoformat(timespec='seconds')}")
    print(f"Total execution time: {format_duration(time.perf_counter() - started)}")
    print("Reports: tests/results/latest_report.md, .json, and .csv")
    print("=" * 56)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
