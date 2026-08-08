"""Generate missing audio, then run all ASR cases and reports."""

from pathlib import Path
import subprocess
import sys


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    generation = subprocess.run(
        [sys.executable, "tests/generate_test_audio.py"], cwd=root, check=False)
    if generation.returncode:
        print("Audio generation did not complete; ASR was not run.", file=sys.stderr)
        return 3
    evaluation = subprocess.run(
        [sys.executable, "evaluation_runner.py", "--no-generate"],
        cwd=root, check=False,
    )
    return evaluation.returncode


if __name__ == "__main__":
    raise SystemExit(main())
