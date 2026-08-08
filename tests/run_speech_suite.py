"""Generate missing audio, then run all ASR cases and reports."""

import argparse
from pathlib import Path
import subprocess
import sys


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cleanup-after", choices=("none", "generated", "all"), default="none",
        help=("Remove audio after the suite: 'generated' preserves human recordings; "
              "'all' also removes human recordings"),
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    try:
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
    finally:
        if args.cleanup_after != "none":
            cleanup_flag = "--remove-all" if args.cleanup_after == "all" else "--remove-generated"
            cleanup = subprocess.run(
                [sys.executable, "tests/generate_test_audio.py", cleanup_flag],
                cwd=root, check=False,
            )
            if cleanup.returncode:
                print("Test-audio cleanup failed.", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
