"""Backward-compatible launcher and import facade for evaluation tooling."""

from evaluation.evaluation_runner import *  # noqa: F401,F403
from evaluation.evaluation_runner import main


if __name__ == "__main__":
    raise SystemExit(main())
