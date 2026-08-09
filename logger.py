"""Small, thread-safe file/console logger retained as the project logging API."""

from datetime import datetime
from pathlib import Path
from threading import Lock
import traceback

_LOCK = Lock()
_OUTPUT = Path(__file__).resolve().parent / "speakscribe.log"


def get_output_path() -> str:
    return str(_OUTPUT)


def log_print(message: object) -> None:
    line = f"{datetime.now().isoformat(timespec='milliseconds')} {message}"
    with _LOCK:
        print(line, flush=True)
        try:
            with _OUTPUT.open("a", encoding="utf-8") as stream:
                stream.write(line + "\n")
        except OSError:
            print("SpeakScribe: could not write log file", flush=True)


def log_exception(context: str, exception: BaseException) -> None:
    """Log a worker failure with its complete cross-thread traceback."""
    log_print(f"[{context}] {type(exception).__name__}: {exception}\n{traceback.format_exc()}")
