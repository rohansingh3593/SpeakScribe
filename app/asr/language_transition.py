"""Thread-safe, lightweight recognition-language transition state."""

from dataclasses import dataclass
from threading import Lock
import time


@dataclass(frozen=True)
class RecognitionSnapshot:
    language: str
    script: str
    generation: int
    switched_at: float
    ready_at: float


class RecognitionState:
    """Coordinates language changes without replacing the model or audio stream."""

    def __init__(self, language: str, script: str):
        now = time.monotonic()
        self._lock = Lock()
        self._snapshot = RecognitionSnapshot(language, script, 0, now, now)

    def snapshot(self) -> RecognitionSnapshot:
        with self._lock:
            return self._snapshot

    def switch(self, language: str, script: str) -> RecognitionSnapshot:
        requested_at = time.monotonic()
        with self._lock:
            previous = self._snapshot
            if previous.language == language and previous.script == script:
                return previous
            self._snapshot = RecognitionSnapshot(
                language, script, previous.generation + 1,
                requested_at, time.monotonic())
            return self._snapshot

    def begin_session(self, language: str, script: str) -> RecognitionSnapshot:
        """Start a fresh capture generation, even when configuration is unchanged."""
        requested_at = time.monotonic()
        with self._lock:
            previous = self._snapshot
            self._snapshot = RecognitionSnapshot(
                language, script, previous.generation + 1,
                requested_at, time.monotonic())
            return self._snapshot

    def invalidate(self) -> RecognitionSnapshot:
        """Immediately make every job from the active live session stale."""
        with self._lock:
            previous = self._snapshot
            now = time.monotonic()
            self._snapshot = RecognitionSnapshot(
                previous.language, previous.script, previous.generation + 1, now, now)
            return self._snapshot

    def is_current(self, generation: int) -> bool:
        with self._lock:
            return generation == self._snapshot.generation
