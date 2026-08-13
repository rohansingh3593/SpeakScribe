"""Immediate live-session invalidation independent of worker lifetime."""

from dataclasses import dataclass
from queue import Empty, Queue
from threading import Event
import time

from app.asr.language_transition import RecognitionState


@dataclass(frozen=True)
class StopMetrics:
    stop_clicked_at: float
    capture_disabled_at: float
    generation_invalidated_at: float
    queues_cleared_at: float
    state_ready_at: float
    generation: int
    jobs_cleared: int

    @property
    def stop_to_ready(self) -> float:
        return self.state_ready_at - self.stop_clicked_at


class LiveSessionBoundary:
    """End one live generation synchronously; workers may retire asynchronously."""

    def __init__(self, recognition_state: RecognitionState, capture_stop: Event,
                 queues: tuple[Queue, ...] = ()):
        self.recognition_state = recognition_state
        self.capture_stop = capture_stop
        self.queues = queues

    @staticmethod
    def clear_queue(queue: Queue) -> int:
        cleared = 0
        while True:
            try:
                queue.get_nowait()
                cleared += 1
            except Empty:
                return cleared

    def stop(self) -> StopMetrics:
        clicked = time.monotonic()
        self.capture_stop.set()
        capture_disabled = time.monotonic()
        generation = self.recognition_state.invalidate().generation
        invalidated = time.monotonic()
        cleared = sum(self.clear_queue(queue) for queue in self.queues)
        queues_cleared = time.monotonic()
        ready = time.monotonic()
        return StopMetrics(clicked, capture_disabled, invalidated, queues_cleared,
                           ready, generation, cleared)
