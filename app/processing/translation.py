"""Lazy, asynchronous MarianMT translation support."""

from queue import Empty, Queue
from threading import Event

from transformers import MarianMTModel, MarianTokenizer

from app.utils.logger import get_logger

LOGGER = get_logger("translation")


class TranslationWorker:
    def __init__(self, model_name: str, queue: Queue, stop_event: Event, signals):
        self.model_name, self.queue, self.stop_event, self.signals = (
            model_name, queue, stop_event, signals)
        self.model = None
        self.tokenizer = None

    def _load(self) -> None:
        if self.model is None:
            LOGGER.debug(f"Loading optional translation model: {self.model_name}")
            self.tokenizer = MarianTokenizer.from_pretrained(self.model_name)
            self.model = MarianMTModel.from_pretrained(self.model_name)

    def run(self) -> None:
        try:
            self._load()
            while not self.stop_event.is_set() or not self.queue.empty():
                try:
                    text = self.queue.get(timeout=0.1)
                except Empty:
                    continue
                inputs = self.tokenizer([text], return_tensors="pt", padding=True)
                output = self.model.generate(**inputs)
                translated = self.tokenizer.batch_decode(output, skip_special_tokens=True)[0]
                self.signals.translation_ready.emit(translated)
        except Exception as exc:
            LOGGER.error("Translation failed: %s", exc, exc_info=True)
            self.signals.error.emit(f"Translation: {exc}")
