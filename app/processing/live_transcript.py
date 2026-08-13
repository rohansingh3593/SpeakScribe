"""Small, UI-independent model for the live transcript document."""

from dataclasses import dataclass


PARAGRAPH_PAUSE_THRESHOLD = 2.0
_LEVELS = {"fast": 0, "balanced": 1, "accurate": 2}


@dataclass
class FinalUtterance:
    utterance_id: int
    text: str
    language: str = ""
    start_time: float = 0.0
    end_time: float = 0.0
    refinement_level: str = "fast"
    paragraph_break: bool = False


class LiveTranscriptModel:
    """Address finals by utterance, while exposing clean natural paragraphs."""

    def __init__(self, paragraph_pause_threshold: float = PARAGRAPH_PAUSE_THRESHOLD):
        self.paragraph_pause_threshold = paragraph_pause_threshold
        self.processing_text = ""
        self.processing_id: int | None = None
        self._finals: dict[int, FinalUtterance] = {}

    def update_partial(self, text: str, utterance_id: int | None = None) -> None:
        self.processing_text = text.strip()
        self.processing_id = utterance_id if self.processing_text else None

    def commit(self, utterance_id: int, text: str, *, language: str = "",
               start_time: float = 0.0, end_time: float = 0.0,
               refinement_level: str = "fast", paragraph_break: bool = False) -> bool:
        text = text.strip()
        if not text:
            return False
        old = self._finals.get(utterance_id)
        if old and _LEVELS.get(refinement_level, 0) < _LEVELS.get(old.refinement_level, 0):
            return False
        self._finals[utterance_id] = FinalUtterance(
            utterance_id, text, language, start_time, end_time,
            refinement_level, paragraph_break)
        if self.processing_id == utterance_id:
            self.update_partial("")
        return old is None or old.text != text

    def clear_processing(self) -> None:
        self.update_partial("")

    def clear(self) -> None:
        self.clear_processing()
        self._finals.clear()

    @property
    def utterances(self) -> list[FinalUtterance]:
        return [self._finals[key] for key in sorted(self._finals)]

    def paragraphs(self) -> list[str]:
        result: list[str] = []
        previous: FinalUtterance | None = None
        for item in self.utterances:
            long_pause = bool(previous and item.start_time and previous.end_time and
                              item.start_time - previous.end_time >=
                              self.paragraph_pause_threshold)
            if not result or item.paragraph_break or long_pause:
                result.append(item.text)
            else:
                result[-1] += " " + item.text
            previous = item
        return result

    def clean_text(self) -> str:
        return "\n\n".join(self.paragraphs())
