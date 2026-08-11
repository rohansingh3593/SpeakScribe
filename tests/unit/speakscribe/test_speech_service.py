from threading import Event
import time

import numpy as np
import pytest

from speakscribe import (
    BaseTranscriptionEngine, MicrophoneError, ServiceStateError, SpeechConfig,
    SpeechToText, TranscriptionError, TranscriptionResult,
)
from speakscribe.audio import BaseAudioRecorder


class FakeRecorder(BaseAudioRecorder):
    def __init__(self, chunks=(), delay=0):
        self.chunks = list(chunks)
        self.delay = delay
        self.started = self.stopped = self.closed = 0
        self.finished_at = None

    def start(self):
        self.started += 1

    def iter_audio(self, stop_event: Event):
        for chunk in self.chunks:
            if stop_event.is_set():
                break
            if self.delay:
                time.sleep(self.delay)
            yield chunk
        self.finished_at = time.monotonic()

    def stop(self):
        self.stopped += 1

    def close(self):
        self.closed += 1
        self.stop()


class FakeEngine(BaseTranscriptionEngine):
    def __init__(self, prefix="text", delay=0):
        self.prefix = prefix
        self.delay = delay
        self.calls = self.closed = 0
        self.first_finished_at = None

    def transcribe(self, audio, sample_rate):
        self.calls += 1
        if self.delay:
            time.sleep(self.delay)
        self.first_finished_at = self.first_finished_at or time.monotonic()
        return TranscriptionResult(f"{self.prefix} {self.calls}", language="en", confidence=.9)

    def close(self):
        self.closed += 1


def cfg(**overrides):
    values = dict(sample_rate=1000, chunk_duration=.1, partial_interval=.2,
                  silence_duration=.2, minimum_speech_duration=.1,
                  pre_speech_duration=0, minimum_rms=.01,
                  max_audio_queue=8, max_asr_queue=2)
    values.update(overrides)
    return SpeechConfig(**values)


def speech_frames(count=6, value=.2):
    return [np.full(100, value, dtype=np.float32) for _ in range(count)]


def silence_frames(count=2):
    return [np.zeros(100, dtype=np.float32) for _ in range(count)]


def collect(prefix="text", frames=None, engine_delay=0):
    recorder = FakeRecorder(frames or speech_frames() + silence_frames())
    engine = FakeEngine(prefix, engine_delay)
    results = list(SpeechToText(cfg(), recorder=recorder, engine=engine).listen_continuously())
    return results, recorder, engine


def test_listen_once_returns_final_structured_result_and_releases_session():
    recorder = FakeRecorder(speech_frames(2) + silence_frames())
    speech = SpeechToText(cfg(), recorder=recorder, engine=FakeEngine("hello"))
    result = speech.listen_once()
    assert result.text.startswith("hello")
    assert result.is_final
    assert result.audio_duration >= .2
    assert result.inference_seconds is not None
    assert not speech.is_running
    assert recorder.started == recorder.stopped == 1


@pytest.mark.parametrize("language,text", [
    ("en", "Today main SQLAlchemy"),
    ("hi", "आज मैं काम कर रहा हूँ"),
    (None, "Aaj main pipeline check karunga"),
])
def test_live_partial_and_final_transcription_for_language_modes(language, text):
    results, _, _ = collect(text)
    partials = [result for result in results if not result.is_final]
    finals = [result for result in results if result.is_final]
    assert len(partials) >= 2
    assert len({result.utterance_id for result in results}) == 1
    assert finals and finals[-1].text


def test_partial_results_are_exposed_before_capture_finishes():
    recorder = FakeRecorder(speech_frames(8) + silence_frames(), delay=.01)
    speech = SpeechToText(cfg(), recorder=recorder, engine=FakeEngine())
    iterator = speech.listen_continuously()
    first = next(iterator)
    assert not first.is_final
    assert recorder.finished_at is None
    iterator.close()


def test_capture_continues_while_slow_asr_is_running_and_queue_stays_bounded():
    results, recorder, engine = collect(frames=speech_frames(20) + silence_frames(),
                                        engine_delay=.04)
    assert recorder.finished_at <= engine.first_finished_at
    assert results[-1].is_final
    assert engine.calls < 11  # obsolete partials were coalesced rather than backlogged
    assert all(result.queue_wait_seconds < 1 for result in results)


def test_short_pause_does_not_finalize_but_silence_does():
    frames = speech_frames(3) + silence_frames(1) + speech_frames(3) + silence_frames(2)
    results, _, _ = collect(frames=frames)
    assert sum(result.is_final for result in results) == 1


def test_callback_worker_delivers_partial_and_final_results():
    speech = SpeechToText(cfg(), recorder=FakeRecorder(speech_frames(4) + silence_frames()),
                          engine=FakeEngine("callback"))
    received = []
    worker = speech.start_continuous(received.append)
    worker.join(timeout=2)
    assert received[0].is_final is False
    assert received[-1].is_final is True
    assert not speech.is_running


def test_transcription_failure_is_chained_as_library_exception():
    class BrokenEngine(BaseTranscriptionEngine):
        def transcribe(self, audio, sample_rate):
            raise RuntimeError("backend exploded")

    speech = SpeechToText(cfg(), recorder=FakeRecorder(speech_frames(2) + silence_frames()),
                          engine=BrokenEngine())
    with pytest.raises(TranscriptionError) as error:
        speech.listen_once()
    assert isinstance(error.value.__cause__, RuntimeError)


def test_microphone_failure_remains_clear():
    class BrokenRecorder(FakeRecorder):
        def start(self):
            raise MicrophoneError("permission denied")

    with pytest.raises(MicrophoneError, match="permission denied"):
        SpeechToText(cfg(), recorder=BrokenRecorder(), engine=FakeEngine()).start()


def test_engine_switching_and_cleanup():
    original = FakeEngine()
    replacement = FakeEngine("replacement")
    speech = SpeechToText(cfg(), recorder=FakeRecorder(speech_frames(2) + silence_frames()),
                          engine=original)
    speech.set_engine(replacement)
    assert original.closed == 1
    speech.start()
    with pytest.raises(ServiceStateError):
        speech.set_engine(FakeEngine())
    speech.stop()
    speech.close()
    assert replacement.closed == 1


def test_configuration_validation():
    assert SpeechConfig(language="en-IN").language == "en-IN"
    with pytest.raises(ValueError):
        SpeechConfig(partial_interval=0)
    with pytest.raises(ValueError):
        SpeechConfig(capture_source="unknown")
