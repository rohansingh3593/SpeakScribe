from threading import Event

import numpy as np
import pytest

from speakscribe import (
    BaseTranscriptionEngine, MicrophoneError, ServiceStateError, SpeechConfig,
    SpeechToText, TranscriptionError, TranscriptionResult,
)
from speakscribe.audio import BaseAudioRecorder


class FakeRecorder(BaseAudioRecorder):
    def __init__(self, chunks=()):
        self.chunks = list(chunks)
        self.started = 0
        self.stopped = 0
        self.closed = 0

    def start(self):
        self.started += 1

    def iter_audio(self, stop_event: Event):
        for chunk in self.chunks:
            if stop_event.is_set():
                break
            yield chunk

    def stop(self):
        self.stopped += 1

    def close(self):
        self.closed += 1
        self.stop()


class FakeEngine(BaseTranscriptionEngine):
    def __init__(self, texts=("hello",)):
        self.texts = iter(texts)
        self.calls = 0
        self.closed = 0

    def transcribe(self, audio, sample_rate):
        self.calls += 1
        return TranscriptionResult(next(self.texts), language="en", confidence=.9)

    def close(self):
        self.closed += 1


def audio_chunks(count=1):
    return [np.ones(1600, dtype=np.float32) for _ in range(count)]


def test_listen_once_returns_structured_result_and_releases_owned_session():
    recorder = FakeRecorder(audio_chunks())
    speech = SpeechToText(recorder=recorder, engine=FakeEngine())
    result = speech.listen_once()
    assert result.text == "hello"
    assert result.language == "en"
    assert result.confidence == .9
    assert not speech.is_running
    assert recorder.started == recorder.stopped == 1


def test_continuous_generator_yields_each_result_without_waiting_for_completion():
    recorder = FakeRecorder(audio_chunks(2))
    speech = SpeechToText(recorder=recorder, engine=FakeEngine(("first", "second")))
    results = speech.listen_continuously()
    assert next(results).text == "first"
    assert speech.is_running
    assert next(results).text == "second"
    with pytest.raises(StopIteration):
        next(results)
    assert not speech.is_running


def test_start_stop_and_context_manager_cleanup_are_idempotent():
    recorder = FakeRecorder()
    engine = FakeEngine()
    speech = SpeechToText(recorder=recorder, engine=engine)
    with speech:
        assert speech.is_running
        speech.start()
        assert recorder.started == 1
    assert not speech.is_running
    assert recorder.closed == 1
    assert engine.closed == 1


def test_callback_worker_delivers_results_and_stops():
    speech = SpeechToText(recorder=FakeRecorder(audio_chunks(2)),
                          engine=FakeEngine(("one", "two")))
    received = []
    worker = speech.start_continuous(received.append)
    worker.join(timeout=2)
    assert [result.text for result in received] == ["one", "two"]
    assert not speech.is_running


def test_transcription_failure_is_chained_as_library_exception():
    class BrokenEngine(BaseTranscriptionEngine):
        def transcribe(self, audio, sample_rate):
            raise RuntimeError("backend exploded")

    speech = SpeechToText(recorder=FakeRecorder(audio_chunks()), engine=BrokenEngine())
    with pytest.raises(TranscriptionError) as error:
        speech.listen_once()
    assert isinstance(error.value.__cause__, RuntimeError)


def test_microphone_failure_remains_a_clear_library_exception():
    class BrokenRecorder(FakeRecorder):
        def start(self):
            raise MicrophoneError("permission denied")

    with pytest.raises(MicrophoneError, match="permission denied"):
        SpeechToText(recorder=BrokenRecorder(), engine=FakeEngine()).start()


def test_engine_can_switch_only_while_stopped():
    original = FakeEngine()
    replacement = FakeEngine(("replacement",))
    speech = SpeechToText(recorder=FakeRecorder(audio_chunks()), engine=original)
    speech.set_engine(replacement)
    assert original.closed == 1
    assert speech.listen_once().text == "replacement"
    speech.start()
    with pytest.raises(ServiceStateError):
        speech.set_engine(FakeEngine())
    speech.stop()


def test_configuration_validation():
    assert SpeechConfig(language="en-IN").language == "en-IN"
    assert SpeechConfig(capture_source="loopback").capture_source == "loopback"
    with pytest.raises(ValueError):
        SpeechConfig(sample_rate=0)
    with pytest.raises(ValueError):
        SpeechConfig(capture_source="unknown")
