"""Fast developer checks for the four primary language transitions."""

import time

import pytest

from app.asr.language_transition import RecognitionState
from tests.switching.support import Engine, evidence, job, run_worker


CASES = [
    ("SW-LIGHT-001", "hi", "en", "Today I need to update the SQLAlchemy dependency."),
    ("SW-LIGHT-002", "en", "hi", "आज मुझे ऑफिस जाने से पहले मीटिंग जॉइन करनी है।"),
    ("SW-LIGHT-003", "auto", "en", "The pipeline has completed successfully."),
    ("SW-LIGHT-004", "en", "auto", "Aaj main SQLAlchemy dependency update kar raha hoon."),
]


@pytest.mark.parametrize("test_id,source,target,transcript", CASES,
                         ids=[case[0] for case in CASES])
def test_basic_switch_first_result(test_id, source, target, transcript, record_switch):
    state = RecognitionState(source, "original")
    switched = state.switch(target, "original")
    first = job(1, target, switched.generation,
                switched_at=switched.switched_at, ready_at=switched.ready_at)
    signals, provider = run_worker(state, [first], Engine({1: transcript}))
    actual = signals.partial_text.values[0][0]
    record_switch(evidence(
        test_id, f"{source} to {target} first transcript",
        "New mode produces a clean first transcript", actual,
        switch_from=source, switch_to=target,
        switch_time=switched.ready_at - switched.switched_at,
        total_latency=time.monotonic() - switched.switched_at,
        first_transcript=actual, notes="Deterministic ASR double; real worker/state path"))
    assert state.snapshot().language == target
    assert actual == transcript
    assert provider.calls == 1


def test_previous_final_history_is_preserved(record_switch):
    history = ["आज मुझे ऑफिस जाना है।"]
    state = RecognitionState("hi", "original")
    switched = state.switch("en", "original")
    signals, _ = run_worker(
        state, [job(2, "en", switched.generation, final=True)],
        Engine({2: "After that I will join the meeting."}))
    history.extend(value[0] for value in signals.final_text.values)
    record_switch(evidence(
        "SW-STATE-004", "Final transcript history preserved",
        "Old Hindi final remains unchanged before new English final", repr(history),
        switch_from="hi", switch_to="en", final_transcript="\n\n".join(history)))
    assert history == ["आज मुझे ऑफिस जाना है।", "After that I will join the meeting."]
