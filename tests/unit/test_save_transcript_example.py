from datetime import datetime, timezone
from io import StringIO

import pytest

from examples.save_transcript import TranscriptWriter, process_result
from speakscribe import TranscriptionResult


@pytest.fixture
def writer(tmp_path):
    with TranscriptWriter(
        tmp_path / "transcripts",
        timestamp=datetime(2026, 8, 14, 10, 11, 12, tzinfo=timezone.utc),
    ) as transcript:
        yield transcript


def result(text: str, *, final: bool, utterance_id: int | None = None):
    return TranscriptionResult(text, is_final=final, utterance_id=utterance_id)


def test_transcript_directory_and_timestamped_file_are_created(tmp_path):
    output_dir = tmp_path / "nested" / "transcripts"
    with TranscriptWriter(
        output_dir,
        timestamp=datetime(2026, 8, 14, 10, 11, 12, 345678),
    ) as transcript:
        assert output_dir.is_dir()
        assert transcript.path.name == "transcript_2026-08-14_10-11-12_345678.txt"
        assert transcript.path.is_file()


@pytest.mark.parametrize("text", [
    "आज हमें project की performance improve करनी है।",
    "After that we need to run the tests.",
    "Kal main Jira ticket update karunga.",
])
def test_utf8_language_text_is_saved_exactly(writer, text):
    assert writer.append_final(text)
    assert writer.path.read_text(encoding="utf-8") == f"{text}\n"


def test_partial_result_is_displayed_but_not_saved(writer):
    output = StringIO()

    process_result(result("आज हमें project", final=False, utterance_id=1),
                   writer, output=output)

    assert output.getvalue() == "\rProcessing: आज हमें project"
    assert writer.path.read_text(encoding="utf-8") == ""


def test_final_result_is_appended_once(writer):
    final = result("Final transcript", final=True, utterance_id=7)

    process_result(final, writer, output=StringIO())
    process_result(final, writer, output=StringIO())

    assert writer.path.read_text(encoding="utf-8") == "Final transcript\n"


def test_empty_final_result_is_ignored(writer):
    assert not writer.append_final("  \n ", utterance_id=2)
    assert writer.path.read_text(encoding="utf-8") == ""


def test_duplicate_final_without_utterance_id_is_not_written_twice(writer):
    assert writer.append_final("Same finalized segment")
    assert not writer.append_final(" Same finalized segment ")
    assert writer.path.read_text(encoding="utf-8") == "Same finalized segment\n"


def test_multiple_final_segments_remain_in_order(writer):
    expected = [
        "आज हमें project की performance improve करनी है।",
        "After that we need to run the tests.",
        "Kal main final report share karunga.",
    ]

    for utterance_id, text in enumerate(expected, start=1):
        assert writer.append_final(text, utterance_id)

    assert writer.path.read_text(encoding="utf-8").splitlines() == expected


def test_each_final_is_flushed_and_synced(writer, monkeypatch):
    calls = []
    monkeypatch.setattr("examples.save_transcript.os.fsync", calls.append)

    assert writer.append_final("Durable final", utterance_id=3)

    assert calls == [writer._stream.fileno()]
    assert writer.path.read_text(encoding="utf-8") == "Durable final\n"


def test_context_manager_closes_file_handle(tmp_path):
    with TranscriptWriter(tmp_path) as transcript:
        transcript.append_final("Closed safely", utterance_id=1)
        assert not transcript.closed

    assert transcript.closed
    assert transcript.path.read_text(encoding="utf-8") == "Closed safely\n"
