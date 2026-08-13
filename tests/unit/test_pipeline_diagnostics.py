import json
from pathlib import Path

import numpy as np

from app.utils.pipeline_diagnostics import PipelineDiagnostics, RootCause


def test_trace_id_is_stable_across_complete_hindi_lifecycle(tmp_path):
    diagnostics = PipelineDiagnostics(tmp_path, enabled=True)
    diagnostics.detected(42, 18, "hi", rms=.081)
    for stage in ("SEGMENT", "QUEUE", "FAST RESULT", "UI", "FINALIZE"):
        diagnostics.stage(42, stage, language="hi")
    diagnostics.terminal(42, "FINAL", "delivered_to_final_ui", RootCause.UI)
    diagnostics.close()

    records = []
    for path in (tmp_path / "pipeline").glob("*.log"):
        records.extend(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines())
    traced = [record for record in records if record["utterance"] == "UTT-000042"]
    assert traced
    assert {record["stage"] for record in traced} >= {
        "VAD START", "SEGMENT", "QUEUE", "FAST RESULT", "UI", "FINALIZE"}
    assert diagnostics.traces[42].terminal == "FINAL"


def test_audio_conservation_assigns_explicit_reason_to_unresolved_speech(tmp_path):
    diagnostics = PipelineDiagnostics(tmp_path, enabled=True)
    diagnostics.detected(7, 2, "hi")
    diagnostics.stage(7, "SEGMENT")
    diagnostics.close("session_stop")
    trace = diagnostics.traces[7]
    assert trace.terminal == "REJECTED"
    assert trace.reason == "session_stop"
    assert "Dropped/rejected: 1" in (tmp_path / "summary.log").read_text()


def test_debug_audio_is_written_by_background_writer(tmp_path):
    diagnostics = PipelineDiagnostics(tmp_path, enabled=True, sample_rate=16_000)
    diagnostics.detected(3, 1, "hi")
    diagnostics.save_audio(3, np.zeros(1600, dtype=np.float32))
    diagnostics.terminal(3, "REJECTED", "test", RootCause.FAST_ASR)
    diagnostics.close()
    assert (tmp_path / "debug_audio" / "UTT-000003_input.wav").stat().st_size > 44


def test_disabled_diagnostics_create_no_files(tmp_path):
    diagnostics = PipelineDiagnostics(tmp_path, enabled=False)
    diagnostics.detected(1, 1, "hi")
    diagnostics.save_audio(1, np.zeros(10))
    diagnostics.close()
    assert list(Path(tmp_path).iterdir()) == []


def test_root_cause_classifies_first_missing_or_explicit_discard_stage(tmp_path):
    diagnostics = PipelineDiagnostics(tmp_path, enabled=False)
    diagnostics.detected(1, 1, "hi")
    diagnostics.stage(1, "SEGMENT")
    assert diagnostics.classify_root_cause(1) is RootCause.QUEUE
    diagnostics.terminal(1, "REJECTED", "stale_generation_after_asr",
                         RootCause.GENERATION_FILTER)
    assert diagnostics.classify_root_cause(1) is RootCause.GENERATION_FILTER
