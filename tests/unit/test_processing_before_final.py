"""Regression checks for the visible Processing → Final lifecycle."""

from pathlib import Path


def test_final_without_partial_is_previewed_before_deferred_commit():
    source = Path("app/main.py").read_text(encoding="utf-8")
    handler = source.split("def show_mode_text", 1)[1].split("def _remove_segment", 1)[0]
    assert "segment_id not in self._processing_seen" in handler
    assert "self.show_mode_text(segment_id, mode_name, text, False" in handler
    assert "QTimer.singleShot" in handler
    assert "self._deliver_deferred_final" in handler


def test_stop_invalidates_deferred_final_preview():
    source = Path("app/main.py").read_text(encoding="utf-8")
    stop = source.split("def stop_listening", 1)[1].split("def _finish_stop_ui", 1)[0]
    deferred = source.split("def _deliver_deferred_final", 1)[1].split(
        "def _remove_segment", 1)[0]
    assert "self._ui_session_epoch += 1" in stop
    assert "expected_epoch != self._ui_session_epoch" in deferred
