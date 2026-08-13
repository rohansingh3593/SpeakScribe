"""Structural regression checks for the compact live-status template."""

from pathlib import Path


def test_template_contains_listening_card_meter_and_status_fields():
    source = Path("app/main.py").read_text(encoding="utf-8")
    for content in ("listeningCard", "Audio level indicator", "● Status: Ready",
                    "▰ Mode: Balanced", "Words: 0", "Characters: 0"):
        assert content in source


def test_meter_reuses_record_timer_instead_of_adding_animation_thread():
    source = Path("app/main.py").read_text(encoding="utf-8")
    timer_method = source.split("def _update_record_timer", 1)[1].split(
        "def _connect_signals", 1)[0]
    assert "audio_level_label.setText" in timer_method
    assert "Thread(" not in timer_method

