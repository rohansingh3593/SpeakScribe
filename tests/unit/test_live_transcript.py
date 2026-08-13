"""UI-LIVE regression coverage for the Processing + Final state model."""

from app.processing.live_transcript import LiveTranscriptModel


def test_ui_live_001_002_partial_replaces_in_place():
    model = LiveTranscriptModel()
    model.update_partial("Today we need", 7)
    assert model.processing_text == "Today we need"
    model.update_partial("Today we need to update", 7)
    assert model.processing_text == "Today we need to update"


def test_ui_live_003_004_final_moves_and_clears_processing():
    model = LiveTranscriptModel()
    model.update_partial("working", 3)
    model.commit(3, "Working is complete.")
    assert model.processing_text == ""
    assert model.clean_text() == "Working is complete."


def test_ui_live_005_006_paragraph_policy():
    model = LiveTranscriptModel(paragraph_pause_threshold=2.0)
    model.commit(1, "Good morning.", start_time=0, end_time=1)
    model.commit(2, "Today we begin.", start_time=1.5, end_time=2)
    model.commit(3, "After a break.", start_time=4.1, end_time=5)
    assert model.paragraphs() == ["Good morning. Today we begin.", "After a break."]


def test_ui_live_007_008_refinements_replace_same_utterance():
    model = LiveTranscriptModel()
    model.commit(9, "We need update dependency.", refinement_level="fast")
    model.commit(9, "We need to update the dependency.", refinement_level="balanced")
    model.commit(9, "We need to update SQLAlchemy dependency.",
                 refinement_level="accurate")
    assert len(model.utterances) == 1
    assert model.clean_text() == "We need to update SQLAlchemy dependency."
    assert not model.commit(9, "obsolete", refinement_level="fast")


def test_ui_live_009_010_011_unicode_and_technical_terms_are_untouched():
    model = LiveTranscriptModel()
    texts = ["सुप्रभात सभी को।", "Today we are discussing the project.",
             "आज हमें SQLAlchemy dependency को update करना है।"]
    for number, value in enumerate(texts):
        model.commit(number, value)
    assert model.clean_text() == " ".join(texts)


def test_ui_live_015_016_stop_and_language_switch_preserve_final():
    model = LiveTranscriptModel()
    model.commit(1, "आज application update करें।", language="hi")
    model.update_partial("unfinished", 2)
    model.clear_processing()
    model.commit(2, "After that run tests.", language="en")
    assert model.clean_text() == "आज application update करें। After that run tests."


def test_ui_live_017_stale_lower_quality_result_is_ignored():
    model = LiveTranscriptModel()
    model.commit(4, "Accurate final.", refinement_level="accurate")
    model.commit(4, "Late fast result.", refinement_level="fast")
    assert model.clean_text() == "Accurate final."


def test_ui_live_018_copy_text_excludes_processing():
    model = LiveTranscriptModel()
    model.commit(1, "Clean final.")
    model.update_partial("debug partial", 2)
    assert model.clean_text() == "Clean final."


def test_ui_live_019_020_long_updates_do_not_mutate_final_model():
    model = LiveTranscriptModel()
    for number in range(3600):
        model.commit(number, f"Sentence {number}.")
    snapshot = model.clean_text()
    for number in range(1000):
        model.update_partial(f"live {number}", 3601)
    assert len(model.utterances) == 3600
    assert model.clean_text() == snapshot
