from pathlib import Path


def test_default_launcher_is_final_only_and_example_retains_live_update():
    root = Path(__file__).resolve().parents[2]
    assert "app.final_only_main" in (root / "main.py").read_text(encoding="utf-8")
    example = (root / "examples/live_update_main.py").read_text(encoding="utf-8")
    assert "from app.main import main" in example


def test_final_only_template_hides_processing_and_ignores_partials():
    source = Path("app/final_only_main.py").read_text(encoding="utf-8")
    assert "self.processing_title.hide()" in source
    assert "self.processing_output.hide()" in source
    assert "if not final:" in source
    assert 'final_metrics["processing_previewed"] = True' in source


def test_performance_comparison_template_documents_source_checkout_boundary():
    source = Path("examples/performance_comparison_template.py").read_text(encoding="utf-8")

    assert "source-checkout-only" in source
    assert "from app.main import MainWindow" in source
    assert "pyqt_library_template.py" in source
    assert "from speakscribe import" not in source
