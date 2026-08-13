"""Final-transcript-only application template used by the default launcher."""

from __future__ import annotations

import sys

from PyQt6.QtWidgets import QApplication

from app.main import MainWindow, configure_logging, parse_args


class FinalOnlyMainWindow(MainWindow):
    """Use the production pipeline while omitting the live Processing preview."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("SpeakScribe — Final Transcript")
        self.processing_title.hide()
        self.processing_output.hide()
        self._processing_seen.clear()

    def show_mode_text(self, segment_id: int, mode_name: str, text: str,
                       final: bool, metrics: dict) -> None:
        """Ignore partial presentation and commit accepted finals immediately."""
        if not final:
            return
        final_metrics = dict(metrics)
        final_metrics["processing_previewed"] = True
        super().show_mode_text(segment_id, mode_name, text, True, final_metrics)

    def show_partial(self, text: str) -> None:
        """The legacy signal path is intentionally final-only in this template."""


def main(argv=None) -> int:
    args, qt_args = parse_args(argv)
    configure_logging(debug=args.debug)
    app = QApplication([sys.argv[0], *qt_args])
    window = FinalOnlyMainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
