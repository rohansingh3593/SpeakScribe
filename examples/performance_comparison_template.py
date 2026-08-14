"""Launch the source-checkout-only three-mode application UI.

This template intentionally launches ``app.main.MainWindow`` because the current
public ``speakscribe`` package does not yet expose the application's independent
Fast/Balanced/Accurate scheduler. It therefore works from a SpeakScribe source
checkout, but it is not a portable installed-library example. For a PyQt consumer in
another repository, start with ``examples/pyqt_library_template.py`` instead.
"""

import sys
from pathlib import Path

# Support the documented direct invocation from a source checkout on Windows
# and POSIX: ``python examples/performance_comparison_template.py``.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PyQt6.QtWidgets import QApplication

from app.main import MainWindow


def main() -> int:
    app = QApplication(sys.argv)
    window = MainWindow()
    window.display_mode.setCurrentText("Compare All")
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
