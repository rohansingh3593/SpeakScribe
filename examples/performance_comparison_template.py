"""Launch the original three-row performance-comparison UI template."""

import sys

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
