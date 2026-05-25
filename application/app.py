import sys
import traceback
from pathlib import Path

from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QIcon

from .main_window import MainWindow
from .screens.splash_screen import show_splash_then


def excepthook(exc_type, exc, tb):
    traceback.print_exception(exc_type, exc, tb)


def main() -> int:
    sys.excepthook = excepthook

    app = QApplication(sys.argv)

    base_dir = Path(__file__).resolve().parent
    icon_path = base_dir / "assets" / "maria_logo.png"

    if icon_path.exists():
        app.setWindowIcon(QIcon(str(icon_path)))

    win = MainWindow()

    if icon_path.exists():
        win.setWindowIcon(QIcon(str(icon_path)))

    win.showMaximized()
    win.setWindowOpacity(0.0)

    show_splash_then(win, ms=6000)

    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
