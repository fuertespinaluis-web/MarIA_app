from __future__ import annotations

from PySide6.QtWidgets import QMainWindow, QMessageBox


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MarIA")
        self.setMinimumSize(1100, 700)

        try:
            try:
                from .screens.main_ui import MainUI
            except ImportError:
                from screens.main_ui import MainUI
        except Exception as e:
            QMessageBox.critical(
                self,
                "MarIA",
                "MarIA no puede arrancar por imports rotos:\n\n"
                f"- No puedo importar .screens.main_ui.MainUI\n  {e}\n\n"
                "Revisa:\n"
                "- que exista apps/MarIA_app/screens/__init__.py\n"
                "- que el archivo se llame main_ui.py\n"
                "- que la clase se llame MainUI"
            )
            self.close()
            return

        self.main_ui = MainUI()
        self.setCentralWidget(self.main_ui)
