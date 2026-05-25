from __future__ import annotations
from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QPixmap, QDragEnterEvent, QDropEvent
from PySide6.QtWidgets import QFrame, QLabel, QVBoxLayout


class LogoLabel(QLabel):
    """Logo que se escala 'contain' y se ve bien (sin deformar)."""
    def __init__(self, path: Path, w: int, h: int, fallback_text: str = ""):
        super().__init__()
        self.setFixedSize(w, h)
        self.setAlignment(Qt.AlignCenter)
        self._path = path
        self._w, self._h = w, h

        if path.exists():
            pm = QPixmap(str(path))
            if not pm.isNull():
                self.setPixmap(pm.scaled(w, h, Qt.KeepAspectRatio, Qt.SmoothTransformation))
                self.setStyleSheet("background: transparent;")
                return

        self.setText(fallback_text)
        self.setStyleSheet("color: rgba(255,255,255,0.9); background: transparent; font-size: 12px;")


class DropZone(QFrame):
    """Caja drag&drop para archivos (mzML, zip, joblib, npy, etc.)."""
    files_dropped = Signal(list)

    def __init__(self, hint: str = "Drop files here or click"):
        super().__init__()
        self.setAcceptDrops(True)
        self.setObjectName("DropZone")

        self.hint_label = QLabel(hint)
        self.hint_label.setAlignment(Qt.AlignCenter)
        self.hint_label.setObjectName("DropHint")

        lay = QVBoxLayout(self)
        lay.setContentsMargins(16, 16, 16, 16)
        lay.addWidget(self.hint_label)

        self._normal_style = """
        #DropZone {
            background: transparent;
            border: 2px solid #0F172A;
        }
        #DropHint {
            color: #64748B;
            font-style: italic;
        }
        """
        self._hover_style = """
        #DropZone {
            background: rgba(8,136,145,0.08);
            border: 2px dashed #088891;
        }
        #DropHint {
            color: #0F172A;
            font-style: italic;
            font-weight: 600;
        }
        """
        self.setStyleSheet(self._normal_style)

    def dragEnterEvent(self, e: QDragEnterEvent):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()
            self.setStyleSheet(self._hover_style)

    def dragLeaveEvent(self, e):
        self.setStyleSheet(self._normal_style)

    def dropEvent(self, e: QDropEvent):
        self.setStyleSheet(self._normal_style)
        paths = []
        for url in e.mimeData().urls():
            p = Path(url.toLocalFile())
            if p.exists():
                paths.append(str(p))
        if paths:
            self.files_dropped.emit(paths)
            e.acceptProposedAction()