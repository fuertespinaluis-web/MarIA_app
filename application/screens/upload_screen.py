from __future__ import annotations

import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QFileDialog, QMessageBox
)

from .widgets import DropZone


# =========================
# Bruker detection
# =========================
def _has_any_case_file(folder: Path, names: set[str]) -> bool:
    try:
        for p in folder.iterdir():
            if p.is_file() and p.name.lower() in names:
                return True
    except Exception:
        return False
    return False


def is_bruker_spectrum_folder(folder: Path) -> bool:
    """
    Un folder es espectro Bruker si contiene:
      - fid
      - acqu o acqus
    """
    if not folder.is_dir():
        return False
    has_fid = _has_any_case_file(folder, {"fid"})
    has_acq = _has_any_case_file(folder, {"acqu", "acqus"})
    return has_fid and has_acq


# =========================
# TXT detection
# =========================
TEXT_SPECTRUM_EXTENSIONS = {".txt"}  # puedes ampliar a {".txt", ".csv", ".tsv"}


def is_text_spectrum_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in TEXT_SPECTRUM_EXTENSIONS


# =========================
# Pseudo ID builder
# =========================
def build_pseudo_id_from_spectrum_path(spectrum_path: Path) -> str:
    """
    Construye un nombre visible corto para el espectro.

    Si spectrum_path es una carpeta Bruker válida, usa esa carpeta.
    Si spectrum_path es un archivo txt, usa su carpeta contenedora.
    """
    current = spectrum_path.parent if spectrum_path.is_file() else spectrum_path
    if not is_bruker_spectrum_folder(current):
        for parent in current.parents:
            if is_bruker_spectrum_folder(parent):
                current = parent
                break
    if is_bruker_spectrum_folder(current) and current.name == "1SLin" and len(current.parents) >= 3:
        current = current.parents[2]
    name = current.name.strip()
    return name if name else spectrum_path.name


# =========================
# Generic hit
# =========================
@dataclass(frozen=True)
class SpectrumHit:
    parent_pseudo_id: str
    spectrum_path: Path   # puede ser carpeta Bruker o archivo txt
    spectrum_type: str    # "bruker" o "text"


def find_supported_spectra_under_parent(parent: Path) -> list[SpectrumHit]:
    """
    Busca bajo 'parent':
      - carpetas Bruker válidas
      - archivos .txt

    Devuelve una lista de SpectrumHit.
    """
    hits: list[SpectrumHit] = []

    # Caso raro: parent ya es un espectro Bruker
    if is_bruker_spectrum_folder(parent):
        hits.append(
            SpectrumHit(
                parent_pseudo_id=build_pseudo_id_from_spectrum_path(parent),
                spectrum_path=parent,
                spectrum_type="bruker",
            )
        )

    # Caso raro: parent ya es un txt
    if is_text_spectrum_file(parent):
        hits.append(
            SpectrumHit(
                parent_pseudo_id=build_pseudo_id_from_spectrum_path(parent),
                spectrum_path=parent,
                spectrum_type="text",
            )
        )

    # Recorrido recursivo
    for p in parent.rglob("*"):
        if p.is_dir() and is_bruker_spectrum_folder(p):
            hits.append(
                SpectrumHit(
                    parent_pseudo_id=build_pseudo_id_from_spectrum_path(p),
                    spectrum_path=p,
                    spectrum_type="bruker",
                )
            )
        elif p.is_file() and is_text_spectrum_file(p):
            hits.append(
                SpectrumHit(
                    parent_pseudo_id=build_pseudo_id_from_spectrum_path(p),
                    spectrum_path=p,
                    spectrum_type="text",
                )
            )

    # unique + sorted
    uniq = sorted(
        {h.spectrum_path: h for h in hits}.values(),
        key=lambda x: str(x.spectrum_path)
    )
    return uniq


def extract_zip_to_temp(zip_path: Path) -> Path:
    tmp_dir = Path(tempfile.mkdtemp(prefix="maria_spectra_"))
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(tmp_dir)
    return tmp_dir


class UploadScreen(QWidget):
    start_clicked = Signal(list)  # lista final de rutas válidas (folders Bruker o archivos txt)

    def __init__(self):
        super().__init__()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(28, 24, 28, 24)
        layout.setSpacing(14)

        title = QLabel("UPLOAD YOUR SPECTRA")
        title.setObjectName("PageTitle")
        title.setAlignment(Qt.AlignHCenter)

        self.drop = DropZone(
            "Drag & drop a PARENT folder, ZIP, or spectrum TXT here\n"
            "or click to select ZIP/TXT files or a folder"
        )
        self.drop.setFixedSize(640, 260)

        self.drop.mousePressEvent = lambda ev: self.open_input_dialog()

        btn_row = QHBoxLayout()
        btn_row.setSpacing(10)

        self.button = QPushButton("Start classification")
        self.button.setObjectName("BigButton")
        self.button.setFixedHeight(56)
        self.button.clicked.connect(self._emit_start)

        btn_row.addStretch(1)
        btn_row.addWidget(self.button)

        self.status = QLabel("")
        self.status.setObjectName("Status")
        self.status.setAlignment(Qt.AlignHCenter)

        layout.addWidget(title)
        layout.addSpacing(8)
        layout.addWidget(self.drop, alignment=Qt.AlignHCenter)
        layout.addLayout(btn_row)
        layout.addWidget(self.status)
        layout.addStretch(1)

        self._selected_paths: list[str] = []
        self._hits: list[SpectrumHit] = []
        self._temp_dirs: list[Path] = []

        self.drop.files_dropped.connect(self._on_paths)

        self.setStyleSheet("""
        #PageTitle { font-size: 20px; font-weight: 900; letter-spacing: 0.5px; }

        #BigButton {
            background: #0E5A7A;
            color: white;
            font-weight: 900;
            border-radius: 2px;
            padding: 10px 20px;
        }
        #BigButton:hover { background: #0B4D69; }

        #SecondaryButton {
            background: rgba(15, 23, 42, 0.06);
            border: 1px solid rgba(15, 23, 42, 0.20);
            color: #0F172A;
            font-weight: 700;
            border-radius: 8px;
            padding: 8px 14px;
        }
        #SecondaryButton:hover { background: rgba(15, 23, 42, 0.10); }

        #Status { color: #334155; }
        """)

    # ---------------- dialogs ----------------
    def _select_multiple_folders(self) -> list[str]:
        selected: list[str] = []
        while True:
            folder = QFileDialog.getExistingDirectory(
                self,
                "Select Bruker folder or parent folder",
                "",
                QFileDialog.ShowDirsOnly,
            )
            if not folder:
                break
            if folder not in selected:
                selected.append(folder)

            answer = QMessageBox.question(
                self,
                "Add another folder?",
                "Do you want to add another folder?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if answer != QMessageBox.Yes:
                break
        return selected

    def open_input_dialog(self):
        folders = self._select_multiple_folders()
        if folders:
            self._on_paths(folders)
            return

        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Select ZIP or TXT spectra",
            "",
            "Supported inputs (*.zip *.txt);;ZIP files (*.zip);;Spectrum TXT (*.txt);;All files (*.*)"
        )
        if files:
            self._on_paths(files)
            return

        folder = QFileDialog.getExistingDirectory(
            self,
            "Select parent folder",
            "",
            QFileDialog.ShowDirsOnly
        )
        if folder:
            self._on_paths([folder])

    # ---------------- pipeline ----------------
    def _on_paths(self, paths: list[str]):
        self._selected_paths = paths
        self._hits = []

        if not paths:
            self.status.setText("No input selected.")
            return

        all_hits: list[SpectrumHit] = []

        for p_str in paths:
            p = Path(p_str)

            if p.is_file() and p.suffix.lower() == ".zip":
                try:
                    extracted = extract_zip_to_temp(p)
                    self._temp_dirs.append(extracted)
                    all_hits.extend(find_supported_spectra_under_parent(extracted))
                except Exception as e:
                    self.status.setText(f"Error extracting ZIP: {p.name} -> {e}")
                    return

            elif p.is_dir():
                all_hits.extend(find_supported_spectra_under_parent(p))

            elif p.is_file() and is_text_spectrum_file(p):
                all_hits.append(
                    SpectrumHit(
                        parent_pseudo_id=build_pseudo_id_from_spectrum_path(p),
                        spectrum_path=p,
                        spectrum_type="text",
                    )
                )

        uniq_map = {h.spectrum_path: h for h in all_hits}
        self._hits = sorted(uniq_map.values(), key=lambda x: str(x.spectrum_path))

        if self._hits:
            parents = sorted({h.parent_pseudo_id for h in self._hits})
            n_bruker = sum(h.spectrum_type == "bruker" for h in self._hits)
            n_text = sum(h.spectrum_type == "text" for h in self._hits)

            self.status.setText(
                f"Found {len(self._hits)} spectra under {len(parents)} pseudo-id(s) "
                f"({n_bruker} Bruker, {n_text} TXT)."
            )
        else:
            self.status.setText(
                "No supported spectra found "
                "(need Bruker folder with fid+acqu/acqus or .txt files)."
            )

    # ---------------- emit ----------------
    def _emit_start(self):
        """
        Emite la lista final de rutas:
          - carpetas Bruker
          - archivos TXT
        """
        if self._hits:
            self.start_clicked.emit([str(h.spectrum_path) for h in self._hits])
            return

        if self._selected_paths:
            self.status.setText("No valid spectra detected. Emitting selected paths anyway.")
            self.start_clicked.emit(self._selected_paths)
        else:
            self.status.setText("No input selected.")
