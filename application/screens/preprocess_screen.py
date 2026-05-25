from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from PySide6.QtCore import Signal, QObject, QThread
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QProgressBar, QFrame, QSizePolicy, QComboBox
)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from ..maldi_imm.SpectrumObject import SpectrumObject
from ..maldi_imm.preprocessing import (
    SequentialPreprocessor,
    VarStabilizer,
    Smoother,
    BaselineCorrecter,
    Binner,
    Trimmer,
    StdThresholder,
)


# -----------------------------
# Supported input detection
# -----------------------------
TEXT_EXTENSIONS = {".txt"}


@dataclass(frozen=True)
class SpectrumInput:
    path: Path
    kind: str   # "bruker" | "txt"


def is_bruker_folder(folder: Path) -> bool:
    if not folder.is_dir():
        return False

    names = {p.name.lower() for p in folder.iterdir() if p.is_file()}
    return "fid" in names and ("acqu" in names or "acqus" in names)


def is_txt_spectrum(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in TEXT_EXTENSIONS


def find_supported_inputs(root: Path) -> list[SpectrumInput]:
    hits: list[SpectrumInput] = []

    if root.is_dir():
        if is_bruker_folder(root):
            hits.append(SpectrumInput(root, "bruker"))

        for p in root.rglob("*"):
            if p.is_dir() and is_bruker_folder(p):
                hits.append(SpectrumInput(p, "bruker"))
            elif p.is_file() and is_txt_spectrum(p):
                hits.append(SpectrumInput(p, "txt"))

    elif root.is_file() and is_txt_spectrum(root):
        hits.append(SpectrumInput(root, "txt"))

    uniq = {h.path.resolve(): h for h in hits}
    return sorted(uniq.values(), key=lambda x: str(x.path))


# -----------------------------
# TXT loader
# -----------------------------
def _read_txt_spectrum_table(txt_path: Path) -> pd.DataFrame:
    """
    Intenta leer un txt de espectro en formatos comunes:
      - 2 columnas separadas por espacios/tabs
      - 2 columnas separadas por coma/;
      - con o sin cabecera
    """
    attempts = [
        dict(sep=r"\s+", engine="python", header=None),
        dict(sep="\t", engine="python", header=None),
        dict(sep=",", engine="python", header=None),
        dict(sep=";", engine="python", header=None),
    ]

    for kwargs in attempts:
        try:
            df = pd.read_csv(txt_path, **kwargs)
            if df.shape[1] >= 2:
                df = df.iloc[:, :2].copy()
                df.columns = ["mz", "intensity"]
                df["mz"] = pd.to_numeric(df["mz"], errors="coerce")
                df["intensity"] = pd.to_numeric(df["intensity"], errors="coerce")
                df = df.dropna(subset=["mz", "intensity"])
                if len(df) > 10:
                    return df
        except Exception:
            pass

    for kwargs in [
        dict(sep=r"\s+", engine="python"),
        dict(sep="\t", engine="python"),
        dict(sep=",", engine="python"),
        dict(sep=";", engine="python"),
    ]:
        try:
            df = pd.read_csv(txt_path, **kwargs)

            mz_col = None
            int_col = None

            for c in df.columns:
                cl = str(c).strip().lower()
                if cl in {"mz", "m/z"}:
                    mz_col = c
                if cl in {"intensity", "int", "i"}:
                    int_col = c

            if mz_col is not None and int_col is not None:
                out = df[[mz_col, int_col]].copy()
                out.columns = ["mz", "intensity"]
                out["mz"] = pd.to_numeric(out["mz"], errors="coerce")
                out["intensity"] = pd.to_numeric(out["intensity"], errors="coerce")
                out = out.dropna(subset=["mz", "intensity"])
                if len(out) > 10:
                    return out
        except Exception:
            pass

    raise ValueError(f"No se pudo interpretar el TXT como espectro: {txt_path}")


def spectrum_from_txt(txt_path: Path) -> SpectrumObject:
    df = _read_txt_spectrum_table(txt_path)

    s = SpectrumObject(
        mz=df["mz"].to_numpy(dtype=float),
        intensity=df["intensity"].to_numpy(dtype=float),
        meta={}
    )
    return s


# -----------------------------
# Pseudo ID
# -----------------------------
def build_pseudo_id_from_path(path: Path) -> str:
    """
    Construye un nombre visible corto para el espectro.

    Para Bruker:
      usa el nombre de la carpeta que contiene `fid` y `acqu/acqus`.
    Para TXT:
      usa el nombre de la carpeta contenedora.
    """
    current = path.parent if path.is_file() else path
    if not is_bruker_folder(current):
        for parent in current.parents:
            if is_bruker_folder(parent):
                current = parent
                break
    if is_bruker_folder(current) and current.name == "1SLin" and len(current.parents) >= 3:
        current = current.parents[2]
    name = current.name.strip()
    return name if name else path.name


def build_pseudo_id(inp: SpectrumInput) -> str:
    return build_pseudo_id_from_path(inp.path)


# -----------------------------
# Worker
# -----------------------------
class PreprocessWorker(QObject):
    progress = Signal(int)
    status = Signal(str)
    finished = Signal(object, object)
    error = Signal(str)

    def __init__(self, input_paths: list[str]):
        super().__init__()
        self.input_paths = input_paths

    def run(self):
        try:
            roots = [Path(p) for p in self.input_paths]
            inputs: list[SpectrumInput] = []

            for r in roots:
                inputs.extend(find_supported_inputs(r))

            inputs = sorted({x.path.resolve(): x for x in inputs}.values(), key=lambda x: str(x.path))

            if not inputs:
                raise ValueError("No supported spectra found (Bruker or TXT).")

            raw: list[SpectrumObject] = []
            n = len(inputs)

            self.status.emit(f"Loading {n} spectra…")

            for i, inp in enumerate(inputs, 1):
                if inp.kind == "bruker":
                    acqu_path = inp.path / "acqus"
                    if not acqu_path.exists():
                        acqu_path = inp.path / "acqu"
                    fid_path = inp.path / "fid"
                    s = SpectrumObject.from_bruker(acqu_path, fid_path)

                elif inp.kind == "txt":
                    s = spectrum_from_txt(inp.path)

                else:
                    raise ValueError(f"Unsupported input type: {inp.kind}")

                pseudo_id = build_pseudo_id(inp)

                s.meta = {
                    "pseudo_id": pseudo_id,
                    "source_path": str(inp.path),
                    "source_type": inp.kind,
                }

                raw.append(s)
                self.progress.emit(int((i / n) * 40))

            preproc = SequentialPreprocessor(
                VarStabilizer(method="sqrt"),
                Smoother(halfwindow=10, polyorder=3),
                BaselineCorrecter(method="SNIP", snip_n_iter=20),
                StdThresholder(factor=1),
                Binner(start=2000.0, stop=20000.0, step=1.0),
                Trimmer(min=2000.0, max=20000.0)
            )

            self.status.emit("Applying preprocessing…")

            pre: list[SpectrumObject] = []
            for i, sp in enumerate(raw, 1):
                sp2 = preproc(sp)

                m = float(np.max(sp2.intensity)) if len(sp2.intensity) else 1.0
                if m > 0:
                    sp2.intensity = sp2.intensity / m

                # conservar metadata
                sp2.meta = dict(getattr(sp, "meta", {}) or {})

                pre.append(sp2)
                self.progress.emit(40 + int((i / n) * 60))

            self.finished.emit(raw, pre)

        except Exception as e:
            self.error.emit(str(e))


# -----------------------------
# Plot
# -----------------------------
class DualPlot(QWidget):
    def __init__(self):
        super().__init__()
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)

        self.fig = Figure(figsize=(11, 3.5), dpi=100)
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.ax1 = self.fig.add_subplot(1, 2, 1)
        self.ax2 = self.fig.add_subplot(1, 2, 2)

        lay.addWidget(self.canvas)

    def update(self, raw: SpectrumObject, pre: SpectrumObject, title_id: str):
        self.ax1.clear()
        self.ax2.clear()

        self.ax1.plot(raw.mz, raw.intensity)
        self.ax1.set(title=f"RAW — {title_id}", xlabel="m/z", ylabel="Intensity")

        self.ax2.plot(pre.mz, pre.intensity)
        self.ax2.set(title=f"PRE — {title_id}", xlabel="m/z", ylabel="Intensity")

        self.fig.tight_layout()
        self.canvas.draw_idle()


# -----------------------------
# Screen
# -----------------------------
class PreprocessScreen(QWidget):
    preprocessing_ready = Signal()

    def __init__(self):
        super().__init__()

        self.input_paths: list[str] = []
        self.spectra_raw: Optional[list[SpectrumObject]] = None
        self.spectra_pre: Optional[list[SpectrumObject]] = None
        self.X_matrix_pre: Optional[np.ndarray] = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 18, 24, 18)
        layout.setSpacing(12)

        header = QFrame()
        h = QVBoxLayout(header)
        h.setContentsMargins(18, 12, 18, 12)
        h.setSpacing(8)

        self.label = QLabel(
            "Applying:\n"
            "Variance stabilization    Smoothing    "
            "Baseline subtraction    Threshold subtraction    "
            "Binning    Normalization (by maximum)"
        )

        row = QHBoxLayout()
        row.setSpacing(10)

        self.lbl_select = QLabel("Select spectrum:")
        self.combo = QComboBox()
        self.combo.setEnabled(False)
        self.combo.currentIndexChanged.connect(self._on_select_index)

        row.addWidget(self.lbl_select)
        row.addWidget(self.combo, 1)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setTextVisible(False)

        self.status = QLabel("")

        h.addWidget(self.label)
        h.addLayout(row)
        h.addWidget(self.progress)
        h.addWidget(self.status)

        self.plots = DualPlot()

        layout.addWidget(header)
        layout.addWidget(self.plots, 1)

    def set_input_paths(self, paths: list[str]):
        self.input_paths = paths
        self._start()

    def _start(self):
        if not self.input_paths:
            self.status.setText("No input.")
            return

        self.combo.blockSignals(True)
        self.combo.clear()
        self.combo.blockSignals(False)
        self.combo.setEnabled(False)
        self.progress.setValue(0)
        self.status.setText("Working…")

        self.thread = QThread(self)
        self.worker = PreprocessWorker(self.input_paths)
        self.worker.moveToThread(self.thread)

        self.thread.started.connect(self.worker.run)
        self.worker.progress.connect(self.progress.setValue)
        self.worker.status.connect(self.status.setText)
        self.worker.finished.connect(self._on_finished)
        self.worker.error.connect(self._on_error)

        self.worker.finished.connect(self.thread.quit)
        self.worker.error.connect(self.thread.quit)
        self.thread.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)

        self.thread.start()

    def _on_finished(self, raw, pre):
        self.spectra_raw = raw
        self.spectra_pre = pre

        self.status.setText(f"✅ Done. {len(pre)} spectra processed.")

        self.combo.blockSignals(True)
        self.combo.clear()

        for i, sp in enumerate(raw):
            pid = sp.meta.get("pseudo_id", f"idx_{i}")
            src_type = sp.meta.get("source_type", "?")
            self.combo.addItem(f"{pid} [{src_type}]", userData=i)

        self.combo.blockSignals(False)
        self.combo.setEnabled(len(raw) > 0)

        if raw:
            idx = random.randrange(len(raw))
            self.combo.setCurrentIndex(idx)
            self._on_select_index(idx)

        try:
            self.X_matrix_pre = np.vstack([s.intensity for s in pre]).astype(float)
        except Exception:
            self.X_matrix_pre = None

        self.preprocessing_ready.emit()

    def _on_select_index(self, idx: int):
        if idx < 0:
            return
        if not self.spectra_raw or not self.spectra_pre:
            return
        if idx >= len(self.spectra_raw) or idx >= len(self.spectra_pre):
            return

        title_id = self.spectra_raw[idx].meta.get("pseudo_id", f"idx_{idx}")
        self.plots.update(self.spectra_raw[idx], self.spectra_pre[idx], title_id)

    def _on_error(self, msg: str):
        self.status.setText(f"❌ Error: {msg}")
        self.combo.setEnabled(False)
