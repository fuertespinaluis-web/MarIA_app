from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QFrame, QStackedWidget, QPushButton
)

from .widgets import LogoLabel
from .upload_screen import UploadScreen
from .preprocess_screen import PreprocessScreen
from .ns_analysis_screen import NsAnalysisScreen
from .species_id_screen import SpeciesIdScreen


TOPBAR = "#088891"
BG = "#ECEDEF"
PANEL = "#FFFFFF"
TEXT = "#0F172A"


class StepButton(QPushButton):
    def __init__(self, text: str):
        super().__init__(text)
        self.setCursor(Qt.PointingHandCursor)
        self.setCheckable(True)
        self.setAutoExclusive(True)
        self.setFixedHeight(38)
        self.setMinimumWidth(210)
        self.setObjectName("StepBtn")


class MainUI(QWidget):
    def __init__(self):
        super().__init__()

        base_dir = Path(__file__).resolve().parents[1]
        img = base_dir / "assets"
        model_dir = img / "model"
        pca_dir = base_dir / "screens" / "PCA" / "PCA_DA_train_only" / "DA_x1p5"

        maria_logo = img / "maria_logo.png"
        ryc_logo = img / "ryc_logo.png"
        uzh_logo = img / "uzh_logo.png"

        self._X_cached = None
        self._pseudo_ids_cached = None

        root = QVBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(10)

        # ---------------- TOP BAR ----------------
        top = QFrame()
        top.setObjectName("TopBar")
        top.setFixedHeight(74)
        top_l = QHBoxLayout(top)
        top_l.setContentsMargins(18, 10, 18, 10)
        top_l.setSpacing(14)

        brand = QLabel("MarIA")
        brand.setObjectName("Brand")
        top_l.addWidget(brand, alignment=Qt.AlignVCenter | Qt.AlignLeft)
        top_l.addStretch(1)

        top_l.addWidget(LogoLabel(ryc_logo, 140, 46, "RyC"), alignment=Qt.AlignVCenter)
        top_l.addWidget(LogoLabel(uzh_logo, 180, 46, "UZH"), alignment=Qt.AlignVCenter)

        circle = QFrame()
        circle.setObjectName("MariaCircle")
        circle.setFixedSize(52, 52)
        circle_l = QVBoxLayout(circle)
        circle_l.setContentsMargins(0, 0, 0, 0)
        circle_l.setAlignment(Qt.AlignCenter)
        circle_l.addWidget(LogoLabel(maria_logo, 44, 44, ""))
        top_l.addWidget(circle, alignment=Qt.AlignVCenter | Qt.AlignRight)

        root.addWidget(top)

        # ---------------- STEPS BAR ----------------
        steps = QFrame()
        steps.setObjectName("StepsBar")
        steps.setFixedHeight(56)
        s_l = QHBoxLayout(steps)
        s_l.setContentsMargins(14, 8, 14, 8)
        s_l.setSpacing(12)

        self.btn_upload = StepButton("Upload data")
        self.btn_pre = StepButton("Preprocess")
        self.btn_ns = StepButton("Non supervised analysis")
        self.btn_species = StepButton("Species identification")

        self.btn_upload.setChecked(True)

        s_l.addStretch(1)
        s_l.addWidget(self.btn_upload)
        s_l.addWidget(self.btn_pre)
        s_l.addWidget(self.btn_ns)
        s_l.addWidget(self.btn_species)
        s_l.addStretch(1)

        root.addWidget(steps)

        # ---------------- STACK ----------------
        content = QFrame()
        content.setObjectName("ContentPanel")
        c_l = QVBoxLayout(content)
        c_l.setContentsMargins(0, 0, 0, 0)

        self.stack = QStackedWidget()

        self.upload = UploadScreen()
        self.preprocess = PreprocessScreen()
        self.ns = NsAnalysisScreen(pca_dir=pca_dir)
        self.species_screen = SpeciesIdScreen(model_dir=model_dir)

        self.stack.addWidget(self.upload)                # idx 0
        self.stack.addWidget(self.preprocess)            # idx 1
        self.loading_panel = self._build_loading_panel()
        self.stack.addWidget(self.loading_panel)         # idx 2
        self.stack.addWidget(self.ns)                    # idx 3
        self.stack.addWidget(self.species_screen)        # idx 4

        c_l.addWidget(self.stack)
        root.addWidget(content, 1)

        # ---------------- WIRING: NAVIGATION ----------------
        self.btn_upload.clicked.connect(lambda: self._go(self.upload, self.btn_upload))
        self.btn_pre.clicked.connect(lambda: self._go(self.preprocess, self.btn_pre))
        self.btn_ns.clicked.connect(self._go_ns_analysis)
        self.btn_species.clicked.connect(self._go_species_results)

        # Upload -> Preprocess
        self.upload.start_clicked.connect(self._start_pipeline)

        # Preprocess -> NS
        if hasattr(self.preprocess, "preprocessing_ready"):
            self.preprocess.preprocessing_ready.connect(self._send_to_ns)

        # ---------------- STYLE ----------------
        self.setStyleSheet(f"""
        QWidget {{
            background: {BG};
            color: {TEXT};
        }}
        #TopBar {{
            background: {TOPBAR};
            border-radius: 14px;
        }}
        #Brand {{
            background: transparent;
            color: white;
            font-size: 22px;
            font-weight: 900;
        }}
        #MariaCircle {{
            background: #0F172A;
            border-radius: 26px;
        }}
        #StepsBar {{
            background: rgba(215, 238, 240, 0.6);
            border-radius: 14px;
            border: 1px solid rgba(0,0,0,0.10);
        }}
        #StepBtn {{
            background: white;
            border: 2px solid #0F172A;
            border-radius: 14px;
            padding: 6px 14px;
        }}
        #StepBtn:checked {{
            background: #67E8E3;
        }}
        #ContentPanel {{
            background: {PANEL};
            border-radius: 14px;
            border: 1px solid rgba(0,0,0,0.10);
        }}
        #LoadingLabel {{
            background: transparent;
            color: #334155;
            font-size: 18px;
            font-weight: 800;
        }}
        """)

    # -------------------------
    # Helpers
    # -------------------------
    def _build_loading_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setAlignment(Qt.AlignCenter)

        label = QLabel("Loading analysis...")
        label.setObjectName("LoadingLabel")
        label.setAlignment(Qt.AlignCenter)
        layout.addWidget(label)
        return panel

    def _go(self, widget: QWidget, btn: StepButton):
        btn.setChecked(True)
        self.stack.setCurrentWidget(widget)

    def _go_ns_analysis(self):
        self.btn_ns.setChecked(True)
        self.stack.setCurrentWidget(self.loading_panel)
        QTimer.singleShot(60, self._finish_go_ns_analysis)

    def _finish_go_ns_analysis(self):
        if self.btn_ns.isChecked() and self.stack.currentWidget() is self.loading_panel:
            self.stack.setCurrentWidget(self.ns)

    def _get_X_from_preprocess(self):
        X = getattr(self.preprocess, "X_matrix_pre", None)
        if X is None:
            X = getattr(self.preprocess, "X_preprocessed", None)
        if X is None:
            X = getattr(self.preprocess, "X_matrix", None)
        return X

    def _get_pseudo_ids(self, X):
        raw = getattr(self.preprocess, "spectra_raw", None)
        if raw:
            pseudo_ids = []
            for i, sp in enumerate(raw):
                meta = getattr(sp, "meta", {}) or {}
                pseudo_id = str(meta.get("pseudo_id", "") or "").strip()
                pseudo_ids.append(pseudo_id or f"new_{i:04d}")
            return pseudo_ids
        return [f"new_{i:04d}" for i in range(X.shape[0])]

    def _get_current_or_cached_data(self):
        X = self._get_X_from_preprocess()
        pseudo_ids = None

        if X is not None:
            pseudo_ids = self._get_pseudo_ids(X)
            self._X_cached = X
            self._pseudo_ids_cached = pseudo_ids
            return X, pseudo_ids

        return self._X_cached, self._pseudo_ids_cached

    # -------------------------
    # Pipeline
    # -------------------------
    def _start_pipeline(self, files: list[str]):
        self._X_cached = None
        self._pseudo_ids_cached = None

        try:
            self.species_screen.set_new_data(None, [])
        except Exception:
            pass

        self.preprocess.set_input_paths(files)
        self._go(self.preprocess, self.btn_pre)

    def _send_to_ns(self):
        X = self._get_X_from_preprocess()
        if X is None:
            return

        pseudo_ids = self._get_pseudo_ids(X)

        self._X_cached = X
        self._pseudo_ids_cached = pseudo_ids

        self.ns.set_new_data(X_new=X, pseudo_ids=pseudo_ids)

    def _go_species_results(self):
        self.stack.setCurrentWidget(self.species_screen)
        self.btn_species.setChecked(True)

        X, pseudo_ids = self._get_current_or_cached_data()
        if X is None or pseudo_ids is None:
            return

        self.species_screen.set_new_data(X=X, pseudo_ids=pseudo_ids)
        self.species_screen.apply_species_consensus()
        self.species_screen.update()
        self.species_screen.repaint()
