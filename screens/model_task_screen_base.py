from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import json
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from PySide6.QtCore import Qt, Signal, QMargins, QUrl
from PySide6.QtGui import QColor, QPainter, QPen, QBrush
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame,
    QPushButton, QComboBox, QStackedWidget,
    QTableWidget, QTableWidgetItem, QHeaderView, QFileDialog
)
from PySide6.QtCharts import QChart, QChartView, QLineSeries, QScatterSeries, QValueAxis
from PySide6.QtWebEngineWidgets import QWebEngineView


TOPBAR = "#088891"
PANEL = "#FFFFFF"
TEXT = "#0F172A"
MUTED = "rgba(15,23,42,0.62)"
BORDER = "rgba(15,23,42,0.12)"
ACCENT_SOFT = "rgba(103, 232, 227, 0.35)"
ACCENT_SOFT_2 = "rgba(8, 136, 145, 0.10)"
ACCENT_STRONG = TOPBAR

CHART_LINE = QColor(249, 115, 22, 80)      # importancias RF en rojo translucido
CHART_GRID = QColor(15, 23, 42, 25)
CHART_SPECTRUM = QColor(8, 136, 145, 220)      # espectro seleccionado destacado

COLOR_GREEN_DARK = QColor("#166534")
COLOR_BLUE_DARK = QColor("#005BBB")
COLOR_YELLOW_DARK = QColor("#CA8A04")
COLOR_SLATE = QColor("#475569")
COLOR_RED_DARK = QColor("#991B1B")


@dataclass
class ResultRow:
    request_number: int
    pseudo_id: str
    rf_predicted_category: str = "—"
    rf_probability: float = np.nan
    consensus_text: str = "—"
    consensus_code: str = "none"


@dataclass
class PeakGroup:
    rank: int
    median_mz: float
    min_mz: float
    max_mz: float
    count: int
    direction: str
    contribution: float
    model_weight: float
    strength: float
    indices: np.ndarray


class ModelTaskScreen(QWidget):
    spectrum_changed = Signal(int)
    row_selected = Signal(int)
    probability_detail_requested = Signal()

    def __init__(
        self,
        title_text: str,
        positive_label: str,
        negative_label: str,
        rf_model_path: Path,
        rf_importances_path: Path,
        rf_params_path: Optional[Path] = None,
        show_consensus_column: bool = False,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)

        self.title_text = title_text
        self.positive_label = positive_label
        self.negative_label = negative_label
        self.show_consensus_column = show_consensus_column

        self.rf_model_path = Path(rf_model_path)
        self.rf_importances_path = Path(rf_importances_path)
        self.rf_params_path = Path(rf_params_path) if rf_params_path is not None else None

        self._rf_model = None

        self._rf_params: dict = {}

        self._rf_importances_from_model: Optional[np.ndarray] = None
        self._rf_importances_ext: Optional[np.ndarray] = None

        self._rows: list[ResultRow] = []
        self._last_X: Optional[np.ndarray] = None
        self._last_pseudo_ids: list[str] = []
        self._row_peak_cache: dict[int, list[PeakGroup]] = {}
        self._load_errors: list[str] = []

        self._build_ui()
        self._apply_style()

        try:
            self._load_rf_bundle()
            print(f"[OK] RF bundle loaded for: {self.title_text}")
        except Exception as e:
            self._load_errors.append(f"RF load failed: {e}")
            print(f"[ERROR] RF bundle failed for {self.title_text}: {e}")

        self._update_formula_or_params()
        self._update_biomarkers()
        if self._load_errors and not self._rows:
            self._render_empty()

    # ---------- public ----------
    def set_new_data(self, X: np.ndarray, pseudo_ids: Sequence[str]):
        if X is None:
            self._last_X = None
            self._last_pseudo_ids = []
            self._rows = []
            self._row_peak_cache = {}
            self._render_empty()
            self._fill_table()
            self._update_table_metrics()
            return

        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError(f"X must be 2D array, got shape={X.shape}")
        if len(pseudo_ids) != X.shape[0]:
            raise ValueError(f"pseudo_ids ({len(pseudo_ids)}) != n_samples ({X.shape[0]})")

        self._last_X = X.copy()
        self._last_pseudo_ids = [str(x) for x in pseudo_ids]
        self._row_peak_cache = {}

        self._ensure_models_loaded()
        self._validate_dimensions(X)

        rf_preds, rf_probs = self._predict_rf_rows(X)

        self._rows = [
            ResultRow(
                request_number=i + 1,
                pseudo_id=str(pid),
                rf_predicted_category=rf_preds[i],
                rf_probability=rf_probs[i],
            )
            for i, pid in enumerate(pseudo_ids)
        ]
        self._post_update_ui()

    # ---------- predictions ----------
    def _ensure_models_loaded(self):
        if self._rf_model is None:
            self._load_rf_bundle()

    def _validate_dimensions(self, X: np.ndarray):
        if self._rf_model is not None:
            n_rf = self._infer_model_n_features(self._rf_model)
            if n_rf is not None and X.shape[1] != n_rf:
                raise ValueError(f"X has {X.shape[1]} features but RF expects {n_rf}")
        if self._rf_importances_ext is not None and len(self._rf_importances_ext) != X.shape[1]:
            raise ValueError("RF importances feature count mismatch")

    def _predict_rf_rows(self, X: np.ndarray) -> tuple[list[str], list[float]]:
        if self._rf_model is None:
            return ["—"] * X.shape[0], [np.nan] * X.shape[0]

        try:
            p_pos = self._predict_positive_probability(self._rf_model, X, "RF")
        except Exception as e:
            msg = f"RF prediction failed: {e}"
            if msg not in self._load_errors:
                self._load_errors.append(msg)
            return ["—"] * X.shape[0], [np.nan] * X.shape[0]

        preds, probs = [], []
        for p in p_pos:
            p = float(p)
            preds.append(self.positive_label if p >= 0.5 else self.negative_label)
            probs.append(p)
        return preds, probs

    def _display_confidence_value(self, predicted_label: str, rf_probability: float) -> float:
        if np.isnan(rf_probability):
            return np.nan
        return float(rf_probability)

    def _display_confidence_text(self, predicted_label: str, rf_probability: float) -> str:
        shown_conf = self._display_confidence_value(predicted_label, rf_probability)
        if np.isnan(shown_conf):
            return "Confidence: —"
        return f"Confidence: {shown_conf * 100.0:.2f}%"

    def _post_update_ui(self):
        self._populate_selector()
        self._render_index(0 if self._rows else -1)
        self._update_biomarkers()
        self._update_formula_or_params()
        self._fill_table()
        self._update_table_metrics()

    # ---------- ui ----------
    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(12)

        self.stack = QStackedWidget()
        root.addWidget(self.stack, 1)

        self.view_main = QWidget()
        v0 = QVBoxLayout(self.view_main)
        v0.setContentsMargins(0, 0, 0, 0)
        v0.setSpacing(12)

        title = QLabel(self.title_text)
        title.setObjectName("Title")
        title.setAlignment(Qt.AlignCenter)
        v0.addWidget(title)

        selector_card = self._card()
        sc = QHBoxLayout(selector_card)
        sc.setContentsMargins(14, 10, 14, 10)
        sc.setSpacing(10)

        lbl = QLabel("Selected spectrum")
        lbl.setObjectName("LabelMuted")

        self.combo = QComboBox()
        self.combo.setObjectName("Combo")
        self.combo.setMinimumWidth(560)
        self.combo.setFixedHeight(38)

        sc.addStretch(1)
        sc.addWidget(lbl, alignment=Qt.AlignVCenter)
        sc.addWidget(self.combo, alignment=Qt.AlignVCenter)
        sc.addStretch(1)
        v0.addWidget(selector_card)

        result_card = self._card()
        rc = QVBoxLayout(result_card)
        rc.setContentsMargins(16, 12, 16, 12)
        rc.setSpacing(8)

        self.caption = QLabel("Your spectra is classified as")
        self.caption.setObjectName("LabelMuted")
        self.caption.setAlignment(Qt.AlignCenter)
        rc.addWidget(self.caption)

        row = QHBoxLayout()
        row.setSpacing(12)

        self.lbl_pred = QLabel("—")
        self.lbl_pred.setObjectName("PredBox")
        self.lbl_pred.setAlignment(Qt.AlignCenter)

        self.lbl_conf = QLabel("Confidence: ?")
        self.lbl_conf.setObjectName("ConfPill")
        self.lbl_conf.setAlignment(Qt.AlignCenter)
        self.lbl_conf.setMinimumWidth(260)

        self.btn_probability_detail = QPushButton("Check confidence")
        self.btn_probability_detail.setObjectName("LinkBtn")
        self.btn_probability_detail.setCursor(Qt.PointingHandCursor)
        self.btn_probability_detail.setFixedHeight(24)
        self.btn_probability_detail.setVisible(False)

        row.addWidget(self.lbl_pred, 1)
        row.addWidget(self.lbl_conf, 0)
        rc.addLayout(row)
        rc.addWidget(self.btn_probability_detail, alignment=Qt.AlignCenter)

        self.lbl_formula = QLabel("")
        self.lbl_formula.setObjectName("Formula")
        self.lbl_formula.setAlignment(Qt.AlignCenter)
        rc.addWidget(self.lbl_formula)

        self.lbl_consensus = QLabel("")
        self.lbl_consensus.setObjectName("ConsensusLabel")
        self.lbl_consensus.setAlignment(Qt.AlignCenter)
        rc.addWidget(self.lbl_consensus)

        v0.addWidget(result_card)

        biom_row = QHBoxLayout()
        biom_row.setSpacing(14)

        self.card_pos = self._card()
        self.card_neg = self._card()
        self.card_pos.setMinimumHeight(400)
        self.card_neg.setMinimumHeight(290)

        self._explain_mode = "decision"
        self.lbl_pos_title = QLabel("")
        self.lbl_pos_title.setObjectName("CardTitle")
        self.lbl_neg_title = QLabel("")
        self.lbl_neg_title.setObjectName("CardTitle")
        self.lbl_pos_help = QLabel("")
        self.lbl_pos_help.setObjectName("LabelMuted")
        self.lbl_pos_help.setWordWrap(True)
        self.lbl_neg_help = QLabel("")
        self.lbl_neg_help.setObjectName("LabelMuted")
        self.lbl_neg_help.setWordWrap(True)

        self.pos_chart = self._make_plotly_chart()
        self.neg_chart = self._make_spike_chart()
        self.pos_chart.setMinimumHeight(340)
        self.neg_chart.setMinimumHeight(180)
        self.neg_table = self._make_feature_table()
        self.btn_mode_decision = self._make_feature_toggle_button("Decision")
        self.btn_mode_scatter = self._make_feature_toggle_button("Overlay")

        lp = QVBoxLayout(self.card_pos)
        lp.setContentsMargins(16, 14, 16, 14)
        lp.setSpacing(10)
        lp.addLayout(
            self._explain_card_header(
                self.lbl_pos_title,
                self.btn_mode_decision,
                self.btn_mode_scatter,
            )
        )
        lp.addWidget(self.lbl_pos_help)
        lp.addWidget(self.pos_chart, 1)

        ln = QVBoxLayout(self.card_neg)
        ln.setContentsMargins(16, 14, 16, 14)
        ln.setSpacing(10)
        ln.addWidget(self.lbl_neg_title)
        ln.addWidget(self.lbl_neg_help)
        ln.addWidget(self.neg_table, 1)

        biom_row.addWidget(self.card_pos, 3)
        biom_row.addWidget(self.card_neg, 2)
        v0.addLayout(biom_row, 1)

        btn_row = QHBoxLayout()
        btn_row.addStretch(1)

        self.btn_complete = QPushButton("Access to complete list")
        self.btn_complete.setObjectName("PrimaryBtn")
        self.btn_complete.setFixedHeight(42)
        self.btn_complete.setMinimumWidth(260)

        btn_row.addWidget(self.btn_complete, alignment=Qt.AlignCenter)
        btn_row.addStretch(1)
        v0.addLayout(btn_row)

        self.stack.addWidget(self.view_main)

        self.view_table = QWidget()
        v1 = QVBoxLayout(self.view_table)
        v1.setContentsMargins(0, 0, 0, 0)
        v1.setSpacing(12)

        top_row = QHBoxLayout()

        self.btn_back = QPushButton("Back to results")
        self.btn_back.setObjectName("GhostBtn")
        self.btn_back.setFixedHeight(38)

        self.btn_export_csv = QPushButton("Export CSV")
        self.btn_export_csv.setObjectName("GhostBtn")
        self.btn_export_csv.setFixedHeight(38)

        t = QLabel("Complete list")
        t.setObjectName("TitleSmall")

        top_row.addWidget(self.btn_back, alignment=Qt.AlignLeft)
        top_row.addSpacing(8)
        top_row.addWidget(self.btn_export_csv, alignment=Qt.AlignLeft)
        top_row.addStretch(1)
        top_row.addWidget(t, alignment=Qt.AlignRight)
        v1.addLayout(top_row)

        metrics_card = self._card()
        mc = QHBoxLayout(metrics_card)
        mc.setContentsMargins(12, 10, 12, 10)

        self.lbl_metrics = QLabel("RF positive-rate: —   |   Positive: —   |   Other/negative: —")
        self.lbl_metrics.setObjectName("LabelMuted")
        self.lbl_metrics.setAlignment(Qt.AlignCenter)
        mc.addWidget(self.lbl_metrics, 1)
        v1.addWidget(metrics_card)

        table_card = self._card()
        tc = QVBoxLayout(table_card)
        tc.setContentsMargins(12, 12, 12, 12)

        n_cols = 6 if self.show_consensus_column else 5
        self.table = QTableWidget(0, n_cols)
        self.table.setObjectName("Table")
        headers = [
            "Request #",
            "Pseudo ID",
            "RF prediction",
            "RF confidence",
            "Top 10 peaks (Da)",
        ]
        if self.show_consensus_column:
            headers.append("Consensus")
        self.table.setHorizontalHeaderLabels(headers)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setSelectionMode(QTableWidget.SingleSelection)
        self.table.verticalHeader().setVisible(False)
        self.table.setAlternatingRowColors(True)

        hh = self.table.horizontalHeader()
        hh.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        hh.setSectionResizeMode(1, QHeaderView.Stretch)
        hh.setSectionResizeMode(2, QHeaderView.Stretch)
        hh.setSectionResizeMode(3, QHeaderView.ResizeToContents)
        hh.setSectionResizeMode(4, QHeaderView.Stretch)
        if self.show_consensus_column:
            hh.setSectionResizeMode(5, QHeaderView.Stretch)

        tc.addWidget(self.table)
        v1.addWidget(table_card, 1)

        self.stack.addWidget(self.view_table)

        self.combo.currentIndexChanged.connect(self._render_index)
        self.btn_complete.clicked.connect(lambda: self.stack.setCurrentIndex(1))
        self.btn_back.clicked.connect(lambda: self.stack.setCurrentIndex(0))
        self.btn_export_csv.clicked.connect(self._export_table_csv)
        self.btn_probability_detail.clicked.connect(self.probability_detail_requested.emit)
        self.table.itemSelectionChanged.connect(self._on_table_selection)
        self.btn_mode_decision.clicked.connect(lambda: self._set_explain_mode("decision"))
        self.btn_mode_scatter.clicked.connect(lambda: self._set_explain_mode("scatter"))

        self.stack.setCurrentIndex(0)
        self._set_explain_mode("decision")
        self._render_empty()
        self._fill_table()
        self._update_table_metrics()

    def _card(self) -> QFrame:
        c = QFrame()
        c.setObjectName("Card")
        return c

    def _apply_style(self):
        self.setStyleSheet(f"""
        QWidget {{
            background: transparent;
            color: {TEXT};
        }}
        QLabel#Title {{
            font-size: 18px;
            font-weight: 900;
            color: {TEXT};
            margin-top: 2px;
        }}
        QLabel#TitleSmall {{
            font-size: 16px;
            font-weight: 900;
        }}
        QLabel#LabelMuted {{
            color: {MUTED};
            font-weight: 800;
            font-size: 12px;
        }}
        QFrame#Card {{
            background: {PANEL};
            border: 1px solid {BORDER};
            border-radius: 16px;
        }}
        QComboBox#Combo {{
            background: rgba(255,255,255,0.95);
            border: 1px solid rgba(15,23,42,0.18);
            border-radius: 12px;
            padding: 6px 10px;
            font-weight: 800;
        }}
        QComboBox#Combo::drop-down {{
            border: 0px;
            width: 28px;
        }}
        QLabel#PredBox {{
            background: {ACCENT_STRONG};
            color: white;
            border-radius: 14px;
            padding: 8px 12px;
            font-size: 18px;
            font-weight: 900;
        }}
        QLabel#ConfPill {{
            background: {ACCENT_SOFT};
            color: {TEXT};
            border: 1px solid rgba(15,23,42,0.10);
            border-radius: 14px;
            padding: 10px 10px;
            font-size: 14px;
            font-weight: 900;
        }}
        QLabel#Formula {{
            color: {MUTED};
            font-size: 11px;
            margin-top: 0px;
        }}
        QLabel#ConsensusLabel {{
            font-size: 12px;
            font-weight: 900;
            margin-top: 2px;
        }}
        QLabel#CardTitle {{
            font-size: 13px;
            font-weight: 900;
            color: {TEXT};
        }}
        QPushButton#PrimaryBtn {{
            background: {ACCENT_STRONG};
            color: white;
            border: 0px;
            border-radius: 14px;
            padding: 10px 14px;
            font-weight: 900;
        }}
        QPushButton#PrimaryBtn:hover {{
            background: #0A7A82;
        }}
        QPushButton#GhostBtn {{
            background: rgba(255,255,255,0.65);
            border: 1px solid rgba(15,23,42,0.14);
            border-radius: 12px;
            padding: 8px 12px;
            font-weight: 900;
        }}
        QPushButton#GhostBtn:hover {{
            background: rgba(103,232,227,0.25);
        }}
        QPushButton#FeatureToggle {{
            background: rgba(255,255,255,0.72);
            border: 1px solid rgba(15,23,42,0.10);
            border-radius: 10px;
            padding: 6px 10px;
            font-size: 11px;
            font-weight: 900;
            color: {TEXT};
        }}
        QPushButton#FeatureToggle[active="true"] {{
            background: rgba(8,136,145,0.14);
            border: 1px solid rgba(8,136,145,0.34);
            color: {ACCENT_STRONG};
        }}
        QPushButton#LinkBtn {{
            background: transparent;
            border: 0px;
            color: {ACCENT_STRONG};
            font-size: 12px;
            font-weight: 900;
            text-decoration: underline;
            padding: 0px;
        }}
        QPushButton#LinkBtn:hover {{
            color: #0A7A82;
        }}
        QTableWidget#Table {{
            background: white;
            border: 1px solid rgba(15,23,42,0.10);
            border-radius: 12px;
            gridline-color: rgba(15,23,42,0.08);
            selection-background-color: rgba(103,232,227,0.25);
        }}
        QHeaderView::section {{
            background: rgba(8,136,145,0.10);
            border: 0px;
            padding: 8px;
            font-weight: 900;
        }}
        QChartView#SpikeChart {{
            background: {ACCENT_SOFT_2};
            border: 1px solid rgba(15,23,42,0.08);
            border-radius: 12px;
        }}
        QWebEngineView#InteractiveChart {{
            background: {ACCENT_SOFT_2};
            border: 1px solid rgba(15,23,42,0.08);
            border-radius: 12px;
        }}
        QTableWidget#FeatureTable {{
            background: rgba(251,252,254,0.98);
            border: 1px solid rgba(15,23,42,0.08);
            border-radius: 12px;
            gridline-color: rgba(15,23,42,0.06);
            selection-background-color: rgba(103,232,227,0.18);
        }}
        """)

    # ---------- loading ----------
    def _load_rf_bundle(self):
        self._load_rf_model()
        self._load_rf_importances()
        self._load_rf_params()

    def _load_rf_model(self):
        print("[LOAD] RF model:", self.rf_model_path)
        if not self.rf_model_path.exists():
            raise FileNotFoundError(f"Missing RF model file: {self.rf_model_path}")

        loaded_model = joblib.load(self.rf_model_path)
        clf = self._extract_classifier(loaded_model, expected="rf")

        predict_source = loaded_model if hasattr(loaded_model, "predict_proba") else clf
        if not hasattr(predict_source, "predict_proba"):
            raise ValueError("Loaded RF object does not implement predict_proba().")

        self._rf_model = loaded_model
        self._rf_importances_from_model = np.asarray(clf.feature_importances_, dtype=float).ravel()

    def _load_rf_importances(self):
        print("[LOAD] RF importances:", self.rf_importances_path)
        if not self.rf_importances_path.exists():
            raise FileNotFoundError(f"Missing RF importances CSV: {self.rf_importances_path}")

        df = pd.read_csv(self.rf_importances_path)
        if {"feature_index", "importance"}.issubset(df.columns):
            value_col = "importance"
        elif {"feature_index", "feature_importance"}.issubset(df.columns):
            value_col = "feature_importance"
        elif {"feature_index", "coef"}.issubset(df.columns):
            value_col = "coef"
        else:
            raise ValueError("RF importances CSV must contain feature_index and importance-like column")

        max_idx = int(df["feature_index"].max())
        arr = np.zeros(max_idx + 1, dtype=float)
        for _, r in df.iterrows():
            arr[int(r["feature_index"])] = float(r[value_col])
        self._rf_importances_ext = arr

    def _load_rf_params(self):
        self._rf_params = {}
        if self.rf_params_path is not None and self.rf_params_path.exists():
            try:
                with open(self.rf_params_path, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
                bp = cfg.get("best_params", cfg) if isinstance(cfg, dict) else {}
                if isinstance(bp, dict):
                    self._rf_params = dict(bp)
            except Exception:
                self._rf_params = {}
        if not self._rf_params:
            try:
                clf = self._extract_classifier(self._rf_model, expected="rf")
                self._rf_params = clf.get_params()
            except Exception:
                self._rf_params = {}

    # ---------- helpers ----------
    def _extract_classifier(self, model, expected: str = "rf"):
        def matches(obj) -> bool:
            if expected == "rf":
                return hasattr(obj, "feature_importances_")
            return False

        visited: set[int] = set()

        def walk(obj):
            if obj is None:
                return None

            obj_id = id(obj)
            if obj_id in visited:
                return None
            visited.add(obj_id)

            if matches(obj):
                return obj

            if hasattr(obj, "named_steps"):
                for step in reversed(list(obj.named_steps.values())):
                    found = walk(step)
                    if found is not None:
                        return found

            nested_attrs = (
                "best_estimator_",
                "estimator",
                "estimator_",
                "base_estimator",
                "base_estimator_",
                "final_estimator_",
                "classifier",
                "classifier_",
                "model",
                "model_",
            )
            for attr in nested_attrs:
                if hasattr(obj, attr):
                    found = walk(getattr(obj, attr))
                    if found is not None:
                        return found

            if hasattr(obj, "calibrated_classifiers_"):
                for calibrated in getattr(obj, "calibrated_classifiers_", []):
                    found = walk(calibrated)
                    if found is not None:
                        return found

            if isinstance(obj, dict):
                for value in obj.values():
                    found = walk(value)
                    if found is not None:
                        return found

            if isinstance(obj, (list, tuple)):
                for value in obj:
                    found = walk(value)
                    if found is not None:
                        return found

            return None

        found = walk(model)
        if found is None:
            raise ValueError(f"Could not extract {expected.upper()} classifier")
        return found

    def _infer_model_n_features(self, model) -> Optional[int]:
        try:
            if hasattr(model, "n_features_in_"):
                return int(model.n_features_in_)

            clf = self._extract_classifier(model, expected="rf")
            if hasattr(clf, "n_features_in_"):
                return int(clf.n_features_in_)
        except Exception:
            return None
        return None

    def _predict_positive_probability(self, model, X: np.ndarray, model_name: str) -> np.ndarray:
        predict_source = model
        if not hasattr(predict_source, "predict_proba"):
            predict_source = self._extract_classifier(model, expected="rf")

        proba = predict_source.predict_proba(X)

        classes = list(getattr(predict_source, "classes_", []))
        if not classes:
            try:
                clf = self._extract_classifier(model, expected="rf")
                classes = list(clf.classes_)
            except Exception:
                classes = None

        if classes is None or len(classes) == 0:
            if proba.shape[1] == 2:
                return np.asarray(proba[:, 1], dtype=float)
            raise ValueError(f"Could not determine classes for {model_name}")

        pos_idx = self._find_positive_class_index(classes)

        if pos_idx is None:
            pos_idx = 1 if proba.shape[1] == 2 else None

        if pos_idx is None:
            raise ValueError(f"Could not identify positive class {self.positive_label}")

        return np.asarray(proba[:, pos_idx], dtype=float)

    @staticmethod
    def _mz_axis_from_feature_count(n_features: int, min_mz: float = 2000.0, bin_size: float = 1.0) -> np.ndarray:
        return min_mz + (np.arange(n_features, dtype=float) + 0.5) * bin_size

    def _get_rf_importances_for_plot(self) -> Optional[np.ndarray]:
        return self._rf_importances_ext if self._rf_importances_ext is not None else self._rf_importances_from_model

    def _get_selected_spectrum(self) -> Optional[np.ndarray]:
        idx = self.combo.currentIndex()
        if self._last_X is None or idx < 0 or idx >= self._last_X.shape[0]:
            return None
        return np.asarray(self._last_X[idx], dtype=float)

    @staticmethod
    def _normalize_class_label(value: object) -> str:
        s = str(value or "").strip().lower()
        s = s.replace("_", " ").replace("-", " ").replace("/", " ")
        s = " ".join(s.split())
        if s.startswith("s "):
            rest = s[2:].strip()
            if "pneumoniae" in rest:
                return "streptococcus pneumoniae"
            if "mitis" in rest or "oralis" in rest:
                return "streptococcus mitis oralis"
        if "pneumoniae" in s:
            return "streptococcus pneumoniae"
        if "mitis" in s or "oralis" in s:
            return "streptococcus mitis oralis"
        return s

    def _find_positive_class_index(self, classes: list[object]) -> Optional[int]:
        target = self._normalize_class_label(self.positive_label)
        for i, c in enumerate(classes):
            if self._normalize_class_label(c) == target:
                return i
        return None

    def _compute_rf_local_feature_contributions(
        self,
        sample: Optional[np.ndarray],
    ) -> Optional[dict[str, np.ndarray | float]]:
        if self._rf_model is None or sample is None:
            return None

        try:
            clf = self._extract_classifier(self._rf_model, expected="rf")
        except Exception:
            return None

        estimators = getattr(clf, "estimators_", None)
        classes = list(getattr(clf, "classes_", []))
        if not estimators or len(classes) != 2:
            return None

        sample = np.asarray(sample, dtype=float).ravel()
        if sample.ndim != 1:
            return None

        pos_idx = self._find_positive_class_index(classes)
        if pos_idx is None:
            pos_idx = 1 if len(classes) == 2 else None
        if pos_idx is None:
            return None
        neg_idx = 1 - pos_idx

        n_features = sample.shape[0]
        bias = np.zeros(2, dtype=float)
        contrib = np.zeros((2, n_features), dtype=float)
        x_row = sample.reshape(1, -1)

        for estimator in estimators:
            tree = getattr(estimator, "tree_", None)
            if tree is None:
                continue

            raw_values = np.asarray(tree.value, dtype=float)
            if raw_values.ndim == 3:
                raw_values = raw_values[:, 0, :]
            if raw_values.shape[1] != 2:
                continue

            sums = raw_values.sum(axis=1, keepdims=True)
            sums[sums == 0.0] = 1.0
            node_probs = raw_values / sums

            node_indicator = estimator.decision_path(x_row)
            node_path = node_indicator.indices[node_indicator.indptr[0]:node_indicator.indptr[1]]
            if len(node_path) == 0:
                continue

            bias += node_probs[0]
            for parent_node, child_node in zip(node_path[:-1], node_path[1:]):
                feature_idx = int(tree.feature[parent_node])
                if feature_idx < 0 or feature_idx >= n_features:
                    continue
                delta = node_probs[child_node] - node_probs[parent_node]
                contrib[:, feature_idx] += delta

        n_trees = float(len(estimators))
        if n_trees <= 0:
            return None

        bias /= n_trees
        contrib /= n_trees

        return {
            "bias_pos": float(bias[pos_idx]),
            "bias_neg": float(bias[neg_idx]),
            "contrib_pos": np.asarray(contrib[pos_idx], dtype=float),
            "contrib_neg": np.asarray(contrib[neg_idx], dtype=float),
        }

    def _top_peak_groups_for_sample(self, sample: Optional[np.ndarray], limit: int = 10) -> list[PeakGroup]:
        imp = self._get_rf_importances_for_plot()
        if imp is None or sample is None:
            return []

        sample = np.asarray(sample, dtype=float).ravel()
        if sample.size != len(imp):
            return []

        local = self._compute_rf_local_feature_contributions(sample)
        if local is None:
            return []

        mz = self._mz_axis_from_feature_count(len(imp))
        signed_scores = np.asarray(local["contrib_pos"], dtype=float) - np.asarray(local["contrib_neg"], dtype=float)
        return self._cluster_top_discriminatory_peaks(mz, signed_scores, imp, top_n=50, window_da=10.0)[:limit]

    def _top_peak_groups_for_row_index(self, row_index: int, limit: int = 10) -> list[PeakGroup]:
        if self._last_X is None or row_index < 0 or row_index >= self._last_X.shape[0]:
            return []
        if row_index not in self._row_peak_cache:
            self._row_peak_cache[row_index] = self._top_peak_groups_for_sample(self._last_X[row_index], limit=10)
        return self._row_peak_cache[row_index][:limit]

    @staticmethod
    def _peak_group_range_text(group: PeakGroup) -> str:
        if group.count > 1:
            return f"{group.min_mz:.0f}-{group.max_mz:.0f}"
        return f"{group.median_mz:.0f}"

    def _peak_groups_summary_text(self, groups: list[PeakGroup]) -> str:
        if not groups:
            return "-"
        return "; ".join(f"{group.rank}. {group.median_mz:.1f}" for group in groups)

    def _peak_groups_tooltip_text(self, groups: list[PeakGroup]) -> str:
        if not groups:
            return "Top local RF peaks are not available for this row."
        return "\n".join(
            (
                f"{group.rank}. {group.median_mz:.1f} Da "
                f"(range {self._peak_group_range_text(group)} Da, "
                f"{group.direction}, contribution {group.contribution:+.4f})"
            )
            for group in groups
        )

    # ---------- export ----------
    def _export_table_csv(self):
        if not self._rows:
            print("[INFO] No rows to export.")
            return

        data = []
        for row_index, r in enumerate(self._rows):
            peak_groups = self._top_peak_groups_for_row_index(row_index, limit=10)
            row_dict = {
                "request_number": r.request_number,
                "pseudo_id": r.pseudo_id,
                "rf_predicted_category": r.rf_predicted_category,
                "rf_probability": None if np.isnan(r.rf_probability) else float(r.rf_probability),
                "top_10_peaks_da": "; ".join(f"{group.median_mz:.1f}" for group in peak_groups),
            }
            for i in range(10):
                group = peak_groups[i] if i < len(peak_groups) else None
                prefix = f"top_peak_{i + 1:02d}"
                row_dict[f"{prefix}_mz_da"] = None if group is None else float(group.median_mz)
                row_dict[f"{prefix}_range_da"] = "" if group is None else self._peak_group_range_text(group)
                row_dict[f"{prefix}_direction"] = "" if group is None else group.direction
                row_dict[f"{prefix}_contribution"] = None if group is None else float(group.contribution)
                row_dict[f"{prefix}_model_weight"] = None if group is None else float(group.model_weight)
            if self.show_consensus_column:
                row_dict["consensus_text"] = r.consensus_text
                row_dict["consensus_code"] = r.consensus_code
            data.append(row_dict)

        df = pd.DataFrame(data)

        safe_name = self.title_text.lower().replace(" ", "_").replace("/", "_")
        default_name = f"{safe_name}_results.csv"

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save results as CSV",
            default_name,
            "CSV files (*.csv)"
        )

        if not file_path:
            return

        try:
            df.to_csv(file_path, index=False, encoding="utf-8-sig")
            print(f"[OK] CSV exported to: {file_path}")
        except Exception as e:
            print(f"[ERROR] Could not export CSV: {e}")

    # ---------- charts ----------
    def _make_spike_chart(self) -> QChartView:
        chart = QChart()
        chart.legend().hide()
        chart.setBackgroundVisible(False)
        chart.setPlotAreaBackgroundVisible(False)
        chart.setMargins(QMargins(8, 8, 8, 8))

        view = QChartView(chart)
        view.setObjectName("SpikeChart")
        view.setRenderHint(QPainter.Antialiasing)
        return view

    def _make_plotly_chart(self) -> QWebEngineView:
        view = QWebEngineView()
        view.setObjectName("InteractiveChart")
        return view

    @staticmethod
    def _local_plotly_script() -> str:
        try:
            from plotly.offline.offline import get_plotlyjs

            plotly_js = get_plotlyjs()
            if plotly_js:
                return f'<script type="text/javascript">\n{plotly_js}\n</script>'
        except Exception:
            pass

        try:
            import plotly

            plotly_js_path = Path(plotly.__file__).resolve().parent / "package_data" / "plotly.min.js"
            if plotly_js_path.exists():
                plotly_js = plotly_js_path.read_text(encoding="utf-8")
                return f'<script type="text/javascript">\n{plotly_js}\n</script>'
        except Exception:
            pass

        return ""

    @staticmethod
    def _plotly_html_fullscreen(fig: go.Figure) -> str:
        html = fig.to_html(
            include_plotlyjs=False,
            full_html=False,
            config={
                "responsive": True,
                "displaylogo": False,
                "scrollZoom": True,
                "modeBarButtonsToRemove": ["select2d", "lasso2d"],
            },
        )
        local_plotly = ModelTaskScreen._local_plotly_script()
        return f"""
        <!doctype html>
        <html>
        <head>
          <meta charset="utf-8"/>
          <meta name="viewport" content="width=device-width, initial-scale=1"/>
          {local_plotly}
          <style>
            html, body {{
              width: 100%;
              height: 100%;
              margin: 0;
              padding: 0;
              overflow: hidden;
              background: rgba(8,136,145,0.06);
              font-family: Arial, sans-serif;
            }}
            #wrap {{
              width: 100%;
              height: 100%;
            }}
            #wrap .plotly-graph-div {{
              width: 100% !important;
              height: 100vh !important;
              min-height: 100vh !important;
            }}
          </style>
        </head>
        <body>
          <div id="wrap">{html}</div>
          <script>
            window.addEventListener('load', function() {{
              if (!window.Plotly) {{
                document.body.innerHTML =
                  '<div style="font-family:Arial,sans-serif;padding:24px;color:#0f172a">' +
                  '<h3 style="margin-top:0">Interactive RF plot could not be loaded</h3>' +
                  '<p>Plotly is not available in this Python environment.</p>' +
                  '</div>';
                return;
              }}
              window.dispatchEvent(new Event('resize'));
              setTimeout(function() {{
                window.dispatchEvent(new Event('resize'));
              }}, 150);
            }});
          </script>
        </body>
        </html>
        """

    def _set_plotly_figure(self, view: QWebEngineView, fig: go.Figure):
        output_dir = Path(__file__).resolve().parent / "fig_rf_local"
        output_dir.mkdir(parents=True, exist_ok=True)

        safe_title = "".join(ch.lower() if ch.isalnum() else "_" for ch in self.title_text).strip("_")
        html_path = output_dir / f"{safe_title or 'rf'}_{id(self)}_local_explanation.html"
        html_path.write_text(self._plotly_html_fullscreen(fig), encoding="utf-8")
        view.setUrl(QUrl.fromLocalFile(str(html_path.resolve())))

    def _make_feature_table(self) -> QTableWidget:
        table = QTableWidget(0, 7)
        table.setObjectName("FeatureTable")
        table.setHorizontalHeaderLabels([
            "Rank", "Median peak (Da)", "Range (Da)", "n", "Direction", "RF contribution", "Model weight"
        ])
        table.verticalHeader().setVisible(False)
        table.setSelectionBehavior(QTableWidget.SelectRows)
        table.setSelectionMode(QTableWidget.NoSelection)
        table.setAlternatingRowColors(True)
        table.setEditTriggers(QTableWidget.NoEditTriggers)
        table.setToolTip(
            "Top 50 discriminative peaks grouped into 10 Da windows.\n"
            "Median peak = median mass of the grouped peaks.\n"
            "RF contribution and model weight are shown as group medians."
        )
        hh = table.horizontalHeader()
        hh.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        hh.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        hh.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        hh.setSectionResizeMode(3, QHeaderView.ResizeToContents)
        hh.setSectionResizeMode(4, QHeaderView.ResizeToContents)
        hh.setSectionResizeMode(5, QHeaderView.ResizeToContents)
        hh.setSectionResizeMode(6, QHeaderView.ResizeToContents)
        return table

    def _make_feature_toggle_button(self, text: str) -> QPushButton:
        btn = QPushButton(text)
        btn.setObjectName("FeatureToggle")
        btn.setFixedHeight(28)
        btn.setCursor(Qt.PointingHandCursor)
        return btn

    def _explain_card_header(self, title: QLabel, *buttons: QPushButton) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(8)
        row.addWidget(title, 1)
        for btn in buttons:
            row.addWidget(btn, 0)
        return row

    def _set_explain_mode(self, mode: str):
        self._explain_mode = mode
        mapping = {
            "decision": self.btn_mode_decision,
            "scatter": self.btn_mode_scatter,
        }
        for key, btn in mapping.items():
            btn.setProperty("active", "true" if key == mode else "false")
            btn.style().unpolish(btn)
            btn.style().polish(btn)
            btn.update()
        if mode == "decision":
            self.lbl_pos_help.setText(
                "Top local RF contributions for the selected spectrum, grouped from the top 50 peaks in 10 Da windows. "
                "Blue pushes toward <i>S. pneumoniae</i>; green pushes toward <i>S. mitis/oralis</i>."
            )
        else:
            self.lbl_pos_help.setText(
                "Sample spectrum with a histogram of the top 50 discriminative peaks grouped into 10 Da windows. "
                "Blue bars push toward <i>S. pneumoniae</i>; green bars push toward <i>S. mitis/oralis</i>."
            )
        self.lbl_neg_help.setText(
            "Top grouped local RF contributions. Peaks within a 10 Da range are represented by their median mass."
        )
        if self._last_X is not None:
            self._update_biomarkers()

    def _cluster_top_discriminatory_peaks(
        self,
        mz: np.ndarray,
        signed_scores: np.ndarray,
        importances: Optional[np.ndarray] = None,
        top_n: int = 50,
        window_da: float = 10.0,
    ) -> list[PeakGroup]:
        mz = np.asarray(mz, dtype=float).ravel()
        scores = np.asarray(signed_scores, dtype=float).ravel()
        if mz.size == 0 or scores.size != mz.size:
            return []

        weights = np.zeros_like(scores)
        if importances is not None:
            imp = np.asarray(importances, dtype=float).ravel()
            if imp.size == scores.size:
                weights = np.nan_to_num(imp, nan=0.0)

        valid = np.isfinite(mz) & np.isfinite(scores) & (np.abs(scores) > 0.0)
        candidate_idx = np.flatnonzero(valid)
        if candidate_idx.size == 0:
            return []

        top_idx = candidate_idx[np.argsort(np.abs(scores[candidate_idx]))[::-1][:top_n]]
        top_idx = top_idx[np.argsort(mz[top_idx])]

        clusters: list[list[int]] = []
        current: list[int] = []
        start_mz: Optional[float] = None
        for idx in top_idx:
            this_mz = float(mz[idx])
            if not current:
                current = [int(idx)]
                start_mz = this_mz
                continue
            if start_mz is not None and this_mz - start_mz <= window_da:
                current.append(int(idx))
            else:
                clusters.append(current)
                current = [int(idx)]
                start_mz = this_mz
        if current:
            clusters.append(current)

        groups: list[PeakGroup] = []
        for cluster in clusters:
            idx_arr = np.asarray(cluster, dtype=int)
            cluster_mz = mz[idx_arr]
            cluster_scores = scores[idx_arr]
            signed_sum = float(np.sum(cluster_scores))
            contribution = float(np.median(cluster_scores))
            if contribution == 0.0 and signed_sum != 0.0:
                contribution = signed_sum / float(len(idx_arr))
            direction = self.positive_label if contribution >= 0 else self.negative_label
            groups.append(
                PeakGroup(
                    rank=0,
                    median_mz=float(np.median(cluster_mz)),
                    min_mz=float(np.min(cluster_mz)),
                    max_mz=float(np.max(cluster_mz)),
                    count=int(idx_arr.size),
                    direction=direction,
                    contribution=contribution,
                    model_weight=float(np.median(weights[idx_arr])) if weights.size else 0.0,
                    strength=float(np.sum(np.abs(cluster_scores))),
                    indices=idx_arr,
                )
            )

        groups.sort(key=lambda group: group.strength, reverse=True)
        for rank, group in enumerate(groups, start=1):
            group.rank = rank
        return groups

    def _set_feature_table_data(self, table: QTableWidget, rows: list[PeakGroup]):
        table.setRowCount(0)
        for group in rows:
            row = table.rowCount()
            table.insertRow(row)
            peak_range = (
                f"{group.min_mz:.0f}-{group.max_mz:.0f}"
                if group.count > 1 else f"{group.median_mz:.0f}"
            )
            items = [
                QTableWidgetItem(str(group.rank)),
                QTableWidgetItem(f"{group.median_mz:.1f}"),
                QTableWidgetItem(peak_range),
                QTableWidgetItem(str(group.count)),
                QTableWidgetItem(group.direction),
                QTableWidgetItem(f"{group.contribution:+.4f}"),
                QTableWidgetItem(f"{group.model_weight:.4f}"),
            ]
            items[0].setTextAlignment(Qt.AlignCenter)
            items[3].setTextAlignment(Qt.AlignCenter)
            items[5].setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            items[6].setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            tooltip = (
                f"Grouped from the top 50 discriminative peaks within a {group.max_mz - group.min_mz:.1f} Da span.\n"
                f"Median peak: {group.median_mz:.1f} Da\n"
                f"Range: {group.min_mz:.1f}-{group.max_mz:.1f} Da\n"
                f"Peaks in group: {group.count}"
            )
            items[1].setToolTip(tooltip)
            items[2].setToolTip(tooltip)
            items[3].setToolTip("Number of top-50 peaks grouped in this 10 Da window")
            items[4].setToolTip("Class direction favored by the median local RF contribution")
            items[5].setToolTip("Median local contribution of this 10 Da peak group")
            items[6].setToolTip("Median global Random Forest importance of this 10 Da peak group")
            for col, item in enumerate(items):
                table.setItem(row, col, item)

    def _clear_chart(self, view: QChartView, x_title: str = "", y_title: str = ""):
        if isinstance(view, QWebEngineView):
            fig = go.Figure()
            fig.update_layout(
                template="plotly_white",
                margin=dict(l=58, r=18, t=14, b=50),
                xaxis_title=x_title,
                yaxis_title=y_title,
                paper_bgcolor="rgba(8,136,145,0.06)",
                plot_bgcolor="rgba(255,255,255,0.72)",
            )
            self._set_plotly_figure(view, fig)
            return

        chart = view.chart()
        chart.removeAllSeries()
        for ax in chart.axes():
            chart.removeAxis(ax)

        base = QLineSeries()
        base.append(0.0, 0.0)
        base.append(1.0, 0.0)
        chart.addSeries(base)

        ax = QValueAxis()
        ax.setRange(0.0, 1.0)
        ax.setTitleText(x_title)
        ax.setGridLineVisible(True)
        ax.setGridLineColor(CHART_GRID)

        ay = QValueAxis()
        ay.setRange(0.0, 1.0)
        ay.setTitleText(y_title)
        ay.setGridLineVisible(True)
        ay.setGridLineColor(CHART_GRID)

        chart.addAxis(ax, Qt.AlignBottom)
        chart.addAxis(ay, Qt.AlignLeft)
        base.attachAxis(ax)
        base.attachAxis(ay)

    def _render_decision_contribution_chart(
        self,
        view: QChartView,
        mz: np.ndarray,
        signed_scores: np.ndarray,
        peak_groups: Optional[list[PeakGroup]] = None,
    ):
        groups = peak_groups if peak_groups is not None else self._cluster_top_discriminatory_peaks(mz, signed_scores)
        groups = groups[:12]
        if isinstance(view, QWebEngineView):
            if not groups:
                self._clear_chart(view, "RF contribution", "Top peaks")
                return

            plot_groups = list(reversed(groups))
            y = np.arange(1, len(plot_groups) + 1)
            scores = np.asarray([group.contribution for group in plot_groups], dtype=float)
            colors = np.where(scores >= 0, "#005BBB", "#166534")
            labels = [
                f"{group.median_mz:.1f} Da" + (f" (n={group.count})" if group.count > 1 else "")
                for group in plot_groups
            ]
            max_abs = max(float(np.max(np.abs(scores))), 1e-6)

            fig = go.Figure()
            for yi, score, color in zip(y, scores, colors):
                fig.add_trace(go.Scatter(
                    x=[0.0, float(score)],
                    y=[int(yi), int(yi)],
                    mode="lines",
                    line=dict(color=color, width=3),
                    hoverinfo="skip",
                    showlegend=False,
                ))
            fig.add_trace(go.Scatter(
                x=scores,
                y=y,
                mode="markers",
                marker=dict(color=colors, size=10, line=dict(color="white", width=1)),
                text=labels,
                customdata=np.asarray([
                    [group.min_mz, group.max_mz, group.count, group.strength]
                    for group in plot_groups
                ], dtype=float),
                hovertemplate=(
                    "<b>%{text}</b><br>"
                    "Median RF contribution: %{x:+.5f}<br>"
                    "Range: %{customdata[0]:.1f}-%{customdata[1]:.1f} Da<br>"
                    "Peaks in group: %{customdata[2]:.0f}<br>"
                    "Group strength: %{customdata[3]:.5f}<extra></extra>"
                ),
                showlegend=False,
            ))
            fig.add_vline(x=0, line_width=2, line_color="#94A3B8")
            fig.update_layout(
                template="plotly_white",
                margin=dict(l=64, r=24, t=14, b=54),
                paper_bgcolor="rgba(8,136,145,0.06)",
                plot_bgcolor="rgba(255,255,255,0.72)",
                xaxis=dict(title="RF contribution", range=[-max_abs * 1.18, max_abs * 1.18], zeroline=False),
                yaxis=dict(title="Top peaks", tickmode="array", tickvals=y, ticktext=labels),
                showlegend=False,
            )
            self._set_plotly_figure(view, fig)
            return

        chart = view.chart()
        chart.removeAllSeries()
        for ax in chart.axes():
            chart.removeAxis(ax)

        if not groups:
            self._clear_chart(view, "RF contribution", "Top peaks")
            return

        max_abs = float(np.max(np.abs([group.contribution for group in groups])))
        max_abs = max(max_abs, 1e-6)

        pos_points = QScatterSeries()
        pos_points.setColor(COLOR_BLUE_DARK)
        pos_points.setMarkerSize(10.0)
        neg_points = QScatterSeries()
        neg_points.setColor(COLOR_GREEN_DARK)
        neg_points.setMarkerSize(10.0)

        for row_idx, group in enumerate(reversed(groups), start=1):
            score = float(group.contribution)
            line = QLineSeries()
            color = COLOR_BLUE_DARK if score >= 0 else COLOR_GREEN_DARK
            line.setPen(QPen(color, 3))
            line.append(0.0, float(row_idx))
            line.append(score, float(row_idx))
            chart.addSeries(line)
            if score >= 0:
                pos_points.append(score, float(row_idx))
            else:
                neg_points.append(score, float(row_idx))

        zero_line = QLineSeries()
        zero_line.setPen(QPen(QColor("#94A3B8"), 2))
        zero_line.append(0.0, 0.5)
        zero_line.append(0.0, float(len(groups)) + 0.5)
        chart.addSeries(zero_line)
        chart.addSeries(pos_points)
        chart.addSeries(neg_points)

        ax = QValueAxis()
        ax.setRange(-max_abs * 1.15, max_abs * 1.15)
        ax.setTitleText("RF contribution")
        ax.setGridLineVisible(True)
        ax.setGridLineColor(CHART_GRID)

        ay = QValueAxis()
        ay.setRange(0.5, float(len(groups)) + 0.5)
        ay.setTitleText("Top peaks")
        ay.setLabelFormat("%.0f")
        ay.setTickCount(len(groups) + 1)
        ay.setGridLineVisible(True)
        ay.setGridLineColor(CHART_GRID)

        chart.addAxis(ax, Qt.AlignBottom)
        chart.addAxis(ay, Qt.AlignLeft)
        for s in chart.series():
            s.attachAxis(ax)
            s.attachAxis(ay)

    def _render_contribution_scatter_chart(
        self,
        view: QChartView,
        mz: np.ndarray,
        signed_scores: np.ndarray,
        selected_spectrum: np.ndarray,
        peak_groups: Optional[list[PeakGroup]] = None,
    ):
        groups = peak_groups if peak_groups is not None else self._cluster_top_discriminatory_peaks(mz, signed_scores)
        if isinstance(view, QWebEngineView):
            selected = np.asarray(selected_spectrum, dtype=float).ravel()
            selected = np.nan_to_num(selected, nan=0.0)
            if selected.size == 0:
                self._clear_chart(view, "Mass (Da)", "Normalized intensity")
                return

            max_selected = float(np.max(selected))
            selected_norm = selected / max_selected if max_selected > 0 else selected
            step = max(1, len(mz) // 4500)

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=mz[::step],
                y=selected_norm[::step],
                mode="lines",
                name="Sample spectrum",
                line=dict(color="rgba(214,40,57,0.52)", width=1.15),
                hovertemplate="Mass: %{x:.0f} Da<br>Normalized intensity: %{y:.3f}<extra></extra>",
            ))

            if groups:
                x_vals = [group.median_mz for group in groups]
                y_vals = [group.count for group in groups]
                widths = [max(4.0, group.max_mz - group.min_mz if group.count > 1 else 10.0) for group in groups]
                colors = [
                    "rgba(0,91,187,0.48)" if group.contribution >= 0 else "rgba(22,101,52,0.48)"
                    for group in groups
                ]
                bar_text = [f"{group.rank}" for group in groups]
                customdata = np.asarray([
                    [
                        group.min_mz,
                        group.max_mz,
                        group.contribution,
                        group.strength,
                        group.model_weight,
                    ]
                    for group in groups
                ], dtype=float)
                fig.add_trace(go.Bar(
                    x=x_vals,
                    y=y_vals,
                    width=widths,
                    text=bar_text,
                    textposition="outside",
                    name="Grouped top-50 peaks",
                    marker=dict(color=colors, line=dict(color="rgba(15,23,42,0.25)", width=1)),
                    opacity=0.92,
                    yaxis="y2",
                    customdata=customdata,
                    hovertemplate=(
                        "<b>Grouped top-50 peak range</b><br>"
                        "Median mass: %{x:.1f} Da<br>"
                        "Range: %{customdata[0]:.1f}-%{customdata[1]:.1f} Da<br>"
                        "Peaks in 10 Da window: %{y}<br>"
                        "Median RF contribution: %{customdata[2]:+.5f}<br>"
                        "Group strength: %{customdata[3]:.5f}<br>"
                        "Median model weight: %{customdata[4]:.5f}<extra></extra>"
                    ),
                ))

            fig.update_layout(
                template="plotly_white",
                margin=dict(l=62, r=56, t=18, b=58),
                paper_bgcolor="rgba(8,136,145,0.06)",
                plot_bgcolor="rgba(255,255,255,0.72)",
                hovermode="closest",
                barmode="overlay",
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1.0,
                    font=dict(size=10),
                ),
                xaxis=dict(
                    title="Mass (Da)",
                    range=[float(mz[0]), float(mz[-1]) if len(mz) else 20000.0],
                    showgrid=True,
                    gridcolor="rgba(15,23,42,0.10)",
                    zeroline=False,
                ),
                yaxis=dict(
                    title="Normalized intensity",
                    range=[0.0, 1.15],
                    showgrid=True,
                    gridcolor="rgba(15,23,42,0.10)",
                    zeroline=False,
                ),
                yaxis2=dict(
                    title="Peaks per 10 Da window",
                    overlaying="y",
                    side="right",
                    rangemode="tozero",
                    showgrid=False,
                    zeroline=False,
                ),
            )
            self._set_plotly_figure(view, fig)
            return

        chart = view.chart()
        chart.removeAllSeries()
        for ax in chart.axes():
            chart.removeAxis(ax)

        selected = np.asarray(selected_spectrum, dtype=float).ravel()
        selected = np.nan_to_num(selected, nan=0.0)
        if selected.size == 0:
            self._clear_chart(view, "Mass (Da)", "Normalized intensity")
            return

        max_selected = float(np.max(selected))
        if max_selected > 0:
            selected_norm = selected / max_selected
        else:
            selected_norm = selected

        ranked_idx = np.argsort(np.abs(signed_scores))[::-1][:80]
        ranked_idx = ranked_idx[np.abs(signed_scores[ranked_idx]) > 0]

        line = QLineSeries()
        line.setPen(QPen(QColor("#D62839"), 2))
        step = max(1, len(mz) // 2500)
        for x, y in zip(mz[::step], selected_norm[::step]):
            line.append(float(x), float(y))
        chart.addSeries(line)

        max_score = max(float(np.max(np.abs(signed_scores[ranked_idx]))), 1e-6) if ranked_idx.size else 1e-6
        for feat_idx in ranked_idx:
            x = float(mz[feat_idx])
            y = float(selected_norm[feat_idx])
            score = abs(float(signed_scores[feat_idx]))
            height = 0.08 + 0.22 * (score / max_score)
            top_y = min(1.0, y + height)
            peak_series = QLineSeries()
            peak_color = QColor(0, 91, 187, 110) if signed_scores[feat_idx] >= 0 else QColor(22, 101, 52, 110)
            peak_series.setPen(QPen(peak_color, 3))
            peak_series.append(x, y)
            peak_series.append(x, top_y)
            chart.addSeries(peak_series)

        ax = QValueAxis()
        ax.setRange(float(mz[0]), float(mz[-1]) if len(mz) else 20000.0)
        ax.setTitleText("Mass (Da)")
        ax.setLabelFormat("%.0f")
        ax.setGridLineVisible(True)
        ax.setGridLineColor(CHART_GRID)

        ay = QValueAxis()
        ay.setRange(0.0, 1.05)
        ay.setTitleText("Normalized intensity")
        ay.setGridLineVisible(True)
        ay.setGridLineColor(CHART_GRID)

        chart.addAxis(ax, Qt.AlignBottom)
        chart.addAxis(ay, Qt.AlignLeft)
        for s in chart.series():
            s.attachAxis(ax)
            s.attachAxis(ay)

    # ---------- rendering ----------
    def _populate_selector(self):
        self.combo.blockSignals(True)
        self.combo.clear()
        for r in self._rows:
            self.combo.addItem(r.pseudo_id)
        self.combo.setCurrentIndex(0 if self.combo.count() else -1)
        self.combo.blockSignals(False)

    def _render_empty(self):
        if self._load_errors:
            self.lbl_pred.setText("Model load error")
            self.lbl_formula.setText(" | ".join(self._load_errors))
        else:
            self.lbl_pred.setText("—")
            self.lbl_formula.setText("")
        self.lbl_conf.setText("Confidence: ?")
        self.lbl_consensus.setText("")
        self.lbl_consensus.setStyleSheet("")
        self._clear_chart(self.pos_chart)
        self._set_feature_table_data(self.neg_table, [])

    def _consensus_color(self, code: str) -> QColor:
        if code == "confirmed":
            return COLOR_GREEN_DARK
        if code == "warning":
            return COLOR_YELLOW_DARK
        if code == "negative":
            return COLOR_SLATE
        return COLOR_RED_DARK

    def _render_index(self, idx: int):
        if idx < 0 or idx >= len(self._rows):
            self._render_empty()
            return

        r = self._rows[idx]

        self.caption.setText("Prediction by Random Forest")
        self.lbl_pred.setText(r.rf_predicted_category)
        self.lbl_conf.setText(self._display_confidence_text(r.rf_predicted_category, r.rf_probability))

        if self.show_consensus_column or r.consensus_text not in {"", "—"}:
            self.lbl_consensus.setText(r.consensus_text)
            c = self._consensus_color(r.consensus_code)
            self.lbl_consensus.setStyleSheet(
                f"color: {c.name()}; font-weight: 900; font-size: 12px;"
            )
        else:
            self.lbl_consensus.setText("")
            self.lbl_consensus.setStyleSheet("")

        self._update_formula_or_params()
        self._update_biomarkers()
        self.spectrum_changed.emit(idx)

    def _update_formula_or_params(self):
        if self._rf_model is None:
            self.lbl_formula.setText("")
            return

        order = [
            "n_estimators", "max_depth", "max_features", "min_samples_split",
            "min_samples_leaf", "criterion", "class_weight", "bootstrap",
        ]
        bits = ["RF (external model)"]
        for k in order:
            if k in self._rf_params:
                bits.append(f"{k}={self._rf_params[k]}")
        self.lbl_formula.setText("   |   ".join(bits))

    def _update_biomarkers(self):
        self._update_rf_importances()

    def _update_rf_importances(self):
        self.lbl_pos_title.setText("RF local explanation")
        self.lbl_neg_title.setText("Top local RF contributions")

        imp = self._get_rf_importances_for_plot()
        selected = self._get_selected_spectrum()

        if imp is None:
            self._clear_chart(self.pos_chart)
            self._set_feature_table_data(self.neg_table, [])
            return

        mz = self._mz_axis_from_feature_count(len(imp))
        if selected is not None:
            selected = np.asarray(selected, dtype=float).ravel()
        if selected is None or selected.size != len(imp):
            self._clear_chart(self.pos_chart)
            self._set_feature_table_data(self.neg_table, [])
            return

        local = self._compute_rf_local_feature_contributions(selected)
        if local is None:
            self.lbl_pos_help.setText(
                "Local RF explanation is not available for this model object. "
                "Prediction still works, but tree-level contributions could not be extracted."
            )
            self._clear_chart(self.pos_chart)
            self._set_feature_table_data(self.neg_table, [])
            return

        signed_scores = np.asarray(local["contrib_pos"], dtype=float) - np.asarray(local["contrib_neg"], dtype=float)
        peak_groups = self._cluster_top_discriminatory_peaks(mz, signed_scores, imp, top_n=50, window_da=10.0)

        if self._explain_mode == "decision":
            self._render_decision_contribution_chart(self.pos_chart, mz, signed_scores, peak_groups)
        else:
            self._render_contribution_scatter_chart(self.pos_chart, mz, signed_scores, selected, peak_groups)

        self._set_feature_table_data(self.neg_table, peak_groups[:12])

    def _fill_table(self):
        self.table.setRowCount(0)
        for row_index, r in enumerate(self._rows):
            row = self.table.rowCount()
            self.table.insertRow(row)

            it0 = QTableWidgetItem(str(r.request_number))
            it1 = QTableWidgetItem(r.pseudo_id)
            it2 = QTableWidgetItem(r.rf_predicted_category)
            shown_conf = self._display_confidence_value(r.rf_predicted_category, r.rf_probability)
            it3 = QTableWidgetItem("—" if np.isnan(shown_conf) else f"{shown_conf * 100.0:.2f}%")

            peak_groups = self._top_peak_groups_for_row_index(row_index, limit=10)
            it4 = QTableWidgetItem(self._peak_groups_summary_text(peak_groups))
            it4.setToolTip(self._peak_groups_tooltip_text(peak_groups))

            items = [it0, it1, it2, it3, it4]

            if self.show_consensus_column:
                it5 = QTableWidgetItem(r.consensus_text)
                it5.setForeground(QBrush(self._consensus_color(r.consensus_code)))
                items.append(it5)

            it3.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)

            for it in items:
                it.setFlags(it.flags() ^ Qt.ItemIsEditable)

            for col, it in enumerate(items):
                self.table.setItem(row, col, it)

    def _update_table_metrics(self):
        total = len(self._rows)
        if total == 0:
            self.lbl_metrics.setText("RF positive-rate: —   |   Positive: —   |   Other/negative: —")
            return

        preds = [r.rf_predicted_category for r in self._rows]
        model_name = "RF"

        valid = [
            p for p in preds
            if p not in {"—", "-", "Blocked", "Not applicable", "Complementary testing required"}
        ]
        if not valid:
            self.lbl_metrics.setText(f"{model_name}: no applicable rows for evaluation")
            return

        hits = sum(1 for p in valid if p == self.positive_label)
        misses = len(valid) - hits
        rate = 100.0 * hits / len(valid)

        self.lbl_metrics.setText(
            f"{model_name} positive-rate: {rate:.2f}%   |   Positive: {hits}   |   Other/negative: {misses}   |   Applicable: {len(valid)}"
        )

    # ---------- events ----------
    def _on_table_selection(self):
        sel = self.table.selectionModel().selectedRows()
        if not sel:
            return
        r = sel[0].row()
        if 0 <= r < len(self._rows):
            self.row_selected.emit(self._rows[r].request_number)
            self.combo.setCurrentIndex(r)
