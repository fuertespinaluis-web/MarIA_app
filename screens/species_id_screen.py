from __future__ import annotations

import base64
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from PySide6.QtCore import Qt, QUrl
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QToolButton,
    QDialog,
    QTextEdit,
)

from .model_task_screen_base import ModelTaskScreen


TOP_GREEN = "#007A33"
TOP_BLUE = "#005BBB"
THRESHOLD_BURGUNDY = "#7A1F3D"
SELECTED_RED = "#D62839"
ZONE_GREY = "rgba(120, 120, 120, 0.10)"


class SpeciesIdScreen(ModelTaskScreen):
    SPECIES_THRESHOLD = 0.619
    REFERENCE_THRESHOLD = 0.474
    OPTIMAL_THRESHOLD = 0.553
    ADJUSTED_THRESHOLD = 0.619

    def __init__(self, model_dir: Path, parent: Optional[QWidget] = None):
        rf_bundle_dir = model_dir / "Spn_species_RF_DA_x1p5"
        super().__init__(
            title_text="Species identification",
            positive_label="Streptococcus pneumoniae",
            negative_label="Streptococcus mitis/oralis",
            rf_model_path=rf_bundle_dir / "rf_spn_species_best.joblib",
            rf_importances_path=rf_bundle_dir / "rf_feature_importances.csv",
            rf_params_path=rf_bundle_dir / "rf_best_params.json",
            show_consensus_column=True,
            parent=parent,
        )
        self.threshold_assets_dir = Path(__file__).resolve().parent / "fig_threshold"
        self.threshold_output_path = self.threshold_assets_dir / "maria_threshold_figure_current.html"
        self._threshold_curve_df: Optional[pd.DataFrame] = None
        self._threshold_hover_df: Optional[pd.DataFrame] = None
        self._threshold_density_df: Optional[pd.DataFrame] = None
        self._threshold_points_df: Optional[pd.DataFrame] = None
        self._build_threshold_view()
        self.btn_probability_detail.setVisible(True)
        self.probability_detail_requested.connect(self._show_probability_detail)
        self.lbl_formula.setVisible(False)
        self.card_pos.setMinimumHeight(250)
        self.card_neg.setMinimumHeight(250)
        self.pos_chart.setMinimumHeight(180)
        self.neg_chart.setMinimumHeight(180)

    def _predict_rf_rows(self, X) -> tuple[list[str], list[float]]:
        if self._rf_model is None:
            return ["-"] * X.shape[0], [np.nan] * X.shape[0]

        try:
            p_pos = self._predict_positive_probability(self._rf_model, X, "RF")
        except Exception as e:
            msg = f"RF prediction failed: {e}"
            if msg not in self._load_errors:
                self._load_errors.append(msg)
            return ["-"] * X.shape[0], [np.nan] * X.shape[0]

        preds, probs = [], []
        for p in p_pos:
            p = float(p)
            preds.append(self.positive_label if p >= self.SPECIES_THRESHOLD else self.negative_label)
            probs.append(p)
        return preds, probs

    def build_species_consensus(self, rf_pred: str, rf_probability: float) -> tuple[str, str]:
        if rf_pred in {"-", "Blocked", "Not applicable"} or np.isnan(rf_probability):
            return rf_pred, "none"
        if rf_probability < self.SPECIES_THRESHOLD:
            return "Not compatible with <i>S. pneumoniae</i>", "negative"
        if rf_pred == self.positive_label:
            return "Confirmed <i>S. pneumoniae</i>", "confirmed"
        return "Not compatible with <i>S. pneumoniae</i>", "negative"

    def apply_species_consensus(self):
        for r in self._rows:
            txt, code = self.build_species_consensus(r.rf_predicted_category, r.rf_probability)
            r.consensus_text = txt
            r.consensus_code = code

        self._post_update_ui()

    def _update_formula_or_params(self):
        self.lbl_formula.setText("")
        self.lbl_formula.setVisible(False)

    def _display_confidence_value(self, predicted_label: str, rf_probability: float) -> float:
        if np.isnan(rf_probability):
            return np.nan
        if predicted_label == self.negative_label:
            return float(1.0 - rf_probability)
        return float(rf_probability)

    def _build_threshold_view(self):
        self.view_threshold = QWidget()
        layout = QVBoxLayout(self.view_threshold)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        top = QHBoxLayout()
        top.setContentsMargins(0, 0, 0, 0)
        top.setSpacing(8)

        self.btn_threshold_back = QPushButton("Back to results")
        self.btn_threshold_back.setObjectName("GhostBtn")
        self.btn_threshold_back.setFixedHeight(38)
        self.btn_threshold_back.clicked.connect(lambda: self.stack.setCurrentIndex(0))

        title = QLabel("Confidence threshold")
        title.setObjectName("TitleSmall")

        top.addWidget(self.btn_threshold_back, alignment=Qt.AlignLeft)
        top.addStretch(1)
        top.addWidget(title, alignment=Qt.AlignRight)
        layout.addLayout(top)

        self.lbl_threshold_summary = QLabel("")
        self.lbl_threshold_summary.setObjectName("LabelMuted")
        self.lbl_threshold_summary.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.lbl_threshold_summary)

        self.lbl_threshold_metrics = QLabel("")
        self.lbl_threshold_metrics.setObjectName("LabelMuted")
        self.lbl_threshold_metrics.setAlignment(Qt.AlignCenter)
        self.lbl_threshold_metrics.setWordWrap(True)
        layout.addWidget(self.lbl_threshold_metrics)

        info_row = QHBoxLayout()
        info_row.setContentsMargins(0, 0, 0, 0)
        info_row.setSpacing(8)

        self.btn_threshold_info = QToolButton()
        self.btn_threshold_info.setText("i")
        self.btn_threshold_info.setObjectName("GhostBtn")
        self.btn_threshold_info.setFixedSize(28, 28)
        self.btn_threshold_info.setToolTip(
            "How to read this plot:\n"
            "Model confidence on the x-axis is the RF confidence for S. pneumoniae.\n"
            "The blue and green curves show the Radius Neighbours confidence (OODTest) for each class nearby.\n"
            "The density band shows where OODTest samples are concentrated.\n"
            "The sample map shows individual OODTest spectra.\n"
            "Dashed lines mark the reference, optimal and adjusted thresholds.\n"
            "Your selected sample is marked in red."
        )
        self.btn_threshold_info.setCursor(Qt.PointingHandCursor)
        self.btn_threshold_info.clicked.connect(self._show_threshold_info_dialog)

        info_hint = QLabel("How to interpret this graph")
        info_hint.setObjectName("LabelMuted")

        top.addSpacing(8)
        top.addWidget(self.btn_threshold_info, alignment=Qt.AlignLeft)
        top.addWidget(info_hint, alignment=Qt.AlignVCenter)

        self.threshold_web = QWebEngineView()
        layout.addWidget(self.threshold_web, 1)

        self.stack.addWidget(self.view_threshold)

    def _show_threshold_info_dialog(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("Confidence View – How to interpret this screen")
        dialog.resize(860, 760)
        images_dir = Path(__file__).resolve().parents[1] / "Images"
        rf_example_path = images_dir / "RF_example.png"
        radius_example_path = images_dir / "radius neighbours.png"
        rf_example_html = ""
        radius_example_html = ""
        if rf_example_path.exists():
            try:
                image_b64 = base64.b64encode(rf_example_path.read_bytes()).decode("ascii")
                rf_example_html = (
                    '<div style="margin:10px 0 18px 0; text-align:center; background:#FFFFFF; '
                    'border:1px solid rgba(15,23,42,0.08); border-radius:14px; padding:12px;">'
                    f'<img src="data:image/png;base64,{image_b64}" '
                    'style="max-width:100%; height:auto; border-radius:10px;" '
                    'alt="Random forest voting example">'
                    '</div>'
                )
            except Exception:
                rf_example_html = ""
        if radius_example_path.exists():
            try:
                image_b64 = base64.b64encode(radius_example_path.read_bytes()).decode("ascii")
                radius_example_html = (
                    '<div style="margin:10px 0 18px 0; text-align:center; background:#FFFFFF; '
                    'border:1px solid rgba(15,23,42,0.08); border-radius:14px; padding:12px;">'
                    f'<img src="data:image/png;base64,{image_b64}" '
                    'style="max-width:100%; height:auto; border-radius:10px;" '
                    'alt="Radius neighbours explanation">'
                    '</div>'
                )
            except Exception:
                radius_example_html = ""
        dialog.setStyleSheet("""
        QDialog {
            background: #F5F7FA;
        }
        QLabel#InfoTitle {
            color: #088891;
            font-size: 22px;
            font-weight: 900;
        }
        QWidget#InfoCard {
            background: #FFFFFF;
            border: 1px solid rgba(15, 23, 42, 0.10);
            border-radius: 16px;
        }
        QTextEdit#InfoBody {
            background: #FBFCFE;
            color: #243244;
            border: 1px solid rgba(8, 136, 145, 0.12);
            border-radius: 12px;
            padding: 14px;
            font-size: 14px;
            line-height: 1.45;
            selection-background-color: rgba(8, 136, 145, 0.18);
        }
        QTextEdit#InfoBody QScrollBar:vertical {
            background: transparent;
            width: 12px;
            margin: 4px 2px 4px 2px;
        }
        QTextEdit#InfoBody QScrollBar::handle:vertical {
            background: rgba(8, 136, 145, 0.35);
            border-radius: 6px;
            min-height: 28px;
        }
        QTextEdit#InfoBody QScrollBar::handle:vertical:hover {
            background: rgba(8, 136, 145, 0.55);
        }
        QTextEdit#InfoBody QScrollBar::add-line:vertical,
        QTextEdit#InfoBody QScrollBar::sub-line:vertical,
        QTextEdit#InfoBody QScrollBar::add-page:vertical,
        QTextEdit#InfoBody QScrollBar::sub-page:vertical {
            background: transparent;
            height: 0px;
        }
        """)

        layout = QVBoxLayout(dialog)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        title = QLabel("Confidence View – How to interpret this screen")
        title.setObjectName("InfoTitle")
        layout.addWidget(title)

        card = QWidget()
        card.setObjectName("InfoCard")
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(16, 16, 16, 16)
        card_layout.setSpacing(12)

        body = QTextEdit()
        body.setObjectName("InfoBody")
        body.setReadOnly(True)
        body.setAcceptRichText(True)
        body.setHtml(f"""
            <div style="color:#243244; font-size:14px; line-height:1.55;">
              <h3 style="margin:0 0 14px 0; color:#1F3B64;">Confidence View – How to interpret this screen</h3>
              <p>This screen is designed to help the clinical microbiologist make safer decisions when differentiating between Streptococcus pneumoniae and closely related species such as S. mitis/oralis, which can sometimes produce very similar MALDI-TOF spectra.</p>
              <p>To improve reliability, MarIA combines two machine learning models:</p>
              <p><b>Random Forest (RF):</b> the main classification model that predicts whether the isolate is S. pneumoniae or S. mitis/oralis.<br>
              <b>Radius Neighbours:</b> an additional confidence model that estimates how trustworthy that prediction is based on how similar cases behaved during external validation inside a fixed local confidence window.</p>
              <p>This means the system not only gives a result, but also indicates how confident it is in that result.</p>
              <h3 style="margin:18px 0 10px 0; color:#1F3B64;">How is the graph organized?</h3>
              <p>The graph is divided into three sections, with the top panel being the most important for routine interpretation.</p>
              <h3 style="margin:18px 0 10px 0; color:#1F3B64;">1. Top panel – Prediction + confidence</h3>
              <p>This panel combines the outputs of both models.</p>
              <p><b>X-axis:</b> confidence of the Random Forest model that the isolate is S. pneumoniae.<br>
              Values close to 100% suggest strong support for S. pneumoniae.<br>
              Values close to 0% suggest strong support for S. mitis/oralis.</p>
              <p>This confidence is calculated from the proportion of decision trees within the Random Forest voting for each category.</p>
              {rf_example_html}
              <p><b>Y-axis:</b> confidence estimated by the Radius Neighbours model.</p>
              <p>This value reflects how often isolates with similar RF confidence scores were correctly classified during external validation. In other words, it answers the question:</p>
              <p style="margin-left:14px;"><i>“When the model gives a result like this, how often has it been right in real test data?”</i></p>
              <p>This provides an extra layer of reassurance beyond the original training dataset and better reflects routine diagnostic performance.</p>
              {radius_example_html}
              <h3 style="margin:18px 0 10px 0; color:#1F3B64;">2. Middle panel – Density heatmap</h3>
              <p>This section shows how samples from the external validation dataset were distributed across the RF confidence scale.</p>
              <p>Warmer colors indicate zones where more samples were found.<br>
              Lighter colors indicate less populated areas.</p>
              <p>In general, most isolates cluster toward the extremes, meaning the model often makes clear high-confidence predictions rather than uncertain intermediate ones.</p>
              <h3 style="margin:18px 0 10px 0; color:#1F3B64;">3. Bottom panel – Sample distribution map</h3>
              <p>This scatter plot shows how real spectra from the external validation set were distributed.</p>
              <p>Each point represents one isolate:</p>
              <p>Blue: S. pneumoniae<br>
              Green: S. mitis/oralis<br>
              Red: S. pseudopneumoniae (included as a minority reference group)</p>
              <p>This allows the user to visualize where each species tends to appear and where overlap may occur.</p>
              <h3 style="margin:18px 0 10px 0; color:#1F3B64;">How should results be interpreted?</h3>
              <p>In most situations, the top panel is the key area to use.</p>
              <p><b>Y-axis (Radius Neighbours confidence)</b></p>
              <p>The closer the value is to 100%, the more often similar samples were correctly classified during validation.</p>
              <p><b>Higher = more reliable prediction</b></p>
              <p><b>X-axis (RF confidence)</b><br>
              Closer to 100% → more likely S. pneumoniae<br>
              Closer to 0% → more likely S. mitis/oralis</p>
              <p><b>Farther from the center = stronger biological separation</b></p>
              <p><b>Practical interpretation</b><br>
              High X + High Y → strong and reliable S. pneumoniae<br>
              Low X + High Y → strong and reliable S. mitis/oralis<br>
              Intermediate X or low Y → less certainty, consider complementary testing</p>
              <h3 style="margin:18px 0 10px 0; color:#1F3B64;">Additional note</h3>
              <p>You can place the mouse cursor over the selected sample to view the exact prediction values and confidence percentages for that isolate.</p>
              <h3 style="margin:18px 0 10px 0; color:#1F3B64;">Support</h3>
              <p>For questions, feedback, or suggestions, please contact:</p>
              <p><b>fuertespinaluis@gmail.com</b></p>
            </div>
        """)
        card_layout.addWidget(body, 1)
        layout.addWidget(card, 1)

        btn_close = QPushButton("Close")
        btn_close.setObjectName("PrimaryBtn")
        btn_close.setFixedHeight(38)
        btn_close.clicked.connect(dialog.accept)

        btn_row = QHBoxLayout()
        btn_row.addStretch(1)
        btn_row.addWidget(btn_close)
        layout.addLayout(btn_row)

        dialog.exec()

    def _selected_row(self):
        idx = self.combo.currentIndex()
        if idx < 0 or idx >= len(self._rows):
            return None
        return self._rows[idx]

    def _show_probability_detail(self):
        row = self._selected_row()
        if row is None or np.isnan(row.rf_probability):
            self.lbl_threshold_summary.setText("No RF confidence is available for the selected spectrum.")
            self.lbl_threshold_metrics.setText("")
            self.threshold_web.setHtml("")
            self.stack.setCurrentWidget(self.view_threshold)
            return

        shown_conf = self._display_confidence_value(row.rf_predicted_category, row.rf_probability)
        conf_pct = float(shown_conf) * 100.0
        prob_pct = float(row.rf_probability) * 100.0
        stats = self._threshold_summary_stats()

        self.lbl_threshold_summary.setText(
            f"{row.pseudo_id} | RF confidence: {conf_pct:.2f}% | {row.rf_predicted_category}"
        )
        self.lbl_threshold_metrics.setText(
            "OODTest | "
            f"Balanced accuracy: {stats['ba_pct_text']} | "
            f"Sensitivity: {stats['pneumo_acc_text']} | "
            f"Specificity: {stats['mitis_acc_text']} | "
            f"PPV: {stats['pneumo_precision_text']} | "
            f"NPV: {stats['mitis_precision_text']}"
        )
        fig = self._build_threshold_figure(
            pseudo_id=row.pseudo_id,
            prob_pct=prob_pct,
            prediction=row.rf_predicted_category,
        )
        html_path = self._write_threshold_figure_html(fig)
        self.threshold_web.setUrl(QUrl.fromLocalFile(str(html_path.resolve())))
        self.stack.setCurrentWidget(self.view_threshold)

    def _load_threshold_assets(self):
        if self._threshold_curve_df is None:
            self._threshold_curve_df = pd.read_csv(
                self.threshold_assets_dir / "rf_da_x1p5_curve_plot_5pct_with_smooth_density_heatmap.csv"
            )
        if self._threshold_hover_df is None:
            self._threshold_hover_df = pd.read_csv(
                self.threshold_assets_dir / "rf_da_x1p5_curve_hover_raw_1pct_with_smooth_density_heatmap.csv"
            )
        if self._threshold_density_df is None:
            self._threshold_density_df = pd.read_csv(
                self.threshold_assets_dir / "rf_da_x1p5_smooth_density_heatmap.csv"
            )
        if self._threshold_points_df is None:
            self._threshold_points_df = pd.read_csv(
                self.threshold_assets_dir / "rf_da_x1p5_probability_points_map.csv"
            )

    @staticmethod
    def _as_float_series(df: pd.DataFrame, column: str) -> pd.Series:
        if column not in df.columns:
            return pd.Series(np.nan, index=df.index, dtype=float)
        return pd.to_numeric(df[column], errors="coerce")

    def _prepared_points_df(self) -> pd.DataFrame:
        self._load_threshold_assets()
        points_df = self._threshold_points_df.copy()
        points_df["prob_pct"] = self._as_float_series(points_df, "prob_pct")
        if points_df["prob_pct"].isna().all() and "rf_da_x1p5_probability" in points_df.columns:
            points_df["prob_pct"] = self._as_float_series(points_df, "rf_da_x1p5_probability") * 100.0
        points_df["point_y"] = self._as_float_series(points_df, "point_y")

        if points_df["point_y"].isna().all():
            rng = np.random.default_rng(42)
            points_df["point_y"] = rng.uniform(0.35, 0.75, size=len(points_df))

        if "truth_label_norm" not in points_df.columns and "truth_label" in points_df.columns:
            points_df["truth_label_norm"] = points_df["truth_label"]

        return points_df.loc[points_df["prob_pct"].notna()].copy()

    def _threshold_summary_stats(self) -> dict[str, float | str | int | None]:
        points_df = self._prepared_points_df()
        truth_series = points_df.get("truth_label_norm", points_df.get("truth_label", "")).astype(str).str.lower()
        scoring_mask = (
            points_df.get("included_in_global_scoring", pd.Series("", index=points_df.index))
            .astype(str)
            .str.strip()
            .str.lower()
            .eq("yes")
        )
        pneumo_mask = truth_series.str.contains("pneumoniae") & ~truth_series.str.contains("pseudo")
        mitis_mask = truth_series.str.contains("mitis")
        evaluable = points_df.loc[scoring_mask & (pneumo_mask | mitis_mask)].copy()

        threshold_pct = self.ADJUSTED_THRESHOLD * 100.0
        pneumo_eval = evaluable.loc[pneumo_mask.reindex(evaluable.index, fill_value=False)]
        mitis_eval = evaluable.loc[mitis_mask.reindex(evaluable.index, fill_value=False)]
        pred_pneumo = evaluable.loc[evaluable["prob_pct"] >= threshold_pct]
        pred_mitis = evaluable.loc[evaluable["prob_pct"] < threshold_pct]

        pneumo_acc = float((pneumo_eval["prob_pct"] >= threshold_pct).mean()) if len(pneumo_eval) else np.nan
        mitis_acc = float((mitis_eval["prob_pct"] < threshold_pct).mean()) if len(mitis_eval) else np.nan
        ba_pct = (
            100.0 * ((pneumo_acc + mitis_acc) / 2.0)
            if np.isfinite(pneumo_acc) and np.isfinite(mitis_acc)
            else np.nan
        )
        pneumo_precision = (
            float(pneumo_mask.reindex(pred_pneumo.index, fill_value=False).mean()) if len(pred_pneumo) else np.nan
        )
        mitis_precision = (
            float(mitis_mask.reindex(pred_mitis.index, fill_value=False).mean()) if len(pred_mitis) else np.nan
        )

        def _fmt_ratio(value: float) -> str:
            return f"{value * 100.0:.1f}%" if np.isfinite(value) else "N/A"

        def _fmt_pct(value: float) -> str:
            return f"{value:.1f}%" if np.isfinite(value) else "N/A"

        return {
            "ba_pct": None if not np.isfinite(ba_pct) else float(ba_pct),
            "ba_pct_text": _fmt_pct(ba_pct),
            "pneumo_acc": None if not np.isfinite(pneumo_acc) else float(pneumo_acc),
            "pneumo_acc_text": _fmt_ratio(pneumo_acc),
            "mitis_acc": None if not np.isfinite(mitis_acc) else float(mitis_acc),
            "mitis_acc_text": _fmt_ratio(mitis_acc),
            "pneumo_precision": None if not np.isfinite(pneumo_precision) else float(pneumo_precision),
            "pneumo_precision_text": _fmt_ratio(pneumo_precision),
            "mitis_precision": None if not np.isfinite(mitis_precision) else float(mitis_precision),
            "mitis_precision_text": _fmt_ratio(mitis_precision),
            "n_evaluable": int(len(evaluable)),
        }

    def _build_threshold_figure(self, pseudo_id: str, prob_pct: float, prediction: str) -> go.Figure:
        self._load_threshold_assets()

        curve_df = self._threshold_curve_df.copy().sort_values("probability_pct").reset_index(drop=True)
        hover_df = self._threshold_hover_df.copy().sort_values("hover_x_pct").reset_index(drop=True)
        density_df = self._threshold_density_df.copy().sort_values("probability_pct").reset_index(drop=True)
        points_df = self._prepared_points_df()
        threshold_map_pct = {
            "reference": self.REFERENCE_THRESHOLD * 100.0,
            "optimal": self.OPTIMAL_THRESHOLD * 100.0,
            "adjusted": self.ADJUSTED_THRESHOLD * 100.0,
        }
        threshold_pct = threshold_map_pct["adjusted"]
        zone_left_pct = min(threshold_map_pct.values())
        zone_right_pct = max(threshold_map_pct.values())

        x_curve = self._as_float_series(curve_df, "probability_pct")
        pneumo_curve = self._as_float_series(curve_df, "real_prob_pneumoniae_pct")
        mitis_curve = self._as_float_series(curve_df, "real_prob_mitis_oralis_pct")
        selected_curve = pneumo_curve if prob_pct >= threshold_pct else mitis_curve
        sample_curve_y = float(np.interp(prob_pct, x_curve.to_numpy(), selected_curve.to_numpy()))
        sample_knn_pneumo = float(np.interp(prob_pct, x_curve.to_numpy(), pneumo_curve.to_numpy()))
        sample_knn_mitis = float(np.interp(prob_pct, x_curve.to_numpy(), mitis_curve.to_numpy()))
        sample_model_mitis = float(100.0 - prob_pct)

        fig = make_subplots(
            rows=3,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.06,
            row_heights=[0.58, 0.16, 0.26],
        )

        for row_idx in [1, 2, 3]:
            fig.add_vrect(
                x0=zone_left_pct,
                x1=zone_right_pct,
                fillcolor=ZONE_GREY,
                line_width=0,
                layer="below",
                row=row_idx,
                col=1,
            )

        hover_custom = np.stack(
            [
                self._as_float_series(hover_df, "window_left_pct").to_numpy(),
                self._as_float_series(hover_df, "window_right_pct").to_numpy(),
                self._as_float_series(hover_df, "n_local").to_numpy(),
                self._as_float_series(hover_df, "n_true_pneumoniae").to_numpy(),
                self._as_float_series(hover_df, "n_true_mitis_oralis").to_numpy(),
                self._as_float_series(hover_df, "hover_x_pct").to_numpy(),
                100.0 - self._as_float_series(hover_df, "hover_x_pct").to_numpy(),
                self._as_float_series(hover_df, "real_prob_pneumoniae_pct").to_numpy(),
                self._as_float_series(hover_df, "real_prob_mitis_oralis_pct").to_numpy(),
            ],
            axis=-1,
        )
        common_knn_hover = (
            "<b>Model confidence for S. pneumoniae</b>: %{customdata[5]:.1f}%<br>"
            "<b>Model confidence for S. mitis/oralis</b>: %{customdata[6]:.1f}%<br>"
            "<b>Radius Neighbours confidence (OODTest) S. pneumoniae</b>: %{customdata[7]:.1f}%<br>"
            "<b>Radius Neighbours confidence (OODTest) S. mitis/oralis</b>: %{customdata[8]:.1f}%<br>"
            "<b>Raw local window</b>: %{customdata[0]:.1f}% - %{customdata[1]:.1f}%<br>"
            "<b>Samples in window</b>: %{customdata[2]:.0f}<br>"
            "<b>True S. pneumoniae</b>: %{customdata[3]:.0f}<br>"
            "<b>True S. mitis/oralis</b>: %{customdata[4]:.0f}<extra></extra>"
        )
        fig.add_trace(
            go.Scatter(
                x=x_curve,
                y=mitis_curve,
                mode="lines+markers",
                name="Radius Neighbours confidence (OODTest) S. mitis/oralis",
                line=dict(color=TOP_GREEN, width=3, shape="spline", smoothing=1.0),
                marker=dict(size=7, color=TOP_GREEN, line=dict(color="white", width=1)),
                customdata=hover_custom,
                hovertemplate=common_knn_hover,
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=x_curve,
                y=pneumo_curve,
                mode="lines+markers",
                name="Radius Neighbours confidence (OODTest) S. pneumoniae",
                line=dict(color=TOP_BLUE, width=3, shape="spline", smoothing=1.0),
                marker=dict(size=7, color=TOP_BLUE, line=dict(color="white", width=1)),
                customdata=hover_custom,
                hovertemplate=common_knn_hover,
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=self._as_float_series(hover_df, "hover_x_pct"),
                y=self._as_float_series(hover_df, "real_prob_pneumoniae_pct"),
                mode="markers",
                showlegend=False,
                marker=dict(size=18, color="rgba(0,0,0,0)"),
                customdata=hover_custom,
                hovertemplate=common_knn_hover,
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=self._as_float_series(hover_df, "hover_x_pct"),
                y=self._as_float_series(hover_df, "real_prob_mitis_oralis_pct"),
                mode="markers",
                showlegend=False,
                marker=dict(size=18, color="rgba(0,0,0,0)"),
                customdata=hover_custom,
                hovertemplate=common_knn_hover,
            ),
            row=1,
            col=1,
        )

        heat_custom = np.stack(
            [
                self._as_float_series(density_df, "window_left_pct").to_numpy(),
                self._as_float_series(density_df, "window_right_pct").to_numpy(),
                self._as_float_series(density_df, "n_samples_window").to_numpy(),
                self._as_float_series(density_df, "pct_samples_window").to_numpy(),
                self._as_float_series(density_df, "pct_pneumo_window").to_numpy(),
                self._as_float_series(density_df, "pct_mitis_window").to_numpy(),
                self._as_float_series(density_df, "pct_pseudo_window").to_numpy(),
            ],
            axis=-1,
        )[np.newaxis, :, :]
        fig.add_trace(
            go.Heatmap(
                x=self._as_float_series(density_df, "probability_pct").to_numpy(),
                y=["Sample density"],
                z=np.array([self._as_float_series(density_df, "pct_samples_window").to_numpy()]),
                customdata=heat_custom,
                colorscale="YlOrRd",
                zmin=0,
                zmax=float(np.nanmax(self._as_float_series(density_df, "pct_samples_window").to_numpy())),
                showscale=True,
                zsmooth="best",
                colorbar=dict(
                    title=dict(text="% samples", side="top"),
                    len=0.24,
                    y=0.48,
                    thickness=22,
                    tickfont=dict(size=10),
                    x=1.03,
                    xpad=8,
                ),
                hovertemplate=(
                    "<b>Model confidence</b>: %{x:.1f}%<br>"
                    "<b>Raw local window</b>: %{customdata[0]:.1f}% - %{customdata[1]:.1f}%<br>"
                    "<b>Samples in window</b>: %{customdata[2]:.0f}<br>"
                    "<b>% total samples in window</b>: %{customdata[3]:.2f}%<br>"
                    "<b>% S. pneumoniae in window</b>: %{customdata[4]:.2f}%<br>"
                    "<b>% S. mitis/oralis in window</b>: %{customdata[5]:.2f}%<br>"
                    "<b>% S. pseudopneumoniae in window</b>: %{customdata[6]:.2f}%<extra></extra>"
                ),
            ),
            row=2,
            col=1,
        )

        point_styles = {
            "Streptococcus mitis/oralis": ("OODTest mitis/oralis", "rgba(0, 132, 61, 0.35)"),
            "Streptococcus pneumoniae": ("OODTest pneumoniae", "rgba(0, 91, 187, 0.35)"),
            "Streptococcus pseudopneumoniae": ("OODTest pseudopneumoniae", "rgba(210, 35, 35, 0.65)"),
        }
        truth_labels = points_df.get("truth_label_norm", points_df.get("truth_label", pd.Series("", index=points_df.index)))
        for truth_label, (trace_name, color) in point_styles.items():
            sub = points_df.loc[truth_labels.astype(str).eq(truth_label)].copy()
            if sub.empty:
                continue
            customdata = np.stack(
                [
                    sub.get("sample_id", pd.Series("", index=sub.index)).astype(str).to_numpy(),
                    sub.get("truth_label_norm", sub.get("truth_label", pd.Series("", index=sub.index))).astype(str).to_numpy(),
                    sub["prob_pct"].astype(float).round(2).astype(str).to_numpy(),
                ],
                axis=-1,
            )
            fig.add_trace(
                go.Scattergl(
                    x=sub["prob_pct"],
                    y=sub["point_y"],
                    mode="markers",
                    name=trace_name,
                    marker=dict(
                        size=7,
                        color=color,
                        symbol="circle",
                        line=dict(color="rgba(0,0,0,0.18)", width=0.4),
                    ),
                    customdata=customdata,
                    hovertemplate=(
                        "<b>%{customdata[0]}</b><br>"
                        "Truth: %{customdata[1]}<br>"
                        "Model confidence: %{customdata[2]}%<extra></extra>"
                    ),
                ),
                row=3,
                col=1,
            )

        fig.add_trace(
            go.Scatter(
                x=[prob_pct],
                y=[sample_curve_y],
                mode="markers",
                name="Selected spectrum",
                marker=dict(color=SELECTED_RED, size=14, symbol="circle", line=dict(color="white", width=2)),
                hovertemplate=(
                    f"<b>{pseudo_id}</b><br>"
                    f"Prediction: {prediction}<br>"
                    f"<b>Model confidence for S. pneumoniae</b>: {prob_pct:.2f}%<br>"
                    f"<b>Model confidence for S. mitis/oralis</b>: {sample_model_mitis:.2f}%<br>"
                    f"<b>Radius Neighbours confidence (OODTest) S. pneumoniae</b>: {sample_knn_pneumo:.1f}%<br>"
                    f"<b>Radius Neighbours confidence (OODTest) S. mitis/oralis</b>: {sample_knn_mitis:.1f}%<br>"
                    "Local support on selected side: %{y:.1f}%<extra></extra>"
                ),
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=[prob_pct],
                y=[0.55],
                mode="markers",
                name="Selected spectrum on map",
                showlegend=False,
                marker=dict(color=SELECTED_RED, size=13, symbol="diamond", line=dict(color="white", width=2)),
                hovertemplate=(
                    f"<b>{pseudo_id}</b><br>"
                    f"Prediction: {prediction}<br>"
                    "Model confidence: %{x:.2f}%<extra></extra>"
                ),
            ),
            row=3,
            col=1,
        )

        for label, x_value in threshold_map_pct.items():
            for row in [1, 2, 3]:
                fig.add_vline(
                    x=x_value,
                    line_width=2,
                    line_dash="dash",
                    line_color=THRESHOLD_BURGUNDY,
                    row=row,
                    col=1,
                )
            fig.add_annotation(
                x=x_value,
                y=101.5,
                text=f"{label}: {x_value:.1f}%",
                showarrow=False,
                font=dict(color=THRESHOLD_BURGUNDY, size=10),
                bgcolor="rgba(255,255,255,0.75)",
                row=1,
                col=1,
            )

        for row in [1, 2, 3]:
            fig.add_vline(
                x=prob_pct,
                line_width=2,
                line_dash="dot",
                line_color=SELECTED_RED,
                row=row,
                col=1,
            )

        fig.add_annotation(
            x=prob_pct,
            y=max(5.0, sample_curve_y + 8.0),
            text=f"{pseudo_id}<br>{prob_pct:.2f}%",
            showarrow=False,
            font=dict(color=SELECTED_RED, size=10),
            bgcolor="rgba(255,255,255,0.78)",
            row=1,
            col=1,
        )

        complementary_mask = points_df["prob_pct"].between(zone_left_pct, zone_right_pct, inclusive="both")
        complementary_n = int(complementary_mask.sum())
        complementary_pct = float(complementary_n / len(points_df) * 100.0) if len(points_df) else 0.0
        fig.add_annotation(
            x=(zone_left_pct + zone_right_pct) / 2.0,
            y="Sample density",
            text=(
                "complementary testing zone"
                f"<br>{complementary_pct:.1f}% samples (n={complementary_n})"
            ),
            showarrow=False,
            font=dict(color="#374151", size=9),
            bgcolor="rgba(255,248,220,0.70)",
            row=2,
            col=1,
        )

        fig.update_layout(
            title=dict(text="How confident could I be?", x=0.5, xanchor="center"),
            template="plotly_white",
            autosize=True,
            height=720,
            font=dict(size=11),
            legend=dict(
                orientation="h",
                yanchor="top",
                y=-0.12,
                xanchor="left",
                x=0,
                font=dict(size=10),
            ),
            margin=dict(l=96, r=52, t=72, b=82),
        )
        fig.update_xaxes(
            title_text="Model confidence for S. pneumoniae (%)",
            range=[0, 100],
            tickmode="linear",
            tick0=0,
            dtick=5,
            row=3,
            col=1,
        )
        fig.update_xaxes(range=[0, 100], tickmode="linear", tick0=0, dtick=5, showticklabels=False, row=1, col=1)
        fig.update_xaxes(range=[0, 100], tickmode="linear", tick0=0, dtick=5, showticklabels=False, row=2, col=1)
        fig.update_yaxes(
            title_text="Radius Neighbours<br>confidence<br>(OODTest)",
            range=[0, 105],
            tickmode="linear",
            tick0=0,
            dtick=10,
            automargin=True,
            row=1,
            col=1,
        )
        fig.update_yaxes(title_text="Sample<br>density", showticklabels=False, automargin=True, row=2, col=1)
        fig.update_yaxes(
            title_text=f"Samples<br>map<br>(n={len(points_df)})",
            range=[0, 1.1],
            showticklabels=False,
            automargin=True,
            row=3,
            col=1,
        )
        return fig

    def _write_threshold_figure_html(self, fig: go.Figure) -> Path:
        html = self._plotly_html_fullscreen(fig)
        self.threshold_output_path.write_text(html, encoding="utf-8")
        return self.threshold_output_path

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
            config={"responsive": True, "displaylogo": False},
        )
        local_plotly = SpeciesIdScreen._local_plotly_script()
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
              background: white;
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
                  '<h3 style="margin-top:0">Threshold figure could not be loaded</h3>' +
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
