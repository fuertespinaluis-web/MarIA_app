from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import joblib
import plotly.graph_objects as go

from PySide6.QtCore import Qt, QUrl
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QSizePolicy
)
from PySide6.QtWebEngineWidgets import QWebEngineView


class NsAnalysisScreen(QWidget):
    """
    PCA overlay (TRAIN + NEW) rendered with Plotly inside QWebEngineView.

    Fixes:
      - Plot uses full available space (no cropping)
      - Responsive HTML/CSS + plotly config responsive=True
      - No fixed height=650 that creates internal scroll/cut
    """

    def __init__(self, pca_dir: Optional[Path] = None):
        super().__init__()

        default_pca_dir = Path(__file__).resolve().parent / "PCA" / "PCA_DA_train_only" / "DA_x1p5"
        self.pca_dir = Path(pca_dir) if pca_dir is not None else default_pca_dir

        self.scaler = None
        self.pca = None
        self.meta = None
        self.train_df: Optional[pd.DataFrame] = None

        self.X_new: Optional[np.ndarray] = None
        self.new_ids: list[str] = []

        root = QVBoxLayout(self)
        root.setContentsMargins(18, 16, 18, 16)   # tighter, more room for plot
        root.setSpacing(10)

        title = QLabel("NON SUPERVISED ANALYSIS — PCA overlay")
        title.setAlignment(Qt.AlignLeft)
        title.setObjectName("Title")

        self.status = QLabel("")
        self.status.setObjectName("Status")

        btnrow = QHBoxLayout()
        btnrow.setSpacing(10)

        self.btn_reload = QPushButton("Reload PCA assets")
        self.btn_reload.clicked.connect(self._load_assets)

        self.btn_render2d = QPushButton("Render PCA 2D")
        self.btn_render2d.clicked.connect(self.render_2d)
        self.btn_render2d.setEnabled(False)

        self.btn_render3d = QPushButton("Render PCA 3D")
        self.btn_render3d.clicked.connect(self.render_3d)
        self.btn_render3d.setEnabled(False)

        btnrow.addWidget(self.btn_reload)
        btnrow.addWidget(self.btn_render2d)
        btnrow.addWidget(self.btn_render3d)
        btnrow.addStretch(1)

        self.web = QWebEngineView()
        self.web.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.web.setMinimumHeight(620)  # you can tune (580–750)
        # Optional: slightly zoom out so everything fits comfortably
        # self.web.setZoomFactor(0.95)

        root.addWidget(title, 0)
        root.addWidget(self.status, 0)
        root.addLayout(btnrow)
        root.addWidget(self.web, 1)  # <-- important stretch

        self.setStyleSheet("""
        #Title { font-size: 18px; font-weight: 900; color: #0F172A; }
        #Status { color: #334155; }
        QPushButton {
            background: rgba(15, 23, 42, 0.06);
            border: 1px solid rgba(15, 23, 42, 0.20);
            color: #0F172A;
            font-weight: 800;
            border-radius: 10px;
            padding: 8px 14px;
        }
        QPushButton:disabled { opacity: 0.5; }
        """)

        self._load_assets()

    # -------------------------
    # Public API (called from MainUI)
    # -------------------------
    def set_new_data(self, X_new: np.ndarray, pseudo_ids: Optional[list[str]] = None):
        X_new = np.asarray(X_new, dtype=float)
        if X_new.ndim != 2:
            self.status.setText(f"❌ X_new must be 2D. Got {X_new.shape}")
            return

        self.X_new = X_new

        if pseudo_ids is None or len(pseudo_ids) != X_new.shape[0]:
            self.new_ids = [f"new_{i:04d}" for i in range(X_new.shape[0])]
        else:
            self.new_ids = [str(x) for x in pseudo_ids]

        if self.train_df is not None and self.pca is not None and self.scaler is not None:
            self.btn_render2d.setEnabled(True)
            self.btn_render3d.setEnabled(True)

        self.status.setText(f"✅ Received {X_new.shape[0]} new spectra. Ready to overlay.")

    # -------------------------
    # Load PCA assets
    # -------------------------
    def _load_assets(self):
        scaler_path = self.pca_dir / "pca_scaler.joblib"
        pca_path = self.pca_dir / "pca_model.joblib"
        coords_path = self.pca_dir / "pca_train_coords.csv"
        meta_path = self.pca_dir / "pca_metadata.joblib"

        self.btn_render2d.setEnabled(False)
        self.btn_render3d.setEnabled(False)

        if not pca_path.exists() or not scaler_path.exists() or not coords_path.exists():
            self.status.setText(
                "❌ Missing PCA files in screens/PCA.\n"
                "Expected:\n"
                " - pca_model.joblib\n"
                " - pca_scaler.joblib\n"
                " - pca_train_coords.csv"
            )
            return

        try:
            self.scaler = joblib.load(scaler_path)
            self.pca = joblib.load(pca_path)
        except Exception as e:
            self.status.setText(f"❌ Error loading scaler/pca: {e}")
            return

        try:
            self.train_df = pd.read_csv(coords_path)
        except Exception as e:
            self.status.setText(f"❌ Error loading pca_train_coords.csv: {e}")
            return

        self.meta = None
        if meta_path.exists():
            try:
                self.meta = joblib.load(meta_path)
            except Exception:
                self.meta = None

        required = {"pseudo_id", "species", "PC1", "PC2", "PC3"}
        miss = required - set(self.train_df.columns)
        if miss:
            self.status.setText(f"❌ pca_train_coords.csv missing columns: {sorted(miss)}")
            self.train_df = None
            return

        if self.X_new is not None:
            self.btn_render2d.setEnabled(True)
            self.btn_render3d.setEnabled(True)

        self.status.setText(
            f"✅ PCA assets loaded: {len(self.train_df)} TRAIN points with species. "
            f"Waiting for new spectra to overlay."
        )

    # -------------------------
    # Projection
    # -------------------------
    def _project_new(self) -> pd.DataFrame:
        if self.X_new is None:
            raise RuntimeError("No new spectra provided.")
        if self.scaler is None or self.pca is None:
            raise RuntimeError("PCA assets not loaded.")

        expected = getattr(self.pca, "n_features_in_", None)
        if expected is not None and self.X_new.shape[1] != expected:
            raise ValueError(f"Feature mismatch: X_new has {self.X_new.shape[1]} features, PCA expects {expected}.")

        X_t = self.scaler.transform(self.X_new) if self.scaler is not None else self.X_new
        Z = self.pca.transform(X_t)

        new_df = pd.DataFrame({
            "pseudo_id": self.new_ids,
            "species": "NEW",
            "PC1": Z[:, 0],
            "PC2": Z[:, 1],
            "PC3": Z[:, 2],
        })

        try:
            new_df.to_csv(self.pca_dir / "pca_new_coords.csv", index=False)
        except Exception:
            pass

        return new_df

    def _title_with_evr(self, base: str) -> str:
        if isinstance(self.meta, dict) and "explained_variance_ratio" in self.meta:
            evr = self.meta["explained_variance_ratio"]
            if isinstance(evr, (list, tuple)) and len(evr) >= 3:
                return f"{base} — EVR: PC1={evr[0]:.3f}, PC2={evr[1]:.3f}, PC3={evr[2]:.3f}"
        return base

    # -------------------------
    # HTML helper: make Plotly responsive and fill container
    # -------------------------
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
        """
        Wrap Plotly HTML so it always fills the QWebEngineView height/width.
        No fixed pixel height -> avoids cropping/scroll.
        """
        html = fig.to_html(
            include_plotlyjs=False,
            full_html=False,
            config={"responsive": True, "displaylogo": False},
        )
        local_plotly = NsAnalysisScreen._local_plotly_script()

        return f"""
        <!doctype html>
        <html>
        <head>
          <meta charset="utf-8"/>
          <meta name="viewport" content="width=device-width, initial-scale=1"/>
          {local_plotly}
          <style>
            html, body {{
              height: 100%;
              width: 100%;
              margin: 0;
              padding: 0;
              overflow: hidden;
              background: white;
              font-family: Arial, sans-serif;
            }}
            #wrap {{
              height: 100%;
              width: 100%;
            }}
          </style>
        </head>
        <body>
          <div id="wrap">
            {html}
          </div>
          <script>
            // Force one resize after load (helps inside QtWebEngine)
            window.addEventListener('load', function() {{
              if (!window.Plotly) {{
                document.body.innerHTML =
                  '<div style="font-family:Arial,sans-serif;padding:24px;color:#0f172a">' +
                  '<h3 style="margin-top:0">PCA figure could not be loaded</h3>' +
                  '<p>Plotly is not available in this Python environment.</p>' +
                  '</div>';
                return;
              }}
              window.dispatchEvent(new Event('resize'));
              setTimeout(() => window.dispatchEvent(new Event('resize')), 150);
            }});
          </script>
        </body>
        </html>
        """

    def _write_plot_html(self, filename: str, html: str) -> Path:
        output_path = self.pca_dir / filename
        output_path.write_text(html, encoding="utf-8")
        return output_path

    # -------------------------
    # Render 3D
    # -------------------------
    def render_3d(self):
        if self.train_df is None:
            self.status.setText("❌ Train PCA coords not loaded.")
            return
        if self.X_new is None:
            self.status.setText("❌ No new spectra yet. Go to Upload → Preprocess first.")
            return

        try:
            new_df = self._project_new()
        except Exception as e:
            self.status.setText(f"❌ Projection failed: {e}")
            return

        train = self.train_df.copy()
        fig = go.Figure()

        for sp in sorted(train["species"].astype(str).unique()):
            sub = train[train["species"].astype(str) == sp]
            fig.add_trace(go.Scatter3d(
                x=sub["PC1"], y=sub["PC2"], z=sub["PC3"],
                mode="markers",
                name=f"TRAIN — {sp}",
                marker=dict(size=2, opacity=0.35),
                text=sub["pseudo_id"],
                hovertemplate="TRAIN<br>species=%{name}<br>pseudo_id=%{text}<br>"
                              "PC1=%{x:.3f}<br>PC2=%{y:.3f}<br>PC3=%{z:.3f}<extra></extra>"
            ))

        fig.add_trace(go.Scatter3d(
            x=new_df["PC1"], y=new_df["PC2"], z=new_df["PC3"],
            mode="markers+text",
            name="NEW",
            marker=dict(size=6, opacity=1.0, color="green"),
            textposition="top center",
            hovertemplate="NEW<br>pseudo_id=%{text}<br>"
                          "PC1=%{x:.3f}<br>PC2=%{y:.3f}<br>PC3=%{z:.3f}<extra></extra>"
        ))

        fig.update_layout(
            title=self._title_with_evr("PCA 3D — TRAIN (species) + NEW (green)"),
            scene=dict(
                xaxis_title="PC1",
                yaxis_title="PC2",
                zaxis_title="PC3",
            ),
            margin=dict(l=0, r=0, t=45, b=0),
            legend=dict(itemsizing="constant"),
            template="plotly_white",
            autosize=True,
        )

        html = self._plotly_html_fullscreen(fig)
        try:
            output_path = self._write_plot_html("pca_overlay_last.html", html)
            self.web.setUrl(QUrl.fromLocalFile(str(output_path.resolve())))
        except Exception as e:
            self.status.setText(f"❌ Could not write/load PCA 3D HTML: {e}")
            return

        self.status.setText("✅ Rendered PCA 3D overlay (NEW in green).")

    # -------------------------
    # Render 2D (PC1 vs PC2)
    # -------------------------
    def render_2d(self):
        if self.train_df is None:
            self.status.setText("❌ Train PCA coords not loaded.")
            return
        if self.X_new is None:
            self.status.setText("❌ No new spectra yet. Go to Upload → Preprocess first.")
            return

        try:
            new_df = self._project_new()
        except Exception as e:
            self.status.setText(f"❌ Projection failed: {e}")
            return

        train = self.train_df.copy()
        fig = go.Figure()

        for sp in sorted(train["species"].astype(str).unique()):
            sub = train[train["species"].astype(str) == sp]
            fig.add_trace(go.Scatter(
                x=sub["PC1"], y=sub["PC2"],
                mode="markers",
                name=f"TRAIN — {sp}",
                marker=dict(size=5, opacity=0.35),
                text=sub["pseudo_id"],
                hovertemplate="TRAIN<br>species=%{name}<br>pseudo_id=%{text}<br>"
                              "PC1=%{x:.3f}<br>PC2=%{y:.3f}<extra></extra>"
            ))

        fig.add_trace(go.Scatter(
            x=new_df["PC1"], y=new_df["PC2"],
            mode="markers+text",
            name="NEW",
            marker=dict(size=10, opacity=1.0, color="green"),
            textposition="top center",
            hovertemplate="NEW<br>pseudo_id=%{text}<br>"
                          "PC1=%{x:.3f}<br>PC2=%{y:.3f}<extra></extra>"
        ))

        fig.update_layout(
            title=self._title_with_evr("PCA 2D (PC1 vs PC2) — TRAIN + NEW"),
            xaxis_title="PC1",
            yaxis_title="PC2",
            margin=dict(l=25, r=25, t=55, b=25),
            legend=dict(itemsizing="constant"),
            template="plotly_white",
            autosize=True,
        )

        html = self._plotly_html_fullscreen(fig)
        try:
            output_path = self._write_plot_html("pca_overlay_last_2d.html", html)
            self.web.setUrl(QUrl.fromLocalFile(str(output_path.resolve())))
        except Exception as e:
            self.status.setText(f"❌ Could not write/load PCA 2D HTML: {e}")
            return

        self.status.setText("✅ Rendered PCA 2D overlay (NEW in green).")
