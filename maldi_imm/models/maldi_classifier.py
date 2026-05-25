import warnings
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.utils.multiclass import type_of_target
from lightgbm import LGBMClassifier

class MaldiClassifier:
    """
    Unified MALDI-TOF classifier wrapper with DRIAMS-style configurations.

    Supports:
      - 'lightgbm' (LGBMClassifier)
      - 'mlp'      (sklearn MLPClassifier)

    Parameters
    ----------
    model : {'lightgbm','mlp'}
        Which base model to use.
    random_state : int or None
        Random seed for reproducibility. If None, a warning is raised.
    class_weight : dict, 'balanced', or None
        Passed to LightGBM (ignored by MLP). Useful for imbalanced data.
    n_jobs : int
        Parallelism for LightGBM and GridSearchCV (where applicable).
    max_iter_mlp : int
        Max iterations for the MLP solver.
    param_grid_override : dict or None
        Optional grid to override the default DRIAMS-style grids.
        Keys must match pipeline param names (e.g., 'lightgbm__n_estimators').
    """

    def __init__(
        self,
        model: str = "lightgbm",
        random_state: int | None = 42,
        class_weight=None,
        n_jobs: int = -1,
        max_iter_mlp: int = 500,
        param_grid_override: dict | None = None,
    ):
        self.model_name = model.lower()
        self.random_state = random_state
        self.class_weight = class_weight
        self.n_jobs = n_jobs
        self.max_iter_mlp = max_iter_mlp
        self.param_grid_override = param_grid_override

        if self.random_state is None:
            warnings.warn("`random_state` is not set; results may be non-reproducible.")

        self.pipeline_: Pipeline | None = None
        self.param_grid_: dict | None = None
        self.search_: GridSearchCV | None = None
        self.best_params_: dict | None = None

        self._build_pipeline_and_grid()

    def fit(
        self,
        X,
        y,
        use_grid_search: bool = True,
        cv: int = 5,
        scoring: str | None = None,
        refit: bool = True,
        verbose: int = 0,
        sample_weight=None,
    ):
        """
        Fit the classifier. Optionally run GridSearchCV.

        scoring:
          - If None, auto-selects ROC AUC variant:
              * binary: 'roc_auc'
              * multiclass: 'roc_auc_ovr'
          - Or pass any sklearn-compatible scoring string / callable.
        """
        if scoring is None:
            scoring = self._default_scoring(y)

        if use_grid_search:
            self.search_ = GridSearchCV(
                estimator=self.pipeline_,
                param_grid=self.param_grid_,
                scoring=scoring,
                cv=cv,
                n_jobs=self.n_jobs,
                refit=refit,
                verbose=verbose,
            )
            self.search_.fit(X, y, **(self._fit_kwargs(sample_weight)))
            self.best_params_ = self.search_.best_params_
            # Keep a ref to the refit estimator (Pipeline) for direct predict/predict_proba
            self.pipeline_ = self.search_.best_estimator_
        else:
            self.pipeline_.fit(X, y, **self._fit_kwargs(sample_weight))

        return self

    def predict(self, X):
        self._ensure_fitted()
        return self.pipeline_.predict(X)

    def predict_proba(self, X):
        self._ensure_fitted()
        # Both LGBMClassifier and MLPClassifier support predict_proba
        return self.pipeline_.predict_proba(X)

    def get_pipeline(self) -> Pipeline:
        """Return the current sklearn Pipeline (best refit if grid-searched)."""
        self._ensure_initialized()
        return self.pipeline_

    def get_param_grid(self) -> dict:
        """Return the hyperparameter grid currently in use."""
        self._ensure_initialized()
        return self.param_grid_

    # ---------- Internal helpers ----------

    def _build_pipeline_and_grid(self):
        if self.model_name == "lightgbm":
            if LGBMClassifier is None:
                raise ImportError("lightgbm is not installed. Please `pip install lightgbm`.")
            lightgbm = LGBMClassifier(
                class_weight=self.class_weight,
                n_jobs=self.n_jobs,
                random_state=self.random_state,
            )
            self.pipeline_ = Pipeline(steps=[("lightgbm", lightgbm)])

            default_grid = {
                "lightgbm__boosting_type": ["gbdt", "dart", "goss", "rf"],
                "lightgbm__n_estimators": [25, 50, 100, 200],
                # DRIAMS-style wide sweep (note: >1 can be unstable; included to match the paper)
                "lightgbm__learning_rate": 10.0 ** np.arange(-3, 4),
            }

        elif self.model_name == "mlp":
            mlp = MLPClassifier(
                max_iter=self.max_iter_mlp,
                random_state=self.random_state,
                solver="adam",
            )
            self.pipeline_ = Pipeline(steps=[("scaler", None), ("mlp", mlp)])

            default_grid = {
                "scaler": ["passthrough", StandardScaler()],
                "mlp__hidden_layer_sizes": [
                    (512, 256, 128),
                    (512, 128, 64),
                    (256, 64),
                    (256, 128),
                ],
                "mlp__activation": ["relu"],
                "mlp__alpha": [1e-4],
            }
        else:
            raise RuntimeError(f'No pipeline or configuration for "{self.model_name}" available.')

        # Allow user overrides
        if self.param_grid_override:
            default_grid.update(self.param_grid_override)

        self.param_grid_ = default_grid

    def _default_scoring(self, y) -> str:
        t = type_of_target(y)
        # 'binary', 'multiclass', 'multilabel-indicator', etc.
        if t == "binary":
            return "roc_auc"
        elif t in ("multiclass", "multiclass-multioutput"):
            return "roc_auc_ovr"
        # Fallback—let sklearn estimator's default .score be used
        warnings.warn(
            f"Unrecognized/complex target type '{t}'. Falling back to estimator's default scoring."
        )
        return None

    def _fit_kwargs(self, sample_weight):
        # Pipeline.fit will forward sample_weight when supported by final estimator.
        # Both LGBMClassifier and MLPClassifier accept sample_weight.
        return {"sample_weight": sample_weight} if sample_weight is not None else {}

    def _ensure_initialized(self):
        if self.pipeline_ is None or self.param_grid_ is None:
            raise RuntimeError("Pipeline not initialized. This should not happen.")

    def _ensure_fitted(self):
        if self.pipeline_ is None:
            raise RuntimeError("Model not initialized.")
        # Rough check: final step has been fitted if it has attribute 'classes_'
        final_est = self.pipeline_.steps[-1][1]
        if not hasattr(final_est, "classes_"):
            raise RuntimeError("Call .fit() before prediction.")

    def __repr__(self):
        name = self.model_name
        return f"MaldiClassifier(model='{name}', random_state={self.random_state})"
