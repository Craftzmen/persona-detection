"""Phase 4 AI Persona Detection models.

This module trains and evaluates multiple classifiers on Phase 3 features and
provides persistence + inference helpers for downstream components.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
import pickle

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.base import ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from app.config import DEFAULT_RANDOM_SEED, PROCESSED_DATA_DIR
from app.utils.logging_utils import setup_logging

try:
    from xgboost import XGBClassifier
except ImportError:  # pragma: no cover - runtime dependency guard
    XGBClassifier = None


logger = setup_logging(__name__)


@dataclass(frozen=True)
class TrainingConfig:
    """Configuration for Phase 4 model training."""

    test_size: float = 0.2
    random_state: int = DEFAULT_RANDOM_SEED
    cv_folds: int = 5
    use_grid_search: bool = False
    top_n_features: int = 20


MODEL_NAMES = {
    "rf": "RandomForest",
    "svm": "SVM",
    "xgb": "XGBoost",
}


def _sigmoid(values: np.ndarray) -> np.ndarray:
    """Convert decision-function values to pseudo-probabilities."""

    clipped = np.clip(values, -20, 20)
    return 1.0 / (1.0 + np.exp(-clipped))


def _ensure_binary_labels(y: pd.Series | np.ndarray | list[Any]) -> np.ndarray:
    """Normalize labels into integer binary values {0, 1}."""

    series = pd.Series(y)
    if series.empty:
        raise ValueError("Label vector y is empty.")

    label_map = {
        "human": 0,
        "real": 0,
        "organic": 0,
        "0": 0,
        "ai": 1,
        "synthetic": 1,
        "bot": 1,
        "generated": 1,
        "1": 1,
    }

    if pd.api.types.is_numeric_dtype(series):
        normalized = pd.to_numeric(series, errors="coerce")
    else:
        normalized = (
            series.astype(str)
            .str.strip()
            .str.lower()
            .map(label_map)
        )

    if normalized.isna().any():
        invalid_count = int(normalized.isna().sum())
        raise ValueError(f"Found {invalid_count} invalid labels in y.")

    unique = sorted(normalized.astype(int).unique().tolist())
    if unique not in ([0], [1], [0, 1]):
        raise ValueError(f"Expected binary labels in y, got values: {unique}")

    return normalized.astype(int).to_numpy()


def _log_matrix_anomalies(X: pd.DataFrame) -> None:
    """Log suspicious quality issues in the feature matrix."""

    if X.empty:
        logger.warning("Feature matrix is empty.")
        return

    null_count = int(X.isna().sum().sum())
    inf_count = int(np.isinf(X.select_dtypes(include=[np.number]).to_numpy()).sum())
    duplicated_columns = int(X.columns.duplicated().sum())
    constant_columns = int((X.nunique(dropna=False) <= 1).sum())

    if null_count > 0:
        logger.warning("Feature matrix has %d null values.", null_count)
    if inf_count > 0:
        logger.warning("Feature matrix has %d infinite values.", inf_count)
    if duplicated_columns > 0:
        logger.warning("Feature matrix has %d duplicated columns.", duplicated_columns)
    if constant_columns > 0:
        logger.warning("Feature matrix has %d constant columns.", constant_columns)


def _prepare_features(
    X: pd.DataFrame | np.ndarray | sparse.spmatrix,
    feature_names: list[str] | None = None,
) -> tuple[pd.DataFrame | np.ndarray | sparse.spmatrix, list[str]]:
    """Prepare feature inputs and return model-ready matrix + feature names."""

    if sparse.issparse(X):
        names = feature_names or [f"f_{i}" for i in range(X.shape[1])]
        if len(names) != X.shape[1]:
            raise ValueError("feature_names length does not match sparse matrix columns.")
        return X, names

    if isinstance(X, pd.DataFrame):
        _log_matrix_anomalies(X)
        numeric = X.apply(pd.to_numeric, errors="coerce")
        numeric = numeric.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return numeric, numeric.columns.astype(str).tolist()

    if isinstance(X, np.ndarray):
        if X.ndim != 2:
            raise ValueError("X must be a 2-dimensional matrix.")
        matrix = np.nan_to_num(X.astype(float), copy=False)
        names = feature_names or [f"f_{i}" for i in range(matrix.shape[1])]
        if len(names) != matrix.shape[1]:
            raise ValueError("feature_names length does not match numpy matrix columns.")
        return matrix, names

    raise TypeError("X must be a pandas DataFrame, numpy array, or scipy sparse matrix.")


def _build_models(
    random_state: int,
    use_grid_search: bool,
) -> dict[str, tuple[ClassifierMixin, dict[str, list[Any]] | None]]:
    """Create model definitions and optional parameter grids."""

    models: dict[str, tuple[ClassifierMixin, dict[str, list[Any]] | None]] = {}

    rf = RandomForestClassifier(
        n_estimators=350,
        random_state=random_state,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )
    rf_grid = {
        "n_estimators": [200, 350],
        "max_depth": [None, 20],
        "min_samples_leaf": [1, 2],
    }
    models["rf"] = (rf, rf_grid if use_grid_search else None)

    svm = Pipeline(
        steps=[
            ("scaler", StandardScaler(with_mean=False)),
            (
                "model",
                SVC(
                    kernel="rbf",
                    class_weight="balanced",
                    probability=True,
                    random_state=random_state,
                ),
            ),
        ]
    )
    svm_grid = {
        "model__C": [1.0, 5.0],
        "model__gamma": ["scale", "auto"],
    }
    models["svm"] = (svm, svm_grid if use_grid_search else None)

    if XGBClassifier is not None:
        xgb = XGBClassifier(
            n_estimators=350,
            max_depth=6,
            learning_rate=0.08,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=random_state,
            n_jobs=-1,
            tree_method="hist",
        )
        xgb_grid = {
            "n_estimators": [250, 350],
            "max_depth": [4, 6],
            "learning_rate": [0.05, 0.08],
        }
        models["xgb"] = (xgb, xgb_grid if use_grid_search else None)
    else:
        logger.warning("xgboost is not installed. XGBoost model will be skipped.")

    return models


def _extract_probability(model: ClassifierMixin, X: Any) -> np.ndarray:
    """Get AI probability scores from a classifier."""

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X)
        if probabilities.ndim == 2 and probabilities.shape[1] > 1:
            return probabilities[:, 1]

    if hasattr(model, "decision_function"):
        decision_values = model.decision_function(X)
        return _sigmoid(np.asarray(decision_values, dtype=float))

    predictions = model.predict(X)
    return np.asarray(predictions, dtype=float)


def evaluate_model(
    model: ClassifierMixin,
    X_test: pd.DataFrame | np.ndarray | sparse.spmatrix,
    y_test: np.ndarray,
    model_name: str,
    plot_confusion: bool = False,
    output_dir: str | Path | None = None,
    interactive: bool = False,
) -> dict[str, Any]:
    """Evaluate a trained classifier on holdout data."""

    y_pred = model.predict(X_test)
    y_prob = _extract_probability(model, X_test)

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "y_pred": y_pred,
        "y_prob": y_prob,
    }

    if plot_confusion:
        visualize_confusion_matrix(
            metrics["confusion_matrix"],
            model_name=model_name,
            output_dir=output_dir,
            interactive=interactive,
        )

    return metrics


def train_model(
    X: pd.DataFrame | np.ndarray | sparse.spmatrix,
    y: pd.Series | np.ndarray | list[Any],
    config: TrainingConfig | None = None,
    plot_confusion: bool = False,
    output_dir: str | Path | None = None,
    interactive: bool = False,
) -> dict[str, Any]:
    """Train RF/SVM/XGBoost, evaluate, and select the best by F1-score."""

    cfg = config or TrainingConfig()
    y_array = _ensure_binary_labels(y)
    X_prepared, feature_names = _prepare_features(X)

    if len(y_array) < 4:
        raise ValueError("At least 4 samples are required for training.")

    X_train, X_test, y_train, y_test = train_test_split(
        X_prepared,
        y_array,
        test_size=cfg.test_size,
        random_state=cfg.random_state,
        stratify=y_array if len(np.unique(y_array)) > 1 else None,
    )

    cv_folds = max(2, min(cfg.cv_folds, int(np.bincount(y_train).min()) if len(np.unique(y_train)) > 1 else 2))
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=cfg.random_state)

    models = _build_models(cfg.random_state, cfg.use_grid_search)
    evaluation: dict[str, dict[str, Any]] = {}
    trained_models: dict[str, ClassifierMixin] = {}

    for key, (base_model, param_grid) in models.items():
        model_name = MODEL_NAMES[key]
        logger.info("Training %s...", model_name)

        if param_grid:
            search = GridSearchCV(
                estimator=base_model,
                param_grid=param_grid,
                scoring="f1",
                cv=cv,
                n_jobs=-1,
                refit=True,
            )
            search.fit(X_train, y_train)
            model = search.best_estimator_
            best_params = search.best_params_
        else:
            model = base_model
            model.fit(X_train, y_train)
            best_params = None

        cv_scores = cross_val_score(model, X_train, y_train, scoring="f1", cv=cv, n_jobs=-1)
        eval_result = evaluate_model(
            model,
            X_test,
            y_test,
            model_name=model_name,
            plot_confusion=plot_confusion,
            output_dir=output_dir,
            interactive=interactive,
        )
        eval_result["cv_f1_mean"] = float(np.mean(cv_scores))
        eval_result["cv_f1_std"] = float(np.std(cv_scores))
        eval_result["best_params"] = best_params

        evaluation[model_name] = eval_result
        trained_models[model_name] = model

    if not evaluation:
        raise RuntimeError("No models were trained. Check dependencies and input data.")

    best_model_name = max(evaluation, key=lambda name: evaluation[name]["f1"])
    best_model = trained_models[best_model_name]

    feature_importance = {
        "RandomForest": get_feature_importance(trained_models.get("RandomForest"), feature_names, cfg.top_n_features),
        "XGBoost": get_feature_importance(trained_models.get("XGBoost"), feature_names, cfg.top_n_features),
    }

    logger.info(
        "Best model selected: %s (F1=%.4f)",
        best_model_name,
        evaluation[best_model_name]["f1"],
    )

    return {
        "best_model_name": best_model_name,
        "best_model": best_model,
        "all_models": trained_models,
        "metrics": evaluation,
        "feature_names": feature_names,
        "feature_importance": feature_importance,
    }


def get_feature_importance(
    model: ClassifierMixin | None,
    feature_names: list[str],
    top_n: int = 20,
) -> pd.DataFrame:
    """Return top feature importance for RF/XGBoost; empty for unsupported models."""

    if model is None or not hasattr(model, "feature_importances_"):
        return pd.DataFrame(columns=["feature", "importance"])

    importances = np.asarray(getattr(model, "feature_importances_"), dtype=float)
    if importances.size != len(feature_names):
        logger.warning("Feature importance length does not match feature names.")
        return pd.DataFrame(columns=["feature", "importance"])

    frame = pd.DataFrame({"feature": feature_names, "importance": importances})
    frame = frame.sort_values("importance", ascending=False).head(max(1, top_n)).reset_index(drop=True)
    return frame


def visualize_confusion_matrix(
    matrix: np.ndarray,
    model_name: str,
    output_dir: str | Path | None = None,
    interactive: bool = False,
) -> None:
    """Visualize confusion matrix in static matplotlib or optional interactive Plotly."""

    out_dir = Path(output_dir) if output_dir else PROCESSED_DATA_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    if interactive:
        try:
            import plotly.express as px

            fig = px.imshow(
                matrix,
                text_auto=True,
                color_continuous_scale="Blues",
                labels={"x": "Predicted", "y": "Actual"},
                x=["Human", "AI"],
                y=["Human", "AI"],
                title=f"Confusion Matrix - {model_name}",
            )
            fig.write_html(out_dir / f"confusion_matrix_{model_name.lower()}.html")
            return
        except ImportError:
            logger.warning("plotly not installed; falling back to matplotlib confusion matrix.")

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not installed; cannot visualize confusion matrix.")
        return

    fig, ax = plt.subplots(figsize=(5, 4.5))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_title(f"Confusion Matrix - {model_name}")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticks([0, 1], labels=["Human", "AI"])
    ax.set_yticks([0, 1], labels=["Human", "AI"])

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center")

    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(out_dir / f"confusion_matrix_{model_name.lower()}.png", dpi=140)
    plt.close(fig)


def visualize_feature_importance(
    importance_df: pd.DataFrame,
    model_name: str,
    output_dir: str | Path | None = None,
    interactive: bool = False,
) -> None:
    """Render top feature importance chart for RF/XGBoost."""

    if importance_df.empty:
        logger.warning("No feature importance data found for %s.", model_name)
        return

    out_dir = Path(output_dir) if output_dir else PROCESSED_DATA_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    if interactive:
        try:
            import plotly.express as px

            fig = px.bar(
                importance_df.sort_values("importance", ascending=True),
                x="importance",
                y="feature",
                orientation="h",
                title=f"Top Feature Importance - {model_name}",
            )
            fig.write_html(out_dir / f"feature_importance_{model_name.lower()}.html")
            return
        except ImportError:
            logger.warning("plotly not installed; falling back to matplotlib feature plot.")

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not installed; cannot visualize feature importance.")
        return

    frame = importance_df.sort_values("importance", ascending=True)
    fig, ax = plt.subplots(figsize=(8.5, max(4.0, 0.32 * len(frame))))
    ax.barh(frame["feature"], frame["importance"], color="#4e79a7")
    ax.set_title(f"Top Feature Importance - {model_name}")
    ax.set_xlabel("Importance")
    fig.tight_layout()
    fig.savefig(out_dir / f"feature_importance_{model_name.lower()}.png", dpi=140)
    plt.close(fig)


def save_model(
    model: ClassifierMixin,
    file_path: str | Path,
    feature_names: Iterable[str],
    model_name: str,
) -> Path:
    """Persist a model bundle as .pkl for later inference."""

    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "model": model,
        "model_name": model_name,
        "feature_names": list(feature_names),
        "saved_at_utc": datetime.now(timezone.utc).isoformat(),
        "label_mapping": {0: "Human", 1: "AI"},
    }

    with path.open("wb") as file_handle:
        pickle.dump(payload, file_handle)

    logger.info("Saved model bundle to %s", path.resolve())
    return path


def load_model(file_path: str | Path) -> dict[str, Any]:
    """Load a saved model bundle from disk."""

    path = Path(file_path)
    with path.open("rb") as file_handle:
        payload = pickle.load(file_handle)

    if not isinstance(payload, dict) or "model" not in payload:
        raise ValueError("Invalid model payload format.")

    return payload


def predict(
    model_bundle: dict[str, Any],
    X_new: pd.DataFrame | np.ndarray | sparse.spmatrix,
    usernames: Iterable[str] | None = None,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """Predict class label and synthetic score for new accounts.

    The output schema is stable for downstream Phase 5 attribution usage.
    """

    model = model_bundle["model"]
    feature_names = model_bundle.get("feature_names", [])
    model_name = model_bundle.get("model_name", "UnknownModel")

    if isinstance(X_new, pd.DataFrame) and feature_names:
        aligned = X_new.copy()
        for col in feature_names:
            if col not in aligned.columns:
                aligned[col] = 0.0
        aligned = aligned[feature_names]
        matrix, _ = _prepare_features(aligned, feature_names=feature_names)
    else:
        matrix, _ = _prepare_features(X_new, feature_names=feature_names if feature_names else None)

    scores = _extract_probability(model, matrix)
    predictions = (scores >= threshold).astype(int)

    if usernames is None:
        usernames = [f"account_{idx}" for idx in range(len(predictions))]

    result = pd.DataFrame(
        {
            "username": list(usernames),
            "predicted_label": predictions,
            "classification": np.where(predictions == 1, "AI", "Human"),
            "synthetic_score": scores,
            "model_name": model_name,
        }
    )

    return result


def predict_usernames_from_feature_frame(
    model_bundle: dict[str, Any],
    feature_frame: pd.DataFrame,
    username_col: str = "username",
    threshold: float = 0.5,
) -> pd.DataFrame:
    """Simple API-like helper for real-time username prediction from a feature row."""

    if username_col not in feature_frame.columns:
        raise ValueError(f"Feature frame must include '{username_col}' column.")

    usernames = feature_frame[username_col].astype(str).tolist()
    X_new = feature_frame.drop(columns=[username_col])
    return predict(model_bundle, X_new=X_new, usernames=usernames, threshold=threshold)
