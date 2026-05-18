"""
Evaluation via 5-fold Stratified Cross-Validation (Out-of-Fold predictions).

Training uses ALL available data. Evaluation re-trains the same model
architecture in each CV fold and collects held-out (out-of-fold) predictions,
giving an honest performance estimate without a separate test split.

- RAW type  : plain Pipeline(StandardScaler + model) with balanced sample weights.
- BALANCED type: resampling inside each fold (ADASYN/SMOTE from model config).
"""
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import pickle
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from imblearn.over_sampling import ADASYN, SMOTE

from src.config import (
    PROCESSED_DIR,
    MODELS_DIR,
    RESULTS_DIR,
    DEFAULT_MODELS,
    TARGET_COLUMN,
    RANDOM_STATE,
)
from src.models import get_evaluation_config
from src.io_utils import read_csv, ensure_dir, write_csv


def _load_all_data() -> tuple[pd.DataFrame, pd.Series]:
    """Load the complete (pre-split) training dataset."""
    csv_dir = PROCESSED_DIR / "csv"
    X_path = csv_dir / "X_train_raw.csv"
    y_path = csv_dir / "y_train_raw.csv"

    if not X_path.exists() or not y_path.exists():
        raise FileNotFoundError("Training datasets not found. Run preprocessing first.")

    X = read_csv(X_path)
    y_df = read_csv(y_path)
    if TARGET_COLUMN not in y_df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' not present in y_train_raw.csv")
    y = y_df[TARGET_COLUMN]
    return X, y


def _load_model(model_name: str, dataset_type: str):
    model_path = MODELS_DIR / model_name / dataset_type / "model.pkl"
    if not model_path.exists():
        return None
    with model_path.open("rb") as f:
        return pickle.load(f)


def _clone_model_from_pipeline(pipeline, model_name: str):
    """
    Extract the fitted estimator from a saved Pipeline and return an
    unfitted clone with the same hyperparameters.
    """
    from src.models import get_model

    saved_model = None
    if hasattr(pipeline, "named_steps"):
        saved_model = pipeline.named_steps.get("model")
    elif hasattr(pipeline, "steps"):
        for name, step in pipeline.steps:
            if name == "model":
                saved_model = step
                break

    if saved_model is None:
        return get_model(model_name, random_state=RANDOM_STATE, class_weight="balanced")

    try:
        params = saved_model.get_params()
        new_model = saved_model.__class__(**params)
        return new_model
    except Exception:
        return get_model(model_name, random_state=RANDOM_STATE, class_weight="balanced")


def _cv_predict(
    model_instance,
    X: np.ndarray,
    y: np.ndarray,
    dataset_type: str,
    eval_config: dict,
    n_splits: int = 5,
):
    """
    Run stratified k-fold CV and return out-of-fold predictions.

    Returns
    -------
    y_pred  : ndarray (n_samples,)
    y_proba : ndarray (n_samples, n_classes) or None
    """
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    n_classes = len(np.unique(y))

    y_pred = np.zeros(len(y), dtype=int)
    y_proba = np.zeros((len(y), n_classes))
    has_proba = hasattr(model_instance, "predict_proba")

    threshold = eval_config.get("threshold") if eval_config.get("apply_threshold", False) else None
    resampler_name = eval_config.get("resampler", "smote")

    for train_idx, val_idx in cv.split(X, y):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr = y[train_idx]

        if dataset_type == "balanced":
            n_neighbors = min(5, min(np.bincount(y_tr)) - 1) if min(np.bincount(y_tr)) > 1 else 1
            if resampler_name == "adasyn":
                resampler = ADASYN(
                    sampling_strategy="auto",
                    random_state=RANDOM_STATE,
                    n_neighbors=n_neighbors,
                )
            else:
                resampler = SMOTE(random_state=RANDOM_STATE, k_neighbors=n_neighbors)
            try:
                X_res, y_res = resampler.fit_resample(X_tr, y_tr)
            except Exception:
                X_res, y_res = X_tr, y_tr

            from sklearn.base import clone as sk_clone
            model_clone = sk_clone(model_instance)
            if resampler_name == "adasyn" and "class_weight" in model_clone.get_params():
                model_clone.set_params(class_weight=None)
            estimator = Pipeline([
                ("scaler", StandardScaler()),
                ("model", model_clone),
            ])
            estimator.fit(X_res, y_res)
        else:
            sample_weights = compute_sample_weight("balanced", y_tr)
            from sklearn.base import clone as sk_clone
            estimator = Pipeline([
                ("scaler", StandardScaler()),
                ("model", sk_clone(model_instance)),
            ])
            try:
                estimator.fit(X_tr, y_tr, model__sample_weight=sample_weights)
            except TypeError:
                estimator.fit(X_tr, y_tr)

        if has_proba:
            try:
                y_proba[val_idx] = estimator.predict_proba(X_val)
            except Exception:
                has_proba = False
        if threshold is not None and has_proba:
            y_pred[val_idx] = (y_proba[val_idx, 1] >= float(threshold)).astype(int)
        else:
            y_pred[val_idx] = estimator.predict(X_val)

    return y_pred, (y_proba if has_proba else None)


def run_evaluation(
    model_names: Optional[Iterable[str]] = None,
    dataset_types: Iterable[str] = ("raw", "balanced"),
    n_splits: int = 5,
) -> Path:
    if model_names is None:
        model_names = DEFAULT_MODELS

    X_df, y_series = _load_all_data()
    X = X_df.values
    y = y_series.values.astype(int)
    classes = np.unique(y)

    rows: list[dict] = []

    for dataset_type in dataset_types:
        for model_name in model_names:
            saved_pipeline = _load_model(model_name, dataset_type)
            if saved_pipeline is None:
                print(f"  Skipping {model_name} [{dataset_type}]: model not found")
                continue

            model_instance = _clone_model_from_pipeline(saved_pipeline, model_name)
            eval_config = get_evaluation_config(model_name)
            model_splits = eval_config.get("cv_folds", n_splits)
            print(f"  Evaluating {model_name} [{dataset_type}] via {model_splits}-fold CV...")

            y_pred, y_proba = _cv_predict(
                model_instance,
                X,
                y,
                dataset_type,
                eval_config,
                model_splits,
            )

            acc = accuracy_score(y, y_pred)
            prec_macro = precision_score(y, y_pred, average="macro", zero_division=0)
            rec_macro = recall_score(y, y_pred, average="macro", zero_division=0)
            f1_macro = f1_score(y, y_pred, average="macro", zero_division=0)
            kappa = cohen_kappa_score(y, y_pred)

            prec_c, rec_c, f1_c, sup_c = precision_recall_fscore_support(
                y, y_pred, average=None, zero_division=0
            )

            roc_auc_macro = float("nan")
            roc_auc_per_class = None
            proba_per_class = {}

            if y_proba is not None:
                try:
                    roc_auc_per_class = []
                    for i, cls in enumerate(classes):
                        try:
                            auc_i = roc_auc_score((y == cls).astype(int), y_proba[:, i])
                        except ValueError:
                            auc_i = float("nan")
                        roc_auc_per_class.append((int(cls), float(auc_i)))
                    roc_auc_macro = float(np.nanmean([v for _, v in roc_auc_per_class]))
                except Exception:
                    roc_auc_macro = float("nan")

                for i, cls in enumerate(classes):
                    proba_per_class[int(cls)] = float(np.mean(y_proba[:, i]))

            row = {
                "model": model_name,
                "dataset_type": dataset_type,
                "accuracy": acc,
                "precision_macro": prec_macro,
                "recall_macro": rec_macro,
                "f1_macro": f1_macro,
                "kappa": kappa,
                "roc_auc_ovr_macro": roc_auc_macro,
                "avg_proba_class_0": proba_per_class.get(0, float("nan")),
                "avg_proba_class_1": proba_per_class.get(1, float("nan")),
            }
            rows.append(row)

            results_dir = RESULTS_DIR / model_name
            ensure_dir(results_dir)

            cm = confusion_matrix(y, y_pred, labels=list(classes))
            n_classes_eval = cm.shape[0]

            sensitivity, specificity = [], []
            for i in range(n_classes_eval):
                tp = cm[i, i]
                fn = cm[i, :].sum() - tp
                fp = cm[:, i].sum() - tp
                tn = cm.sum() - (tp + fn + fp)
                sensitivity.append(tp / (tp + fn) if (tp + fn) > 0 else 0.0)
                specificity.append(tn / (tn + fp) if (tn + fp) > 0 else 0.0)

            df_per_class = pd.DataFrame({
                "class": list(classes),
                "precision": prec_c,
                "recall": rec_c,
                "f1_score": f1_c,
                "sensitivity": sensitivity,
                "specificity": specificity,
                "roc_auc": [v for _, v in roc_auc_per_class] if roc_auc_per_class else [float("nan")] * len(classes),
                "support": sup_c,
            })
            write_csv(df_per_class, results_dir / f"per_class_metrics_{dataset_type}.csv")

            cm_df = pd.DataFrame(cm, index=list(classes), columns=list(classes))
            write_csv(cm_df, results_dir / f"confusion_matrix_{dataset_type}.csv")

            if roc_auc_per_class is not None:
                df_roc = pd.DataFrame({
                    "class": [c for c, _ in roc_auc_per_class],
                    "roc_auc_ovr": [v for _, v in roc_auc_per_class],
                })
                write_csv(df_roc, results_dir / f"roc_auc_per_class_{dataset_type}.csv")

    if not rows:
        raise RuntimeError("No models found to evaluate.")

    df = pd.DataFrame(rows)
    ensure_dir(RESULTS_DIR)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"evaluation_summary_{timestamp}.csv"
    write_csv(df, out_path)
    write_csv(df, RESULTS_DIR / "evaluation_summary_latest.csv")

    return out_path
