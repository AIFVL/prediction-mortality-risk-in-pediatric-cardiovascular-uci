"""
Per-model feature importance from each trained pipeline.

Uses native model signals when available and permutation importance as fallback.
"""
from __future__ import annotations

import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from src.config import PROCESSED_DIR, RESULTS_DIR, MODELS_DIR, RANDOM_STATE
from src.io_utils import save_json, load_json, ensure_dir, read_csv
from src.utils.data_loader import load_train_split


TOP_N_DEFAULT = 15

METHOD_LABELS = {
    "feature_importances": "Importancia nativa del modelo",
    "coefficient_magnitude": "Magnitud de coeficientes",
    "permutation_importance": "Permutation importance",
}


def _model_key(model_name: str, dataset_type: str) -> str:
    return f"{model_name}__{dataset_type}"


def _fi_path(model_name: str, dataset_type: str) -> Path:
    return RESULTS_DIR / model_name / f"feature_importance_{dataset_type}.json"


def _load_pipeline(model_name: str, dataset_type: str):
    path = MODELS_DIR / model_name / dataset_type / "model.pkl"
    if not path.exists():
        return None
    with path.open("rb") as f:
        return pickle.load(f)


def _estimator_from_pipeline(pipeline):
    if hasattr(pipeline, "named_steps"):
        return pipeline.named_steps.get("model")
    if hasattr(pipeline, "steps"):
        for name, step in pipeline.steps:
            if name == "model":
                return step
    return pipeline


def _importance_method_for_model(model_name: str, estimator) -> str:
    if hasattr(estimator, "feature_importances_"):
        return "feature_importances"
    if hasattr(estimator, "coef_"):
        return "coefficient_magnitude"
    return "permutation_importance"


def _raw_importances(
    pipeline,
    model_name: str,
    X: pd.DataFrame,
    y: pd.Series,
) -> Tuple[np.ndarray, str]:
    estimator = _estimator_from_pipeline(pipeline)
    method = _importance_method_for_model(model_name, estimator)

    if method == "feature_importances":
        return np.asarray(estimator.feature_importances_, dtype=float), method

    if method == "coefficient_magnitude":
        coef = np.abs(estimator.coef_)
        values = coef.mean(axis=0) if coef.ndim > 1 else coef
        return np.asarray(values, dtype=float), method

    result = permutation_importance(
        pipeline,
        X,
        y,
        n_repeats=5,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        scoring="recall_macro",
    )
    return np.asarray(result.importances_mean, dtype=float), method


def _normalize_scores(values: np.ndarray) -> np.ndarray:
    values = np.maximum(values, 0.0)
    total = values.sum()
    if total <= 0:
        return values
    return values / total


def compute_model_feature_importance(
    pipeline,
    model_name: str,
    X: pd.DataFrame,
    y: pd.Series,
    top_n: int = TOP_N_DEFAULT,
) -> Tuple[List[Dict[str, Any]], str]:
    raw, method = _raw_importances(pipeline, model_name, X, y)
    if len(raw) != len(X.columns):
        raise ValueError(
            f"Importance length ({len(raw)}) != features ({len(X.columns)}) for {model_name}"
        )

    normalized = _normalize_scores(raw)
    importance_df = (
        pd.DataFrame({"variable": X.columns, "importance": normalized})
        .sort_values("importance", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )

    features = [
        {
            "rank": int(i + 1),
            "variable": str(row["variable"]),
            "importance": float(row["importance"]),
        }
        for i, row in importance_df.iterrows()
    ]
    return features, method


def build_feature_importance_payload(
    model_name: str,
    dataset_type: str,
    top_n: int = TOP_N_DEFAULT,
) -> Dict[str, Any]:
    pipeline = _load_pipeline(model_name, dataset_type)
    if pipeline is None:
        raise FileNotFoundError(
            f"Modelo entrenado no encontrado: {model_name} [{dataset_type}]"
        )

    X, y = load_train_split(dataset_type)
    features, method = compute_model_feature_importance(
        pipeline, model_name, X, y, top_n=top_n
    )

    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "method": method,
        "method_label": METHOD_LABELS.get(method, method),
        "model": model_name,
        "dataset_type": dataset_type,
        "top_n": top_n,
        "n_features": int(X.shape[1]),
        "n_samples": int(len(y)),
        "features": features,
    }


def save_model_feature_importance(
    model_name: str,
    dataset_type: str,
    top_n: int = TOP_N_DEFAULT,
) -> Path:
    payload = build_feature_importance_payload(model_name, dataset_type, top_n=top_n)
    out_path = _fi_path(model_name, dataset_type)
    ensure_dir(out_path.parent)
    save_json(payload, out_path)
    print(
        f"  Feature importance {model_name} [{dataset_type}] ({payload['method']}) -> {out_path}"
    )
    return out_path


def load_model_feature_importance(
    model_name: str,
    dataset_type: str,
) -> Optional[Dict[str, Any]]:
    path = _fi_path(model_name, dataset_type)
    if not path.exists():
        return None
    return load_json(path)


def _list_models_from_comparison() -> List[Tuple[str, str]]:
    comparison_path = RESULTS_DIR / "model_comparison.csv"
    if not comparison_path.exists():
        latest = RESULTS_DIR / "evaluation_summary_latest.csv"
        if latest.exists():
            comparison_path = latest
        else:
            return []

    df = read_csv(comparison_path)
    return [(str(r["model"]), str(r["dataset_type"])) for _, r in df.iterrows()]


def run_feature_importance(top_n: int = TOP_N_DEFAULT) -> Path:
    """Compute and persist feature importance for every model in the comparison table."""
    pairs = _list_models_from_comparison()
    if not pairs:
        raise FileNotFoundError(
            "No hay comparación de modelos. Ejecuta train, evaluate y compare primero."
        )

    index: Dict[str, Any] = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "top_n": top_n,
        "models": {},
    }

    for model_name, dataset_type in pairs:
        key = _model_key(model_name, dataset_type)
        try:
            save_model_feature_importance(model_name, dataset_type, top_n=top_n)
            payload = load_model_feature_importance(model_name, dataset_type)
            if payload:
                index["models"][key] = {
                    "model": model_name,
                    "dataset_type": dataset_type,
                    "method": payload.get("method"),
                    "n_features": len(payload.get("features", [])),
                }
        except Exception as exc:
            print(f"  Warning: feature importance skipped for {key}: {exc}")
            index["models"][key] = {"model": model_name, "dataset_type": dataset_type, "error": str(exc)}

    ensure_dir(PROCESSED_DIR)
    index_path = PROCESSED_DIR / "feature_importance_index.json"
    save_json(index, index_path)
    return index_path
