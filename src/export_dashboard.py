"""
Export pipeline results to a JSON file consumed by the dashboard.
"""
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import math

import numpy as np
import pandas as pd

from src.config import RESULTS_DIR, PROCESSED_DIR, MODELS_DIR, DEFAULT_MODELS, TARGET_COLUMN
from src.io_utils import ensure_dir, load_json, save_json
from src.feature_importance import (
    load_model_feature_importance,
    save_model_feature_importance,
)


CLASS_LABELS = {0: "No murió", 1: "Murió"}


def _safe(val: Any) -> Any:
    if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
        return None
    if isinstance(val, (np.integer,)):
        return int(val)
    if isinstance(val, (np.floating,)):
        return float(val)
    return val


def _row_to_dict(row: pd.Series) -> Dict:
    return {k: _safe(v) for k, v in row.items()}


def _class_counts_from_csv(dataset_type: str) -> Dict[str, int]:
    y_path = PROCESSED_DIR / "csv" / f"y_train_{dataset_type}.csv"
    if not y_path.exists():
        return {}

    y_df = pd.read_csv(y_path)
    if TARGET_COLUMN in y_df.columns:
        y = y_df[TARGET_COLUMN]
    elif len(y_df.columns) == 1:
        y = y_df.iloc[:, 0]
    else:
        return {}

    return {str(k): int(v) for k, v in y.value_counts().sort_index().items()}


def _load_confusion_matrix(path: Path) -> List[List[int]]:
    cm_df = pd.read_csv(path)

    first_col = str(cm_df.columns[0])
    if first_col.startswith("Unnamed") or first_col == "":
        cm_df = cm_df.drop(columns=[cm_df.columns[0]])

    cm_df = cm_df.apply(pd.to_numeric, errors="coerce").fillna(0).astype(int)
    return cm_df.values.tolist()


def build_dashboard_data() -> Dict:
    data: Dict[str, Any] = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "models": [],
        "best_model": None,
        "class_distribution": {},
        "comparison_table": [],
        "rachs1": None,
    }

    target_classes_path = PROCESSED_DIR / "target_classes.json"
    if target_classes_path.exists():
        target_info = load_json(target_classes_path)
        raw_classes = target_info.get("classes", [])
        data["target_classes"] = {str(i): name for i, name in enumerate(raw_classes)}
    else:
        data["target_classes"] = {str(k): v for k, v in CLASS_LABELS.items()}

    raw_counts = _class_counts_from_csv("raw")
    balanced_counts = _class_counts_from_csv("balanced")
    if raw_counts or balanced_counts:
        data["class_distribution"] = {
            "raw": raw_counts,
            "balanced": balanced_counts,
        }

    balanced_meta_path = PROCESSED_DIR / "metadatos_balanced.json"
    if balanced_meta_path.exists():
        balanced_meta = load_json(balanced_meta_path)
        if not data["class_distribution"]:
            data["class_distribution"] = {
                "raw": balanced_meta.get("clases_raw", {}),
                "balanced": balanced_meta.get("clases_balanced", {}),
            }
        data["balance_method"] = balanced_meta.get("balance_method")

    comparison_path = RESULTS_DIR / "model_comparison.csv"
    if not comparison_path.exists():
        latest_path = RESULTS_DIR / "evaluation_summary_latest.csv"
        if latest_path.exists():
            comparison_path = latest_path

    if not comparison_path.exists():
        data["status"] = "no_results"
        return data

    df = pd.read_csv(comparison_path)
    data["comparison_table"] = [_row_to_dict(row) for _, row in df.iterrows()]

    # Prioritize Recall Macro as the metric for "best model"
    if "recall_macro" in df.columns:
        best_idx = df["recall_macro"].idxmax()
    else:
        best_idx = df["accuracy"].idxmax()

    best_row = df.loc[best_idx]
    data["best_model"] = {
        "model": best_row.get("model", ""),
        "dataset_type": best_row.get("dataset_type", ""),
        "accuracy": _safe(best_row.get("accuracy")),
        "recall_macro": _safe(best_row.get("recall_macro")),
        "precision_macro": _safe(best_row.get("precision_macro")),
        "f1_macro": _safe(best_row.get("f1_macro")),
        "kappa": _safe(best_row.get("kappa")),
        "roc_auc_ovr_macro": _safe(best_row.get("roc_auc_ovr_macro")),
    }

    models_detail: List[Dict] = []
    for _, row in df.iterrows():
        model_name = row["model"]
        dataset_type = row["dataset_type"]

        model_entry: Dict[str, Any] = {
            "model": model_name,
            "dataset_type": dataset_type,
            "metrics": {
                "accuracy": _safe(row.get("accuracy")),
                "precision_macro": _safe(row.get("precision_macro")),
                "recall_macro": _safe(row.get("recall_macro")),
                "f1_macro": _safe(row.get("f1_macro")),
                "kappa": _safe(row.get("kappa")),
                "roc_auc_ovr_macro": _safe(row.get("roc_auc_ovr_macro")),
            },
            "avg_prediction_probabilities": {
                "class_0": _safe(row.get("avg_proba_class_0")),
                "class_1": _safe(row.get("avg_proba_class_1")),
            },
            "per_class_metrics": [],
            "confusion_matrix": [],
        }

        per_class_path = RESULTS_DIR / model_name / f"per_class_metrics_{dataset_type}.csv"
        if per_class_path.exists():
            pc_df = pd.read_csv(per_class_path)
            model_entry["per_class_metrics"] = [_row_to_dict(r) for _, r in pc_df.iterrows()]

        cm_path = RESULTS_DIR / model_name / f"confusion_matrix_{dataset_type}.csv"
        if cm_path.exists():
            model_entry["confusion_matrix"] = _load_confusion_matrix(cm_path)

        fi = load_model_feature_importance(model_name, dataset_type)
        if not fi:
            try:
                save_model_feature_importance(model_name, dataset_type)
                fi = load_model_feature_importance(model_name, dataset_type)
            except Exception:
                fi = None
        if fi:
            model_entry["feature_importance"] = fi

        models_detail.append(model_entry)

    data["models"] = models_detail

    rachs1_path = RESULTS_DIR / "rachs1" / "rachs1_metrics.json"
    if rachs1_path.exists():
        data["rachs1"] = load_json(rachs1_path)

    data["status"] = "ok"
    return data


def run_export_dashboard() -> Path:
    ensure_dir(RESULTS_DIR)
    dashboard_data = build_dashboard_data()
    out_path = RESULTS_DIR / "dashboard_data.json"
    save_json(dashboard_data, out_path)
    print(f"  Dashboard data exported -> {out_path}")
    return out_path
