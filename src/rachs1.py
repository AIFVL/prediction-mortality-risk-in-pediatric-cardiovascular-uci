"""
RACHS-1 baseline metrics computation.

Loads dataset_limpio.xlsx, computes RACHS-1 predictions (score <= 3 → class 0,
score > 3 → class 1) vs the binary mortality ground truth, and saves
the resulting metrics to data/results/rachs1/rachs1_metrics.json.
"""
from pathlib import Path
from typing import Optional
import math
import unicodedata

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    matthews_corrcoef,
    precision_recall_fscore_support,
    roc_auc_score,
)

from src.config import DATASETS_DIR, RESULTS_DIR
from src.io_utils import ensure_dir, save_json


RACHS1_RESULTS_DIR = RESULTS_DIR / "rachs1"

MORTALITY_BINARY_MAP = {
    "no murio": 0,
    "no murió": 0,
    "murio": 1,
    "murió": 1,
    "murio (>30d)": 1,
    "murió (>30d)": 1,
    "murio (<30d)": 1,
    "murió (<30d)": 1,
    "murió en los primeros 30 dias": 1,
    "murio en los primeros 30 dias": 1,
    "murió despues de 30 dias": 1,
    "murio despues de 30 dias": 1,
}

NORMALIZED_MORTALITY_BINARY_MAP = {
    "no murio": 0,
    "murio": 1,
    "murio (>30d)": 1,
    "murio (<30d)": 1,
    "murio en los primeros 30 dias": 1,
    "murio despues de 30 dias": 1,
}


def _normalize_label(value) -> str:
    text = str(value).strip().lower()
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")


def _safe(val):
    if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
        return None
    if isinstance(val, (np.integer,)):
        return int(val)
    if isinstance(val, (np.floating,)):
        return float(val)
    return val


def run_rachs1(dataset_path: Optional[str] = None) -> Path:
    if dataset_path is None:
        candidates = [
            DATASETS_DIR / "dataset_limpio.xlsx",
            DATASETS_DIR / "dataset_inicial.xlsx",
        ]
        source_file = None
        for c in candidates:
            if c.exists():
                source_file = c
                break
        if source_file is None:
            raise FileNotFoundError(
                "Could not find dataset_limpio.xlsx or dataset_inicial.xlsx in data/datasets/"
            )
    else:
        source_file = Path(dataset_path)
        if not source_file.exists():
            raise FileNotFoundError(f"RACHS-1 source file not found: {source_file}")

    if source_file.suffix.lower() in (".xlsx", ".xls"):
        df = pd.read_excel(source_file)
    else:
        df = pd.read_csv(source_file)

    df.columns = df.columns.str.strip()

    rachs_col = None
    for col in df.columns:
        if "RACHS" in col.upper():
            rachs_col = col
            break

    if rachs_col is None:
        raise ValueError("No RACHS-1 column found in the dataset.")

    mort_col = None
    for col in df.columns:
        if "MORTALIDAD" in col.upper() and "GENERAL" not in col.upper() and "30" not in col.upper():
            mort_col = col
            break
    if mort_col is None:
        for col in df.columns:
            if col.strip().lower() in ("mortalidad",):
                mort_col = col
                break
    if mort_col is None:
        for col in df.columns:
            if "MORTALIDAD" in col.upper():
                mort_col = col
                break

    if mort_col is None:
        raise ValueError("No Mortalidad column found in the dataset.")

    df_rachs = df[df[rachs_col].astype(str).str.strip().str.upper() != "NO APLICA"].copy()
    df_rachs[rachs_col] = pd.to_numeric(df_rachs[rachs_col], errors="coerce")
    df_rachs = df_rachs.dropna(subset=[rachs_col])

    n_total = len(df)
    n_valid = len(df_rachs)
    n_excluded = n_total - n_valid

    y_real_series = df_rachs[mort_col].map(_normalize_label).map(NORMALIZED_MORTALITY_BINARY_MAP)
    unknown_mortality = sorted(df_rachs.loc[y_real_series.isna(), mort_col].dropna().astype(str).unique())
    if unknown_mortality:
        raise ValueError(f"Unknown Mortalidad labels in RACHS-1 data: {unknown_mortality}")
    y_real = y_real_series.astype(int).values

    y_pred_rachs = (df_rachs[rachs_col] > 3).astype(int).values
    y_score_rachs = (df_rachs[rachs_col] / 6.0).values

    acc = accuracy_score(y_real, y_pred_rachs)
    from sklearn.metrics import recall_score, precision_score, f1_score
    prec_macro = precision_score(y_real, y_pred_rachs, average="macro", zero_division=0)
    rec_macro = recall_score(y_real, y_pred_rachs, average="macro", zero_division=0)
    f1_macro = f1_score(y_real, y_pred_rachs, average="macro", zero_division=0)
    kappa = cohen_kappa_score(y_real, y_pred_rachs)

    prec_c, rec_c, f1_c, sup_c = precision_recall_fscore_support(
        y_real, y_pred_rachs, average=None, zero_division=0, labels=[0, 1]
    )

    try:
        roc_auc = float(roc_auc_score(y_real, y_score_rachs))
    except ValueError:
        roc_auc = float("nan")

    cm = confusion_matrix(y_real, y_pred_rachs, labels=[0, 1])

    sensitivity, specificity = [], []
    for i in range(2):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = cm.sum() - (tp + fn + fp)
        sensitivity.append(float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0)
        specificity.append(float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0)

    per_class = []
    for i in range(2):
        per_class.append({
            "class": i,
            "precision": _safe(prec_c[i]),
            "recall": _safe(rec_c[i]),
            "f1_score": _safe(f1_c[i]),
            "sensitivity": _safe(sensitivity[i]),
            "specificity": _safe(specificity[i]),
            "support": _safe(sup_c[i]),
        })

    result = {
        "source_file": str(source_file),
        "n_total": n_total,
        "n_valid_rachs1": n_valid,
        "n_excluded_no_aplica": n_excluded,
        "class_distribution": {
            "y_real": {str(k): int(v) for k, v in zip(*np.unique(y_real, return_counts=True))},
            "y_pred": {str(k): int(v) for k, v in zip(*np.unique(y_pred_rachs, return_counts=True))},
        },
        "metrics": {
            "accuracy": _safe(acc),
            "precision_macro": _safe(prec_macro),
            "recall_macro": _safe(rec_macro),
            "f1_macro": _safe(f1_macro),
            "kappa": _safe(kappa),
            "roc_auc": _safe(roc_auc),
        },
        "per_class_metrics": per_class,
        "confusion_matrix": cm.tolist(),
    }

    ensure_dir(RACHS1_RESULTS_DIR)
    out_path = RACHS1_RESULTS_DIR / "rachs1_metrics.json"
    save_json(result, out_path)
    print(f"  RACHS-1 metrics saved -> {out_path}")
    return out_path
