"""
Consistent data-loading helpers for the PDG pipeline.

All code that reads processed CSVs or metadata should go through
these functions rather than calling io_utils directly, so the
loading contract stays in one place.
"""
from pathlib import Path

import pandas as pd

from src.config import PROCESSED_DIR, TARGET_COLUMN
from src.io_utils import read_csv, load_json


def load_train_split(dataset_type: str = "raw") -> tuple[pd.DataFrame, pd.Series]:
    """
    Load X and y for a given dataset type ('raw' or 'balanced').

    Returns
    -------
    X : pd.DataFrame
    y : pd.Series
    """
    csv_dir = PROCESSED_DIR / "csv"
    X_path = csv_dir / f"X_train_{dataset_type}.csv"
    y_path = csv_dir / f"y_train_{dataset_type}.csv"

    if not X_path.exists() or not y_path.exists():
        raise FileNotFoundError(
            f"Processed data for type '{dataset_type}' not found. "
            "Run preprocessing and balancing first."
        )

    X = read_csv(X_path)
    y_df = read_csv(y_path)

    if TARGET_COLUMN not in y_df.columns:
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' not present in {y_path}"
        )

    return X, y_df[TARGET_COLUMN]


def load_all_data() -> tuple[pd.DataFrame, pd.Series]:
    """Shorthand for load_train_split('raw') — loads the full unbalanced dataset."""
    return load_train_split("raw")


def load_metadata(meta_file: str = "metadatos_generated.json") -> dict:
    """Load preprocessing metadata from the processed directory."""
    path = PROCESSED_DIR / meta_file
    if not path.exists():
        return {}
    return load_json(path)


def list_available_models(models_dir: Path) -> list[str]:
    """
    Return the names of model subdirectories that contain a model.pkl file
    (at least one dataset_type variant).
    """
    available = []
    if not models_dir.exists():
        return available
    for model_dir in sorted(models_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        has_pkl = any((model_dir / dt / "model.pkl").exists() for dt in ("raw", "balanced"))
        if has_pkl:
            available.append(model_dir.name)
    return available
