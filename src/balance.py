from pathlib import Path
from collections import Counter
from typing import Optional

import pandas as pd
from imblearn.over_sampling import ADASYN

from src.config import PROCESSED_DIR, RANDOM_STATE, TARGET_COLUMN
from src.io_utils import read_csv, write_csv, save_json, ensure_dir


DESIRED_CLASS_PROPORTIONS = {0: 0.70, 1: 0.30}


def run_balance(version_tag: Optional[str] = None) -> None:
    csv_dir = PROCESSED_DIR / "csv"
    X_train_path = csv_dir / "X_train_raw.csv"
    y_train_path = csv_dir / "y_train_raw.csv"

    if not X_train_path.exists() or not y_train_path.exists():
        raise FileNotFoundError("Training raw datasets not found. Run preprocessing first.")

    X_train = read_csv(X_train_path)
    y_train_df = read_csv(y_train_path)

    if TARGET_COLUMN not in y_train_df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' not found in y_train_raw.csv")

    y_train = y_train_df[TARGET_COLUMN]

    k_neighbors = min(5, min(Counter(y_train).values()) - 1)
    if k_neighbors < 1:
        k_neighbors = 1

    counts_raw = Counter(y_train)
    majority_class = max(counts_raw, key=counts_raw.get)
    majority_count = counts_raw[majority_class]
    desired_majority_proportion = DESIRED_CLASS_PROPORTIONS.get(majority_class, 0.70)
    desired_total = int(majority_count / desired_majority_proportion)
    sampling_strategy = {
        cls: int(desired_total * proportion)
        for cls, proportion in DESIRED_CLASS_PROPORTIONS.items()
        if cls != majority_class and int(desired_total * proportion) > counts_raw.get(cls, 0)
    }

    if sampling_strategy:
        resampler = ADASYN(
            sampling_strategy=sampling_strategy,
            random_state=RANDOM_STATE,
            n_neighbors=k_neighbors,
        )
        X_res, y_res = resampler.fit_resample(X_train, y_train)
        balance_method = "adasyn_custom_70_30"
    else:
        X_res, y_res = X_train.copy(), y_train.copy()
        balance_method = "none"

    X_balanced_path = csv_dir / "X_train_balanced.csv"
    y_balanced_path = csv_dir / "y_train_balanced.csv"
    write_csv(X_res, X_balanced_path)
    write_csv(pd.DataFrame({TARGET_COLUMN: y_res}), y_balanced_path)

    ensure_dir(PROCESSED_DIR)
    meta_path = PROCESSED_DIR / "metadatos_balanced.json"

    counts_balanced = Counter(y_res)

    meta = {
        "X_train_raw_shape": list(X_train.shape),
        "y_train_raw_shape": [int(y_train.shape[0])],
        "clases_raw": {str(k): int(v) for k, v in counts_raw.items()},
        "X_train_balanced_shape": list(X_res.shape),
        "y_train_balanced_shape": [int(y_res.shape[0])],
        "clases_balanced": {str(k): int(v) for k, v in counts_balanced.items()},
        "balance_method": balance_method,
        "desired_class_proportions": {str(k): v for k, v in DESIRED_CLASS_PROPORTIONS.items()},
        "sampling_strategy": {str(k): int(v) for k, v in sampling_strategy.items()},
        "variable_objetivo": TARGET_COLUMN,
        "random_state": RANDOM_STATE,
        "version_tag": version_tag,
    }

    save_json(meta, meta_path)
