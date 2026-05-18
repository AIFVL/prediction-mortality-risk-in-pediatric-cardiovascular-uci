from pathlib import Path
from datetime import datetime
from typing import Optional

import pandas as pd
import pickle

from src.config import MODELS_DIR, RESULTS_DIR
from src.io_utils import read_csv, write_csv, ensure_dir


def run_predict(
    input_path: str,
    model_name: str,
    dataset_type: str = "raw",
    output_name: Optional[str] = None,
) -> Path:
    input_file = Path(input_path)
    if not input_file.exists():
        raise FileNotFoundError(f"Input data file not found: {input_file}")

    df = read_csv(input_file)

    model_path = MODELS_DIR / model_name / dataset_type / "model.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}")

    with model_path.open("rb") as f:
        model = pickle.load(f)

    preds = model.predict(df)

    if hasattr(model, "predict_proba"):
        try:
            proba = model.predict_proba(df)
        except Exception:
            proba = None
    else:
        proba = None

    result_df = df.copy()
    result_df["prediction"] = preds

    if proba is not None:
        for i in range(proba.shape[1]):
            result_df[f"prob_class_{i}"] = proba[:, i]

    ensure_dir(RESULTS_DIR / "predictions")

    if output_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_name = f"predictions_{model_name}_{dataset_type}_{timestamp}.csv"

    out_path = RESULTS_DIR / "predictions" / output_name
    write_csv(result_df, out_path)

    return out_path
