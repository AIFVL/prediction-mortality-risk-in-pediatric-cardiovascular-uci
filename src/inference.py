"""
Batch inference for new patient data.

Uses the same notebook-2 preprocessing artifacts saved during training.
"""
from __future__ import annotations

import io
import json
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.config import MODELS_DIR, PROCESSED_DIR, DEFAULT_MODELS, TARGET_COLUMN, MONGODB_URI
from src.preprocess import preprocess_for_inference, INFERENCE_DROP_COLS

try:
    from src.db.mongo_store import MongoStore
except ImportError:
    MongoStore = None


def _parse_upload(content: bytes, filename: str) -> pd.DataFrame:
    ext = Path(filename).suffix.lower()
    if ext in (".xlsx", ".xls"):
        return pd.read_excel(io.BytesIO(content))
    if ext == ".csv":
        return pd.read_csv(io.BytesIO(content))
    if ext == ".txt":
        try:
            return pd.read_csv(io.BytesIO(content), sep="\t")
        except Exception:
            return pd.read_csv(io.BytesIO(content))
    raise ValueError(f"Formato no soportado: {ext}. Usa CSV, XLSX o TXT.")


def _load_pipeline(model_name: str, dataset_type: str):
    path = MODELS_DIR / model_name / dataset_type / "model.pkl"
    
    if not path.exists() and MONGODB_URI:
        store = MongoStore.from_env()
        if store:
            # Buscar el ID del modelo en la colección de modelos
            model_doc = store.models.find_one(
                {"model": model_name, "dataset_type": dataset_type},
                sort=[("stored_at", -1)]
            )
            if model_doc and "model_file_id" in model_doc:
                print(f"Downloading model {model_name} ({dataset_type}) from MongoDB...")
                store.download_file_by_id(model_doc["model_file_id"], path)

    if not path.exists():
        return None
    with path.open("rb") as f:
        return pickle.load(f)


def _best_model_key() -> Optional[tuple[str, str]]:
    data_path = PROCESSED_DIR.parent / "results" / "dashboard_data.json"
    
    data = None
    if MONGODB_URI:
        store = MongoStore.from_env()
        if store:
            data = store.get_latest_dashboard_data()
    
    if not data and data_path.exists():
        try:
            with open(data_path, encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            pass

    if data:
        bm = data.get("best_model")
        if bm:
            return bm["model"], bm["dataset_type"]
    return None


def run_batch_inference(file_content: bytes, filename: str) -> dict:
    """
    Parse an uploaded file and run all available models on the patients.
    """
    df_raw = _parse_upload(file_content, filename)
    n_patients = len(df_raw)

    if n_patients == 0:
        raise ValueError("El archivo no contiene pacientes.")

    skip_cols = set(INFERENCE_DROP_COLS)
    display_cols = [c for c in df_raw.columns if c not in skip_cols][:8]

    patient_rows = []
    for i in range(n_patients):
        row: dict = {"patient_idx": i + 1}
        for col in display_cols:
            val = df_raw[col].iloc[i]
            row[col] = "—" if pd.isna(val) else str(val)
        patient_rows.append(row)

    X = preprocess_for_inference(df_raw)
    if len(X) != n_patients:
        raise ValueError(
            f"Tras el preprocesamiento quedaron {len(X)} filas de {n_patients} pacientes. "
            "Revisa que los registros tengan datos clínicos mínimos (p. ej. IMC o edad)."
        )

    best_key = _best_model_key()
    model_predictions: list[dict] = []

    for model_name in DEFAULT_MODELS:
        for dataset_type in ("raw", "balanced"):
            pipeline = _load_pipeline(model_name, dataset_type)
            if pipeline is None:
                continue
            try:
                preds = pipeline.predict(X).tolist()
                proba: Optional[list[float]] = None
                if hasattr(pipeline, "predict_proba"):
                    try:
                        p = pipeline.predict_proba(X)
                        proba = [round(float(v), 4) for v in p[:, 1]]
                    except Exception:
                        pass

                n_died = int(sum(p == 1 for p in preds))
                avg_prob = round(float(np.mean(proba)), 4) if proba else None

                is_best = (
                    best_key is not None
                    and best_key[0] == model_name
                    and best_key[1] == dataset_type
                )

                model_predictions.append({
                    "model": model_name,
                    "dataset_type": dataset_type,
                    "is_best": is_best,
                    "predictions": preds,
                    "probabilities": proba,
                    "n_predicted_died": n_died,
                    "n_predicted_survived": n_patients - n_died,
                    "avg_prob_died": avg_prob,
                })
            except Exception as exc:
                model_predictions.append({
                    "model": model_name,
                    "dataset_type": dataset_type,
                    "is_best": False,
                    "error": str(exc),
                })

    return {
        "n_patients": n_patients,
        "filename": filename,
        "display_columns": display_cols,
        "patient_rows": patient_rows,
        "model_predictions": model_predictions,
        "best_model_key": list(best_key) if best_key else None,
    }
