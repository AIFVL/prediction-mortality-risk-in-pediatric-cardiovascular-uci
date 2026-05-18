from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

try:
    import gridfs
    from bson import ObjectId
    from pymongo import ASCENDING, DESCENDING, MongoClient
    from pymongo.collection import Collection
    from pymongo.database import Database
except Exception as exc:  # pragma: no cover
    raise ImportError(
        "MongoDB support requires 'pymongo' (and its dependencies). "
        "Install it via requirements.txt."
    ) from exc

from src.config import (
    MONGODB_URI,
    MONGODB_DB_NAME,
    MONGODB_CONNECT_TIMEOUT_MS,
    PROCESSED_DIR,
    RESULTS_DIR,
    MODELS_DIR,
    DATASETS_DIR,
    PDG_MONGO_STORE_PROCESSED_PICKLES,
)


@dataclass(frozen=True)
class MongoStoreConfig:
    uri: str
    db_name: str
    connect_timeout_ms: int = 3000


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


class MongoStore:
    """PDG persistence layer on top of MongoDB + GridFS.

    MongoDB is optional; use `MongoStore.from_env()` and check for None.

    Collections used:
    - datasets
    - pipeline_executions
    - models
    - model_evaluations
    - model_comparisons
    - dashboard_data
    - preprocess_artifacts
    - prediction_sessions

    GridFS default bucket: fs.files / fs.chunks
    """

    def __init__(self, cfg: MongoStoreConfig):
        self.cfg = cfg
        self.client: MongoClient = MongoClient(
            cfg.uri,
            serverSelectionTimeoutMS=cfg.connect_timeout_ms,
            connectTimeoutMS=cfg.connect_timeout_ms,
        )
        self.db: Database = self.client[cfg.db_name]
        self.fs = gridfs.GridFS(self.db)

        self.datasets: Collection = self.db["datasets"]
        self.pipeline_executions: Collection = self.db["pipeline_executions"]
        self.models: Collection = self.db["models"]
        self.model_evaluations: Collection = self.db["model_evaluations"]
        self.model_comparisons: Collection = self.db["model_comparisons"]
        self.dashboard_data: Collection = self.db["dashboard_data"]
        self.preprocess_artifacts: Collection = self.db["preprocess_artifacts"]
        self.prediction_sessions: Collection = self.db["prediction_sessions"]

    @staticmethod
    def from_env() -> Optional["MongoStore"]:
        if not MONGODB_URI:
            return None
        cfg = MongoStoreConfig(
            uri=MONGODB_URI,
            db_name=MONGODB_DB_NAME,
            connect_timeout_ms=MONGODB_CONNECT_TIMEOUT_MS,
        )
        return MongoStore(cfg)

    def ping(self) -> bool:
        try:
            self.client.admin.command("ping")
            return True
        except Exception:
            return False

    # ----------------------- GridFS helpers -----------------------

    def upload_path(
        self,
        path: Path,
        *,
        filename: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        content_type: Optional[str] = None,
    ) -> ObjectId:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(str(path))

        file_metadata = dict(metadata or {})
        file_metadata.setdefault("sha256", _sha256_file(path))
        file_metadata.setdefault("size_bytes", path.stat().st_size)
        file_metadata.setdefault("source_path", str(path))

        with path.open("rb") as f:
            return self.fs.put(
                f,
                filename=filename or path.name,
                metadata=file_metadata,
                content_type=content_type,
            )

    def download_file_by_name(self, filename: str, destination_path: Path) -> bool:
        """Download the latest file from GridFS with a given filename."""
        try:
            grid_out = self.fs.find_one({"filename": filename}, sort=[("uploadDate", DESCENDING)])
            if not grid_out:
                return False

            destination_path = Path(destination_path)
            destination_path.parent.mkdir(parents=True, exist_ok=True)

            with destination_path.open("wb") as f:
                f.write(grid_out.read())
            return True
        except Exception:
            return False

    def download_file_by_id(self, file_id: ObjectId, destination_path: Path) -> bool:
        """Download a specific file from GridFS by its ID."""
        try:
            grid_out = self.fs.get(file_id)
            destination_path = Path(destination_path)
            destination_path.parent.mkdir(parents=True, exist_ok=True)
            with destination_path.open("wb") as f:
                f.write(grid_out.read())
            return True
        except Exception:
            return False

    # ----------------------- High-level persistence -----------------------

    def upsert_pipeline_execution(
        self,
        execution_id: str,
        *,
        status: str,
        started_at: Optional[datetime] = None,
        completed_at: Optional[datetime] = None,
        input_path: Optional[str] = None,
        steps: Optional[str] = None,
        models: Optional[str] = None,
        log_lines: Optional[List[str]] = None,
        message: Optional[str] = None,
        error: Optional[str] = None,
    ) -> None:
        doc: Dict[str, Any] = {
            "execution_id": execution_id,
            "status": status,
            "updated_at": _utc_now(),
        }
        if started_at is not None:
            doc["started_at"] = started_at
        if completed_at is not None:
            doc["completed_at"] = completed_at
        if input_path is not None:
            doc["input_path"] = input_path
        if steps is not None:
            doc["steps"] = steps
        if models is not None:
            doc["models"] = models
        if message is not None:
            doc["message"] = message
        if error is not None:
            doc["error"] = error
        if log_lines is not None:
            doc["log"] = log_lines

        self.pipeline_executions.update_one(
            {"execution_id": execution_id},
            {"$set": doc, "$setOnInsert": {"created_at": _utc_now()}},
            upsert=True,
        )

    def save_dashboard_data(self, data: Dict[str, Any], *, execution_id: Optional[str] = None) -> ObjectId:
        payload = dict(data)
        payload["stored_at"] = _utc_now()
        if execution_id:
            payload["execution_id"] = execution_id
        return self.dashboard_data.insert_one(payload).inserted_id

    def get_latest_dashboard_data(self) -> Optional[Dict[str, Any]]:
        doc = self.dashboard_data.find_one(sort=[("stored_at", DESCENDING)])
        if not doc:
            return None
        doc.pop("_id", None)
        return doc

    def persist_dataset_files(
        self,
        *,
        execution_id: str,
        raw_csv_path: Optional[Path],
        input_path: Optional[Path] = None,
    ) -> Optional[ObjectId]:
        if raw_csv_path is None:
            raw_csv_path = None

        input_file_id = None
        if input_path is not None:
            ip = Path(input_path)
            if ip.exists() and ip.is_file():
                try:
                    input_file_id = self.upload_path(
                        ip,
                        metadata={
                            "execution_id": execution_id,
                            "kind": "input_file",
                        },
                    )
                except Exception:
                    input_file_id = None

        if raw_csv_path is None:
            # Still store a dataset document if input file exists
            if input_file_id is None:
                return None
            doc = {
                "execution_id": execution_id,
                "stored_at": _utc_now(),
                "input_file_id": input_file_id,
            }
            return self.datasets.insert_one(doc).inserted_id

        raw_csv_path = Path(raw_csv_path)
        meta_path = raw_csv_path.with_name(raw_csv_path.stem + "_metadata.json")

        csv_file_id = self.upload_path(
            raw_csv_path,
            metadata={
                "execution_id": execution_id,
                "kind": "raw_dataset_csv",
            },
            content_type="text/csv",
        )

        meta: Optional[Dict[str, Any]] = None
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception:
                meta = None

        doc: Dict[str, Any] = {
            "execution_id": execution_id,
            "filename": raw_csv_path.name,
            "stored_at": _utc_now(),
            "csv_file_id": csv_file_id,
            "csv_sha256": _sha256_file(raw_csv_path),
        }
        if input_file_id is not None:
            doc["input_file_id"] = input_file_id
        if meta:
            doc.update(meta)

        result = self.datasets.insert_one(doc)
        return result.inserted_id

    def persist_preprocess_artifacts(self, *, execution_id: str) -> Dict[str, Any]:
        artifacts: Dict[str, Any] = {"execution_id": execution_id, "stored_at": _utc_now(), "files": {}}

        candidates = [
            (PROCESSED_DIR / "preprocessor.pkl", "preprocessor"),
            (PROCESSED_DIR / "preprocess_artifacts.joblib", "preprocess_artifacts"),
            (PROCESSED_DIR / "target_classes.json", "target_classes"),
        ]

        for path, key in candidates:
            if path.exists():
                file_id = self.upload_path(path, metadata={"execution_id": execution_id, "kind": key})
                artifacts["files"][key] = {"file_id": file_id, "filename": path.name, "size_bytes": path.stat().st_size}

        # Optionally store heavy processed pickles
        if PDG_MONGO_STORE_PROCESSED_PICKLES:
            pickles_dir = PROCESSED_DIR / "pickle"
            if pickles_dir.exists():
                for pkl in pickles_dir.glob("*.pkl"):
                    file_id = self.upload_path(pkl, metadata={"execution_id": execution_id, "kind": "processed_pickle"})
                    artifacts["files"].setdefault("processed_pickles", []).append(
                        {"file_id": file_id, "filename": pkl.name, "size_bytes": pkl.stat().st_size}
                    )

        self.preprocess_artifacts.insert_one(artifacts)
        return artifacts

    def persist_models_and_results(
        self,
        *,
        execution_id: str,
        model_names: Optional[Iterable[str]] = None,
        dataset_types: Iterable[str] = ("raw", "balanced"),
    ) -> None:
        """Persist model.pkl + metrics (comparison/evaluation) from filesystem outputs."""

        comparison_path = RESULTS_DIR / "model_comparison.csv"
        if not comparison_path.exists():
            latest_path = RESULTS_DIR / "evaluation_summary_latest.csv"
            if latest_path.exists():
                comparison_path = latest_path

        rows: List[Dict[str, Any]] = []
        if comparison_path.exists():
            import pandas as pd

            df = pd.read_csv(comparison_path)
            rows = df.to_dict(orient="records")
            self.model_comparisons.insert_one(
                {
                    "execution_id": execution_id,
                    "stored_at": _utc_now(),
                    "source_file": str(comparison_path),
                    "rows": rows,
                }
            )

        # Helper: index metrics per (model, dataset_type)
        metrics_index: Dict[tuple, Dict[str, Any]] = {}
        for r in rows:
            key = (r.get("model"), r.get("dataset_type"))
            metrics_index[key] = r

        for dataset_type in dataset_types:
            for model_name in (model_names or []):
                model_path = MODELS_DIR / str(model_name) / str(dataset_type) / "model.pkl"
                if not model_path.exists():
                    continue

                model_file_id = self.upload_path(
                    model_path,
                    metadata={
                        "execution_id": execution_id,
                        "model": model_name,
                        "dataset_type": dataset_type,
                        "kind": "model_pkl",
                    },
                )

                metrics = metrics_index.get((model_name, dataset_type), {})
                model_doc = {
                    "execution_id": execution_id,
                    "model": model_name,
                    "dataset_type": dataset_type,
                    "stored_at": _utc_now(),
                    "model_file_id": model_file_id,
                    "model_sha256": _sha256_file(model_path),
                    "metrics": metrics,
                }

                res = self.models.update_one(
                    {"execution_id": execution_id, "model": model_name, "dataset_type": dataset_type},
                    {"$set": model_doc, "$setOnInsert": {"created_at": _utc_now()}},
                    upsert=True,
                )

                model_ref = self.models.find_one(
                    {"execution_id": execution_id, "model": model_name, "dataset_type": dataset_type},
                    {"_id": 1},
                )
                if not model_ref:
                    continue
                model_id = model_ref["_id"]

                # Attach evaluation artifacts if present
                eval_doc: Dict[str, Any] = {
                    "execution_id": execution_id,
                    "model_id": model_id,
                    "model": model_name,
                    "dataset_type": dataset_type,
                    "stored_at": _utc_now(),
                }

                cm_path = RESULTS_DIR / str(model_name) / f"confusion_matrix_{dataset_type}.csv"
                if cm_path.exists():
                    try:
                        import pandas as pd

                        cm_df = pd.read_csv(cm_path)
                        first_col = str(cm_df.columns[0])
                        if first_col.startswith("Unnamed") or first_col == "":
                            cm_df = cm_df.drop(columns=[cm_df.columns[0]])
                        cm_df = cm_df.apply(pd.to_numeric, errors="coerce").fillna(0).astype(int)
                        eval_doc["confusion_matrix"] = cm_df.values.tolist()
                    except Exception:
                        pass

                pc_path = RESULTS_DIR / str(model_name) / f"per_class_metrics_{dataset_type}.csv"
                if pc_path.exists():
                    try:
                        import pandas as pd

                        pc_df = pd.read_csv(pc_path)
                        eval_doc["per_class_metrics"] = pc_df.to_dict(orient="records")
                    except Exception:
                        pass

                fi_path = RESULTS_DIR / str(model_name) / f"feature_importance_{dataset_type}.json"
                if fi_path.exists():
                    try:
                        eval_doc["feature_importance"] = json.loads(fi_path.read_text(encoding="utf-8"))
                    except Exception:
                        pass

                self.model_evaluations.update_one(
                    {"execution_id": execution_id, "model_id": model_id},
                    {"$set": eval_doc, "$setOnInsert": {"created_at": _utc_now()}},
                    upsert=True,
                )

    def save_prediction_session(
        self,
        prediction_id: str,
        file_content: bytes,
        filename: str,
        n_patients: int,
        model_predictions: List[Dict[str, Any]],
        best_model_info: Optional[Dict[str, str]],
        patient_display_data: Optional[List[Dict[str, Any]]] = None,
    ) -> ObjectId:
        """Save a prediction session with uploaded file and model predictions."""
        # Upload the patient file to GridFS
        input_file_id = self.fs.put(
            file_content,
            filename=filename,
            metadata={
                "prediction_id": prediction_id,
                "kind": "prediction_input",
                "size_bytes": len(file_content),
            },
            content_type="application/octet-stream",
        )

        # Create prediction session document
        doc: Dict[str, Any] = {
            "prediction_id": prediction_id,
            "uploaded_at": _utc_now(),
            "input_file_id": input_file_id,
            "filename": filename,
            "n_patients": n_patients,
            "model_predictions": model_predictions,
            "stored_at": _utc_now(),
        }

        if best_model_info:
            doc["best_model_info"] = best_model_info

        if patient_display_data:
            doc["patient_display_data"] = patient_display_data

        return self.prediction_sessions.insert_one(doc).inserted_id

    # ----------------------- Indexes (used by init script) -----------------------

    def ensure_indexes(self) -> None:
        self.datasets.create_index([("execution_id", ASCENDING)])
        self.datasets.create_index([("stored_at", DESCENDING)])

        self.pipeline_executions.create_index([("execution_id", ASCENDING)], unique=True)
        self.pipeline_executions.create_index([("updated_at", DESCENDING)])

        self.models.create_index(
            [("execution_id", ASCENDING), ("model", ASCENDING), ("dataset_type", ASCENDING)],
            unique=True,
        )
        self.models.create_index([("stored_at", DESCENDING)])

        self.model_evaluations.create_index([("execution_id", ASCENDING)])
        self.model_evaluations.create_index([("model_id", ASCENDING)])

        self.model_comparisons.create_index([("stored_at", DESCENDING)])
        self.model_comparisons.create_index([("execution_id", ASCENDING)])

        self.dashboard_data.create_index([("stored_at", DESCENDING)])
        self.preprocess_artifacts.create_index([("stored_at", DESCENDING)])
        
        self.prediction_sessions.create_index([("prediction_id", ASCENDING)], unique=True)
        self.prediction_sessions.create_index([("uploaded_at", DESCENDING)])
