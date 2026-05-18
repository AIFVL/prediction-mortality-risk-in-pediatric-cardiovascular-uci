"""
FastAPI backend — serves dashboard data and triggers pipeline steps.
"""
import subprocess
import threading
import json
import io
from pathlib import Path
from datetime import datetime
from typing import Optional
import uuid

from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from src.config import PREPROCESS_MODE

try:
    from src.db.mongo_store import MongoStore
except Exception:
    MongoStore = None  # type: ignore

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "data" / "results"
DASHBOARD_DATA_PATH = RESULTS_DIR / "dashboard_data.json"
FRONTEND_DIR = BASE_DIR / "dashboard" / "dist"

app = FastAPI(title="PDG Dashboard API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_pipeline_status = {
    "running": False,
    "last_run": None,
    "last_status": "idle",
    "last_message": "",
    "log": [],
    "execution_id": None,
}


class PipelineRequest(BaseModel):
    input_path: Optional[str] = "data/datasets/dataset_inicial.xlsx"
    models: Optional[str] = None
    use_gridsearch: bool = False
    preprocess_mode: str = PREPROCESS_MODE
    steps: str = "ingest,preprocess,balance,train,evaluate,compare,feature_importance,export,rachs1"


def _run_pipeline_thread(req: PipelineRequest, execution_id: str):
    _pipeline_status["running"] = True
    _pipeline_status["last_status"] = "running"
    _pipeline_status["log"] = []
    _pipeline_status["last_run"] = datetime.now().isoformat(timespec="seconds")
    _pipeline_status["execution_id"] = execution_id

    store = None
    if MongoStore is not None:
        try:
            store = MongoStore.from_env()
        except Exception:
            store = None
    if store is not None and store.ping():
        try:
            store.upsert_pipeline_execution(
                execution_id,
                status="running",
                started_at=datetime.now(),
                input_path=req.input_path,
                steps=req.steps,
                models=req.models,
                message="Pipeline iniciado.",
            )
        except Exception:
            pass

    cmd = [
        "python", "-u", "train_pipeline.py",
        "--input-path", req.input_path,
        "--steps", req.steps,
        "--execution-id", execution_id,
    ]
    if req.preprocess_mode:
        cmd += ["--preprocess-mode", req.preprocess_mode]
    if req.models:
        cmd += ["--models", req.models]
    if req.use_gridsearch:
        cmd.append("--use-gridsearch")

    try:
        proc = subprocess.Popen(
            cmd,
            cwd=str(BASE_DIR),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        for line in proc.stdout:
            line = line.rstrip()
            _pipeline_status["log"].append(line)
        proc.wait()
        if proc.returncode == 0:
            _pipeline_status["last_status"] = "success"
            _pipeline_status["last_message"] = "Pipeline completado exitosamente."
        else:
            _pipeline_status["last_status"] = "error"
            _pipeline_status["last_message"] = "El pipeline finalizó con errores. Revisa el log."
    except Exception as e:
        _pipeline_status["last_status"] = "error"
        _pipeline_status["last_message"] = str(e)
    finally:
        if store is not None and store.ping():
            try:
                store.upsert_pipeline_execution(
                    execution_id,
                    status=_pipeline_status.get("last_status", "unknown"),
                    completed_at=datetime.now(),
                    log_lines=_pipeline_status.get("log", [])[-5000:],
                    message=_pipeline_status.get("last_message", ""),
                    error=_pipeline_status.get("last_message") if _pipeline_status.get("last_status") == "error" else None,
                )
            except Exception:
                pass
        _pipeline_status["running"] = False


@app.get("/api/health")
def health():
    mongo = {"enabled": False, "ok": None}
    if MongoStore is not None:
        try:
            store = MongoStore.from_env()
            if store is not None:
                mongo["enabled"] = True
                mongo["ok"] = store.ping()
        except Exception:
            mongo = {"enabled": True, "ok": False}
    return {"status": "ok", "time": datetime.now().isoformat(), "mongo": mongo}


@app.get("/api/results")
def get_results():
    if MongoStore is not None:
        try:
            store = MongoStore.from_env()
            if store is not None and store.ping():
                latest = store.get_latest_dashboard_data()
                if latest:
                    return latest
        except Exception:
            pass

    if not DASHBOARD_DATA_PATH.exists():
        return JSONResponse({"status": "no_results", "message": "No hay resultados aún. Ejecuta el pipeline."})
    with open(DASHBOARD_DATA_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


@app.post("/api/pipeline/run")
def run_pipeline(req: PipelineRequest, background_tasks: BackgroundTasks):
    if _pipeline_status["running"]:
        raise HTTPException(status_code=409, detail="Pipeline ya está en ejecución.")
    execution_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    background_tasks.add_task(_run_pipeline_thread, req, execution_id)
    return {"message": "Pipeline iniciado.", "status": "running", "execution_id": execution_id}


@app.get("/api/pipeline/status")
def pipeline_status():
    return _pipeline_status


@app.get("/api/pipeline/log")
def pipeline_log():
    return {"log": _pipeline_status.get("log", [])}


@app.get("/api/pipeline/executions")
def list_pipeline_executions(limit: int = 20):
    """List recent pipeline executions (MongoDB only)."""
    if MongoStore is None:
        return {"enabled": False, "executions": []}

    try:
        store = MongoStore.from_env()
        if store is None or not store.ping():
            return {"enabled": False, "executions": []}

        docs = list(
            store.pipeline_executions.find(
                {},
                {
                    "_id": 0,
                    "execution_id": 1,
                    "status": 1,
                    "created_at": 1,
                    "started_at": 1,
                    "completed_at": 1,
                    "input_path": 1,
                    "steps": 1,
                    "models": 1,
                    "message": 1,
                },
            )
            .sort("updated_at", -1)
            .limit(max(1, min(limit, 200)))
        )
        return {"enabled": True, "executions": docs}
    except Exception:
        return {"enabled": True, "executions": []}


@app.post("/api/predict-upload")
async def predict_upload(file: UploadFile = File(...)):
    """
    Accept a CSV / XLSX / TXT file with new patient data and return
    predictions from every trained model.
    Saves prediction session to MongoDB if available.
    """
    allowed = {".csv", ".xlsx", ".xls", ".txt"}
    ext = Path(file.filename).suffix.lower()
    if ext not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"Tipo de archivo no soportado: '{ext}'. Usa CSV, XLSX o TXT.",
        )

    content = await file.read()
    if len(content) == 0:
        raise HTTPException(status_code=400, detail="El archivo está vacío.")

    try:
        from src.inference import run_batch_inference
        result = run_batch_inference(content, file.filename)
        
        # Save prediction session to MongoDB if available
        if MongoStore is not None:
            try:
                store = MongoStore.from_env()
                if store is not None and store.ping():
                    prediction_id = f"pred_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
                    store.save_prediction_session(
                        prediction_id=prediction_id,
                        file_content=content,
                        filename=file.filename,
                        n_patients=result.get("n_patients", 0),
                        model_predictions=result.get("model_predictions", []),
                        best_model_info=result.get("best_model_key"),
                        patient_display_data=result.get("patient_rows", []),
                    )
            except Exception:
                pass  # MongoDB storage is optional; don't fail if unavailable
        
        return JSONResponse(result)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Error en la predicción: {exc}")


if FRONTEND_DIR.exists():
    app.mount("/assets", StaticFiles(directory=str(FRONTEND_DIR / "assets")), name="assets")

    @app.get("/{full_path:path}")
    def serve_spa(full_path: str):
        index = FRONTEND_DIR / "index.html"
        if index.exists():
            return FileResponse(str(index))
        raise HTTPException(status_code=404, detail="Frontend not built yet.")
