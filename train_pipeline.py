from argparse import ArgumentParser
from pathlib import Path
from typing import List
from datetime import datetime
import uuid

from src.config import DEFAULT_INPUT_PATH, PREPROCESS_MODE, DEFAULT_MODELS
from src.data_ingest import run_ingest
from src.preprocess import run_preprocessing
from src.balance import run_balance
from src.train import run_training
from src.evaluate import run_evaluation
from src.compare import run_comparison
from src.export_dashboard import run_export_dashboard
from src.feature_importance import run_feature_importance
from src.rachs1 import run_rachs1
from src.utils.logging_config import setup_pipeline_logging, get_logger

try:
    from src.db.mongo_store import MongoStore
except Exception:
    MongoStore = None  # type: ignore


def parse_args() -> ArgumentParser:
    parser = ArgumentParser(description="End-to-end training pipeline for PDG project")
    parser.add_argument(
        "--input-path",
        type=str,
        default=str(DEFAULT_INPUT_PATH),
        help=f"Raw data file to ingest (xlsx or csv). Default: {DEFAULT_INPUT_PATH}",
    )
    parser.add_argument("--raw-path", type=str, default=None, help="Existing raw dataset CSV in data/datasets/")
    parser.add_argument("--version-tag", type=str, default=None, help="Optional version tag for this run")
    parser.add_argument(
        "--preprocess-mode",
        type=str,
        default=PREPROCESS_MODE,
        help="Preprocessing strategy to use (e.g., pipeline, notebook_v1)",
    )
    parser.add_argument(
        "--steps",
        type=str,
        default="ingest,preprocess,balance,train,evaluate,compare,feature_importance,export,rachs1",
        help="Comma-separated list of steps to run",
    )
    parser.add_argument(
        "--models",
        type=str,
        default=None,
        help="Comma-separated list of models: logistic_regression,random_forest,svm,xgboost",
    )
    parser.add_argument("--use-gridsearch", action="store_true", help="Use GridSearchCV during training")
    parser.add_argument("--log-file", type=str, default="logs/pipeline.log", help="Path to write pipeline log")
    parser.add_argument(
        "--execution-id",
        type=str,
        default=None,
        help="Optional execution id (used by API + MongoDB persistence).",
    )
    return parser


def main() -> None:
    parser = parse_args()
    args = parser.parse_args()

    execution_id = args.execution_id
    if not execution_id:
        execution_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"

    setup_pipeline_logging(log_file=args.log_file)
    logger = get_logger("pdg.pipeline")

    steps: List[str] = [s.strip() for s in args.steps.split(",") if s.strip()]

    raw_dataset_path: Path | None = None

    logger.info("=" * 60)
    logger.info("PDG - Pipeline de Entrenamiento")
    logger.info("=" * 60)

    if "ingest" in steps:
        logger.info("[1/8] Ingesta de datos...")
        raw_dataset_path = run_ingest(args.input_path, version_tag=args.version_tag)
        logger.info("  Dataset almacenado en: %s", raw_dataset_path)
    elif args.raw_path:
        raw_dataset_path = Path(args.raw_path)

    if "preprocess" in steps:
        if raw_dataset_path is None:
            if args.raw_path:
                raw_dataset_path = Path(args.raw_path)
            else:
                raise ValueError("Raw dataset path is required for preprocessing (use --raw-path or --input-path)")
        logger.info("[2/8] Preprocesamiento (binario: 0=no murió, 1=murió)...")
        run_preprocessing(
            str(raw_dataset_path),
            version_tag=args.version_tag,
            mode=args.preprocess_mode,
        )
        logger.info("  Preprocesamiento completado.")

    if "balance" in steps:
        logger.info("[3/8] Balanceo de clases (ADASYN 70/30)...")
        run_balance(version_tag=args.version_tag)
        logger.info("  Balanceo completado.")

    model_list = None
    if args.models:
        model_list = [m.strip() for m in args.models.split(",") if m.strip()]

    if "train" in steps:
        logger.info("[4/8] Entrenamiento de modelos (datos completos)...")
        run_training(model_names=model_list, use_gridsearch=args.use_gridsearch)
        logger.info("  Entrenamiento completado.")

    if "evaluate" in steps:
        logger.info("[5/8] Evaluación via 5-Fold Stratified CV (Out-of-Fold)...")
        out_path = run_evaluation(model_names=model_list)
        logger.info("  Evaluación guardada en: %s", out_path)

    if "compare" in steps:
        logger.info("[6/8] Comparación de modelos...")
        comparison_path = run_comparison()
        logger.info("  Comparación guardada en: %s", comparison_path)

    if "feature_importance" in steps:
        logger.info("[6b/8] Importancia de variables (todos los modelos entrenados)...")
        try:
            run_feature_importance()
            logger.info("  Importancia de variables completada.")
        except (FileNotFoundError, ValueError) as exc:
            logger.warning("  No se pudo calcular importancia: %s", exc)

    if "export" in steps:
        logger.info("[7/8] Exportando datos para el dashboard...")
        dashboard_path = run_export_dashboard()
        logger.info("  Dashboard data: %s", dashboard_path)

    if "rachs1" in steps:
        logger.info("[8/8] Calculando métricas RACHS-1...")
        try:
            rachs1_path = run_rachs1()
            logger.info("  RACHS-1 metrics: %s", rachs1_path)
            if "export" in steps:
                logger.info("  Re-exportando dashboard con datos RACHS-1...")
                run_export_dashboard()
        except FileNotFoundError as exc:
            logger.warning("  No se pudo calcular RACHS-1: %s", exc)
        except Exception as exc:
            logger.warning("  Error en RACHS-1: %s", exc)

    logger.info("=" * 60)
    logger.info("Pipeline completado exitosamente!")
    logger.info("=" * 60)

    # Optional persistence to MongoDB (when MONGODB_URI is configured)
    if MongoStore is not None:
        try:
            store = MongoStore.from_env()
        except Exception:
            store = None
        if store is not None and store.ping():
            try:
                store.upsert_pipeline_execution(
                    execution_id,
                    status="success",
                    started_at=None,
                    completed_at=datetime.now(),
                    input_path=args.input_path,
                    steps=args.steps,
                    models=args.models,
                    message="Pipeline completado exitosamente.",
                )
            except Exception:
                pass

            # Persist raw dataset + preprocess artifacts + models/results when present
            try:
                store.persist_dataset_files(
                    execution_id=execution_id,
                    raw_csv_path=raw_dataset_path,
                    input_path=Path(args.input_path) if args.input_path else None,
                )
            except Exception:
                pass

            try:
                store.persist_preprocess_artifacts(execution_id=execution_id)
            except Exception:
                pass

            try:
                names = model_list or DEFAULT_MODELS
                store.persist_models_and_results(execution_id=execution_id, model_names=names)
            except Exception:
                pass

            try:
                from src.export_dashboard import build_dashboard_data

                store.save_dashboard_data(build_dashboard_data(), execution_id=execution_id)
            except Exception:
                pass


if __name__ == "__main__":
    main()
