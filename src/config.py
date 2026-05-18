"""
Central configuration for the PDG pipeline.

Settings are loaded from config/*.yml when present; hardcoded defaults
are used as fallback so the project works even without YAML files.
"""
import os
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    # Optional dependency / best-effort load.
    pass

BASE_DIR = Path(__file__).resolve().parents[1]
CONFIG_DIR = BASE_DIR / "config"
LEGACY_CONFIGS_DIR = BASE_DIR / "configs"


def _load_yml(name: str) -> dict:
    """Load a YAML config file, returning an empty dict if missing."""
    path = CONFIG_DIR / name
    if not path.exists():
        path = LEGACY_CONFIGS_DIR / name
    if not path.exists():
        return {}
    try:
        import yaml
        with open(path, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


_train_cfg = _load_yml("training.yml")
_extract_cfg = _load_yml("extraction.yml")
_valid_cfg = _load_yml("validation.yml")

DATASETS_DIR = BASE_DIR / "data" / "datasets"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
RESULTS_DIR = BASE_DIR / "data" / "results"
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"
DEFAULT_INPUT_PATH = DATASETS_DIR / "dataset_inicial.xlsx"

# MongoDB (optional). Enabled when MONGODB_URI is set.
MONGODB_URI: str | None = os.getenv("MONGODB_URI") or None
MONGODB_DB_NAME: str = os.getenv("MONGODB_DB_NAME", "pdg_db")
MONGODB_CONNECT_TIMEOUT_MS: int = int(os.getenv("MONGODB_CONNECT_TIMEOUT_MS", "3000"))

# Storage flags (optional)
PDG_MONGO_STORE_PROCESSED_PICKLES: bool = os.getenv("PDG_MONGO_STORE_PROCESSED_PICKLES", "0").lower() in (
    "1",
    "true",
    "yes",
)

TARGET_COLUMN: str = _extract_cfg.get("target_column", "Mortalidad")
PREPROCESS_MODE: str = _extract_cfg.get("preprocess_mode", "pipeline")
RANDOM_STATE: int = _train_cfg.get("random_state", 42)
CV_N_SPLITS: int = _valid_cfg.get("n_splits", 5)
CV_SCORING: str = _valid_cfg.get("scoring", "recall_macro")

DEFAULT_MODELS: list[str] = _train_cfg.get(
    "models",
    ["logistic_regression", "random_forest", "svm", "xgboost", "decision_tree"],
)
