"""
Generic trainer for scikit-learn compatible models.

Responsibilities
----------------
- Build a StandardScaler + model Pipeline.
- Optionally run GridSearchCV using the model's validation_grid.
- Persist the best estimator to <model_dir>/model.pkl.
- Provide a helper to load a model's config.yml from its folder.

This module is the single place that knows how to fit and save a model.
src/train.py uses it as an orchestrator; it does not duplicate this logic.
"""
import importlib.util
import pickle
from pathlib import Path
from typing import Optional

import pandas as pd
import yaml
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

from src.io_utils import ensure_dir, write_csv
from src.config import RESULTS_DIR
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


def load_model_config_yml(model_file: str) -> dict:
    """
    Load config.yml from the same directory as a model.py file.

    Intended to be called inside a per-model model.py as:
        cfg = load_model_config_yml(__file__)
    """
    cfg_path = Path(model_file).parent / "config.yml"
    if not cfg_path.exists():
        return {}
    with open(cfg_path) as f:
        return yaml.safe_load(f) or {}


def _load_model_module(model_name: str, models_dir: Path):
    """
    Dynamically import models/<model_name>/model.py and return the module.
    Returns None if the file does not exist.
    """
    model_py = models_dir / model_name / "model.py"
    if not model_py.exists():
        return None
    spec = importlib.util.spec_from_file_location(
        f"models.{model_name}.model", model_py
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_model_from_folder(
    model_name: str,
    models_dir: Path,
    random_state: int = 42,
    class_weight=None,
):
    """
    Build a model instance by delegating to models/<model_name>/model.py.
    Returns None if the per-model file does not exist.
    """
    module = _load_model_module(model_name, models_dir)
    if module is None or not hasattr(module, "build_model"):
        return None
    return module.build_model(random_state=random_state, class_weight=class_weight)


def get_validation_grid_from_folder(model_name: str, models_dir: Path) -> dict:
    """
    Load the validation_grid from models/<model_name>/config.yml via model.py.
    Returns an empty dict if not available.
    """
    module = _load_model_module(model_name, models_dir)
    if module is None or not hasattr(module, "get_validation_grid"):
        return {}
    return module.get_validation_grid()


def get_training_config_from_folder(model_name: str, models_dir: Path) -> dict:
    """Load optional training settings from models/<model_name>/model.py."""
    module = _load_model_module(model_name, models_dir)
    if module is None or not hasattr(module, "get_training_config"):
        return {}
    return module.get_training_config()


def get_evaluation_config_from_folder(model_name: str, models_dir: Path) -> dict:
    """Load optional evaluation settings from models/<model_name>/model.py."""
    module = _load_model_module(model_name, models_dir)
    if module is None or not hasattr(module, "get_evaluation_config"):
        return {}
    return module.get_evaluation_config()


def train_single_model(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    model_name: str,
    dataset_type: str,
    model_dir: Path,
    use_gridsearch: bool = False,
    param_grid: Optional[dict] = None,
    scoring: str = "recall_macro",
    cv: int = 3,
    fit_sample_weight: bool = False,
) -> Pipeline:
    """
    Fit a scikit-learn compatible model inside a StandardScaler pipeline,
    optionally using GridSearchCV, and save the best estimator.

    Parameters
    ----------
    model        : unfitted estimator
    X, y         : training data
    model_name   : used for naming result files
    dataset_type : 'raw' or 'smote'
    model_dir    : directory where model.pkl is saved
    use_gridsearch : whether to run GridSearchCV
    param_grid   : parameter grid for GridSearchCV (keys must use 'model__' prefix)
    scoring      : GridSearchCV scoring metric
    cv           : number of CV folds in GridSearchCV
    fit_sample_weight : whether to pass balanced sample weights to model.fit

    Returns
    -------
    best_estimator : fitted Pipeline
    """
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", model),
    ])

    ensure_dir(model_dir)
    fit_params = {}
    if fit_sample_weight:
        fit_params["model__sample_weight"] = compute_sample_weight("balanced", y)

    if use_gridsearch and param_grid:
        logger.info("  Running GridSearchCV for %s [%s]...", model_name, dataset_type)
        search = GridSearchCV(
            pipeline,
            param_grid=param_grid,
            scoring=scoring,
            cv=cv,
            n_jobs=-1,
            refit=True,
        )
        try:
            search.fit(X, y, **fit_params)
        except TypeError:
            logger.warning("  Sample weights not supported by %s; fitting without them.", model_name)
            search.fit(X, y)
        best_estimator = search.best_estimator_

        results_dir = RESULTS_DIR / model_name
        ensure_dir(results_dir)
        results_df = pd.DataFrame(search.cv_results_)
        write_csv(results_df, results_dir / f"gridsearch_results_{dataset_type}.csv")
        logger.info(
            "  Best params for %s [%s]: %s", model_name, dataset_type, search.best_params_
        )
    else:
        try:
            pipeline.fit(X, y, **fit_params)
        except TypeError:
            logger.warning("  Sample weights not supported by %s; fitting without them.", model_name)
            pipeline.fit(X, y)
        best_estimator = pipeline

    model_path = model_dir / "model.pkl"
    with model_path.open("wb") as f:
        pickle.dump(best_estimator, f)

    logger.info("  Saved %s [%s] -> %s", model_name, dataset_type, model_path)
    return best_estimator
