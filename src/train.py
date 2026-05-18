"""
Training orchestrator for the PDG pipeline.

This module is a thin facade: it iterates over model/dataset combinations
and delegates the actual fit-and-save work to src/utils/base_trainer.
"""
from typing import Iterable, Optional

from src.config import (
    PROCESSED_DIR,
    MODELS_DIR,
    RANDOM_STATE,
    DEFAULT_MODELS,
    TARGET_COLUMN,
    CV_SCORING,
)
from src.utils.data_loader import load_train_split
from src.utils.base_trainer import train_single_model
from src.utils.logging_config import get_logger
from src.models import get_model, get_param_grid, get_training_config

logger = get_logger(__name__)


def run_training(
    model_names: Optional[Iterable[str]] = None,
    dataset_types: Iterable[str] = ("raw", "balanced"),
    use_gridsearch: bool = False,
    scoring: str = CV_SCORING,
    cv: int = 3,
) -> None:
    """
    Train each model on each dataset type and save model.pkl files.

    Parameters
    ----------
    model_names   : models to train (defaults to DEFAULT_MODELS from config)
    dataset_types : which data variants to use ('raw', 'balanced')
    use_gridsearch: whether to run GridSearchCV
    scoring       : GridSearchCV scoring metric
    cv            : GridSearchCV CV folds
    """
    if model_names is None:
        model_names = DEFAULT_MODELS

    for dataset_type in dataset_types:
        try:
            X, y = load_train_split(dataset_type)
        except FileNotFoundError as exc:
            logger.warning("Skipping dataset_type='%s': %s", dataset_type, exc)
            continue

        for model_name in model_names:
            logger.info("Training %s [%s]...", model_name, dataset_type)
            try:
                model = get_model(
                    model_name,
                    random_state=RANDOM_STATE,
                    class_weight="balanced",
                )
            except (ImportError, ValueError) as exc:
                logger.warning("Skipping %s: %s", model_name, exc)
                continue

            param_grid = get_param_grid(model_name) if use_gridsearch else {}
            model_training_cfg = get_training_config(model_name)
            model_scoring = model_training_cfg.get("scoring", scoring)
            model_cv = model_training_cfg.get("cv_folds", cv)
            fit_sample_weight = model_training_cfg.get("fit_sample_weight", False)

            train_single_model(
                model=model,
                X=X,
                y=y,
                model_name=model_name,
                dataset_type=dataset_type,
                model_dir=MODELS_DIR / model_name / dataset_type,
                use_gridsearch=use_gridsearch,
                param_grid=param_grid,
                scoring=model_scoring,
                cv=model_cv,
                fit_sample_weight=fit_sample_weight,
            )
