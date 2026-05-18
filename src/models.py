"""
Model factory for the PDG pipeline.

Resolution order for each model
--------------------------------
1. Load from  models/<name>/model.py  (per-model file, config-driven).
2. Fall back to the inline definitions below (for robustness / legacy).

Adding a new model
------------------
Create  models/<new_name>/model.py  with ``build_model()`` and
``get_validation_grid()``, plus  models/<new_name>/config.yml``.
No changes to this file are needed.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from src.config import MODELS_DIR, RANDOM_STATE
from src.utils.base_trainer import (
    build_model_from_folder,
    get_evaluation_config_from_folder,
    get_training_config_from_folder,
    get_validation_grid_from_folder,
)


def get_model(
    model_name: str,
    random_state: int = RANDOM_STATE,
    class_weight: str | None = None,
) -> Any:
    """
    Return an unfitted estimator for *model_name*.

    Tries the per-model model.py first; falls back to inline defaults.
    """
    name = model_name.lower()

    model = build_model_from_folder(name, MODELS_DIR, random_state, class_weight)
    if model is not None:
        return model

    return _build_inline(name, random_state, class_weight)


def get_param_grid(model_name: str) -> Dict[str, list]:
    """
    Return the GridSearchCV parameter grid for *model_name*.

    Tries the per-model config.yml first; falls back to inline defaults.
    """
    name = model_name.lower()

    grid = get_validation_grid_from_folder(name, MODELS_DIR)
    if grid:
        return grid

    return _inline_param_grid(name)


def get_training_config(model_name: str) -> Dict[str, Any]:
    """Return per-model training settings from models/<name>/config.yml."""
    name = model_name.lower()
    return get_training_config_from_folder(name, MODELS_DIR)


def get_evaluation_config(model_name: str) -> Dict[str, Any]:
    """Return per-model evaluation settings from models/<name>/config.yml."""
    name = model_name.lower()
    return get_evaluation_config_from_folder(name, MODELS_DIR)


def _build_inline(name: str, random_state: int, class_weight) -> Any:
    """Inline fallback model definitions (used when model.py is absent)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    from sklearn.tree import DecisionTreeClassifier

    if name == "logistic_regression":
        return LogisticRegression(
            solver="lbfgs", max_iter=1000, class_weight=class_weight
        )
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            random_state=random_state,
            class_weight=class_weight,
            n_jobs=-1,
        )
    if name == "svm":
        return SVC(
            kernel="rbf",
            probability=True,
            class_weight=class_weight,
            random_state=random_state,
        )
    if name == "xgboost":
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:
            raise ImportError(
                "xgboost is not installed but 'xgboost' model was requested"
            ) from exc
        return XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=random_state,
        )
    if name == "decision_tree":
        return DecisionTreeClassifier(
            class_weight=class_weight,
            random_state=random_state,
        )
    raise ValueError(f"Unknown model name: '{name}'")


def _inline_param_grid(name: str) -> Dict[str, list]:
    """Inline fallback parameter grids."""
    if name == "logistic_regression":
        return {"model__C": [0.1, 1.0, 10.0]}
    if name == "random_forest":
        return {"model__n_estimators": [100, 200], "model__max_depth": [None, 10]}
    if name == "svm":
        return {"model__C": [0.5, 1.0, 2.0], "model__gamma": ["scale", "auto"]}
    if name == "xgboost":
        return {
            "model__n_estimators": [200, 300],
            "model__max_depth": [3, 5],
            "model__learning_rate": [0.05, 0.1],
        }
    if name == "decision_tree":
        return {
            "model__max_depth": [3, 5, 7, 10, None],
            "model__min_samples_split": [2, 5, 10, 20],
            "model__min_samples_leaf": [1, 2, 4, 8],
            "model__criterion": ["gini", "entropy"],
        }
    return {}
