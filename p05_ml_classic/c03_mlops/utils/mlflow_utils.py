"""
Small helpers shared by the MLOps examples: a local, database-backed MLflow setup and the churn data.

Everything lives in the outputs/ folder next to the chapter scripts:
    outputs/mlflow.db        - SQLite backend store (runs, params, metrics, tags, model registry)
    outputs/mlartifacts/     - artifact store (models, plots, files)
A database-backed store is required for the model registry (the plain file store does not support it).
"""

import os
from pathlib import Path

import mlflow
import pandas as pd

from p05_ml_classic.c01_gradient_boosted_trees.utils.churn_data import make_churn_dataset, train_val_test_split

CHAPTER_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = CHAPTER_DIR / "outputs"
EXPERIMENT_NAME = "churn-classification"
MODEL_NAME = "churn-classifier"
SEED = 0


def tracking_uri() -> str:
    """SQLite tracking URI under outputs/ (absolute path, so it works from any working directory)."""
    return f"sqlite:///{OUTPUT_DIR / 'mlflow.db'}"


def setup_mlflow(experiment_name: str = EXPERIMENT_NAME) -> str:
    """
    Point MLflow at the local SQLite store and create/select the experiment.

    The artifact location is set explicitly when the experiment is created; otherwise MLflow would write
    artifacts to ./mlruns relative to the current working directory.
    Returns the experiment id.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    mlflow.set_tracking_uri(tracking_uri())
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(
            experiment_name, artifact_location=(OUTPUT_DIR / "mlartifacts").as_uri()
        )
    else:
        experiment_id = experiment.experiment_id
    mlflow.set_experiment(experiment_name)
    return experiment_id


def load_churn_splits(
    n_samples: int = 20_000,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Reference churn data (numeric frame) split 60/20/20 into train / validation / test."""
    X, y = make_churn_dataset(n_samples=n_samples, seed=SEED, categorical="codes")
    return train_val_test_split(X, y, seed=SEED)
