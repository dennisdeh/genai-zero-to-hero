"""
MLflow set-up: a local SQLite tracking store and artefact folder under this chapter's outputs/ folder.

View the results with:
    mlflow ui --backend-store-uri sqlite:///p07_llms/c07_validation/outputs/mlflow.db
(run from the repository root, then open http://127.0.0.1:5000).
"""

import os

import mlflow

OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs"
)


def setup_mlflow(experiment_name: str) -> str:
    """
    Point MLflow to outputs/mlflow.db, create the experiment with its artefacts in outputs/mlartifacts and
    activate it. Returns the tracking URI.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    tracking_uri = "sqlite:///" + os.path.join(OUTPUT_DIR, "mlflow.db")
    mlflow.set_tracking_uri(tracking_uri)
    if mlflow.get_experiment_by_name(experiment_name) is None:
        artifact_dir = os.path.join(OUTPUT_DIR, "mlartifacts", experiment_name)
        mlflow.create_experiment(
            experiment_name, artifact_location="file://" + artifact_dir
        )
    mlflow.set_experiment(experiment_name)
    return tracking_uri
