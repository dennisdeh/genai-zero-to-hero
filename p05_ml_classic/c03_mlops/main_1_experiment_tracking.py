"""
Experiment tracking with MLflow: log, compare and query model-selection runs.

This script demonstrates:
    1. A local MLflow setup with a SQLite backend store and an artifact folder under outputs/.
    2. One MLflow run per candidate model (logistic regression, random forest, HistGradientBoosting with
       different hyperparameters), logging
         - parameters and tags (model family, data version, purpose),
         - validation metrics, plus a metric HISTORY (validation loss per boosting iteration, using `step`),
         - artifacts: a ROC-curve figure and the feature list,
         - the fitted model with a signature (input/output schema) and an input example.
    3. Comparing runs programmatically with mlflow.search_runs (a pandas DataFrame) - the same information
       is shown in the MLflow UI.

How to run (from the repository root):
    python -m p05_ml_classic.c03_mlops.main_1_experiment_tracking
Then inspect the runs in the UI:
    mlflow ui --backend-store-uri sqlite:///p05_ml_classic/c03_mlops/outputs/mlflow.db --port 5000
"""

import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import pandas as pd
from mlflow.models import infer_signature
from sklearn.base import ClassifierMixin
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import RocCurveDisplay, accuracy_score, f1_score, log_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from p05_ml_classic.c03_mlops.utils.mlflow_utils import EXPERIMENT_NAME, SEED, load_churn_splits, setup_mlflow

DATA_VERSION = "synthetic-churn-v1-seed0"
# MLflow saves scikit-learn models with skops (a safer alternative to pickle): on loading, only types that were
# explicitly trusted are reconstructed. Plain sklearn estimators are trusted by default; the low-level tree
# structures below are not, so we declare them (they are stored in the model's MLmodel file for loading).
SKOPS_TRUSTED_TYPES = {
    "random_forest": ["sklearn.tree._tree.Tree"],
    "hist_gradient_boosting": ["sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor"],
}


def candidate_models() -> list[tuple[str, str, dict, ClassifierMixin]]:
    """(run name, model family, hyperparameters, estimator) for every configuration we want to compare."""
    candidates = []
    for C in [0.01, 1.0]:
        params = {"C": C, "max_iter": 2000}
        candidates.append((f"logreg_C{C}", "logistic_regression", params,
                           make_pipeline(StandardScaler(), LogisticRegression(**params))))
    params = {"n_estimators": 200, "min_samples_leaf": 20}
    candidates.append(("random_forest", "random_forest", params,
                       RandomForestClassifier(**params, n_jobs=-1, random_state=SEED)))
    for lr, leaves in [(0.1, 31), (0.05, 15)]:
        params = {"learning_rate": lr, "max_leaf_nodes": leaves, "max_iter": 1000, "l2_regularization": 1.0}
        candidates.append((f"hgb_lr{lr}_leaves{leaves}", "hist_gradient_boosting", params,
                           HistGradientBoostingClassifier(**params, early_stopping=True, n_iter_no_change=30,
                                                          scoring="loss", random_state=SEED)))
    return candidates


def train_and_log(run_name: str, family: str, params: dict, model: ClassifierMixin, data: tuple) -> str:
    """Fit one model inside an MLflow run and log everything needed to reproduce and compare it."""
    X_train, X_val, y_train, y_val = data
    with mlflow.start_run(run_name=run_name) as run:
        # Tags: searchable metadata that are not hyperparameters
        mlflow.set_tags({"model_family": family, "data_version": DATA_VERSION, "stage": "model-selection"})
        mlflow.log_params(params)
        mlflow.log_params({"n_train": len(X_train), "n_val": len(X_val), "n_features": X_train.shape[1]})

        t0 = time.perf_counter()
        if isinstance(model, HistGradientBoostingClassifier):
            model.fit(X_train, y_train, X_val=X_val, y_val=y_val)  # early stopping on our validation set
            # Metric history: one value per boosting iteration -> rendered as a curve in the MLflow UI
            for step, score in enumerate(model.validation_score_):
                mlflow.log_metric("val_loss_curve", -score, step=step)
            mlflow.log_metric("n_trees", model.n_iter_)
        else:
            model.fit(X_train, y_train)
        mlflow.log_metric("fit_time_s", time.perf_counter() - t0)

        p_val = model.predict_proba(X_val)[:, 1]
        metrics = {
            "val_roc_auc": roc_auc_score(y_val, p_val),
            "val_log_loss": log_loss(y_val, p_val),
            "val_accuracy": accuracy_score(y_val, p_val >= 0.5),
            "val_f1": f1_score(y_val, p_val >= 0.5),
        }
        mlflow.log_metrics(metrics)

        # Artifacts: a figure and a small JSON file
        fig, ax = plt.subplots(figsize=(5, 5))
        RocCurveDisplay.from_predictions(y_val, p_val, ax=ax, name=run_name)
        mlflow.log_figure(fig, "plots/roc_curve_val.png")
        plt.close(fig)
        mlflow.log_dict({"features": list(X_train.columns), "target": "churn (1 = churned)"}, "data/schema.json")

        # The model: signature = input schema (column names/types) + output schema; input_example is stored too.
        # pyfunc_predict_fn="predict_proba" makes the generic pyfunc / REST interface return probabilities.
        signature = infer_signature(X_train, model.predict_proba(X_train.head(100)))
        model_info = mlflow.sklearn.log_model(
            model,
            name="model",
            signature=signature,
            input_example=X_train.head(3),
            pyfunc_predict_fn="predict_proba",
            skops_trusted_types=SKOPS_TRUSTED_TYPES.get(family),
        )
        mlflow.set_tag("model_uri", model_info.model_uri)  # handy for registering the model later
        print(f"  {run_name:28s} val ROC-AUC={metrics['val_roc_auc']:.4f}  log-loss={metrics['val_log_loss']:.4f}"
              f"  (run_id={run.info.run_id[:8]}...)")
        return run.info.run_id


def run_experiments() -> pd.DataFrame:
    """Train and log all candidates, then return the runs ordered by validation ROC-AUC."""
    setup_mlflow()
    X_train, X_val, _, y_train, y_val, _ = load_churn_splits()
    print(f"Logging runs to experiment '{EXPERIMENT_NAME}' at {mlflow.get_tracking_uri()}")
    with threadpool_limits(limits=1, user_api="openmp"):  # small data: one OpenMP thread is fastest
        for run_name, family, params, model in candidate_models():
            train_and_log(run_name, family, params, model, (X_train, X_val, y_train, y_val))

    # Query runs like a database table: filter on tags/params/metrics, order by a metric
    runs = mlflow.search_runs(
        experiment_names=[EXPERIMENT_NAME],
        filter_string="tags.stage = 'model-selection'",
        order_by=["metrics.val_roc_auc DESC"],
    )
    return runs


if __name__ == "__main__":
    # 1.-2. Train and log all candidate models
    runs = run_experiments()

    # 3. Compare (all runs ever logged with stage=model-selection, best first)
    cols = ["tags.mlflow.runName", "tags.model_family", "metrics.val_roc_auc", "metrics.val_log_loss",
            "metrics.fit_time_s", "start_time"]
    print(f"\nmlflow.search_runs: {len(runs)} runs in total (re-running this script adds new runs):")
    print(runs[cols].head(10).to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    best = runs.iloc[0]
    print(f"\nBest run: {best['tags.mlflow.runName']} (run_id={best.run_id}), val ROC-AUC={best['metrics.val_roc_auc']:.4f}")
    # Per-family best, a typical "leaderboard" query done in pandas
    leaderboard = runs.groupby("tags.model_family")["metrics.val_roc_auc"].max().sort_values(ascending=False)
    print("\nBest validation ROC-AUC per model family:")
    print(leaderboard.round(4).to_string())
    print("\nNext: python -m p05_ml_classic.c03_mlops.main_2_model_registry  (registers the best model)")
