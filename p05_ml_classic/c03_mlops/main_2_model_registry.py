"""
MLflow model registry: register the best run's model, manage versions with aliases, load and predict.

This script demonstrates:
    1. Selecting the best run of the model-selection experiment with mlflow.search_runs (runs
       main_1_experiment_tracking first if the experiment is empty).
    2. A final, one-off evaluation of the selected model on the untouched TEST set, logged to its run.
    3. Registering the model as a new version of the registered model "churn-classifier", with a description
       and tags, and pointing the alias "champion" at it (aliases replace the deprecated stages
       Staging/Production).
    4. Loading the model back by alias (models:/churn-classifier@champion) and by version
       (models:/churn-classifier/<version>), both through the generic pyfunc interface and the sklearn flavour,
       and checking that the predictions agree.
    5. How to serve the model as a REST API (printed instructions; see the README/book).

How to run (from the repository root), after or without main_1:
    python -m p05_ml_classic.c03_mlops.main_2_model_registry
"""

import mlflow
import numpy as np
from mlflow import MlflowClient
from sklearn.metrics import log_loss, roc_auc_score

from p05_ml_classic.c03_mlops.main_1_experiment_tracking import run_experiments
from p05_ml_classic.c03_mlops.utils.mlflow_utils import (
    EXPERIMENT_NAME,
    MODEL_NAME,
    load_churn_splits,
    setup_mlflow,
    tracking_uri,
)

CHAMPION_ALIAS = "champion"


def best_run():
    """Return the best model-selection run (highest validation ROC-AUC) as a pandas Series."""
    runs = mlflow.search_runs(
        experiment_names=[EXPERIMENT_NAME],
        filter_string="tags.stage = 'model-selection' and attributes.status = 'FINISHED'",
        order_by=["metrics.val_roc_auc DESC"],
    )
    if runs.empty:
        print("No model-selection runs found - running main_1_experiment_tracking first ...")
        runs = run_experiments()
    return runs.iloc[0]


if __name__ == "__main__":
    setup_mlflow()
    client = MlflowClient()

    # 1. Pick the best run
    best = best_run()
    run_id, model_uri = best.run_id, best["tags.model_uri"]
    print(f"Best run: {best['tags.mlflow.runName']} ({run_id}), val ROC-AUC={best['metrics.val_roc_auc']:.4f}")

    # 2. Evaluate ONCE on the test set and log the result to the same run
    _, _, X_test, _, _, y_test = load_churn_splits()
    sk_model = mlflow.sklearn.load_model(model_uri)
    p_test = sk_model.predict_proba(X_test)[:, 1]
    test_metrics = {"test_roc_auc": roc_auc_score(y_test, p_test), "test_log_loss": log_loss(y_test, p_test)}
    with mlflow.start_run(run_id=run_id):  # re-open the finished run to add metrics
        mlflow.log_metrics(test_metrics)
    print(f"Test ROC-AUC={test_metrics['test_roc_auc']:.4f}, test log-loss={test_metrics['test_log_loss']:.4f}")

    # 3. Register a new model version and point the 'champion' alias at it
    mv = mlflow.register_model(model_uri, MODEL_NAME)
    client.update_registered_model(MODEL_NAME, description="Predicts the probability that a customer churns.")
    client.update_model_version(
        MODEL_NAME, mv.version,
        description=f"{best['tags.model_family']} selected by validation ROC-AUC from run {run_id}.",
    )
    for key, value in {**test_metrics, "model_family": best["tags.model_family"], "validated": "true"}.items():
        client.set_model_version_tag(MODEL_NAME, mv.version, key, str(value))
    client.set_registered_model_alias(MODEL_NAME, CHAMPION_ALIAS, mv.version)
    print(f"Registered '{MODEL_NAME}' version {mv.version} and set alias '@{CHAMPION_ALIAS}' -> v{mv.version}")

    versions = client.search_model_versions(f"name = '{MODEL_NAME}'")
    print(f"All versions of '{MODEL_NAME}':")
    for v in sorted(versions, key=lambda v: int(v.version)):
        v = client.get_model_version(MODEL_NAME, v.version)  # search results may omit aliases
        print(f"  v{v.version}: aliases={list(v.aliases)}, family={v.tags.get('model_family')}, "
              f"test_roc_auc={float(v.tags.get('test_roc_auc', 'nan')):.4f}")

    # 4. Load by alias and by version - consumers only need the name and alias, not the run id
    champion = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}@{CHAMPION_ALIAS}")
    by_version = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}/{mv.version}")
    new_customers = X_test.head(5)
    proba_alias = np.asarray(champion.predict(new_customers))[:, 1]  # pyfunc_predict_fn = predict_proba
    proba_version = np.asarray(by_version.predict(new_customers))[:, 1]
    print("\nChurn probabilities for 5 new customers (pyfunc, loaded via alias):", np.round(proba_alias, 3))
    print("Identical when loaded by version number:", bool(np.allclose(proba_alias, proba_version)))
    print("Input schema enforced by the signature:", champion.metadata.get_input_schema().input_names())

    # 5. Serving instructions
    print(
        "\nServe the champion as a REST API (in a separate terminal, from the repository root):\n"
        f"  export MLFLOW_TRACKING_URI={tracking_uri()}\n"
        f"  mlflow models serve -m 'models:/{MODEL_NAME}@{CHAMPION_ALIAS}' -p 5001 --env-manager local\n"
        "and query it with:\n"
        "  curl -X POST http://127.0.0.1:5001/invocations -H 'Content-Type: application/json' \\\n"
        f"       -d '{{\"dataframe_split\": {new_customers.head(1).to_json(orient='split', index=False)}}}'\n"
        "Browse runs and the registry with:\n"
        f"  mlflow ui --backend-store-uri {tracking_uri()} --port 5000"
    )
