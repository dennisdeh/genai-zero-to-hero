"""
Monitoring a deployed model for data drift, and a simple automated retraining trigger (champion/challenger).

This script demonstrates (run main_2_model_registry first, so that a '@champion' model exists):
    1. Loading the production model from the registry by alias.
    2. Simulating weekly batches of production data whose distribution drifts over time
       (more short-tenure customers, higher prices, more fibre - plus a change in price sensitivity).
    3. Drift metrics per feature: the Population Stability Index (PSI) and the Kolmogorov-Smirnov test,
       plus the model's ROC-AUC once the labels of the batch are known; all logged to an MLflow
       "monitoring" run as metric histories (step = week).
    4. A retraining trigger (PSI > 0.2 or a drop in ROC-AUC). When it fires, a challenger is retrained on the
       reference data plus recent labelled data, registered as a new version with alias '@challenger', compared
       with the champion on held-out recent data, and promoted to '@champion' if it is better.

How to run (from the repository root):
    python -m p05_ml_classic.c03_mlops.main_3_drift_monitoring
"""

import mlflow
import numpy as np
import pandas as pd
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
from mlflow.models import infer_signature
from scipy.stats import ks_2samp
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from threadpoolctl import threadpool_limits

from p05_ml_classic.c01_gradient_boosted_trees.utils.churn_data import make_churn_dataset
from p05_ml_classic.c03_mlops.main_1_experiment_tracking import SKOPS_TRUSTED_TYPES
from p05_ml_classic.c03_mlops.utils.mlflow_utils import MODEL_NAME, load_churn_splits, setup_mlflow

PSI_THRESHOLD = 0.2  # rule of thumb: < 0.1 stable, 0.1-0.2 moderate shift, > 0.2 significant shift
AUC_DROP_THRESHOLD = 0.02
WEEKLY_DRIFT = [0.0, 0.1, 0.3, 0.6, 1.0]  # drift intensity of the simulated weekly batches
BATCH_SIZE = 4000


def psi(reference: pd.Series, current: pd.Series, n_bins: int = 10, eps: float = 1e-4) -> float:
    """
    Population Stability Index: sum_b (cur_b - ref_b) * ln(cur_b / ref_b) over bins b, where ref_b and cur_b
    are the fractions of reference / current observations in bin b. Bins are reference quantiles for continuous
    features and the distinct values for discrete ones.
    """
    if reference.nunique() <= n_bins:
        cats = np.union1d(reference.unique(), current.unique())
        ref_frac = reference.value_counts(normalize=True).reindex(cats, fill_value=0).to_numpy()
        cur_frac = current.value_counts(normalize=True).reindex(cats, fill_value=0).to_numpy()
    else:
        edges = np.unique(np.quantile(reference, np.linspace(0, 1, n_bins + 1)[1:-1]))
        edges = np.concatenate([[-np.inf], edges, [np.inf]])
        ref_frac = np.histogram(reference, edges)[0] / len(reference)
        cur_frac = np.histogram(current, edges)[0] / len(current)
    ref_frac, cur_frac = np.clip(ref_frac, eps, None), np.clip(cur_frac, eps, None)
    return float(np.sum((cur_frac - ref_frac) * np.log(cur_frac / ref_frac)))


def drift_report(reference: pd.DataFrame, current: pd.DataFrame) -> pd.DataFrame:
    """PSI and KS-test p-value for every feature."""
    rows = []
    for col in reference.columns:
        rows.append({"feature": col, "psi": psi(reference[col], current[col]),
                     "ks_pvalue": ks_2samp(reference[col], current[col]).pvalue})
    return pd.DataFrame(rows).set_index("feature")


def retrain_challenger(champion, X_ref, y_ref, X_new, y_new, family: str, week: int) -> tuple[str, object]:
    """Retrain a copy of the champion's pipeline on reference + recent data and log it as a new MLflow run."""
    X_fit, y_fit = pd.concat([X_ref, X_new]), pd.concat([y_ref, y_new])
    challenger = clone(champion)  # same model type and hyperparameters, untrained
    if isinstance(challenger, HistGradientBoostingClassifier):
        challenger.set_params(early_stopping=False, max_iter=max(champion.n_iter_, 50))
    with mlflow.start_run(run_name=f"retrain_week{week}", nested=True) as run:  # child of the monitoring run
        mlflow.set_tags({"stage": "retraining", "model_family": family, "trigger_week": week})
        mlflow.log_params({"n_reference": len(X_ref), "n_recent": len(X_new)})
        challenger.fit(X_fit, y_fit)
        info = mlflow.sklearn.log_model(
            challenger, name="model", signature=infer_signature(X_fit, challenger.predict_proba(X_fit.head(100))),
            input_example=X_fit.head(3), pyfunc_predict_fn="predict_proba",
            skops_trusted_types=SKOPS_TRUSTED_TYPES.get(family),
        )
        mlflow.set_tag("model_uri", info.model_uri)
    print(f"  retrained challenger in run {run.info.run_id[:8]}... on {len(X_fit)} rows")
    return info.model_uri, challenger


if __name__ == "__main__":
    setup_mlflow()
    client = MlflowClient()

    # 1. Production model and its reference data / performance
    try:
        champion_mv = client.get_model_version_by_alias(MODEL_NAME, "champion")
    except MlflowException:
        raise SystemExit("No '@champion' model found - run main_2_model_registry first.")
    champion = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@champion")
    family = champion_mv.tags.get("model_family", "unknown")
    ref_auc = float(champion_mv.tags["test_roc_auc"])
    X_ref, _, _, y_ref, _, _ = load_churn_splits()
    print(f"Champion: '{MODEL_NAME}' v{champion_mv.version} ({family}), reference test ROC-AUC={ref_auc:.4f}")

    limiter = threadpool_limits(limits=1, user_api="openmp")
    history = []
    retrained = False
    with mlflow.start_run(run_name="monitoring") as monitoring_run:
        mlflow.set_tags({"stage": "monitoring", "champion_version": champion_mv.version})
        for week, drift in enumerate(WEEKLY_DRIFT, start=1):
            # 2. A new batch of production data (labels arrive later, e.g. after the churn window)
            X_new, y_new = make_churn_dataset(n_samples=BATCH_SIZE, seed=100 + week, drift=drift, categorical="codes")

            # 3. Drift and performance metrics
            report = drift_report(X_ref, X_new)
            auc = roc_auc_score(y_new, champion.predict_proba(X_new)[:, 1])
            max_psi_feature = report["psi"].idxmax()
            mlflow.log_metrics({f"psi_{f}": v for f, v in report["psi"].items()}, step=week)
            mlflow.log_metrics({"max_psi": report["psi"].max(), "roc_auc": auc, "churn_rate": y_new.mean()}, step=week)
            drift_alert = bool(report["psi"].max() > PSI_THRESHOLD)
            perf_alert = bool(auc < ref_auc - AUC_DROP_THRESHOLD)
            history.append({"week": week, "drift": drift, "churn_rate": y_new.mean(), "roc_auc": auc,
                            "max_psi": report["psi"].max(), "max_psi_feature": max_psi_feature,
                            "n_ks_p<0.01": int((report["ks_pvalue"] < 0.01).sum()),
                            "alert": " ".join(n for n, a in [("DRIFT", drift_alert), ("PERF", perf_alert)] if a) or "-"})
            print(f"Week {week}: churn rate={y_new.mean():.3f}, ROC-AUC={auc:.4f}, max PSI={report['psi'].max():.3f} "
                  f"({max_psi_feature}){'  -> ALERT' if drift_alert or perf_alert else ''}")

            # 4. Retraining trigger (once in this demo): champion/challenger on held-out recent data
            if (drift_alert or perf_alert) and not retrained:
                print("  drifted batch: top PSI features ->",
                      ", ".join(f"{f}={v:.2f}" for f, v in report["psi"].nlargest(3).items()))
                X_fit_new, X_hold, y_fit_new, y_hold = train_test_split(
                    X_new, y_new, test_size=0.5, stratify=y_new, random_state=0
                )
                uri, challenger = retrain_challenger(champion, X_ref, y_ref, X_fit_new, y_fit_new, family, week)
                mv = mlflow.register_model(uri, MODEL_NAME)
                client.set_registered_model_alias(MODEL_NAME, "challenger", mv.version)
                auc_champ = roc_auc_score(y_hold, champion.predict_proba(X_hold)[:, 1])
                auc_chall = roc_auc_score(y_hold, challenger.predict_proba(X_hold)[:, 1])
                print(f"  hold-out ROC-AUC on recent data: champion v{champion_mv.version}={auc_champ:.4f}, "
                      f"challenger v{mv.version}={auc_chall:.4f}")
                client.set_model_version_tag(MODEL_NAME, mv.version, "model_family", family)
                client.set_model_version_tag(MODEL_NAME, mv.version, "test_roc_auc", str(auc_chall))
                if auc_chall > auc_champ:
                    client.set_registered_model_alias(MODEL_NAME, "champion", mv.version)
                    print(f"  -> promoted v{mv.version} to '@champion' (previous champion v{champion_mv.version} kept for rollback)")
                    # Continue monitoring with the new model; its training data becomes the new reference
                    champion, champion_mv = challenger, client.get_model_version(MODEL_NAME, mv.version)
                    X_ref, y_ref = pd.concat([X_ref, X_fit_new]), pd.concat([y_ref, y_fit_new])
                    ref_auc = auc_chall
                else:
                    print("  -> challenger not better; champion stays")
                retrained = True
    limiter.restore_original_limits()

    print("\nMonitoring summary:")
    print(pd.DataFrame(history).round(4).to_string(index=False))
    print(f"\nMetric histories are in MLflow run 'monitoring' ({monitoring_run.info.run_id}); open the MLflow UI "
          "and plot max_psi / roc_auc against step to see the drift over time.")
