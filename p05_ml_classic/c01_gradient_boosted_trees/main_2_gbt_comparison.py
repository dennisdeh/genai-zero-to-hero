"""
Gradient-boosted trees in practice: scikit-learn HistGradientBoosting vs XGBoost vs LightGBM.

This script demonstrates, on the synthetic (offline) churn dataset with a train / validation / test split:
    1. Fair baselines: a one-hot + logistic regression pipeline and a random forest.
    2. Three gradient-boosting libraries with native categorical-feature support and EARLY STOPPING on the
       same validation set: we allow many trees and stop when the validation log-loss has not improved
       for 50 rounds.
    3. A comparison table on the untouched test set (ROC-AUC, log-loss, accuracy, number of trees, fit time),
       including the best achievable score computed from the TRUE churn probabilities.
    4. Validation learning curves and gain-based feature importances.

Plots are saved to the outputs/ folder next to this script.

How to run (from the repository root):
    python -m p05_ml_classic.c01_gradient_boosted_trees.main_2_gbt_comparison
"""

import os
import time

import matplotlib

matplotlib.use("Agg")
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from threadpoolctl import threadpool_limits

from p05_ml_classic.c01_gradient_boosted_trees.utils.churn_data import (
    CATEGORICAL_FEATURES,
    make_churn_dataset,
    train_val_test_split,
)

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
SEED = 0
# For ~10k rows a single thread is fastest: multi-threading overhead dominates, and on a busy machine OpenMP
# threads can even slow training down dramatically (oversubscription). Increase for large datasets.
N_JOBS = 1
MAX_TREES = 2000  # upper bound; early stopping decides the actual number
EARLY_STOPPING_ROUNDS = 50
LEARNING_RATE = 0.05


def make_baselines(numeric_features: list[str]) -> dict[str, Pipeline]:
    """Logistic regression (one-hot + scaling) and random forest (ordinal-encoded categoricals)."""
    logreg = Pipeline(
        [
            (
                "prep",
                ColumnTransformer(
                    [
                        ("num", StandardScaler(), numeric_features),
                        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
                    ]
                ),
            ),
            ("clf", LogisticRegression(max_iter=2000)),
        ]
    )
    forest = Pipeline(
        [
            (
                "prep",
                ColumnTransformer(
                    [("cat", OrdinalEncoder(), CATEGORICAL_FEATURES)], remainder="passthrough"
                ),
            ),
            ("clf", RandomForestClassifier(n_estimators=300, min_samples_leaf=20, n_jobs=N_JOBS, random_state=SEED)),
        ]
    )
    return {"Logistic regression (baseline)": logreg, "Random forest (baseline)": forest}


def fit_hist_gbt(X_train, y_train, X_val, y_val) -> tuple[HistGradientBoostingClassifier, list[float]]:
    """scikit-learn HistGradientBoosting with an explicit validation set for early stopping."""
    model = HistGradientBoostingClassifier(
        learning_rate=LEARNING_RATE,
        max_iter=MAX_TREES,
        max_leaf_nodes=15,
        l2_regularization=1.0,
        categorical_features="from_dtype",  # pandas 'category' columns are handled natively
        early_stopping=True,
        scoring="loss",
        n_iter_no_change=EARLY_STOPPING_ROUNDS,
        random_state=SEED,
    )
    with threadpool_limits(limits=N_JOBS, user_api="openmp"):  # HGB parallelises with OpenMP
        model.fit(X_train, y_train, X_val=X_val, y_val=y_val)
    # validation_score_ stores the NEGATIVE loss (higher is better); entry 0 is the initial prediction
    curve = list(-model.validation_score_[1:])
    return model, curve


def fit_xgboost(X_train, y_train, X_val, y_val) -> tuple[xgb.XGBClassifier, list[float]]:
    """XGBoost (histogram method) with native categorical support and early stopping."""
    model = xgb.XGBClassifier(
        n_estimators=MAX_TREES,
        learning_rate=LEARNING_RATE,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        tree_method="hist",
        enable_categorical=True,
        eval_metric="logloss",
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        n_jobs=N_JOBS,
        random_state=SEED,
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    curve = model.evals_result()["validation_0"]["logloss"]
    return model, curve


def fit_lightgbm(X_train, y_train, X_val, y_val) -> tuple[lgb.LGBMClassifier, list[float]]:
    """LightGBM (leaf-wise growth) with native categorical support and early stopping."""
    model = lgb.LGBMClassifier(
        n_estimators=MAX_TREES,
        learning_rate=LEARNING_RATE,
        num_leaves=15,
        min_child_samples=20,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        n_jobs=N_JOBS,
        random_state=SEED,
        verbose=-1,
    )
    model.fit(
        X_train,
        y_train,
        eval_X=(X_val,),
        eval_y=(y_val,),
        eval_metric="binary_logloss",
        callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False), lgb.log_evaluation(0)],
    )
    curve = model.evals_result_["valid_0"]["binary_logloss"]
    return model, curve


def plot_learning_curves(curves: dict[str, list[float]], best_iters: dict[str, int], path: str) -> None:
    """Validation log-loss per boosting round for each library, with the early-stopping point marked."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, curve in curves.items():
        line = ax.plot(np.arange(1, len(curve) + 1), curve, label=name)[0]
        b = best_iters[name]
        ax.scatter([b], [curve[b - 1]], color=line.get_color(), zorder=3)
    ax.set(xlabel="boosting round", ylabel="validation log-loss", title="Early stopping on the validation set")
    ax.set_ylim(min(min(c) for c in curves.values()) - 0.005, 0.5)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_importances(importances: pd.DataFrame, path: str) -> None:
    """Horizontal bar chart of normalised gain importances per library."""
    ax = importances.sort_values(importances.columns[0]).plot.barh(figsize=(8, 6))
    ax.set(xlabel="normalised total gain", title="Gain-based feature importance")
    ax.figure.tight_layout()
    ax.figure.savefig(path, dpi=120)
    plt.close(ax.figure)


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Data: 20k customers, native pandas categoricals, 60/20/20 split
    X, y, true_p = make_churn_dataset(n_samples=20_000, seed=SEED, categorical="category", return_proba=True)
    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(X, y, seed=SEED)
    numeric_features = [c for c in X.columns if c not in CATEGORICAL_FEATURES]
    print(f"Train={len(X_train)}, val={len(X_val)}, test={len(X_test)}; churn rate={y.mean():.3f}")

    results = []

    def record(name: str, model, fit_time: float, n_trees: int | None) -> None:
        """Evaluate a fitted model on the test set and append a result row."""
        p = model.predict_proba(X_test)[:, 1]
        results.append(
            {
                "model": name,
                "test_roc_auc": roc_auc_score(y_test, p),
                "test_log_loss": log_loss(y_test, p),
                "test_accuracy": accuracy_score(y_test, (p >= 0.5).astype(int)),
                "n_trees": n_trees,
                "fit_time_s": fit_time,
            }
        )

    # 2. Baselines (trained on train only; they do not use the validation set)
    for name, model in make_baselines(numeric_features).items():
        t0 = time.perf_counter()
        model.fit(X_train, y_train)
        n_trees = model["clf"].n_estimators if hasattr(model["clf"], "n_estimators") else None
        record(name, model, time.perf_counter() - t0, n_trees)

    # 3. Gradient-boosted trees with early stopping on the validation set
    curves, best_iters, boosters = {}, {}, {}
    for name, fit_fn in [
        ("HistGradientBoosting", fit_hist_gbt),
        ("XGBoost", fit_xgboost),
        ("LightGBM", fit_lightgbm),
    ]:
        t0 = time.perf_counter()
        model, curve = fit_fn(X_train, y_train, X_val, y_val)
        fit_time = time.perf_counter() - t0
        if name == "HistGradientBoosting":
            # HGB keeps all trees up to the point where it stopped (best + n_iter_no_change), no rollback
            best = int(np.argmin(curve)) + 1
            n_used = model.n_iter_
        elif name == "XGBoost":
            best = model.best_iteration + 1
            n_used = best  # predict() automatically uses the best iteration
        else:
            best = model.best_iteration_
            n_used = best
        curves[name], best_iters[name], boosters[name] = curve, best, model
        record(name, model, fit_time, n_used)

    # 4. Results on the untouched test set
    table = pd.DataFrame(results).set_index("model")
    best_auc = roc_auc_score(y_test, true_p[X_test.index])
    best_ll = log_loss(y_test, true_p[X_test.index])
    print("\nTest-set results:")
    print(table.round(4).to_string())
    print(f"Best achievable with the TRUE probabilities: ROC-AUC={best_auc:.4f}, log-loss={best_ll:.4f}")

    # 5. Learning curves and feature importances
    plot_learning_curves(curves, best_iters, os.path.join(OUTPUT_DIR, "gbt_learning_curves.png"))
    xgb_total_gain = pd.Series(boosters["XGBoost"].get_booster().get_score(importance_type="total_gain"))
    lgb_gain = pd.Series(
        boosters["LightGBM"].booster_.feature_importance(importance_type="gain"),
        index=boosters["LightGBM"].booster_.feature_name(),
    )
    importances = pd.DataFrame(
        {"XGBoost": xgb_total_gain / xgb_total_gain.sum(), "LightGBM": lgb_gain / lgb_gain.sum()}
    ).fillna(0.0)
    print("\nNormalised total-gain importances (the generator gives total_charges, num_products and noise NO effect):")
    print(importances.sort_values("LightGBM", ascending=False).round(3).to_string())
    plot_importances(importances, os.path.join(OUTPUT_DIR, "gbt_feature_importance.png"))
    print(f"\nPlots saved to {OUTPUT_DIR}")
