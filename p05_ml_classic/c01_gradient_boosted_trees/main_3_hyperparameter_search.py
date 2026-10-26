"""
Hyperparameter search for gradient-boosted trees (LightGBM) on the synthetic churn dataset.

This script demonstrates:
    1. A reference model with library default hyperparameters.
    2. Random search with cross-validation (scikit-learn RandomizedSearchCV) over the most important
       boosting hyperparameters, sampled from sensible distributions (log-uniform for rates/penalties).
    3. (Optional) Bayesian-style optimisation with Optuna (TPE sampler) using early stopping on a validation set,
       if the optional dependency `optuna` is installed - otherwise this step is skipped.
    4. Evaluation of every tuned model ONCE on the untouched test set.

How to run (from the repository root):
    python -m p05_ml_classic.c01_gradient_boosted_trees.main_3_hyperparameter_search
"""

import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import loguniform, randint, uniform
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

from p05_ml_classic.c01_gradient_boosted_trees.utils.churn_data import make_churn_dataset, train_val_test_split

SEED = 0
N_ITER_RANDOM = 20  # number of random configurations (x 3 CV folds)
N_TRIALS_OPTUNA = 25


def fmt_params(params: dict) -> dict:
    """Round floats (incl. numpy scalars) for readable printing."""
    out = {}
    for k, v in params.items():
        if isinstance(v, (float, np.floating)):
            v = round(float(v), 4)
        elif isinstance(v, np.integer):
            v = int(v)
        out[k] = v
    return out


def evaluate(name: str, model: lgb.LGBMClassifier, X_test: pd.DataFrame, y_test: pd.Series, seconds: float) -> dict:
    """Test-set metrics for a fitted classifier."""
    p = model.predict_proba(X_test)[:, 1]
    return {"model": name, "test_roc_auc": roc_auc_score(y_test, p), "test_log_loss": log_loss(y_test, p), "search_time_s": seconds}


def random_search(X_train: pd.DataFrame, y_train: pd.Series) -> tuple[RandomizedSearchCV, float]:
    """RandomizedSearchCV over LightGBM hyperparameters with 3-fold stratified CV, optimising log-loss."""
    param_distributions = {
        "n_estimators": randint(100, 600),
        "learning_rate": loguniform(0.01, 0.2),
        "num_leaves": randint(4, 64),  # tree complexity (LightGBM grows trees leaf-wise)
        "min_child_samples": randint(5, 200),  # minimum samples per leaf (regularisation)
        "subsample": uniform(0.5, 0.5),  # row subsampling per tree, in [0.5, 1.0]
        "colsample_bytree": uniform(0.5, 0.5),  # feature subsampling per tree
        "reg_lambda": loguniform(1e-3, 10.0),  # L2 penalty on leaf values
    }
    search = RandomizedSearchCV(
        lgb.LGBMClassifier(subsample_freq=1, n_jobs=1, random_state=SEED, verbose=-1),
        param_distributions,
        n_iter=N_ITER_RANDOM,
        scoring="neg_log_loss",
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED),
        n_jobs=-1,  # parallelise over configurations/folds (separate processes), each model single-threaded
        random_state=SEED,
    )
    t0 = time.perf_counter()
    search.fit(X_train, y_train)
    return search, time.perf_counter() - t0


def optuna_search(X_train, y_train, X_val, y_val) -> tuple[lgb.LGBMClassifier | None, float]:
    """
    Optuna study: each trial trains LightGBM with early stopping on the validation set; the TPE sampler
    proposes new hyperparameters based on the results of previous trials. Returns (None, 0) if optuna is missing.
    """
    try:
        import optuna
    except ImportError:
        print("\n[optuna] not installed - skipping (pip install optuna to enable this step)")
        return None, 0.0
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def make_model(params: dict) -> lgb.LGBMClassifier:
        return lgb.LGBMClassifier(
            n_estimators=2000, subsample_freq=1, n_jobs=1, random_state=SEED, verbose=-1, **params
        )

    def objective(trial: "optuna.Trial") -> float:
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 4, 64, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 200, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        }
        model = make_model(params)
        model.fit(
            X_train,
            y_train,
            eval_X=(X_val,),
            eval_y=(y_val,),
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )
        trial.set_user_attr("n_estimators", model.best_iteration_)
        return model.best_score_["valid_0"]["binary_logloss"]

    t0 = time.perf_counter()
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=N_TRIALS_OPTUNA)
    elapsed = time.perf_counter() - t0
    best_params = dict(study.best_params)
    n_best = study.best_trial.user_attrs["n_estimators"]
    print(f"\n[optuna] best validation log-loss={study.best_value:.4f} with {n_best} trees and params:")
    print(fmt_params(best_params))
    # Refit with the early-stopped number of trees (validation data used only for model selection)
    final = make_model(best_params).set_params(n_estimators=n_best).fit(X_train, y_train)
    return final, elapsed


if __name__ == "__main__":
    # 1. Data (same split as main_2)
    X, y = make_churn_dataset(n_samples=20_000, seed=SEED, categorical="category")
    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(X, y, seed=SEED)
    results = []

    # 2. Reference: LightGBM defaults (100 trees, 31 leaves, learning rate 0.1)
    t0 = time.perf_counter()
    default = lgb.LGBMClassifier(n_jobs=1, random_state=SEED, verbose=-1).fit(X_train, y_train)
    results.append(evaluate("LightGBM defaults", default, X_test, y_test, time.perf_counter() - t0))

    # 3. Random search with 3-fold cross-validation
    search, seconds = random_search(X_train, y_train)
    print(f"[random search] best CV log-loss={-search.best_score_:.4f} with params:")
    print(fmt_params(search.best_params_))
    top = pd.DataFrame(search.cv_results_).sort_values("rank_test_score")
    print("Top-5 configurations (mean CV log-loss +/- std):")
    for _, row in top.head(5).iterrows():
        print(f"  {-row.mean_test_score:.4f} +/- {row.std_test_score:.4f}  leaves={row.param_num_leaves}, "
              f"lr={row.param_learning_rate:.3f}, trees={row.param_n_estimators}")
    results.append(evaluate("LightGBM random search (CV)", search.best_estimator_, X_test, y_test, seconds))

    # 4. Optional Optuna study with early stopping
    optuna_model, seconds = optuna_search(X_train, y_train, X_val, y_val)
    if optuna_model is not None:
        results.append(evaluate("LightGBM Optuna (TPE + early stopping)", optuna_model, X_test, y_test, seconds))

    # 5. Final comparison on the test set
    print("\nTest-set comparison:")
    print(pd.DataFrame(results).round(4).to_string(index=False))
    print(
        "\nNote: differences of ~0.001-0.002 in log-loss are within the noise of a 4k test set; tuning matters "
        "most when defaults are far from a good configuration (e.g. too many leaves on small data)."
    )
