"""
Explaining a gradient-boosted tree model with SHAP (SHapley Additive exPlanations).

This script demonstrates, for a LightGBM churn model trained on the synthetic churn dataset of c01:
    1. TreeExplainer: exact, fast Shapley values for tree ensembles (on the log-odds scale).
    2. The additivity ("local accuracy") property: base value + sum of SHAP values = model output.
    3. Global explanations: mean |SHAP| bar chart and the beeswarm summary plot.
    4. Dependence plots: SHAP value of a feature vs its value, coloured by an interacting feature.
       Because the data are synthetic we can compare with the TRUE effects (e.g. the fibre x support-calls
       interaction and the U-shaped age effect).
    5. SHAP interaction values, which separate main effects from pairwise interactions.
    6. Local explanation: a waterfall plot for the customer with the highest predicted churn risk.

Plots are saved to the outputs/ folder next to this script.

How to run (from the repository root):
    python -m p05_ml_classic.c02_model_explainability.main_1_shap
"""

import os

import matplotlib

matplotlib.use("Agg")
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import roc_auc_score

from p05_ml_classic.c01_gradient_boosted_trees.utils.churn_data import (
    CONTRACTS,
    make_churn_dataset,
    train_val_test_split,
)

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
SEED = 0
N_EXPLAIN = 2000  # number of test customers to explain (TreeExplainer is fast, but plots get crowded)


def train_model(X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series) -> lgb.LGBMClassifier:
    """LightGBM with early stopping (as in c01). Categorical features are integer codes here (numeric frame)."""
    model = lgb.LGBMClassifier(
        n_estimators=2000, learning_rate=0.05, num_leaves=15, subsample=0.8, subsample_freq=1,
        colsample_bytree=0.8, n_jobs=1, random_state=SEED, verbose=-1,
    )
    model.fit(X_train, y_train, eval_X=(X_val,), eval_y=(y_val,), callbacks=[lgb.early_stopping(50, verbose=False)])
    return model


def save_current_figure(name: str) -> None:
    """SHAP draws on the current matplotlib figure; save and close it."""
    fig = plt.gcf()
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, name), dpi=120, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Data and model (categoricals as integer codes: contract 0=month-to-month, 1=one-year, 2=two-year)
    X, y = make_churn_dataset(n_samples=20_000, seed=SEED, categorical="codes")
    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(X, y, seed=SEED)
    model = train_model(X_train, y_train, X_val, y_val)
    print(f"LightGBM: {model.best_iteration_} trees, test ROC-AUC={roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]):.4f}")

    # 2. SHAP values with TreeExplainer (exact for trees, polynomial time; explains the raw log-odds output)
    X_explain = X_test.iloc[:N_EXPLAIN]
    explainer = shap.TreeExplainer(model)
    explanation = explainer(X_explain)  # shap.Explanation: .values (n, p), .base_values (n,), .data (n, p)
    print(f"SHAP values shape: {explanation.values.shape}; base value (mean log-odds) = {explanation.base_values[0]:.3f}")

    # 3. Additivity check: phi_0 + sum_j phi_j(x) = f(x) on the log-odds scale
    raw = model.predict(X_explain, raw_score=True)
    max_err = np.max(np.abs(explanation.base_values + explanation.values.sum(axis=1) - raw))
    print(f"Additivity check: max |base + sum(SHAP) - model log-odds| = {max_err:.2e}")

    # 4. Global importance = mean |SHAP value| per feature
    global_imp = pd.Series(np.abs(explanation.values).mean(axis=0), index=X.columns).sort_values(ascending=False)
    print("\nGlobal importance (mean |SHAP|, log-odds units). total_charges, num_products and noise have no TRUE effect:")
    print(global_imp.round(4).to_string())
    shap.plots.bar(explanation, max_display=12, show=False)
    save_current_figure("shap_bar.png")
    shap.plots.beeswarm(explanation, max_display=12, show=False)
    save_current_figure("shap_beeswarm.png")

    # 5. Dependence plots: compare with the true data-generating process
    #    - support_calls: saturating effect + extra jump at >= 3 calls only for fibre customers (interaction)
    #    - tenure_months: decreasing, log-shaped effect
    #    - age: U-shaped effect (invisible to a linear model)
    #    - monthly_charges: steeper slope for month-to-month contracts (interaction with contract)
    for feature, colour in [
        ("support_calls", "has_fibre"),
        ("tenure_months", "contract"),
        ("age", None),
        ("monthly_charges", "contract"),
    ]:
        kwargs = {"color": explanation[:, colour]} if colour else {}
        shap.plots.scatter(explanation[:, feature], show=False, **kwargs)
        plt.title(f"SHAP dependence: {feature}" + (f" (colour: {colour})" if colour else ""))
        save_current_figure(f"shap_dependence_{feature}.png")

    # 6. SHAP interaction values: split each prediction into main effects (diagonal) and pairwise interactions
    #    (off-diagonal). The true model has two interactions: contract x monthly_charges and has_fibre x support_calls.
    X_inter = X_explain.iloc[:1000]  # O(T L D^2) per sample - noticeably slower than plain SHAP values
    inter = explainer.shap_interaction_values(X_inter)  # shape (n, p, p)
    strength = np.abs(inter).mean(axis=0)
    np.fill_diagonal(strength, 0.0)
    pairs = pd.DataFrame(strength, index=X.columns, columns=X.columns).stack()
    pairs = pairs[[a < b for a, b in pairs.index]].sort_values(ascending=False)  # each unordered pair once
    print("\nStrongest pairwise SHAP interactions (mean |phi_ij|, log-odds):")
    print(pairs.head(5).round(4).to_string())
    a, b = X.columns.get_loc("support_calls"), X.columns.get_loc("has_fibre")
    df = pd.DataFrame(
        {
            "calls": X_inter["support_calls"].clip(upper=5),
            "fibre": X_inter["has_fibre"],
            "main_effect": inter[:, a, a],
            "interaction_with_fibre": 2 * inter[:, a, b],  # phi_ab + phi_ba
        }
    )
    print("support_calls: mean main effect and interaction with has_fibre, by number of calls and fibre:")
    print(df.groupby(["calls", "fibre"])[["main_effect", "interaction_with_fibre"]].mean().unstack().round(2).to_string())

    # 7. Local explanation: the customer with the highest predicted churn probability
    proba = model.predict_proba(X_explain)[:, 1]
    i = int(np.argmax(proba))
    customer = X_explain.iloc[i]
    print(f"\nHighest-risk customer (row {X_explain.index[i]}): predicted churn probability {proba[i]:.3f}")
    print(f"  contract={CONTRACTS[int(customer.contract)]}, " + ", ".join(f"{k}={v:g}" for k, v in customer.drop('contract').items()))
    contrib = pd.Series(explanation.values[i], index=X.columns).sort_values(key=np.abs, ascending=False)
    print("  Top contributions (log-odds):", ", ".join(f"{k}={v:+.2f}" for k, v in contrib.head(5).items()))
    shap.plots.waterfall(explanation[i], max_display=11, show=False)
    save_current_figure("shap_waterfall_highest_risk.png")
    print(f"\nPlots saved to {OUTPUT_DIR}")
