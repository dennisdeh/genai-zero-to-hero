"""
Model-agnostic explanations with scikit-learn: permutation importance, partial dependence and ICE curves,
and the classic pitfalls of feature importance.

This script demonstrates, on the synthetic churn dataset of c01 (where the TRUE effects are known):
    1. Impurity-based importance (MDI) of a random forest is biased towards high-cardinality / continuous
       features - even a pure-noise column gets a large share - while permutation importance on held-out
       data correctly assigns it ~0.
    2. Correlated features: adding a near-copy of tenure_months splits (and hides) its permutation importance,
       because permuting one copy leaves the information available in the other.
    3. Partial dependence (PDP, the average effect) and individual conditional expectation (ICE, one curve
       per customer) for a gradient-boosted model, plus group-wise PDPs revealing the contract x price and
       fibre x support-calls interactions (and a pitfall of the fast "recursion" PDP method).

Plots are saved to the outputs/ folder next to this script.

How to run (from the repository root):
    python -m p05_ml_classic.c02_model_explainability.main_2_permutation_pdp
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import PartialDependenceDisplay, partial_dependence, permutation_importance
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits

from p05_ml_classic.c01_gradient_boosted_trees.utils.churn_data import make_churn_dataset, train_val_test_split

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
SEED = 0
N_REPEATS = 5  # permutation repeats per feature (mean +/- std reported)


def permutation_table(model, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """Permutation importance (drop in ROC-AUC when a column is shuffled) on held-out data."""
    res = permutation_importance(model, X, y, scoring="roc_auc", n_repeats=N_REPEATS, random_state=SEED, n_jobs=1)
    return pd.DataFrame({"perm_mean": res.importances_mean, "perm_std": res.importances_std}, index=X.columns)


def plot_mdi_vs_permutation(table: pd.DataFrame, path: str) -> None:
    """Side-by-side bar charts of impurity-based and permutation importance."""
    table = table.sort_values("perm_mean")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    axes[0].barh(table.index, table["mdi"], color="tab:orange")
    axes[0].set(title="Impurity-based importance (MDI, train)", xlabel="mean decrease in impurity")
    axes[1].barh(table.index, table["perm_mean"], xerr=table["perm_std"], color="tab:blue")
    axes[1].set(title="Permutation importance (test)", xlabel="drop in ROC-AUC")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def fit_hgb(X: pd.DataFrame, y: pd.Series) -> HistGradientBoostingClassifier:
    """A reasonably regularised gradient-boosted model (fixed number of trees to keep the script simple)."""
    model = HistGradientBoostingClassifier(
        learning_rate=0.05, max_iter=200, max_leaf_nodes=15, l2_regularization=1.0, early_stopping=False, random_state=SEED
    )
    return model.fit(X, y)


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    # Single OpenMP thread: fastest for this data size and robust on busy machines
    limiter = threadpool_limits(limits=1, user_api="openmp")

    # 1. Data (numeric frame: categoricals as integer codes) - 10k customers keeps the forest fast
    X, y = make_churn_dataset(n_samples=10_000, seed=SEED, categorical="codes")
    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(X, y, seed=SEED)
    X_train = pd.concat([X_train, X_val])  # no early stopping here: use train+val for fitting
    y_train = pd.concat([y_train, y_val])

    # 2. Pitfall 1: impurity-based importance bias (fully grown random forest overfits the noise column)
    rf = RandomForestClassifier(n_estimators=200, min_samples_leaf=1, n_jobs=-1, random_state=SEED).fit(X_train, y_train)
    print(f"Random forest test ROC-AUC={roc_auc_score(y_test, rf.predict_proba(X_test)[:, 1]):.4f}")
    table = permutation_table(rf, X_test, y_test)
    table.insert(0, "mdi", rf.feature_importances_)
    print("\nMDI vs permutation importance (noise, num_products and total_charges have NO true effect):")
    print(table.sort_values("mdi", ascending=False).round(4).to_string())
    print(f"-> MDI ranks 'noise' #{table['mdi'].rank(ascending=False)['noise']:.0f} of {len(table)}, "
          f"permutation importance ranks it #{table['perm_mean'].rank(ascending=False)['noise']:.0f}")
    plot_mdi_vs_permutation(table, os.path.join(OUTPUT_DIR, "importance_mdi_vs_permutation.png"))

    # 3. Pitfall 2: correlated features share (and hide) importance
    hgb = fit_hgb(X_train, y_train)
    base = permutation_table(hgb, X_test, y_test)
    rng = np.random.default_rng(SEED)
    X_train_c = X_train.assign(tenure_copy=X_train["tenure_months"] + rng.normal(0, 1, len(X_train)))
    X_test_c = X_test.assign(tenure_copy=X_test["tenure_months"] + rng.normal(0, 1, len(X_test)))
    hgb_c = fit_hgb(X_train_c, y_train)
    with_copy = permutation_table(hgb_c, X_test_c, y_test)
    comp = pd.DataFrame({"original": base["perm_mean"], "with near-copy of tenure": with_copy["perm_mean"]})
    print(f"\nHGB test ROC-AUC: original={roc_auc_score(y_test, hgb.predict_proba(X_test)[:, 1]):.4f}, "
          f"with tenure_copy={roc_auc_score(y_test, hgb_c.predict_proba(X_test_c)[:, 1]):.4f}")
    print("Permutation importance (drop in ROC-AUC) before/after adding tenure_copy (corr ~0.999 with tenure):")
    print(comp.loc[["tenure_months", "tenure_copy", "total_charges", "contract"]].round(4).to_string())
    print("-> the information did not become less useful, it is just spread over two columns. Remedy: permute "
          "groups of correlated features together, or drop/merge redundant features before interpreting.")

    # 4. Partial dependence + ICE (on the original HGB model)
    features = ["tenure_months", "monthly_charges", "age", "support_calls"]
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
    PartialDependenceDisplay.from_estimator(
        hgb,
        X_test,
        features,
        kind="both",  # thin lines: ICE (one per customer), thick line: PDP (their average)
        centered=True,  # start all ICE curves at 0 so that differences in shape are visible
        response_method="decision_function",  # log-odds scale, where effects are additive
        subsample=150,
        grid_resolution=30,
        random_state=SEED,
        ax=axes,
        ice_lines_kw={"alpha": 0.15, "linewidth": 0.6},
        pd_line_kw={"color": "red", "linewidth": 2.5},
    )
    fig.suptitle("Centred ICE curves (blue) and partial dependence (red), log-odds of churn", y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "pdp_ice.png"), dpi=120, bbox_inches="tight")
    plt.close(fig)

    # 5. Interactions: partial dependence of one feature computed separately within groups of another feature.
    #    If the curves are parallel there is no interaction; different slopes/jumps reveal one.
    #    method="brute" averages predictions over the rows we pass; the faster default for trees ("recursion")
    #    uses the training distribution stored in the trees and would silently ignore our subsetting.
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    true_slopes = {0: 0.04, 1: 0.01, 2: 0.01}
    for code, name in enumerate(["month-to-month", "one-year", "two-year"]):
        sub = X_test[X_test["contract"] == code]
        res = partial_dependence(hgb, sub, ["monthly_charges"], grid_resolution=20, method="brute",
                                 response_method="decision_function")
        grid, avg = res["grid_values"][0], res["average"][0]
        slope = np.polyfit(grid, avg, 1)[0]
        axes[0].plot(grid, avg, "o-", ms=3, label=f"{name} (slope {slope:+.3f})")
        print(f"PDP slope of monthly_charges for {name:15s}: {slope:+.4f} log-odds per unit (true: {true_slopes[code]:+.4f})")
    for fibre in [0, 1]:
        sub = X_test[X_test["has_fibre"] == fibre]
        res = partial_dependence(hgb, sub, ["support_calls"], method="brute", response_method="decision_function")
        axes[1].plot(res["grid_values"][0], res["average"][0], "o-", label=f"has_fibre={fibre}")
    axes[0].set(title="Price effect depends on the contract", xlabel="monthly_charges", ylabel="partial dependence (log-odds)")
    axes[1].set(title="Support calls hurt more for fibre customers", xlabel="support_calls")
    axes[0].legend(fontsize=8)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "pdp_interactions.png"), dpi=120)
    plt.close(fig)
    limiter.restore_original_limits()
    print(f"\nPlots saved to {OUTPUT_DIR}")
