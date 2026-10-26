"""
Beyond linear models: decision trees, random forests and support vector machines (SVMs).

This script demonstrates:
    1. Decision boundaries of a logistic regression, decision trees of different depths, a random forest and
       linear/RBF SVMs on the non-linear "two moons" toy dataset (bias-variance trade-off made visible).
    2. Cost-complexity pruning of a decision tree (choosing ccp_alpha by cross-validation) and the
       human-readable rules of a small tree.
    3. Random forests: out-of-bag (OOB) error as a free validation estimate, and its dependence on the
       number of trees.
    4. SVM hyperparameters C and gamma tuned by grid search.
    5. A fair 5-fold cross-validated comparison of all model families on the breast cancer dataset.

Plots are saved to the outputs/ folder next to this script.

How to run (from the repository root):
    python -m p05_ml_classic.c00_linear_models.main_3_trees_rf_svm
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.datasets import load_breast_cancer, make_moons
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, export_text

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
SEED = 42


def plot_decision_boundaries(
    models: dict[str, ClassifierMixin], X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray, path: str
) -> pd.DataFrame:
    """Fit each model on the moons data, draw its decision boundary and return train/test accuracies."""
    n = len(models)
    ncols = 4
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.6 * nrows))
    rows = []
    for ax, (name, model) in zip(axes.ravel(), models.items()):
        model.fit(X_train, y_train)
        train_acc, test_acc = model.score(X_train, y_train), model.score(X_test, y_test)
        rows.append({"model": name, "train_acc": train_acc, "test_acc": test_acc})
        DecisionBoundaryDisplay.from_estimator(model, X_train, response_method="predict", alpha=0.3, cmap="coolwarm", ax=ax)
        ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap="coolwarm", s=10, edgecolors="k", linewidths=0.2)
        ax.set_title(f"{name}\ntrain={train_acc:.2f}, test={test_acc:.2f}", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return pd.DataFrame(rows)


def prune_tree_by_cv(X: pd.DataFrame, y: np.ndarray, cv: StratifiedKFold) -> tuple[DecisionTreeClassifier, pd.DataFrame]:
    """
    Minimal cost-complexity pruning: grow a full tree, compute the sequence of effective alphas at which
    subtrees are pruned away, and pick the alpha with the best cross-validated accuracy.
    """
    path = DecisionTreeClassifier(random_state=SEED).cost_complexity_pruning_path(X, y)
    alphas = np.unique(path.ccp_alphas[:-1])  # the last alpha prunes the tree down to the root
    search = GridSearchCV(DecisionTreeClassifier(random_state=SEED), {"ccp_alpha": alphas}, cv=cv)
    search.fit(X, y)
    res = pd.DataFrame(search.cv_results_)[["param_ccp_alpha", "mean_test_score"]]
    return search.best_estimator_, res


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    # 1. Toy data: two interleaving half-moons with noise (not linearly separable)
    X_m, y_m = make_moons(n_samples=500, noise=0.3, random_state=SEED)
    Xm_train, Xm_test, ym_train, ym_test = train_test_split(X_m, y_m, test_size=0.3, random_state=SEED)

    # 2. Decision boundaries of different model families (and complexities)
    toy_models = {
        "Logistic regression": LogisticRegression(),
        "Tree (max_depth=2)": DecisionTreeClassifier(max_depth=2, random_state=SEED),
        "Tree (max_depth=5)": DecisionTreeClassifier(max_depth=5, random_state=SEED),
        "Tree (unlimited depth)": DecisionTreeClassifier(random_state=SEED),
        "Random forest (300 trees)": RandomForestClassifier(n_estimators=300, min_samples_leaf=3, random_state=SEED),
        "Linear SVM": make_pipeline(StandardScaler(), SVC(kernel="linear", C=1.0)),
        "RBF SVM (C=1, gamma=1)": make_pipeline(StandardScaler(), SVC(kernel="rbf", C=1.0, gamma=1.0)),
        "RBF SVM (C=1000, gamma=50)": make_pipeline(StandardScaler(), SVC(kernel="rbf", C=1000, gamma=50)),
    }
    acc = plot_decision_boundaries(toy_models, Xm_train, ym_train, Xm_test, ym_test, os.path.join(OUTPUT_DIR, "decision_boundaries.png"))
    print("Two moons - train vs test accuracy (large gap = overfitting / high variance):")
    print(acc.round(3).to_string(index=False))

    # 3. Real tabular data: breast cancer (1 = malignant)
    data = load_breast_cancer(as_frame=True)
    X, y = data.data, (data.target == 0).astype(int).to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=SEED)

    # 3a. Cost-complexity pruning of a single tree
    pruned_tree, prune_res = prune_tree_by_cv(X_train, y_train, cv)
    full_tree = DecisionTreeClassifier(random_state=SEED).fit(X_train, y_train)
    print(
        f"\nFull tree: {full_tree.get_n_leaves()} leaves, test acc={full_tree.score(X_test, y_test):.3f} | "
        f"pruned tree (ccp_alpha={pruned_tree.ccp_alpha:.4f}): {pruned_tree.get_n_leaves()} leaves, "
        f"test acc={pruned_tree.score(X_test, y_test):.3f}"
    )
    print("Rules of the pruned tree (class 1 = malignant):")
    print(export_text(pruned_tree, feature_names=list(X.columns), max_depth=4))

    # 3b. Random forest: OOB accuracy vs number of trees (warm_start adds trees to an existing forest)
    rf = RandomForestClassifier(oob_score=True, warm_start=True, random_state=SEED, n_jobs=-1)
    n_trees_grid = [15, 25, 50, 100, 200, 400]
    oob = []
    for n in n_trees_grid:
        rf.set_params(n_estimators=n)
        rf.fit(X_train, y_train)
        oob.append(rf.oob_score_)
    print("Random forest OOB accuracy vs number of trees:", {n: round(float(s), 4) for n, s in zip(n_trees_grid, oob)})
    print(f"Random forest test accuracy (400 trees): {rf.score(X_test, y_test):.3f}")
    imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
    print("Top-5 impurity-based importances (see c02 for why these can be misleading):")
    print(imp.head(5).round(3).to_string())
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(n_trees_grid, 1 - np.array(oob), "o-")
    ax.set(xscale="log", xlabel="number of trees", ylabel="OOB error", title="Random forest: OOB error vs ensemble size")
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "rf_oob_error.png"), dpi=120)
    plt.close(fig)

    # 3c. SVM with RBF kernel: tune C (margin softness) and gamma (kernel width) jointly
    svm_search = GridSearchCV(
        make_pipeline(StandardScaler(), SVC(kernel="rbf")),
        {"svc__C": [0.1, 1, 10, 100], "svc__gamma": [0.001, 0.01, 0.1, 1]},
        cv=cv,
        n_jobs=-1,
    )
    svm_search.fit(X_train, y_train)
    print(f"\nRBF SVM best params: {svm_search.best_params_}, CV acc={svm_search.best_score_:.4f}, "
          f"test acc={svm_search.score(X_test, y_test):.3f}, support vectors={svm_search.best_estimator_[-1].n_support_.sum()}")

    # 4. Fair comparison: identical 5-fold CV splits on the training set for every model family
    candidates = {
        "Logistic regression": make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=5000)),
        "Decision tree (pruned)": DecisionTreeClassifier(ccp_alpha=pruned_tree.ccp_alpha, random_state=SEED),
        "Random forest": RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1),
        "RBF SVM (tuned)": svm_search.best_estimator_,
    }
    rows = []
    for name, model in candidates.items():
        scores = cross_validate(model, X_train, y_train, cv=cv, scoring=["accuracy", "roc_auc"])
        rows.append(
            {
                "model": name,
                "cv_accuracy": f"{scores['test_accuracy'].mean():.3f} +/- {scores['test_accuracy'].std():.3f}",
                "cv_roc_auc": f"{scores['test_roc_auc'].mean():.3f} +/- {scores['test_roc_auc'].std():.3f}",
                "fit_time_s": round(scores["fit_time"].mean(), 3),
            }
        )
    print("\n5-fold cross-validated comparison on the breast cancer training set:")
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"\nPlots saved to {OUTPUT_DIR}")
