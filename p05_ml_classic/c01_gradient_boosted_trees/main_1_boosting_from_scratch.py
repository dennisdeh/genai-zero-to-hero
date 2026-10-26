"""
Gradient boosting from scratch: what XGBoost, LightGBM and HistGradientBoosting do under the hood.

This script demonstrates:
    1. Gradient boosting for regression with squared loss: each new shallow tree is fitted to the residuals
       (= negative gradient) of the current ensemble, and added with a small learning rate (shrinkage).
       We plot the ensemble after 1, 5, 20 and 200 trees on a noisy sine curve and verify that our
       implementation matches scikit-learn's GradientBoostingRegressor.
    2. Gradient boosting for binary classification with the log-loss, using second-order (Newton) leaf
       values - the trick popularised by XGBoost - on the synthetic churn dataset, compared with
       scikit-learn's GradientBoostingClassifier.
    3. The effect of the learning rate on training and validation loss (why we need early stopping).

Plots are saved to the outputs/ folder next to this script.

How to run (from the repository root):
    python -m p05_ml_classic.c01_gradient_boosted_trees.main_1_boosting_from_scratch
"""

import os
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.tree import DecisionTreeRegressor

from p05_ml_classic.c01_gradient_boosted_trees.utils.churn_data import make_churn_dataset, train_val_test_split

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
SEED = 0


class SimpleGradientBoostingRegressor:
    """
    Minimal gradient boosting with squared loss L(y, F) = 1/2 (y - F)^2.

    F_0(x) = mean(y);  F_m(x) = F_{m-1}(x) + lr * h_m(x), where h_m is a regression tree fitted to the
    negative gradient r_i = -dL/dF = y_i - F_{m-1}(x_i), i.e. to the current residuals.
    """

    def __init__(self, n_estimators: int = 100, learning_rate: float = 0.1, max_depth: int = 3):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.trees: list[DecisionTreeRegressor] = []
        self.init_: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SimpleGradientBoostingRegressor":
        self.init_ = float(np.mean(y))
        F = np.full(len(y), self.init_)
        self.trees = []
        for _ in range(self.n_estimators):
            residuals = y - F  # negative gradient of the squared loss
            tree = DecisionTreeRegressor(max_depth=self.max_depth, random_state=SEED).fit(X, residuals)
            F += self.learning_rate * tree.predict(X)
            self.trees.append(tree)
        return self

    def staged_predict(self, X: np.ndarray):
        """Yield the prediction after each boosting iteration."""
        F = np.full(len(X), self.init_)
        for tree in self.trees:
            F = F + self.learning_rate * tree.predict(X)
            yield F

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.init_ + self.learning_rate * np.sum([t.predict(X) for t in self.trees], axis=0)


class SimpleGradientBoostingClassifier:
    """
    Binary gradient boosting with the log-loss on the log-odds scale F (p = sigmoid(F)).

    gradient   g_i = p_i - y_i,   hessian  h_i = p_i (1 - p_i).
    Each tree is fitted to the negative gradient (y - p) to find the *structure* (splits); the value of every
    leaf j is then set by a Newton step  w_j = sum_{i in j} (y_i - p_i) / (sum_{i in j} h_i + lambda),
    which is exactly the second-order approximation used by XGBoost/LightGBM (lambda = L2 penalty on leaves).
    """

    def __init__(
        self, n_estimators: int = 200, learning_rate: float = 0.1, max_depth: int = 3, reg_lambda: float = 1.0
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.reg_lambda = reg_lambda
        self.trees: list[tuple[DecisionTreeRegressor, dict[int, float]]] = []
        self.init_: float = 0.0

    @staticmethod
    def _sigmoid(F: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-F))

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SimpleGradientBoostingClassifier":
        p0 = np.clip(np.mean(y), 1e-6, 1 - 1e-6)
        self.init_ = float(np.log(p0 / (1 - p0)))  # log-odds of the base rate
        F = np.full(len(y), self.init_)
        self.trees = []
        for _ in range(self.n_estimators):
            p = self._sigmoid(F)
            neg_grad, hess = y - p, p * (1 - p)
            tree = DecisionTreeRegressor(max_depth=self.max_depth, random_state=SEED).fit(X, neg_grad)
            leaf_ids = tree.apply(X)
            leaf_values = {
                leaf: neg_grad[leaf_ids == leaf].sum() / (hess[leaf_ids == leaf].sum() + self.reg_lambda)
                for leaf in np.unique(leaf_ids)
            }
            F += self.learning_rate * np.vectorize(leaf_values.get)(leaf_ids)
            self.trees.append((tree, leaf_values))
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        F = np.full(len(X), self.init_)
        for tree, leaf_values in self.trees:
            F += self.learning_rate * np.vectorize(leaf_values.get)(tree.apply(X))
        return F

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        p = self._sigmoid(self.decision_function(X))
        return np.column_stack([1 - p, p])


def regression_demo() -> None:
    """Boosting stumps-of-depth-2 on a noisy sine curve; compare with scikit-learn."""
    rng = np.random.default_rng(SEED)
    X = np.sort(rng.uniform(0, 6, 300)).reshape(-1, 1)
    y = np.sin(X).ravel() + 0.3 * X.ravel() + rng.normal(0, 0.25, 300)

    ours = SimpleGradientBoostingRegressor(n_estimators=200, learning_rate=0.1, max_depth=2).fit(X, y)
    sk = GradientBoostingRegressor(
        n_estimators=200, learning_rate=0.1, max_depth=2, random_state=SEED
    ).fit(X, y)
    diff = np.max(np.abs(ours.predict(X) - sk.predict(X)))
    print(f"[regression] max |ours - sklearn GradientBoostingRegressor| = {diff:.2e} (should be ~0)")

    x_grid = np.linspace(0, 6, 500).reshape(-1, 1)
    stages = list(ours.staged_predict(x_grid))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(X, y, s=8, color="grey", alpha=0.5, label="data")
    for m in [1, 5, 20, 200]:
        ax.plot(x_grid, stages[m - 1], lw=2, label=f"after {m} trees")
    ax.set(title="Gradient boosting: each tree corrects the residuals of the ensemble", xlabel="x", ylabel="y")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "boosting_stages_regression.png"), dpi=120)
    plt.close(fig)


def classification_demo() -> None:
    """Newton-boosting with log-loss on the churn data, compared with scikit-learn; learning-rate study."""
    X, y, true_p = make_churn_dataset(n_samples=8_000, seed=SEED, categorical="codes", return_proba=True)
    X_train, X_val, _, y_train, y_val, _ = train_val_test_split(X, y, seed=SEED)
    Xtr, Xva, ytr, yva = X_train.to_numpy(), X_val.to_numpy(), y_train.to_numpy(), y_val.to_numpy()
    print(f"\n[classification] churn data: train={len(ytr)}, val={len(yva)}, churn rate={ytr.mean():.3f}, "
          f"best achievable val ROC-AUC (true probabilities)={roc_auc_score(yva, true_p[X_val.index]):.4f}")

    t0 = time.perf_counter()
    ours = SimpleGradientBoostingClassifier(n_estimators=150, learning_rate=0.1, max_depth=3).fit(Xtr, ytr)
    t_ours = time.perf_counter() - t0
    t0 = time.perf_counter()
    sk = GradientBoostingClassifier(n_estimators=150, learning_rate=0.1, max_depth=3, random_state=SEED).fit(Xtr, ytr)
    t_sk = time.perf_counter() - t0
    for name, model, t in [("ours (Newton leaves)", ours, t_ours), ("sklearn GradientBoostingClassifier", sk, t_sk)]:
        p = model.predict_proba(Xva)[:, 1]
        print(f"  {name:36s} val log-loss={log_loss(yva, p):.4f}  val ROC-AUC={roc_auc_score(yva, p):.4f}  fit={t:.1f}s")

    # Learning rate vs number of trees: small learning rates need more trees but generalise better
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for lr in [1.0, 0.3, 0.1, 0.03]:
        gb = GradientBoostingClassifier(n_estimators=400, learning_rate=lr, max_depth=3, random_state=SEED).fit(Xtr, ytr)
        train_loss = [log_loss(ytr, p[:, 1]) for p in gb.staged_predict_proba(Xtr)]
        val_loss = [log_loss(yva, p[:, 1]) for p in gb.staged_predict_proba(Xva)]
        best = int(np.argmin(val_loss))
        axes[0].plot(train_loss, label=f"lr={lr}")
        axes[1].plot(val_loss, label=f"lr={lr} (best: {val_loss[best]:.4f} @ {best + 1} trees)")
        axes[1].scatter([best], [val_loss[best]], s=25)
    axes[0].set(title="Training log-loss keeps decreasing", xlabel="number of trees", ylabel="log-loss")
    axes[1].set(title="Validation log-loss: U-shape -> early stopping", xlabel="number of trees")
    axes[0].legend()
    axes[1].legend(fontsize=8)
    axes[0].set_ylim(0.0, 0.6)
    axes[1].set_ylim(0.40, 0.55)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "boosting_learning_rate.png"), dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Regression: boosting = fitting residuals stage-wise
    regression_demo()

    # 2./3. Classification with log-loss + learning-rate study
    classification_demo()
    print(f"\nPlots saved to {OUTPUT_DIR}")
