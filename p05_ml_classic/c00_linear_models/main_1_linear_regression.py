"""
Linear regression from scratch and with scikit-learn.

This script demonstrates:
    1. Ordinary least squares (OLS) solved in closed form (normal equation) and with batch gradient descent,
       compared with scikit-learn's LinearRegression on the offline diabetes dataset.
    2. Regularised linear models (Ridge, Lasso, ElasticNet) whose regularisation strength is chosen by
       cross-validation, on an expanded (degree-2 polynomial) feature set where plain OLS overfits.
    3. Regularisation paths (how coefficients shrink as the penalty grows).
    4. Residual diagnostics (residuals vs fitted values, Q-Q plot, histogram).

Plots are saved to the outputs/ folder next to this script.

How to run (from the repository root):
    python -m p05_ml_classic.c00_linear_models.main_1_linear_regression
"""

import os

import matplotlib

matplotlib.use("Agg")  # non-interactive backend: write files, never open windows
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.datasets import load_diabetes
from sklearn.linear_model import (
    ElasticNetCV,
    LassoCV,
    LinearRegression,
    RidgeCV,
    lasso_path,
)
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
SEED = 42


def ols_normal_equation(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Solve OLS in closed form: beta = (X^T X)^{-1} X^T y.

    A column of ones is prepended for the intercept. We use np.linalg.solve instead of explicitly
    inverting X^T X, which is faster and numerically more stable.
    Returns the parameter vector [intercept, beta_1, ..., beta_p].
    """
    X1 = np.column_stack([np.ones(len(X)), X])
    return np.linalg.solve(X1.T @ X1, X1.T @ y)


def ols_gradient_descent(
    X: np.ndarray, y: np.ndarray, lr: float = 0.1, n_iter: int = 2000
) -> tuple[np.ndarray, list[float]]:
    """
    Minimise the mean squared error with batch gradient descent.

    Loss: L(beta) = 1/n * ||y - X1 beta||^2, gradient: -2/n * X1^T (y - X1 beta).
    Gradient descent converges much faster when the features are standardised (similar scales give a
    well-conditioned loss surface), so X should be standardised before calling this function.
    Returns the parameters and the loss history.
    """
    n = len(X)
    X1 = np.column_stack([np.ones(n), X])
    beta = np.zeros(X1.shape[1])
    history = []
    for _ in range(n_iter):
        residual = y - X1 @ beta
        history.append(float(np.mean(residual**2)))
        grad = -2.0 / n * X1.T @ residual
        beta -= lr * grad
    return beta, history


def evaluate(name: str, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Return a dictionary with RMSE and R^2 for a set of predictions."""
    return {
        "model": name,
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def plot_regularisation_paths(X: np.ndarray, y: np.ndarray, feature_names: list[str], path: str) -> None:
    """
    Plot Ridge and Lasso coefficient paths as a function of the regularisation strength alpha.

    Both use scikit-learn's conventions, which differ: Ridge minimises ||y - Xw||^2 + alpha ||w||^2 (summed loss),
    Lasso minimises 1/(2n) ||y - Xw||^2 + alpha ||w||_1 (mean loss), so the two alpha axes are not comparable.
    """
    ridge_alphas = np.logspace(-1, 5.5, 60)
    ridge_coefs = []
    p = X.shape[1]
    for a in ridge_alphas:
        # closed-form ridge on standardised features (sklearn convention): (X^T X + alpha I)^{-1} X^T y (y centred)
        ridge_coefs.append(np.linalg.solve(X.T @ X + a * np.eye(p), X.T @ (y - y.mean())))
    ridge_coefs = np.array(ridge_coefs)
    lasso_alphas, lasso_coefs, _ = lasso_path(X, y - y.mean(), alphas=np.logspace(-2, 1.5, 60))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for j, name in enumerate(feature_names):
        axes[0].plot(ridge_alphas, ridge_coefs[:, j], label=name)
        axes[1].plot(lasso_alphas, lasso_coefs[j], label=name)
    titles = ["Ridge (L2): coefficients shrink smoothly", "Lasso (L1): coefficients hit zero"]
    xlabels = [
        r"Ridge alpha:  $\|y - Xw\|^2 + \alpha\|w\|_2^2$  (sklearn Ridge)",
        r"Lasso alpha:  $\frac{1}{2n}\|y - Xw\|^2 + \alpha\|w\|_1$  (sklearn Lasso)",
    ]
    for ax, title, xlabel in zip(axes, titles, xlabels):
        ax.set_xscale("log")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("coefficient (standardised features)")
        ax.set_title(title)
        ax.axhline(0, color="black", lw=0.5)
    axes[1].legend(loc="upper right", fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_residual_diagnostics(y_true: np.ndarray, y_pred: np.ndarray, path: str) -> None:
    """Residuals vs fitted, normal Q-Q plot and histogram of the residuals."""
    residuals = y_true - y_pred
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    axes[0].scatter(y_pred, residuals, alpha=0.6)
    axes[0].axhline(0, color="red", lw=1)
    axes[0].set_xlabel("fitted value")
    axes[0].set_ylabel("residual")
    axes[0].set_title("Residuals vs fitted (look for patterns / funnels)")
    stats.probplot(residuals, dist="norm", plot=axes[1])
    axes[1].set_title("Normal Q-Q plot of residuals")
    axes[2].hist(residuals, bins=25, edgecolor="black")
    axes[2].set_title("Residual distribution")
    axes[2].set_xlabel("residual")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Load data: 442 diabetes patients, 10 baseline variables, target = disease progression after one year
    data = load_diabetes(as_frame=True, scaled=False)
    X_df, y = data.data, data.target.to_numpy()
    feature_names = list(X_df.columns)
    print(f"Diabetes dataset: {X_df.shape[0]} samples, {X_df.shape[1]} features: {feature_names}")
    X_train_df, X_test_df, y_train, y_test = train_test_split(X_df, y, test_size=0.25, random_state=SEED)

    # 2. Standardise features (fit the scaler on the training data only, to avoid leakage)
    scaler = StandardScaler().fit(X_train_df)
    X_train = scaler.transform(X_train_df)
    X_test = scaler.transform(X_test_df)

    # 3. OLS three ways: normal equation, gradient descent, scikit-learn
    beta_ne = ols_normal_equation(X_train, y_train)
    beta_gd, loss_history = ols_gradient_descent(X_train, y_train, lr=0.2, n_iter=20_000)
    sk_ols = LinearRegression().fit(X_train, y_train)
    beta_sk = np.concatenate([[sk_ols.intercept_], sk_ols.coef_])

    coef_table = pd.DataFrame(
        {"normal_eq": beta_ne, "grad_descent": beta_gd, "sklearn": beta_sk},
        index=["intercept"] + feature_names,
    )
    print("\nOLS coefficients (standardised features) - the three methods should agree:")
    print(coef_table.round(3).to_string())
    print(f"Max |normal_eq - sklearn| = {np.max(np.abs(beta_ne - beta_sk)):.2e}")
    print(f"Max |grad_descent - sklearn| = {np.max(np.abs(beta_gd - beta_sk)):.2e}")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(loss_history)
    ax.set_yscale("log")
    ax.set_xlabel("iteration")
    ax.set_ylabel("training MSE")
    ax.set_title("Gradient descent on the OLS loss")
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "linreg_gradient_descent_loss.png"), dpi=120)
    plt.close(fig)

    results = [evaluate("OLS (10 raw features)", y_test, sk_ols.predict(X_test))]

    # 4. Regularisation: expand to all degree-2 polynomial terms (65 features for ~330 training samples).
    #    Plain OLS now has many more parameters and starts to overfit; regularisation helps.
    cv = KFold(n_splits=5, shuffle=True, random_state=SEED)
    alphas = np.logspace(-3, 3, 50)  # Ridge (closed form, cheap)
    l1_alphas = np.logspace(-1.5, 1.5, 30)  # Lasso/ElasticNet: very small alphas converge slowly (collinear features)
    poly = lambda: PolynomialFeatures(degree=2, include_bias=False)  # noqa: E731  (fresh transformer)
    models: dict[str, Pipeline] = {
        "OLS (poly-2)": make_pipeline(poly(), StandardScaler(), LinearRegression()),
        "RidgeCV (poly-2)": make_pipeline(poly(), StandardScaler(), RidgeCV(alphas=alphas)),
        "LassoCV (poly-2)": make_pipeline(poly(), StandardScaler(), LassoCV(alphas=l1_alphas, cv=cv, max_iter=50_000)),
        "ElasticNetCV (poly-2)": make_pipeline(
            poly(),
            StandardScaler(),
            ElasticNetCV(l1_ratio=[0.1, 0.5, 0.9], alphas=l1_alphas, cv=cv, max_iter=50_000),
        ),
    }
    for name, model in models.items():
        model.fit(X_train_df, y_train)
        results.append(evaluate(name, y_test, model.predict(X_test_df)))
        est = model[-1]
        if hasattr(est, "alpha_"):
            n_nonzero = int(np.sum(np.abs(est.coef_) > 1e-8))
            extra = f", l1_ratio={est.l1_ratio_}" if hasattr(est, "l1_ratio_") and not isinstance(est.l1_ratio_, list) else ""
            print(f"{name:24s}: alpha={est.alpha_:.4g}{extra}, non-zero coefficients={n_nonzero}/{est.coef_.size}")

    print("\nTest-set performance (lower RMSE / higher R^2 is better):")
    print(pd.DataFrame(results).round(3).to_string(index=False))

    # 5. Regularisation paths on the 10 standardised raw features
    plot_regularisation_paths(X_train, y_train, feature_names, os.path.join(OUTPUT_DIR, "linreg_regularisation_paths.png"))

    # 6. Residual diagnostics for the plain OLS model on the test set
    y_pred = sk_ols.predict(X_test)
    residuals = y_test - y_pred
    shapiro_p = stats.shapiro(residuals).pvalue
    # Correlation between |residual| and fitted value: a crude check for heteroscedasticity
    hetero_corr = stats.spearmanr(np.abs(residuals), y_pred).statistic
    print(
        f"\nResidual diagnostics (OLS, test set): mean={residuals.mean():.2f}, skew={stats.skew(residuals):.2f}, "
        f"Shapiro-Wilk p={shapiro_p:.3f}, Spearman(|resid|, fitted)={hetero_corr:.2f}"
    )
    plot_residual_diagnostics(y_test, y_pred, os.path.join(OUTPUT_DIR, "linreg_residuals.png"))
    print(f"\nPlots saved to {OUTPUT_DIR}")
