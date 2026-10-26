"""
Synthetic, offline "customer churn" dataset used throughout p05 (gradient boosting, explainability, MLOps).

Why synthetic? It runs offline, is large enough for boosting to shine, and - crucially for teaching - we
know the TRUE data-generating process. This lets us check whether a model (and its explanations) recovers
the real effects, e.g. that the noise feature is unimportant or that churn falls with tenure.

The true churn log-odds are
    logit p = -1.0
              + 1.2 * [contract = month-to-month] - 0.9 * [contract = two-year]
              - 1.0 * log(1 + tenure_months / 6)                         (non-linear, decreasing)
              + (0.01 + 0.03 * [contract = month-to-month]) * (monthly_charges - 65)   (interaction)
              + 0.4 * has_fibre
              + 0.3 * min(support_calls, 5)                              (saturating)
              + 1.2 * has_fibre * [support_calls >= 3]                   (interaction)
              + 0.7 * [payment_method = electronic_check]
              + 0.0015 * ((age - 45)^2 - 340)                            (U-shaped)
num_products, total_charges and noise have NO direct effect (total_charges is strongly correlated with
tenure and monthly_charges, which matters for feature-importance pitfalls in c02).
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

CONTRACTS = ["month-to-month", "one-year", "two-year"]
PAYMENT_METHODS = ["electronic_check", "bank_transfer", "credit_card", "mailed_check"]
CATEGORICAL_FEATURES = ["contract", "payment_method"]
TARGET = "churn"


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def make_churn_dataset(
    n_samples: int = 20_000,
    seed: int = 0,
    drift: float = 0.0,
    categorical: str = "category",
    return_proba: bool = False,
) -> tuple[pd.DataFrame, pd.Series] | tuple[pd.DataFrame, pd.Series, np.ndarray]:
    """
    Generate the synthetic churn dataset.

    Args:
        n_samples: number of customers.
        seed: random seed (different seeds give independent samples from the same distribution).
        drift: 0 = reference distribution. Values > 0 simulate data drift (covariate shift): more recent,
            short-tenure customers, higher prices and more fibre contracts - and a slightly stronger price
            sensitivity (concept drift). Used in the MLOps chapter.
        categorical: "category" -> pandas categorical dtype (native categorical support in HistGradientBoosting,
            XGBoost, LightGBM); "codes" -> integer codes 0..k-1 (purely numeric frame, convenient for SHAP/MLflow).
        return_proba: also return the true churn probabilities (to compute the best achievable score).

    Returns:
        X (features), y (0/1 churn label) and optionally the true probabilities.
    """
    rng = np.random.default_rng(seed)
    n = n_samples

    has_fibre = rng.binomial(1, min(0.45 + 0.2 * drift, 0.95), n)
    tenure = np.clip(rng.gamma(shape=2.0, scale=14.0 / (1.0 + drift), size=n), 1, 72).round()
    # Longer-tenured customers are more likely to be on long contracts
    p_long = _sigmoid(-1.5 + 0.06 * tenure)
    u = rng.random(n)
    contract_idx = np.where(u < 1 - p_long, 0, np.where(u < 1 - p_long / 2, 1, 2))
    monthly = np.clip(25 + 70 * rng.beta(2, 2, n) + 15 * has_fibre + 10 * drift, 18, 130).round(2)
    total = (tenure * monthly * rng.normal(1.0, 0.05, n)).round(2)
    support_calls = rng.poisson(1.0 + 0.6 * has_fibre)
    payment_idx = rng.choice(len(PAYMENT_METHODS), size=n, p=[0.35, 0.22, 0.22, 0.21])
    age = rng.integers(18, 81, n)
    num_products = rng.integers(1, 6, n)
    noise = rng.normal(0, 1, n)

    logit = (
        -1.0
        + 1.2 * (contract_idx == 0)
        - 0.9 * (contract_idx == 2)
        - 1.0 * np.log1p(tenure / 6)
        + (0.01 + 0.03 * (contract_idx == 0) + 0.01 * drift) * (monthly - 65)
        + 0.4 * has_fibre
        + 0.3 * np.minimum(support_calls, 5)
        + 1.2 * has_fibre * (support_calls >= 3)
        + 0.7 * (payment_idx == 0)
        + 0.0015 * ((age - 45) ** 2 - 340)
    )
    proba = _sigmoid(logit)
    y = pd.Series(rng.binomial(1, proba), name=TARGET)

    X = pd.DataFrame(
        {
            "tenure_months": tenure,
            "monthly_charges": monthly,
            "total_charges": total,
            "contract": contract_idx,
            "payment_method": payment_idx,
            "has_fibre": has_fibre,
            "support_calls": support_calls,
            "age": age,
            "num_products": num_products,
            "noise": noise,
        }
    )
    if categorical == "category":
        X["contract"] = pd.Categorical.from_codes(contract_idx, CONTRACTS)
        X["payment_method"] = pd.Categorical.from_codes(payment_idx, PAYMENT_METHODS)
    elif categorical != "codes":
        raise ValueError("categorical must be 'category' or 'codes'")
    # Use float64 for every numeric column: avoids MLflow schema warnings about integer columns with missing values
    num_cols = [c for c in X.columns if c not in CATEGORICAL_FEATURES or categorical == "codes"]
    X[num_cols] = X[num_cols].astype("float64")

    if return_proba:
        return X, y, proba
    return X, y


def train_val_test_split(
    X: pd.DataFrame, y: pd.Series, val_size: float = 0.2, test_size: float = 0.2, seed: int = 0
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Stratified split into train / validation / test sets (default 60/20/20)."""
    X_tmp, X_test, y_tmp, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=seed)
    rel_val = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(X_tmp, y_tmp, test_size=rel_val, stratify=y_tmp, random_state=seed)
    return X_train, X_val, X_test, y_train, y_val, y_test


if __name__ == "__main__":
    X, y, p = make_churn_dataset(return_proba=True)
    print(X.head())
    print(X.dtypes)
    print(f"churn rate={y.mean():.3f}")
    from sklearn.metrics import roc_auc_score

    print(f"Bayes-optimal ROC-AUC={roc_auc_score(y, p):.4f}")
    Xd, yd = make_churn_dataset(drift=1.0, seed=1)
    print(f"drifted churn rate={yd.mean():.3f}, mean tenure {X.tenure_months.mean():.1f} -> {Xd.tenure_months.mean():.1f}")
