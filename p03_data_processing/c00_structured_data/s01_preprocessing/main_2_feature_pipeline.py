"""
Feature engineering and a leakage-free preprocessing pipeline with scikit-learn.

Starting from the cleaned customers & transactions data (see main_1_cleaning.py), this script

1. builds a customer-level modelling table (aggregated purchase behaviour + customer attributes)
2. splits it into train and test sets *before* any statistics are learned
3. assembles a Pipeline: stateless ratio features -> ColumnTransformer (winsorising, imputation, log
   transform, scaling, one-hot and ordinal encoding) -> logistic regression for churn prediction
4. evaluates it with cross-validation on the training set and once on the held-out test set
5. demonstrates why preprocessing must live inside the pipeline: feature selection on the full data
   yields impressive - but fake - accuracy on pure noise
6. saves the fitted pipeline and uses it to score a new, raw customer record

Run from the repository root:
    python -m p03_data_processing.c00_structured_data.s01_preprocessing.main_2_feature_pipeline
"""

import os

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline, make_pipeline

from p03_data_processing.c00_structured_data.s01_preprocessing.utils.cleaning import load_clean_data
from p03_data_processing.c00_structured_data.s01_preprocessing.utils.features import (
    TARGET,
    build_customer_features,
    build_preprocessor,
    to_model_frame,
)

PATH_OUTPUTS = os.path.join(os.path.dirname(__file__), "outputs")
SEED = 42


def build_model_pipeline() -> Pipeline:
    """Preprocessing and a logistic regression classifier in one estimator."""
    return Pipeline([
        ("preprocess", build_preprocessor()),
        ("model", LogisticRegression(max_iter=2000, class_weight="balanced")),
    ])


def demonstrate_leakage(n_samples: int = 200, n_features: int = 2000, seed: int = 0) -> None:
    """
    Show the effect of data leakage on a dataset where the label is pure noise.

    Selecting the "best" features on the full dataset before cross-validation lets the selection see the
    validation folds, so the model looks far better than chance. Inside the pipeline, the selection is
    refitted on each training fold and the honest score is ~0.5.
    """
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_samples, n_features))
    y = rng.integers(0, 2, n_samples)
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)

    X_selected = SelectKBest(f_classif, k=20).fit_transform(X, y)  # WRONG: uses all labels
    leaky = cross_val_score(LogisticRegression(max_iter=1000), X_selected, y, cv=cv).mean()
    pipe = make_pipeline(SelectKBest(f_classif, k=20), LogisticRegression(max_iter=1000))
    honest = cross_val_score(pipe, X, y, cv=cv).mean()
    print(f"random labels, {n_features} noise features:")
    print(f"  feature selection BEFORE cross-validation (leaky): accuracy = {leaky:.2f}")
    print(f"  feature selection INSIDE the pipeline (honest):   accuracy = {honest:.2f}  (chance = 0.50)")


if __name__ == "__main__":
    # 1. Load clean data and build the customer-level modelling table
    customers, transactions = load_clean_data(n_customers=2000, seed=SEED)
    data = build_customer_features(customers, transactions)
    X, y = to_model_frame(data), data[TARGET]
    print(f"modelling table: {X.shape[0]} customers x {X.shape[1]} raw features, churn rate {y.mean():.1%}")
    print(X.head().to_string())

    # 2. Split FIRST - everything that learns from data (imputation values, scaling, clipping thresholds,
    #    categories) is fitted on the training set only
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
    print(f"\ntrain: {len(X_train)} rows, test: {len(X_test)} rows")

    # 3. Pipeline: preprocessing + model; inspect the transformed feature matrix
    pipeline = build_model_pipeline()
    preprocessor = build_preprocessor().fit(X_train)
    feature_names = preprocessor.get_feature_names_out()
    X_train_t = pd.DataFrame(preprocessor.transform(X_train), columns=feature_names, index=X_train.index)
    print(f"\n{len(feature_names)} model features after preprocessing:")
    print(X_train_t.describe().T[["mean", "std", "min", "max"]].round(2).to_string())

    # 4. Cross-validation on the training data, then a single evaluation on the test set
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    cv_auc = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="roc_auc")
    print(f"\n5-fold CV ROC AUC on train: {cv_auc.mean():.3f} +/- {cv_auc.std():.3f}")
    pipeline.fit(X_train, y_train)
    proba = pipeline.predict_proba(X_test)[:, 1]
    print(f"test ROC AUC: {roc_auc_score(y_test, proba):.3f}")
    print(classification_report(y_test, pipeline.predict(X_test), target_names=["stays", "churns"]))

    coefficients = pd.Series(pipeline.named_steps["model"].coef_[0], index=feature_names)
    print("largest coefficients (standardised features):")
    print(coefficients.reindex(coefficients.abs().sort_values(ascending=False).index).head(8).round(3).to_string())

    # 5. Why the pipeline matters: leakage demonstration
    print()
    demonstrate_leakage()

    # 6. Persist the fitted pipeline and score a new raw record (missing values and unseen categories are fine)
    os.makedirs(PATH_OUTPUTS, exist_ok=True)
    path_model = os.path.join(PATH_OUTPUTS, "churn_pipeline.joblib")
    joblib.dump(pipeline, path_model)
    loaded = joblib.load(path_model)
    new_customer = pd.DataFrame([{
        "age": 35, "support_tickets": 4, "n_transactions": 2, "days_since_last_tx": 200, "tenure_days": 600,
        "n_categories": 1, "refund_share": 0.0, "annual_income": np.nan, "total_spend": 90.0, "avg_amount": 45.0,
        "country": "Finland", "segment": "Consumer", "favourite_category": "Groceries", "plan": "Basic",
    }])
    print(f"\npipeline saved to {path_model}")
    print(f"churn probability of a new customer (unknown country, missing income): "
          f"{loaded.predict_proba(new_customer[X.columns])[0, 1]:.2f}")
