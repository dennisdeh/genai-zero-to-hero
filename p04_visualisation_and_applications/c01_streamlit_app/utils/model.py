"""
Churn model for the dashboard: re-uses the feature pipeline of part 3.

The model only uses features that are known *before* a customer churns. "Days since the last purchase"
is deliberately not a feature: churned customers stop buying, so it is a consequence of the label
(target leakage) - see the preprocessing section of part 3.
"""

from dataclasses import dataclass

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from p03_data_processing.c00_structured_data.s01_preprocessing.utils.features import (
    TARGET,
    build_customer_features,
    build_preprocessor,
    to_model_frame,
)


@dataclass
class TrainedModel:
    """A fitted pipeline with its evaluation and the information needed to build an input form."""

    pipeline: Pipeline
    test_auc: float
    coefficients: pd.Series
    feature_frame: pd.DataFrame  # training inputs, used for default values and ranges in the form


def train_churn_model(customers: pd.DataFrame, transactions: pd.DataFrame, seed: int = 42) -> TrainedModel:
    """Build features, split, fit the logistic regression pipeline and evaluate it on the test set."""
    customers = customers.assign(country=customers["country"].astype(object).replace("Unknown", pd.NA))
    data = build_customer_features(customers, transactions)
    X, y = to_model_frame(data), data[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=seed)
    pipeline = Pipeline([
        ("preprocess", build_preprocessor()),
        ("model", LogisticRegression(max_iter=2000, class_weight="balanced")),
    ]).fit(X_train, y_train)
    auc = roc_auc_score(y_test, pipeline.predict_proba(X_test)[:, 1])
    names = pipeline.named_steps["preprocess"].get_feature_names_out()
    coefficients = pd.Series(pipeline.named_steps["model"].coef_[0], index=names)
    return TrainedModel(pipeline, float(auc), coefficients, X_train)


def predict_churn(model: TrainedModel, record: dict) -> float:
    """Churn probability of a single customer given as a dict of raw feature values."""
    X = pd.DataFrame([record])[model.feature_frame.columns]
    return float(model.pipeline.predict_proba(X)[0, 1])
