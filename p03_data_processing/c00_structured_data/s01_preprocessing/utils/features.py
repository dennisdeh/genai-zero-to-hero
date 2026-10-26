"""
Feature engineering for the customer churn example.

- ``build_customer_features`` aggregates the transactions per customer and joins them to the customers.
  These are *row-wise* aggregations (each customer only uses its own transactions), so they can be
  computed before the train/test split without leaking information between rows.
- ``add_ratio_features`` creates ratio features; it is stateless and used inside the sklearn pipeline.
- ``build_preprocessor`` returns the ColumnTransformer with all learned preprocessing steps, which
  must only ever be fitted on training data.
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, OrdinalEncoder, StandardScaler

from p03_data_processing.c00_structured_data.s01_preprocessing.utils.cleaning import QuantileClipper
from p03_data_processing.c00_structured_data.utils.synthetic_data import PLANS

NUMERIC_FEATURES = ["age", "support_tickets", "n_transactions", "days_since_last_tx", "tenure_days",
                    "n_categories", "spend_per_month", "refund_share"]
SKEWED_FEATURES = ["annual_income", "total_spend", "avg_amount"]
NOMINAL_FEATURES = ["country", "segment", "favourite_category"]
ORDINAL_FEATURES = ["plan"]
TARGET = "churned"


def build_customer_features(customers: pd.DataFrame, transactions: pd.DataFrame,
                            reference_date: pd.Timestamp | None = None) -> pd.DataFrame:
    """
    Aggregate transactions per customer and join the aggregates to the customer table.

    :param customers: clean customers
    :param transactions: clean transactions
    :param reference_date: "today" for recency/tenure features (default: last transaction timestamp)
    :return: one row per customer with raw attributes and aggregated behaviour features
    """
    if reference_date is None:
        reference_date = transactions["timestamp"].max()
    purchases = transactions[~transactions["is_refund"]]
    agg = purchases.groupby("customer_id").agg(
        n_transactions=("transaction_id", "count"),
        total_spend=("amount", "sum"),
        avg_amount=("amount", "mean"),
        last_tx=("timestamp", "max"),
        n_categories=("category", "nunique"),
    )
    agg["favourite_category"] = purchases.groupby("customer_id")["category"].agg(
        lambda s: s.mode().iloc[0] if not s.mode().empty else np.nan).astype(str)
    refunds = transactions.groupby("customer_id")["is_refund"].mean().rename("refund_share")
    df = customers.merge(agg, on="customer_id", how="left").merge(refunds, on="customer_id", how="left")

    # customers without purchases: counts are 0, recency stays missing (imputed in the pipeline)
    df["n_transactions"] = df["n_transactions"].fillna(0)
    df["total_spend"] = df["total_spend"].fillna(0.0)
    df["n_categories"] = df["n_categories"].fillna(0)
    df["refund_share"] = df["refund_share"].fillna(0.0)
    df["days_since_last_tx"] = (reference_date - df["last_tx"]).dt.days
    df["tenure_days"] = (reference_date - df["signup_date"]).dt.days
    df = df.drop(columns=["last_tx"])
    # nullable pandas dtypes -> plain floats (sklearn expects numpy compatible input)
    for col in ["age", "support_tickets"]:
        df[col] = df[col].astype("Float64").astype(float)
    df[TARGET] = df[TARGET].astype(int)
    return df


def add_ratio_features(X: pd.DataFrame) -> pd.DataFrame:
    """Stateless feature engineering: spend per month of tenure (safe to apply inside the pipeline)."""
    X = X.copy()
    months = (X["tenure_days"] / 30.0).clip(lower=1.0)
    X["spend_per_month"] = X["total_spend"] / months
    return X


def ratio_feature_names(transformer, input_features) -> list[str]:
    """Output feature names of ``add_ratio_features`` (needed for get_feature_names_out)."""
    return [*input_features, "spend_per_month"]


def build_preprocessor() -> Pipeline:
    """
    Preprocessing pipeline: ratio features followed by a ColumnTransformer with one branch per column type.

    - numeric: winsorise (thresholds learned on train) -> median imputation (+ missing indicator) -> scaling
    - skewed money amounts: median imputation -> log1p -> scaling
    - nominal categories: most frequent imputation -> one-hot encoding (unknown categories are ignored)
    - ordinal plan: most frequent imputation -> ordinal encoding Basic < Standard < Premium
    """
    numeric = Pipeline([
        ("clip", QuantileClipper(0.01, 0.99)),
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
    ])
    skewed = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
        ("scale", StandardScaler()),
    ])
    nominal = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    ordinal = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("ordinal", OrdinalEncoder(categories=[PLANS])),
    ])
    columns = ColumnTransformer(
        [
            ("num", numeric, NUMERIC_FEATURES),
            ("skewed", skewed, SKEWED_FEATURES),
            ("nominal", nominal, NOMINAL_FEATURES),
            ("ordinal", ordinal, ORDINAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )
    return Pipeline([
        ("ratios", FunctionTransformer(add_ratio_features, feature_names_out=ratio_feature_names)),
        ("columns", columns),
    ])


def to_model_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Select the raw model input columns and convert categoricals to plain objects for sklearn."""
    engineered = {"spend_per_month"}  # created inside the pipeline by add_ratio_features
    cols = [c for c in NUMERIC_FEATURES + SKEWED_FEATURES + NOMINAL_FEATURES + ORDINAL_FEATURES
            if c not in engineered]
    X = df[cols].copy()
    for col in NOMINAL_FEATURES + ORDINAL_FEATURES:
        X[col] = X[col].astype(object).where(X[col].notna(), np.nan)
    return X
