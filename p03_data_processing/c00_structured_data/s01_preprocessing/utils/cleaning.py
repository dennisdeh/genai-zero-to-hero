"""
Reusable cleaning helpers for tabular data (pandas) and a small scikit-learn transformer for outliers.

The functions are deliberately small and composable; ``clean_customers`` and ``clean_transactions``
chain them into a complete cleaning procedure for the synthetic shop data of this chapter, and
``load_clean_data`` returns ready-to-use clean data (used by the feature pipeline, the data quality
section and the Streamlit app in part 4).
"""

import re

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from p03_data_processing.c00_structured_data.utils.synthetic_data import (
    CATEGORIES,
    CATEGORY_VARIANTS,
    COUNTRIES,
    COUNTRY_VARIANTS,
    PAYMENT_METHODS,
    PLANS,
    SEGMENT_VARIANTS,
    SEGMENTS,
    generate_raw_data,
)

DATE_FORMATS = ["%Y-%m-%d", "%d.%m.%Y", "%Y/%m/%d", "%B %d, %Y"]
DATETIME_FORMATS = ["%Y-%m-%d %H:%M:%S", "%d.%m.%Y %H:%M", "%Y-%m-%dT%H:%M:%S"]
EMAIL_PATTERN = r"^[\w.+-]+@[\w-]+\.[\w.]+$"


# ----------------------------------------------------------------------------------------------------
# generic helpers
# ----------------------------------------------------------------------------------------------------
def standardise_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Convert column names to snake_case, e.g. ' Annual Income (EUR)' -> 'annual_income_eur'."""
    def to_snake(name: str) -> str:
        name = re.sub(r"[^0-9a-zA-Z]+", "_", name.strip().lower())
        return name.strip("_")

    return df.rename(columns={c: to_snake(c) for c in df.columns})


def normalise_text(s: pd.Series) -> pd.Series:
    """Lower-case, strip and collapse whitespace/underscores/hyphens of a text column."""
    return s.astype("string").str.strip().str.lower().str.replace(r"[\s_\-]+", " ", regex=True)


def build_category_lookup(variants: dict[str, list[str]]) -> dict[str, str]:
    """Build a lookup {normalised spelling -> canonical value} from canonical values and their variants."""
    lookup = {}
    for canonical, spellings in variants.items():
        for spelling in [canonical, *spellings]:
            key = normalise_text(pd.Series([spelling])).iloc[0]
            lookup[key] = canonical
    return lookup


def standardise_categories(s: pd.Series, variants: dict[str, list[str]], categories: list[str]) -> pd.Series:
    """
    Map spelling variants to canonical categories; unknown values become missing.

    :param s: raw text column
    :param variants: canonical value -> list of known variants
    :param categories: the allowed (canonical) categories, in the desired order
    :return: categorical series
    """
    lookup = build_category_lookup(variants)
    mapped = normalise_text(s).map(lookup)
    return pd.Categorical(mapped, categories=categories)


def parse_mixed_dates(s: pd.Series, formats: list[str]) -> pd.Series:
    """
    Parse a text column that contains dates in several known formats.

    Each format is tried in turn on the values that are still unparsed. Being explicit about the
    formats avoids silent day/month swaps (is 03.04.2023 the 3rd of April or the 4th of March?).
    Values matching none of the formats become NaT.
    """
    result = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")
    for fmt in formats:
        todo = result.isna() & s.notna()
        if not todo.any():
            break
        result[todo] = pd.to_datetime(s[todo], format=fmt, errors="coerce")
    return result


def parse_numeric(s: pd.Series) -> pd.Series:
    """Convert text such as 'EUR 52,000', '€45.90' or '52000.0' to floats (unparsable -> NaN)."""
    cleaned = s.astype("string").str.replace(r"[^0-9.\-]", "", regex=True)
    return pd.to_numeric(cleaned, errors="coerce").astype(float)


def parse_boolean(s: pd.Series) -> pd.Series:
    """Map common boolean encodings ('yes'/'no', '1'/'0', 'True'/'False') to a nullable boolean."""
    mapping = {"yes": True, "y": True, "1": True, "true": True, "no": False, "n": False, "0": False, "false": False}
    return normalise_text(s).map(mapping).astype("boolean")


def iqr_bounds(s: pd.Series, k: float = 1.5) -> tuple[float, float]:
    """Tukey fences: [Q1 - k*IQR, Q3 + k*IQR]."""
    q1, q3 = s.quantile([0.25, 0.75])
    iqr = q3 - q1
    return q1 - k * iqr, q3 + k * iqr


def winsorise(s: pd.Series, lower_q: float = 0.01, upper_q: float = 0.99) -> pd.Series:
    """Clip values to the given lower/upper quantiles (winsorising)."""
    lower, upper = s.quantile([lower_q, upper_q])
    return s.clip(lower, upper)


def drop_near_duplicates(df: pd.DataFrame, key: str) -> pd.DataFrame:
    """For rows sharing the same key keep the most complete one (fewest missing values)."""
    completeness = df.notna().sum(axis=1)
    return (
        df.assign(_completeness=completeness)
        .sort_values([key, "_completeness"], ascending=[True, False])
        .drop_duplicates(subset=key, keep="first")
        .drop(columns="_completeness")
    )


class QuantileClipper(BaseEstimator, TransformerMixin):
    """
    scikit-learn transformer that winsorises numeric features.

    The clipping thresholds are *learned in fit* (on the training data only) and re-used in transform,
    which avoids leaking information from the test set into the preprocessing.
    """

    def __init__(self, lower_q: float = 0.01, upper_q: float = 0.99):
        self.lower_q = lower_q
        self.upper_q = upper_q

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.lower_ = np.nanquantile(X, self.lower_q, axis=0)
        self.upper_ = np.nanquantile(X, self.upper_q, axis=0)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        return np.clip(X, self.lower_, self.upper_)  # NaN stays NaN

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            return np.array([f"x{i}" for i in range(self.n_features_in_)], dtype=object)
        return np.asarray(input_features, dtype=object)


# ----------------------------------------------------------------------------------------------------
# dataset-specific cleaning procedures
# ----------------------------------------------------------------------------------------------------
def clean_customers(raw: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    """
    Clean the raw customers table.

    :param raw: raw customers data (all text, messy column names)
    :param verbose: print what each step changes
    :return: clean customers data frame with proper dtypes and one row per customer
    """
    def log(msg: str) -> None:
        if verbose:
            print(f"  - {msg}")

    df = standardise_column_names(raw).rename(columns={"annual_income_eur": "annual_income", "e_mail": "email"})

    # 1. duplicates: exact duplicates first, then several rows per customer id
    n0 = len(df)
    df = df.drop_duplicates()
    log(f"removed {n0 - len(df)} exact duplicate rows")
    n1 = len(df)
    df = drop_near_duplicates(df, key="customer_id")
    log(f"removed {n1 - len(df)} near-duplicate rows (kept the most complete record per customer)")

    # 2. types: dates, numbers, booleans
    df["signup_date"] = parse_mixed_dates(df["signup_date"], DATE_FORMATS)
    df["age"] = parse_numeric(df["age"])
    df["annual_income"] = parse_numeric(df["annual_income"])
    df["support_tickets"] = parse_numeric(df["support_tickets"]).astype("Int64")
    df["churned"] = parse_boolean(df["churned"])
    log(f"unparsable/missing signup dates: {df['signup_date'].isna().sum()}")

    # 3. categories
    df["country"] = standardise_categories(df["country"], COUNTRY_VARIANTS, COUNTRIES)
    df["segment"] = standardise_categories(df["segment"], SEGMENT_VARIANTS, SEGMENTS)
    df["plan"] = pd.Categorical(normalise_text(df["plan"]).str.title(), categories=PLANS, ordered=True)

    # 4. domain rules: invalid values become missing (they are imputed later in the model pipeline)
    invalid_age = ~df["age"].between(18, 100) & df["age"].notna()
    df.loc[invalid_age, "age"] = np.nan
    log(f"set {invalid_age.sum()} impossible ages (outside 18-100) to missing")
    implausible_income = df["annual_income"] > 1_000_000
    df.loc[implausible_income, "annual_income"] = np.nan
    log(f"set {implausible_income.sum()} implausible incomes (> 1M EUR, data-entry errors) to missing")
    invalid_email = ~df["email"].fillna("").str.match(EMAIL_PATTERN) & df["email"].notna()
    df.loc[invalid_email, "email"] = np.nan
    log(f"set {invalid_email.sum()} invalid email addresses to missing")
    df["age"] = df["age"].astype("Int64")

    # 5. a row without a label is useless for supervised learning
    n2 = len(df)
    df = df[df["churned"].notna()]
    log(f"dropped {n2 - len(df)} rows without churn label")
    return df.sort_values("customer_id").reset_index(drop=True)


def clean_transactions(raw: pd.DataFrame, valid_customer_ids: pd.Series | None = None,
                       verbose: bool = False) -> pd.DataFrame:
    """
    Clean the raw transactions table.

    :param raw: raw transactions data (all text, messy column names)
    :param valid_customer_ids: if given, transactions of unknown customers are removed (referential integrity)
    :param verbose: print what each step changes
    :return: clean transactions data frame
    """
    def log(msg: str) -> None:
        if verbose:
            print(f"  - {msg}")

    df = standardise_column_names(raw)

    # 1. duplicates (the transaction id is the primary key)
    n0 = len(df)
    df = df.drop_duplicates().drop_duplicates(subset="transaction_id")
    log(f"removed {n0 - len(df)} duplicate transactions")

    # 2. types
    df["timestamp"] = parse_mixed_dates(df["timestamp"], DATETIME_FORMATS)
    df["amount"] = parse_numeric(df["amount"])
    df["category"] = standardise_categories(df["category"], CATEGORY_VARIANTS, CATEGORIES)
    df["payment_method"] = pd.Categorical(normalise_text(df["payment_method"]).str.replace(" ", "_"),
                                          categories=PAYMENT_METHODS)
    df["is_refund"] = df["amount"] < 0
    log(f"refunds (negative amounts, kept and flagged): {df['is_refund'].sum()}")

    # 3. referential integrity: every transaction must belong to a known customer
    if valid_customer_ids is not None:
        known = df["customer_id"].isin(valid_customer_ids)
        log(f"removed {(~known).sum()} orphaned transactions (unknown customer_id)")
        df = df[known]
    return df.sort_values("timestamp").reset_index(drop=True)


def load_clean_data(n_customers: int = 2000, seed: int = 42, drift: float = 0.0,
                    id_offset: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate raw data and clean it (deterministic for a fixed seed).

    :return: (customers, transactions) clean data frames
    """
    customers_raw, transactions_raw = generate_raw_data(n_customers=n_customers, seed=seed, drift=drift,
                                                        id_offset=id_offset)
    customers = clean_customers(customers_raw)
    transactions = clean_transactions(transactions_raw, valid_customer_ids=customers["customer_id"])
    return customers, transactions
