"""
Declarative data validation with pandera.

Instead of writing each check by hand (main_1_quality_report.py), pandera lets us describe the expected
data as a *schema* - a class with one typed field per column plus column- and table-level checks.
The schema is executable documentation: it can validate data frames, report every failure at once
(lazy validation) and guard function inputs/outputs via the ``@pa.check_types`` decorator.

This script
1. defines schemas for the clean customers and transactions tables
2. validates the clean data (passes)
3. validates the raw data lazily and summarises all failure cases
4. uses ``@pa.check_types`` so that a function refuses invalid input

Run from the repository root:
    python -m p03_data_processing.c00_structured_data.s02_dqa.main_3_pandera_validation
"""

import warnings

import pandas as pd
import pandera.pandas as pa
from pandera.typing import DataFrame, Series

from p03_data_processing.c00_structured_data.s01_preprocessing.utils.cleaning import (
    EMAIL_PATTERN,
    clean_customers,
    clean_transactions,
    standardise_column_names,
)
from p03_data_processing.c00_structured_data.utils.synthetic_data import (
    CATEGORIES,
    COUNTRIES,
    PLANS,
    SEGMENTS,
    generate_raw_data,
)

OBSERVATION_END = pd.Timestamp("2024-12-31 23:59:59")
# pandera tries to interpret raw text dates when reporting dtype failures; silence pandas' format warnings
warnings.filterwarnings("ignore", message="Parsing dates in", category=UserWarning)


class CustomerSchema(pa.DataFrameModel):
    """Expected structure and content of the clean customers table."""

    customer_id: Series[str] = pa.Field(unique=True, str_matches=r"^C\d{6}$")
    signup_date: Series[pd.Timestamp] = pa.Field(nullable=True, le=OBSERVATION_END)
    age: Series[pd.Int64Dtype] = pa.Field(nullable=True, in_range={"min_value": 18, "max_value": 100})
    country: Series[pd.CategoricalDtype] = pa.Field(nullable=True, isin=COUNTRIES)
    segment: Series[pd.CategoricalDtype] = pa.Field(isin=SEGMENTS)
    plan: Series[pd.CategoricalDtype] = pa.Field(isin=PLANS)
    annual_income: Series[float] = pa.Field(nullable=True, in_range={"min_value": 0, "max_value": 1_000_000})
    support_tickets: Series[pd.Int64Dtype] = pa.Field(nullable=True, ge=0)
    email: Series[str] = pa.Field(nullable=True, str_matches=EMAIL_PATTERN)
    churned: Series[pd.BooleanDtype] = pa.Field(nullable=False)

    @pa.dataframe_check
    @classmethod
    def income_mostly_present(cls, df: pd.DataFrame) -> bool:
        """Table-level check: at most 10 % of the incomes may be missing."""
        return df["annual_income"].isna().mean() <= 0.10

    class Config:
        strict = True  # no unexpected columns


class TransactionSchema(pa.DataFrameModel):
    """Expected structure and content of the clean transactions table."""

    transaction_id: Series[str] = pa.Field(unique=True)
    customer_id: Series[str]
    timestamp: Series[pd.Timestamp] = pa.Field(le=OBSERVATION_END)
    # extreme but possible bulk orders exist (they are winsorised for modelling), so the range is generous
    amount: Series[float] = pa.Field(in_range={"min_value": -50_000, "max_value": 50_000})
    category: Series[pd.CategoricalDtype] = pa.Field(isin=CATEGORIES)
    payment_method: Series[pd.CategoricalDtype] = pa.Field(nullable=True)
    is_refund: Series[bool]

    @pa.dataframe_check
    @classmethod
    def refund_flag_consistent(cls, df: pd.DataFrame) -> pd.Series:
        """Row-wise cross-column check: is_refund is True exactly for negative amounts."""
        return df["is_refund"] == (df["amount"] < 0)


@pa.check_types
def churn_rate_by_segment(customers: DataFrame[CustomerSchema]) -> pd.Series:
    """Business logic that only accepts validated customers (the decorator validates the argument)."""
    return customers.groupby("segment", observed=True)["churned"].mean()


def summarise_failures(err: pa.errors.SchemaErrors) -> pd.DataFrame:
    """Condense pandera's failure cases to one row per (column, check) with a few examples."""
    fc = err.failure_cases
    return (
        fc.groupby(["column", "check"], dropna=False)["failure_case"]
        .agg(n_failures="size", examples=lambda s: list(pd.unique(s.astype(str)))[:3])
        .reset_index()
        .sort_values("n_failures", ascending=False)
    )


if __name__ == "__main__":
    # 1. Data: raw (standardised column names only) and clean
    customers_raw, transactions_raw = generate_raw_data(n_customers=2000, seed=42)
    customers_raw = standardise_column_names(customers_raw).rename(
        columns={"annual_income_eur": "annual_income", "e_mail": "email"})
    transactions_raw = standardise_column_names(transactions_raw)
    customers = clean_customers(customers_raw)
    transactions = clean_transactions(transactions_raw, valid_customer_ids=customers["customer_id"])

    # 2. Clean data passes the schemas
    CustomerSchema.validate(customers, lazy=True)
    TransactionSchema.validate(transactions, lazy=True)
    print("clean customers and transactions: schema validation passed")

    # 3. Raw data: collect *all* failures (lazy=True) instead of stopping at the first one
    for name, schema, df in [("raw customers", CustomerSchema, customers_raw),
                             ("raw transactions", TransactionSchema, transactions_raw)]:
        try:
            schema.validate(df, lazy=True)
        except pa.errors.SchemaErrors as err:
            summary = summarise_failures(err)
            print(f"\n{name}: {len(err.failure_cases)} failure cases in {len(summary)} (column, check) pairs")
            print(summary.to_string(index=False, max_colwidth=70))

    # 4. Guarding functions: valid input works, invalid input raises a clear error
    print("\nchurn rate by segment (validated input):")
    print(churn_rate_by_segment(customers).round(3).to_string())
    try:
        churn_rate_by_segment(customers_raw)
    except (pa.errors.SchemaError, pa.errors.SchemaErrors) as err:
        print(f"\nrejected raw input: {type(err).__name__}: {str(err).splitlines()[0][:150]}")
