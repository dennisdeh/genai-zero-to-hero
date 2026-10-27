"""
Cleaning messy tabular data with pandas.

This script walks through a typical cleaning procedure on a synthetic (but realistically messy)
customers & transactions dataset of an online shop:

1. load the raw CSV exports (everything is text)
2. inspect the data: dtypes, missing values, inconsistent categories, duplicates
3. clean the customers table: column names, duplicates, dates, numbers, categories, domain rules
4. clean the transactions table: parse amounts/timestamps, flag refunds, enforce referential integrity
5. detect outliers with the IQR rule and treat them by winsorising
6. compare with the ground truth and save the clean data (parquet keeps the dtypes)

Run from the repository root:
    python -m p03_data_processing.c00_structured_data.s01_preprocessing.main_1_cleaning
"""

import os

import numpy as np
import pandas as pd

from p03_data_processing.c00_structured_data.s01_preprocessing.utils.cleaning import (
    clean_customers,
    clean_transactions,
    iqr_bounds,
    standardise_column_names,
    winsorise,
)
from p03_data_processing.c00_structured_data.utils.synthetic_data import generate_clean_data, generate_raw_data

PATH_OUTPUTS = os.path.join(os.path.dirname(__file__), "outputs")
N_CUSTOMERS = 2000
SEED = 42


def save_raw_exports(path: str) -> tuple[str, str]:
    """Generate the messy raw data and store it as CSV files (our "source system export")."""
    os.makedirs(path, exist_ok=True)
    customers_raw, transactions_raw = generate_raw_data(n_customers=N_CUSTOMERS, seed=SEED)
    path_c = os.path.join(path, "customers_raw.csv")
    path_t = os.path.join(path, "transactions_raw.csv")
    customers_raw.to_csv(path_c, index=False)
    transactions_raw.to_csv(path_t, index=False)
    return path_c, path_t


def inspect(df: pd.DataFrame, name: str, categorical: list[str]) -> None:
    """Print a quick overview: shape, dtypes, missing values, duplicates and category spellings."""
    print(f"\n--- {name}: {df.shape[0]} rows x {df.shape[1]} columns ---")
    overview = pd.DataFrame(
        {"dtype": df.dtypes.astype(str), "missing": df.isna().sum(), "missing_%": (100 * df.isna().mean()).round(1),
         "n_unique": df.nunique()}
    )
    print(overview.to_string())
    print(f"exact duplicate rows: {df.duplicated().sum()}")
    for col in categorical:
        values = df[col].value_counts(dropna=False)
        print(f"distinct spellings in '{col}' ({len(values)}): {[str(v) for v in values.index[:12]]}")


def outlier_report(s: pd.Series, name: str) -> None:
    """Print the IQR fences, the number of outliers and the effect of winsorising."""
    lower, upper = iqr_bounds(s)
    n_out = ((s < lower) | (s > upper)).sum()
    clipped = winsorise(s, 0.01, 0.99)
    print(f"\n{name}: IQR fences [{lower:,.2f}, {upper:,.2f}] -> {n_out} outliers ({n_out / len(s):.1%})")
    print(f"  before winsorising: mean={s.mean():,.2f}  std={s.std():,.2f}  max={s.max():,.2f}")
    print(f"  after  winsorising: mean={clipped.mean():,.2f}  std={clipped.std():,.2f}  max={clipped.max():,.2f}")
    print(f"  median (robust, unchanged): {s.median():,.2f}")
    if (s > 0).all():
        # for right-skewed (log-normal like) data the IQR rule is more sensible on the log scale
        log_s = np.log(s)
        log_lower, log_upper = iqr_bounds(log_s)
        n_log = ((log_s < log_lower) | (log_s > log_upper)).sum()
        print(f"  IQR rule on log scale: fences [{np.exp(log_lower):,.2f}, {np.exp(log_upper):,.2f}] "
              f"-> {n_log} outliers ({n_log / len(s):.1%})")


if __name__ == "__main__":
    # 1. Load the raw exports (all columns as text - never trust the inferred dtypes of raw data)
    path_c, path_t = save_raw_exports(PATH_OUTPUTS)
    customers_raw = pd.read_csv(path_c, dtype=str)
    transactions_raw = pd.read_csv(path_t, dtype=str)

    # 2. Inspect
    inspect(customers_raw, "raw customers", categorical=["country", "Segment", "Plan", "Churned?"])
    inspect(transactions_raw, "raw transactions", categorical=["Category", "Payment Method"])
    print("\nexamples of raw values:")
    print(customers_raw[["Signup Date ", "AGE", "Annual Income (EUR)", "E-mail"]].sample(8, random_state=1).to_string())
    print("column names after standardisation:", list(standardise_column_names(customers_raw).columns))

    # 3. Clean customers
    print("\n=== cleaning customers ===")
    customers = clean_customers(customers_raw, verbose=True)
    inspect(customers, "clean customers", categorical=["country", "segment", "plan", "churned"])

    # 4. Clean transactions (and enforce referential integrity against the clean customers)
    print("\n=== cleaning transactions ===")
    transactions = clean_transactions(transactions_raw, valid_customer_ids=customers["customer_id"], verbose=True)
    inspect(transactions, "clean transactions", categorical=["category", "payment_method"])

    # 5. Outliers: detect with IQR fences, treat by winsorising (here for descriptive statistics;
    #    for modelling the clipping thresholds must be learned on the training data only -> main_2)
    purchases = transactions.loc[~transactions["is_refund"], "amount"]
    outlier_report(purchases, "transaction amount")
    outlier_report(customers["annual_income"].dropna(), "annual income")

    # 6. Sanity check against the ground truth that we know because the data is synthetic
    truth_customers, truth_transactions = generate_clean_data(n_customers=N_CUSTOMERS, seed=SEED)
    merged = customers.merge(truth_customers, on="customer_id", suffixes=("", "_true"))
    for col in ["country", "segment", "plan", "churned"]:
        agreement = (merged[col].astype(str) == merged[f"{col}_true"].astype(str)).mean()
        print(f"agreement with ground truth for '{col}': {agreement:.1%}")
    date_ok = (merged["signup_date"] == merged["signup_date_true"]).mean()
    print(f"agreement with ground truth for 'signup_date': {date_ok:.1%} (rest: missing/unparsable)")
    print(f"customers kept: {len(customers)} of {len(truth_customers)}; "
          f"transactions kept: {len(transactions)} of {len(truth_transactions)}")

    # 7. Save the clean data; parquet keeps dtypes (dates, categories), CSV would lose them
    try:
        customers.to_parquet(os.path.join(PATH_OUTPUTS, "customers_clean.parquet"), index=False)
        transactions.to_parquet(os.path.join(PATH_OUTPUTS, "transactions_clean.parquet"), index=False)
        print(f"\nclean data saved as parquet in {PATH_OUTPUTS}")
    except ImportError:
        customers.to_csv(os.path.join(PATH_OUTPUTS, "customers_clean.csv"), index=False)
        transactions.to_csv(os.path.join(PATH_OUTPUTS, "transactions_clean.csv"), index=False)
        print(f"\npyarrow not installed - clean data saved as CSV in {PATH_OUTPUTS}")
