"""
Data quality report: profile a dataset and validate it against explicit expectations.

The expectations (a small "data contract") for the customers and transactions tables are written as
reusable checks (see utils/checks.py). The same suite is run on

- the raw exports (only the column names are standardised), where many checks fail, and
- the cleaned data from the preprocessing section, where all error-level checks should pass.

The result is a pass/fail report per check plus a column profile, printed and saved as HTML/CSV in
outputs/. In a data pipeline the verdict decides whether the data may be used downstream
("BLOCKED" if any error-level check fails).

Run from the repository root:
    python -m p03_data_processing.c00_structured_data.s02_dqa.main_1_quality_report
"""

import os

import pandas as pd

from p03_data_processing.c00_structured_data.s01_preprocessing.utils.cleaning import (
    EMAIL_PATTERN,
    clean_customers,
    clean_transactions,
    standardise_column_names,
)
from p03_data_processing.c00_structured_data.s02_dqa.utils import checks as dq
from p03_data_processing.c00_structured_data.utils.synthetic_data import (
    CATEGORIES,
    COUNTRIES,
    PAYMENT_METHODS,
    PLANS,
    SEGMENTS,
    generate_raw_data,
)

PATH_OUTPUTS = os.path.join(os.path.dirname(__file__), "outputs")
OBSERVATION_END = pd.Timestamp("2024-12-31 23:59:59")


def customer_suite(customers: pd.DataFrame) -> list[dq.CheckResult]:
    """Expectations for the customers table."""
    expected = ["customer_id", "signup_date", "age", "country", "segment", "plan", "annual_income",
                "support_tickets", "email", "churned"]
    return [
        dq.check_row_count(customers, min_rows=1000),
        dq.check_columns_present(customers, expected),
        dq.check_unique(customers, "customer_id"),
        dq.check_dtype(customers, "signup_date", "datetime"),
        dq.check_dtype(customers, "age", "numeric"),
        dq.check_dtype(customers, "annual_income", "numeric"),
        dq.check_dtype(customers, "churned", "bool"),
        dq.check_completeness(customers, "customer_id", 0.0),
        dq.check_completeness(customers, "churned", 0.0),
        dq.check_completeness(customers, "age", 0.10, severity=dq.WARNING),
        dq.check_completeness(customers, "annual_income", 0.10, severity=dq.WARNING),
        dq.check_range(customers, "age", 18, 100),
        dq.check_range(customers, "annual_income", 0, 1_000_000),
        dq.check_range(customers, "support_tickets", 0, 50),
        dq.check_allowed_values(customers, "country", COUNTRIES),
        dq.check_allowed_values(customers, "segment", SEGMENTS),
        dq.check_allowed_values(customers, "plan", PLANS),
        dq.check_regex(customers, "email", EMAIL_PATTERN),
        dq.check_regex(customers, "customer_id", r"^C\d{6}$"),
        dq.check_not_in_future(customers, "signup_date", now=OBSERVATION_END),
    ]


def transaction_suite(transactions: pd.DataFrame, customers: pd.DataFrame) -> list[dq.CheckResult]:
    """Expectations for the transactions table (incl. referential integrity with the customers)."""
    return [
        dq.check_row_count(transactions, min_rows=5000),
        dq.check_unique(transactions, "transaction_id"),
        dq.check_unique(transactions, list(transactions.columns), severity=dq.WARNING),
        dq.check_dtype(transactions, "timestamp", "datetime"),
        dq.check_dtype(transactions, "amount", "numeric"),
        dq.check_completeness(transactions, "amount", 0.0),
        dq.check_completeness(transactions, "payment_method", 0.10, severity=dq.WARNING),
        dq.check_range(transactions, "amount", -10_000, 10_000, max_violations=0.01),
        dq.check_allowed_values(transactions, "category", CATEGORIES),
        dq.check_allowed_values(transactions, "payment_method", PAYMENT_METHODS),
        dq.check_foreign_key(transactions, "customer_id", customers, "customer_id"),
        dq.check_not_in_future(transactions, "timestamp", now=OBSERVATION_END),
    ]


def run_and_print(name: str, results: list[dq.CheckResult]) -> pd.DataFrame:
    """Run a suite, print the report and its verdict."""
    report = dq.run_suite(results)
    print(f"\n=== {name} ===")
    print(report.to_string(index=False, max_colwidth=60))
    print(dq.summarise(report))
    return report


if __name__ == "__main__":
    # 1. Raw data (as exported) and clean data (output of the preprocessing section)
    customers_raw, transactions_raw = generate_raw_data(n_customers=2000, seed=42)
    customers_raw = standardise_column_names(customers_raw).rename(
        columns={"annual_income_eur": "annual_income", "e_mail": "email"})
    transactions_raw = standardise_column_names(transactions_raw)
    customers = clean_customers(customers_raw)
    transactions = clean_transactions(transactions_raw, valid_customer_ids=customers["customer_id"])

    # 2. Column profiles - the first thing to look at for any new dataset
    print("profile of the clean customers table:")
    print(dq.profile(customers).to_string(index=False))

    # 3. Validate raw and clean data against the same expectations
    reports = {
        "raw customers": run_and_print("raw customers", customer_suite(customers_raw)),
        "raw transactions": run_and_print("raw transactions", transaction_suite(transactions_raw, customers_raw)),
        "clean customers": run_and_print("clean customers", customer_suite(customers)),
        "clean transactions": run_and_print("clean transactions", transaction_suite(transactions, customers)),
    }

    # 4. Save the report (CSV for machines, HTML for humans)
    os.makedirs(PATH_OUTPUTS, exist_ok=True)
    combined = pd.concat({k: v for k, v in reports.items()}, names=["table"]).reset_index(level=0)
    combined.to_csv(os.path.join(PATH_OUTPUTS, "dq_report.csv"), index=False)
    html_parts = ["<html><head><meta charset='utf-8'><title>Data quality report</title></head><body>",
                  "<h1>Data quality report</h1>"]
    for name, report in reports.items():
        html_parts += [f"<h2>{name}</h2><p>{dq.summarise(report)}</p>", report.to_html(index=False)]
    html_parts += ["<h2>Profile: clean customers</h2>", dq.profile(customers).to_html(index=False), "</body></html>"]
    with open(os.path.join(PATH_OUTPUTS, "dq_report.html"), "w", encoding="utf-8") as f:
        f.write("\n".join(html_parts))
    print(f"\nreports saved to {PATH_OUTPUTS} (dq_report.csv, dq_report.html)")
