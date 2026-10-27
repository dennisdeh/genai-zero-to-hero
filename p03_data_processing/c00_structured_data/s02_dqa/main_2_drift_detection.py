"""
Drift detection: has the distribution of a new data batch changed compared to the reference data?

A model trained on last year's customers silently degrades if this year's customers look different.
This script compares a reference batch (think: the training data) with

- a new batch from the *same* process (control: nothing should be flagged as relevant drift), and
- a new batch generated with a distribution shift (older, richer customers, more electronics purchases).

For numeric columns we use the Population Stability Index (PSI) and the two-sample Kolmogorov-Smirnov
test, for categorical columns PSI and the chi-squared test (see utils/drift.py). The script also shows
why p-values alone are a poor alarm for large batches.

Run from the repository root:
    python -m p03_data_processing.c00_structured_data.s02_dqa.main_2_drift_detection
"""

import pandas as pd
from scipy.stats import ks_2samp

from p03_data_processing.c00_structured_data.s01_preprocessing.utils.cleaning import load_clean_data
from p03_data_processing.c00_structured_data.s01_preprocessing.utils.features import build_customer_features
from p03_data_processing.c00_structured_data.s02_dqa.utils.drift import drift_report, psi_label, psi_numeric

NUMERIC = ["age", "annual_income", "support_tickets", "n_transactions", "total_spend", "avg_amount"]
CATEGORICAL = ["country", "segment", "plan", "favourite_category"]


def load_batch(seed: int, drift: float, id_offset: int, n_customers: int = 2000) -> pd.DataFrame:
    """Generate, clean and aggregate one batch of customers."""
    customers, transactions = load_clean_data(n_customers=n_customers, seed=seed, drift=drift, id_offset=id_offset)
    return build_customer_features(customers, transactions)


def print_report(name: str, report: pd.DataFrame) -> None:
    """Pretty-print a drift report and list the columns that need attention."""
    print(f"\n=== {name} ===")
    cols = ["column", "type", "psi", "psi_verdict", "test", "statistic", "p_value", "ref_mean", "cur_mean"]
    print(report[cols].to_string(index=False, float_format=lambda x: f"{x:,.4g}"))
    flagged = report.loc[report["psi_verdict"] != "stable", "column"].tolist()
    print(f"columns with PSI >= 0.1: {flagged or 'none'}")


if __name__ == "__main__":
    # 1. Reference batch and two new batches (same process vs. shifted process)
    reference = load_batch(seed=1, drift=0.0, id_offset=0)
    same_process = load_batch(seed=2, drift=0.0, id_offset=100_000)
    shifted = load_batch(seed=3, drift=1.0, id_offset=200_000)
    print(f"batches: reference={len(reference)}, same process={len(same_process)}, shifted={len(shifted)} customers")

    # 2. Drift reports
    print_report("reference vs. new batch from the same process", drift_report(reference, same_process,
                                                                                NUMERIC, CATEGORICAL))
    print_report("reference vs. shifted batch", drift_report(reference, shifted, NUMERIC, CATEGORICAL))

    # 3. Statistical significance vs. effect size: a tiny shift (+1 year of age) in ever larger samples
    print("\n=== p-values vs. PSI for a tiny shift (age + 1 year) ===")
    ages = reference["age"].dropna()
    for n in [500, 2_000, 10_000]:
        # bootstrap samples of the reference ages; the "new" sample is shifted by one year
        ref = ages.sample(n, replace=True, random_state=0)
        cur = ages.sample(n, replace=True, random_state=1) + 1
        p = ks_2samp(ref, cur).pvalue
        psi = psi_numeric(ref, cur)
        print(f"n={n:>6}: KS p-value={p:.2e} (significant: {p < 0.01}), PSI={psi:.3f} ({psi_label(psi)})")
    print("-> with enough data every difference becomes 'significant'; monitor effect sizes with thresholds.")
