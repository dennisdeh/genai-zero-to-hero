"""
Detecting distribution drift between a reference batch and a new batch of data.

- Population Stability Index (PSI): binned comparison of two distributions, popular in credit risk.
  PSI = sum_i (a_i - e_i) * ln(a_i / e_i), where e_i/a_i are the shares of the reference/new data in bin i.
  Rule of thumb: < 0.1 stable, 0.1-0.25 moderate shift, > 0.25 significant shift.
- Kolmogorov-Smirnov test (numeric): maximum distance between the two empirical CDFs.
- Chi-squared test (categorical): are the category frequencies independent of the batch?

Note that with large samples statistical tests flag even tiny, irrelevant differences as "significant";
effect sizes such as the PSI or the KS statistic itself are often more useful for monitoring.
"""

import numpy as np
import pandas as pd
from scipy import stats

PSI_THRESHOLDS = (0.1, 0.25)


def psi_numeric(reference: pd.Series, current: pd.Series, n_bins: int = 10, eps: float = 1e-4) -> float:
    """PSI of a numeric feature, with quantile bins defined on the reference data."""
    ref, cur = reference.dropna().to_numpy(dtype=float), current.dropna().to_numpy(dtype=float)
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, n_bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.histogram(ref, edges)[0] / len(ref)
    a = np.histogram(cur, edges)[0] / len(cur)
    e, a = np.clip(e, eps, None), np.clip(a, eps, None)
    return float(np.sum((a - e) * np.log(a / e)))


def psi_categorical(reference: pd.Series, current: pd.Series, eps: float = 1e-4) -> float:
    """PSI of a categorical feature (each category is a bin; missing is its own category)."""
    ref = reference.astype(object).fillna("<missing>").value_counts(normalize=True)
    cur = current.astype(object).fillna("<missing>").value_counts(normalize=True)
    cats = ref.index.union(cur.index)
    e = ref.reindex(cats, fill_value=0).clip(lower=eps)
    a = cur.reindex(cats, fill_value=0).clip(lower=eps)
    return float(np.sum((a - e) * np.log(a / e)))


def psi_label(psi: float) -> str:
    """Interpret a PSI value with the usual rule of thumb."""
    if psi < PSI_THRESHOLDS[0]:
        return "stable"
    return "moderate" if psi < PSI_THRESHOLDS[1] else "significant"


def drift_report(reference: pd.DataFrame, current: pd.DataFrame, numeric: list[str], categorical: list[str],
                 alpha: float = 0.01) -> pd.DataFrame:
    """
    Compare two batches column by column.

    :param reference: reference batch (e.g. the training data)
    :param current: new batch
    :param numeric: numeric columns (PSI + two-sample KS test)
    :param categorical: categorical columns (PSI + chi-squared test)
    :param alpha: significance level of the tests
    :return: one row per column with PSI, test statistic, p-value and verdicts
    """
    rows = []
    for col in numeric:
        ks = stats.ks_2samp(reference[col].dropna(), current[col].dropna())
        rows.append({"column": col, "type": "numeric", "psi": psi_numeric(reference[col], current[col]),
                     "test": "KS", "statistic": ks.statistic, "p_value": ks.pvalue,
                     "ref_mean": reference[col].mean(), "cur_mean": current[col].mean()})
    for col in categorical:
        ref = reference[col].astype(object).fillna("<missing>")
        cur = current[col].astype(object).fillna("<missing>")
        table = pd.crosstab(np.r_[np.zeros(len(ref)), np.ones(len(cur))], np.r_[ref.to_numpy(), cur.to_numpy()])
        chi2 = stats.chi2_contingency(table)
        rows.append({"column": col, "type": "categorical", "psi": psi_categorical(reference[col], current[col]),
                     "test": "chi2", "statistic": chi2.statistic, "p_value": chi2.pvalue,
                     "ref_mean": np.nan, "cur_mean": np.nan})
    report = pd.DataFrame(rows)
    report["psi_verdict"] = report["psi"].map(psi_label)
    report["test_significant"] = report["p_value"] < alpha
    return report
