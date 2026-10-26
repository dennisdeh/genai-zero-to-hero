"""
Reusable data quality checks for pandas data frames.

Every check returns a ``CheckResult`` (name, column, passed, severity, observed value, details), so
checks can be combined into a suite and reported uniformly. The checks cover the classic data
quality dimensions:

- schema/validity:       expected columns and dtypes, allowed values, regex patterns, value ranges
- completeness:          share of missing values per column
- uniqueness:            primary keys / combinations of columns
- consistency/integrity: foreign keys referencing an existing parent table, ordered dates
- timeliness:            dates not in the future / data not older than expected

Only pandas is needed; the same ideas are implemented by libraries such as pandera or Great Expectations.
"""

from dataclasses import asdict, dataclass

import pandas as pd

ERROR = "error"  # the data must not be used (block the pipeline)
WARNING = "warning"  # the data can be used, but somebody should have a look


@dataclass
class CheckResult:
    """Result of a single data quality check."""

    check: str
    column: str
    passed: bool
    severity: str
    observed: str
    details: str = ""


def _fmt_share(x: float) -> str:
    return f"{x:.2%}"


def check_columns_present(df: pd.DataFrame, expected: list[str], severity: str = ERROR) -> CheckResult:
    """All expected columns exist (extra columns are reported but allowed)."""
    missing = [c for c in expected if c not in df.columns]
    extra = [c for c in df.columns if c not in expected]
    return CheckResult("columns_present", "*", not missing, severity, f"{len(missing)} missing",
                       f"missing={missing} extra={extra}")


def check_dtype(df: pd.DataFrame, column: str, kind: str, severity: str = ERROR) -> CheckResult:
    """
    The column has the expected kind of dtype.

    :param kind: one of 'numeric', 'integer', 'datetime', 'bool', 'category', 'string'
    """
    tests = {
        "numeric": pd.api.types.is_numeric_dtype,
        "integer": pd.api.types.is_integer_dtype,
        "datetime": pd.api.types.is_datetime64_any_dtype,
        "bool": pd.api.types.is_bool_dtype,
        "category": lambda s: isinstance(s.dtype, pd.CategoricalDtype),
        "string": lambda s: pd.api.types.is_string_dtype(s) or pd.api.types.is_object_dtype(s),
    }
    if column not in df.columns:
        return CheckResult("dtype", column, False, severity, "column missing", f"expected {kind}")
    ok = bool(tests[kind](df[column]))
    return CheckResult("dtype", column, ok, severity, str(df[column].dtype), f"expected {kind}")


def check_completeness(df: pd.DataFrame, column: str, max_missing: float = 0.0,
                       severity: str = ERROR) -> CheckResult:
    """The share of missing values is at most ``max_missing``."""
    share = float(df[column].isna().mean())
    return CheckResult("completeness", column, share <= max_missing, severity, _fmt_share(share),
                       f"max allowed {_fmt_share(max_missing)}")


def check_unique(df: pd.DataFrame, columns: str | list[str], severity: str = ERROR) -> CheckResult:
    """No duplicated values (or value combinations) in the given column(s)."""
    cols = [columns] if isinstance(columns, str) else columns
    n_dup = int(df.duplicated(subset=cols).sum())
    label = "+".join(cols) if len(cols) <= 3 else f"<{len(cols)} columns>"
    return CheckResult("unique", label, n_dup == 0, severity, f"{n_dup} duplicates")


def check_range(df: pd.DataFrame, column: str, min_value=None, max_value=None, max_violations: float = 0.0,
                severity: str = ERROR) -> CheckResult:
    """Non-missing values lie in [min_value, max_value]; values that cannot be compared count as violations."""
    values = df[column].dropna()
    if not pd.api.types.is_numeric_dtype(values) and not pd.api.types.is_datetime64_any_dtype(values):
        values = pd.to_numeric(values, errors="coerce")
    ok = pd.Series(True, index=values.index)
    if min_value is not None:
        ok &= values >= min_value
    if max_value is not None:
        ok &= values <= max_value
    share = float((~ok.fillna(False).astype(bool)).mean()) if len(values) else 0.0
    return CheckResult("range", column, share <= max_violations, severity, f"{_fmt_share(share)} outside",
                       f"allowed [{min_value}, {max_value}]")


def check_allowed_values(df: pd.DataFrame, column: str, allowed: list, severity: str = ERROR) -> CheckResult:
    """Every non-missing value is one of the allowed values."""
    values = df[column].dropna().astype(str)
    invalid = values[~values.isin([str(a) for a in allowed])]
    examples = list(invalid.value_counts().index[:5])
    return CheckResult("allowed_values", column, invalid.empty, severity, f"{len(invalid)} invalid",
                       f"examples={examples}")


def check_regex(df: pd.DataFrame, column: str, pattern: str, max_violations: float = 0.0,
                severity: str = WARNING) -> CheckResult:
    """Non-missing values match a regular expression (e.g. emails, ids)."""
    values = df[column].dropna().astype(str)
    share = float((~values.str.match(pattern)).mean()) if len(values) else 0.0
    return CheckResult("regex", column, share <= max_violations, severity, f"{_fmt_share(share)} not matching",
                       f"pattern={pattern}")


def check_foreign_key(child: pd.DataFrame, fk: str, parent: pd.DataFrame, pk: str,
                      severity: str = ERROR) -> CheckResult:
    """Referential integrity: every foreign key value exists as a primary key in the parent table."""
    orphans = ~child[fk].isin(parent[pk])
    return CheckResult("foreign_key", fk, not orphans.any(), severity, f"{int(orphans.sum())} orphans",
                       f"references {pk} of parent table")


def check_not_in_future(df: pd.DataFrame, column: str, now: pd.Timestamp | None = None,
                        severity: str = ERROR) -> CheckResult:
    """Timestamps must not lie in the future (a classic symptom of time zone or format errors)."""
    now = now or pd.Timestamp.now()
    values = df[column]
    if not pd.api.types.is_datetime64_any_dtype(values):
        values = pd.to_datetime(values, errors="coerce", format="mixed")
    n_future = int((values > now).sum())
    return CheckResult("not_in_future", column, n_future == 0, severity, f"{n_future} in future")


def check_row_count(df: pd.DataFrame, min_rows: int, max_rows: int | None = None,
                    severity: str = ERROR) -> CheckResult:
    """The number of rows is plausible (e.g. an empty or truncated export is caught early)."""
    n = len(df)
    ok = n >= min_rows and (max_rows is None or n <= max_rows)
    return CheckResult("row_count", "*", ok, severity, str(n), f"expected [{min_rows}, {max_rows}]")


def run_suite(results: list[CheckResult]) -> pd.DataFrame:
    """Collect check results into a report data frame."""
    report = pd.DataFrame([asdict(r) for r in results])
    report["status"] = report["passed"].map({True: "PASS", False: "FAIL"})
    return report[["status", "severity", "check", "column", "observed", "details"]]


def summarise(report: pd.DataFrame) -> str:
    """One-line summary; the data is 'blocked' if any error-level check fails."""
    failed = report[report["status"] == "FAIL"]
    n_err = int((failed["severity"] == ERROR).sum())
    n_warn = int((failed["severity"] == WARNING).sum())
    verdict = "BLOCKED" if n_err else "OK"
    return f"{verdict}: {len(report) - len(failed)}/{len(report)} checks passed, {n_err} errors, {n_warn} warnings"


def profile(df: pd.DataFrame) -> pd.DataFrame:
    """Column profile: dtype, missing share, number of distinct values, min/max and the most frequent value."""
    rows = []
    for col in df.columns:
        s = df[col]
        comparable = pd.api.types.is_numeric_dtype(s) or pd.api.types.is_datetime64_any_dtype(s)
        rows.append({
            "column": col,
            "dtype": str(s.dtype),
            "missing_%": round(100 * s.isna().mean(), 2),
            "n_unique": s.nunique(),
            "min": s.min() if comparable and s.notna().any() else "",
            "max": s.max() if comparable and s.notna().any() else "",
            "top": s.mode().iloc[0] if s.notna().any() else "",
        })
    return pd.DataFrame(rows)
