### Data Quality Assurance
Cleaning fixes the problems we know about; data quality assurance (DQA) makes sure we notice the
problems we do not know about yet - every time new data arrives. The idea is to write down the
expectations about the data as executable checks and to run them automatically before the data is
used, similar to unit tests for code. A failed check *blocks* the pipeline (error) or notifies
someone (warning).

#### Dimensions of data quality
| dimension | question | example check |
|---|---|---|
| schema / validity | right columns, types and formats? | `signup_date` is a datetime, `email` matches a regex |
| completeness | how much is missing? | at most 10 % missing incomes |
| uniqueness | one row per entity? | `customer_id` is unique |
| accuracy / plausibility | are the values possible? | age in [18, 100], amount in [-10 000, 10 000] |
| consistency / integrity | do tables fit together? | every transaction references an existing customer |
| timeliness | is the data current? | no timestamps in the future, last record not older than a day |
| stability | does new data look like old data? | distribution drift (PSI, KS test) |

#### A data quality report from reusable checks (`main_1_quality_report.py`)
`utils/checks.py` implements each check as a small function that returns a `CheckResult` with the
check name, column, pass/fail, severity, the observed value and details, for example

```python
dq.check_completeness(customers, "annual_income", max_missing=0.10, severity=dq.WARNING)
dq.check_range(customers, "age", 18, 100)
dq.check_foreign_key(transactions, "customer_id", customers, "customer_id")
```

A *suite* is simply a list of such calls (`customer_suite`, `transaction_suite`); `run_suite` turns the
results into a report table and `summarise` gives the verdict ("BLOCKED" if any error-level check
fails). The script also prints a column *profile* (dtype, missing share, distinct values, min/max,
most frequent value) - the first thing to look at for any new dataset.

The same suites run on the raw exports and on the cleaned data. On the raw customers 11 of 20
checks fail (duplicated IDs, text instead of dates and numbers, 24 spellings of five countries,
invalid emails, ...); on the clean data all checks pass. The report is saved as CSV (for machines,
e.g. to track quality over time) and as HTML (for humans) in `outputs/`.

#### Drift detection (`main_2_drift_detection.py`)
A model trained on last year's customers may silently degrade when the customer base changes
(*data drift* or *covariate shift*). `utils/drift.py` compares a reference batch (e.g. the training
data) with a new batch column by column:

- **Population Stability Index (PSI).** The reference values are split into (here 10 quantile) bins;
  with $e_i$ and $a_i$ the shares of the reference and the new data in bin $i$,
  $$\mathrm{PSI} = \sum_i (a_i - e_i) \ln(a_i / e_i)$$
  Categories are used directly as bins. Rule of thumb: below 0.1 stable, 0.1-0.25 moderate shift,
  above 0.25 significant shift.
- **Two-sample Kolmogorov-Smirnov test** (numeric): the statistic is the largest vertical distance
  between the two empirical cumulative distribution functions (`scipy.stats.ks_2samp`). The test
  assumes continuous data; for discrete columns with many ties (counts such as support tickets) its
  p-values are conservative, i.e. it flags drift less often than the nominal significance level.
- **Chi-squared test** (categorical): are the category frequencies independent of the batch
  (`scipy.stats.chi2_contingency`)?

A new batch from the same process shows PSI values below 0.03 everywhere, whereas the shifted batch
(`drift=1.0`) is flagged for age, income, spending, country and favourite category - but not for
support tickets or the number of purchases, which were not shifted.

The last part of the script illustrates an important practical point: with large samples, statistical
tests flag even irrelevant differences. Shifting all ages by one year is "significant" for 2 000 and
10 000 samples (p-values down to $10^{-12}$), while the PSI stays well below 0.1. For monitoring,
effect sizes with agreed thresholds are more useful than p-values.

#### Declarative validation with pandera (`main_3_pandera_validation.py`)
Libraries such as pandera and Great Expectations package the same ideas. With pandera the expected
table is a class - executable documentation of the data contract:

```python
class CustomerSchema(pa.DataFrameModel):
    customer_id: Series[str] = pa.Field(unique=True, str_matches=r"^C\d{6}$")
    age: Series[pd.Int64Dtype] = pa.Field(nullable=True, in_range={"min_value": 18, "max_value": 100})
    country: Series[pd.CategoricalDtype] = pa.Field(nullable=True, isin=COUNTRIES)
    ...

    @pa.dataframe_check
    @classmethod
    def income_mostly_present(cls, df) -> bool:
        return df["annual_income"].isna().mean() <= 0.10
```

`CustomerSchema.validate(df, lazy=True)` collects *all* failures into a table of failure cases
(column, check, offending value) instead of stopping at the first one; the script condenses it to one
row per column and check. The decorator `@pa.check_types` validates function arguments annotated as
`DataFrame[CustomerSchema]`, so business logic cannot be called with unvalidated data. Cross-column
rules (e.g. `is_refund` is true exactly for negative amounts) are written as `@pa.dataframe_check`.

Hand-written checks are transparent and dependency-free; a library offers more checks, better error
reports and integrations. In both cases the important part is the same: the expectations are explicit,
versioned with the code and run automatically.
