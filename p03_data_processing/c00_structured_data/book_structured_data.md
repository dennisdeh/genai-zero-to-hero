## Structured Data
Structured (tabular) data is organised in rows and columns with a fixed meaning per column: customer
tables, transactions, sensor readings, financial statements. It is still the most common data type
in industry, and classical machine learning (part 5) works directly on it.

### The running example: an online shop
To have realistic problems without depending on downloads, the module
`c00_structured_data/utils/synthetic_data.py` generates the data of an imaginary online shop:

- **customers**: `customer_id`, `signup_date`, `age`, `country`, `segment` (Consumer, Small Business,
  Corporate), `plan` (Basic < Standard < Premium), `annual_income`, `support_tickets`, `email` and the
  label `churned`,
- **transactions**: `transaction_id`, `customer_id` (foreign key), `timestamp`, `amount`, `category`,
  `payment_method`.

The data is generated in two stages. `generate_clean_data` simulates a known data-generating process:
a latent *engagement* score per customer drives the number of purchases, the number of support
tickets and the churn probability (via a logistic model), so the churn label is learnable from the
data. `make_messy` then injects the problems seen in real exports:

| problem | example |
|---|---|
| messy column names | `"Signup Date "`, `"Annual Income (EUR)"`, `"Churned?"` |
| mixed date formats | `2023-04-03`, `03.04.2023`, `2023/04/03`, `April 03, 2023`, `not available` |
| numbers as text | `"52,000"`, `"EUR 52000"`, `"€45.90"` |
| inconsistent categories | `Denmark`, `denmark`, `DK`, `Danmark`, `" Denmark "` |
| mixed booleans | `yes`/`no`, `1`/`0`, `True`/`False` |
| invalid values | ages `-1`, `0`, `999`; incomes multiplied by 50; emails without `@` |
| missing values | 1-5 % per column |
| duplicates | exact copies and near duplicates (same customer, some fields blank) |
| orphaned keys | transactions referencing non-existent customers |
| outliers | refunds (negative amounts) and extreme amounts (x100) |

`generate_raw_data` finally writes the result to CSV in memory and reads it back as text, so the raw
data looks exactly like a freshly read export. Because we know the clean ground truth, we can measure
how well the cleaning recovers it - a luxury rarely available in practice.

The `drift` argument shifts the distributions (older and richer customers, more electronics
purchases); this is used to simulate a new data batch in the data quality section.

### Overview of the sections
- `s01_preprocessing`: cleaning with pandas (`main_1_cleaning.py`) and feature engineering with a
  scikit-learn `Pipeline`/`ColumnTransformer` that avoids data leakage (`main_2_feature_pipeline.py`).
- `s02_dqa`: data quality assurance - a pass/fail data quality report from reusable checks
  (`main_1_quality_report.py`), drift detection with PSI, Kolmogorov-Smirnov and chi-squared tests
  (`main_2_drift_detection.py`) and declarative schemas with pandera (`main_3_pandera_validation.py`).
