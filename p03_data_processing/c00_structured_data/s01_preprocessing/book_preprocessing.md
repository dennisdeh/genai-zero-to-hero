### Preprocessing: Cleaning and Feature Pipelines
Preprocessing has two quite different parts:

1. **Cleaning** fixes the representation of the data so that it means what it says: correct types,
   one spelling per category, one row per entity, impossible values removed. Cleaning rules come from
   domain knowledge and are applied identically to all data, including future data.
2. **Feature preprocessing** transforms clean data into what a model needs: imputing missing values,
   scaling, encoding categories, creating new features. These steps *learn statistics from data*
   (a median, a standard deviation, the list of categories) and must therefore be fitted on the
   training data only.

#### Cleaning with pandas (`main_1_cleaning.py`)
The reusable functions are in `utils/cleaning.py`; the script applies them step by step and prints
what changes.

**Read raw data as text.** `pd.read_csv(path, dtype=str)` prevents pandas from guessing types:
a column with `"52,000"` and `"EUR 52000"` would otherwise silently become `object`, and an ID such as
`00123` would lose its leading zeros. Types are then set explicitly.

**Inspect first.** For each column: dtype, number and share of missing values, number of distinct
values, and the distinct spellings of categorical columns. This overview already reveals most problems
(e.g. 25 spellings of five countries).

**Column names** are converted to `snake_case` (`"Annual Income (EUR)"` → `annual_income_eur`), so that
they are valid Python identifiers and consistent across sources.

**Duplicates.** Exact duplicates are dropped with `drop_duplicates()`. Duplicates *of an entity*
(several rows with the same `customer_id`) need a rule; here we keep the most complete record. Other
common rules are "latest update wins" (sort by a timestamp) or aggregating the rows.

**Dates.** `parse_mixed_dates` tries an explicit list of formats one after the other. Being explicit
matters: `03.04.2023` is the 3rd of April in Europe but would be read as the 4th of March by a US-centric
parser. Unparsable entries (`"not available"`) become `NaT` and are counted.

**Numbers stored as text** are cleaned with a regular expression that removes everything except
digits, the decimal point and the minus sign, then converted with `pd.to_numeric(errors="coerce")`.
Watch out for locales that use the comma as decimal separator (`45,90`) - such data needs a separate
rule.

**Categories.** All variants are normalised (strip, lower-case, collapse separators) and mapped to a
canonical value through a lookup table; unknown spellings become missing instead of creating new
categories. The result is stored as a pandas `Categorical`, which documents the allowed values and
saves memory. Ordered categories (`Basic < Standard < Premium`) are declared as ordered.

**Domain rules.** Values that are impossible rather than unusual are set to missing: ages outside
18-100 (sentinels such as `-1` or `999` are typical), incomes above 1 million EUR (data-entry errors in
this dataset), emails that do not match a pattern. They will be imputed later, inside the model
pipeline.

**Referential integrity.** Transactions whose `customer_id` does not exist in the customer table are
removed (and counted - in a real pipeline they would be reported to the data owner).

**Outliers.** An outlier is not necessarily an error. The script uses Tukey's rule: values outside
$[Q_1 - 1.5\,\mathrm{IQR},\; Q_3 + 1.5\,\mathrm{IQR}]$ with $\mathrm{IQR} = Q_3 - Q_1$ are flagged.
For right-skewed quantities such as amounts and incomes the rule flags far too many points (7.8 % of
the purchases); on the log scale it flags 0.6 %, which is more sensible. A common treatment is
*winsorising*: clipping values at, say, the 1st and 99th percentiles. This keeps the rows but limits
the influence of extreme values on means, variances and linear models; the median is unaffected
anyway. Refunds (negative amounts) are legitimate and are flagged with `is_refund` instead of removed.

Finally the cleaned tables are compared with the known ground truth and saved as parquet, which -
unlike CSV - preserves the dtypes (dates, categories, nullable integers).

#### Feature engineering and leakage-free pipelines (`main_2_feature_pipeline.py`)
The goal is to predict churn per customer. `utils/features.py` first builds a customer-level table by
aggregating each customer's transactions: number of purchases, total and average spend, number of
distinct categories, favourite category, share of refunds, days since the last purchase and tenure.
These aggregations are *row-wise* - each customer only uses its own transactions - so computing them
before the train/test split leaks nothing.

Everything that learns from the data across rows goes into a scikit-learn `Pipeline`:

```python
numeric = Pipeline([("clip", QuantileClipper(0.01, 0.99)),
                    ("impute", SimpleImputer(strategy="median", add_indicator=True)),
                    ("scale", StandardScaler())])
skewed = Pipeline([("impute", SimpleImputer(strategy="median")),
                   ("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
                   ("scale", StandardScaler())])
nominal = Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
ordinal = Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                    ("ordinal", OrdinalEncoder(categories=[["Basic", "Standard", "Premium"]]))])
columns = ColumnTransformer([("num", numeric, NUMERIC_FEATURES), ("skewed", skewed, SKEWED_FEATURES),
                             ("nominal", nominal, NOMINAL_FEATURES), ("ordinal", ordinal, ["plan"])])
```

- The `ColumnTransformer` applies a different sub-pipeline to each group of columns and concatenates
  the results; `get_feature_names_out()` returns readable names such as `nominal__country_Denmark`.
- `QuantileClipper` (in `utils/cleaning.py`) is a small custom transformer: it learns the clipping
  thresholds in `fit` and applies them in `transform` - winsorising without leakage.
- `SimpleImputer(add_indicator=True)` adds a binary "was missing" column; missingness is often
  informative in itself.
- Money amounts are log-transformed before scaling because they are right-skewed.
- `OneHotEncoder(handle_unknown="ignore")` encodes a category never seen in training (e.g. a new
  country) as all zeros instead of failing at prediction time.
- A stateless `FunctionTransformer` adds the ratio feature `spend_per_month` as the first pipeline
  step, so the same feature engineering is applied at inference time.

The data is split with `train_test_split(..., stratify=y)` **before** anything is fitted. The pipeline
(preprocessing + logistic regression) is evaluated with 5-fold cross-validation on the training set and
once on the test set (ROC AUC about 0.94). The pipeline is saved with `joblib` and can score a raw record
directly, including missing values and unseen categories.

**Why the pipeline matters - a leakage demonstration.** `demonstrate_leakage()` creates 200 samples
with 2000 pure-noise features and *random* labels. Selecting the 20 "best" features on the full data
and then cross-validating gives an accuracy of about 0.77, although there is nothing to learn. With
the selection inside the pipeline it is refitted on every training fold, and the honest accuracy is
about 0.5. The same mechanism - more subtly - applies to imputation, scaling and target encoding.

A caveat about this example: the strongest feature is the number of days since the last purchase,
because in the simulation churned customers stop buying. In a real project features must be computed
from data available *before* the moment of prediction (a time-based cut-off), otherwise the label
leaks into the features through time.
