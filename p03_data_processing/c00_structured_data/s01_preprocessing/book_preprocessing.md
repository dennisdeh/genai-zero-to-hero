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
(e.g. 24 different spellings of five countries, plus missing values).

**Column names** are converted to `snake_case` (`"Annual Income (EUR)"` → `annual_income_eur`), so that
they are valid Python identifiers and consistent across sources.

**Duplicates.** Exact duplicates are dropped with `drop_duplicates()`. Duplicates *of an entity*
(several rows with the same `customer_id`) need a rule; here we keep the most complete record. Other
common rules are "latest update wins" (sort by a timestamp) or aggregating the rows.

**Dates.** `parse_mixed_dates` tries an explicit list of formats one after the other. Being explicit
matters: `03.04.2023` is the 3rd of April in Europe but would be read as the 4th of March by a US-centric
parser. Unparsable entries (`"not available"`) become `NaT` and are counted.

**Numbers stored as text** are parsed with a *whitelist*: a regular expression describes the formats
we expect - an optional sign, an optional currency symbol or code (`€`, `EUR`), digits with optional
comma thousands separators and one decimal point, or accounting parentheses (`(12.50)` means $-12.50$).
Everything else becomes `NaN` instead of being guessed. The tempting shortcut of deleting every
character except digits, `.` and `-` silently misreads values: `1e5` becomes 15, `1.234,56` becomes
1.23456, and `(12.50)` or `−12.50` (with the Unicode minus sign) lose their sign - which matters here,
because a refund is defined as a negative amount. Locales that use the comma as decimal separator
(`45,90`) need a separate, explicit rule.

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
distinct categories, favourite category, share of refunds and tenure. These aggregations are
*row-wise* - each customer only uses its own transactions - so computing them before the train/test
split leaks nothing *between rows*. (The days since the last purchase are computed as well, but are
deliberately not a model feature - see the second leakage lesson below.)

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
ordinal = Pipeline([("ordinal", OrdinalEncoder(categories=[["Basic", "Standard", "Premium"]],
                                               handle_unknown="use_encoded_value", unknown_value=np.nan)),
                    ("impute", SimpleImputer(strategy="most_frequent"))])
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
  country) as all zeros instead of failing at prediction time. The `OrdinalEncoder` would raise an
  error for an unknown plan by default; with `handle_unknown="use_encoded_value", unknown_value=np.nan`
  it encodes it as missing, and the following imputer fills it in.
- A stateless `FunctionTransformer` adds the ratio feature `spend_per_month` as the first pipeline
  step, so the same feature engineering is applied at inference time.

The data is split with `train_test_split(..., stratify=y)` **before** anything is fitted. The pipeline
(preprocessing + logistic regression) is evaluated with 5-fold cross-validation on the training set and
once on the test set (ROC AUC about 0.77; 0.75 in cross-validation). That is a realistic value: the
simulated churn depends on a latent "engagement" that the data only shows indirectly, through the
number of purchases and support tickets - which are, accordingly, the largest coefficients. The
pipeline is saved with `joblib` and can score a raw record directly, including missing values and
unseen categories (a new country, or a plan `"Enterprise"` that did not exist in the training data).

**Leakage lesson 1: preprocessing must live inside the pipeline.** `demonstrate_leakage()` creates 200 samples
with 2000 pure-noise features and *random* labels. Selecting the 20 "best" features on the full data
and then cross-validating gives an accuracy of about 0.77, although there is nothing to learn. With
the selection inside the pipeline it is refitted on every training fold, and the honest accuracy is
about 0.5. The same mechanism - more subtly - applies to imputation, scaling and target encoding.

**Leakage lesson 2: target leakage through post-outcome features.** A first version of this example
also used the number of days since the last purchase - and reached a test ROC AUC of 0.94, with that
feature as by far the largest coefficient. The pipeline was perfectly leakage-free in the sense of
lesson 1, yet the result was worthless: in the data, churned customers *stop buying*, so a long gap
since the last purchase is a *consequence* of churn, measured after it happened. On its own, the
recency feature relative to the tenure ranks churners almost perfectly (AUC 0.97).
`demonstrate_target_leakage()` reproduces the comparison: 0.94 with the feature, 0.77 without. A model
that needs to know whether a customer has already left cannot warn us *before* they leave.

The rule: every feature may only use information that is available *at the moment of prediction*.
The proper fix in a real project is a *cut-off date*: features are computed only from data before the
cut-off, and the label (did the customer churn in, say, the following 90 days?) only from data after
it. Suspiciously good results, and a single feature that dominates all others, are the typical
symptoms of target leakage - investigate them before celebrating.
