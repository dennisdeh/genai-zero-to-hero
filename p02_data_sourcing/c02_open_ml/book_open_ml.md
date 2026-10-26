## OpenML

OpenML (https://www.openml.org) is an open, collaborative platform for machine learning research. It hosts
thousands of datasets together with *tasks* and *runs*:

- a **dataset** is a table with rich, machine-readable metadata: feature names and types, nominal values,
  missing values, the default target, licence, citation, the original source and a list of computed
  **qualities** (number of instances and features, number of classes, class imbalance, ...),
- a **task** combines a dataset with a target and an evaluation procedure (e.g. 10-fold cross-validation
  with fixed splits), so that results obtained by different people are directly comparable,
- a **run** is the result of applying a model (a *flow*) to a task; runs are shared and can be compared.

Well-known benchmark suites such as OpenML-CC18 are defined as collections of OpenML tasks. For data sourcing
the main attraction is that datasets are curated, versioned, uniformly formatted and searchable by their
properties. Downloading requires no account; an API key is only needed to upload.

### Searching and downloading with `openml`
`main_1_openml_search.py` uses the official `openml` Python package:

1. **Cache.** `openml.config.set_root_cache_directory(...)` sets where downloaded files are cached; the second
   run is served from disk. A short retry policy (`set_retry_policy("human", n_retries=2)`) makes the script
   fail quickly when the server is unreachable.
2. **Search.** `openml.datasets.list_datasets(output_format="dataframe", number_instances="1000..10000",
   number_classes="2")` filters on the server (ranges are written `min..max`); the resulting DataFrame contains
   the qualities as columns, so the search is refined with pandas, e.g. no missing values and only the newest
   version of each dataset name. An imbalance ratio is computed from majority and minority class sizes.
3. **Download.** `openml.datasets.get_dataset(31, download_data=True, download_qualities=True,
   download_features_meta_data=True)` fetches the *credit-g* dataset (German credit risk) with all metadata.
   Use the numeric id rather than the name: names are not unique across versions.
4. **Inspect.** Description, licence, citation, original URL, default target, selected qualities and the
   feature table (index, name, data type, number of missing values and categories) are printed. Reading the
   metadata *before* modelling avoids many surprises - e.g. ID columns or "ignore" attributes, or targets
   that leak information.
5. **Extract.** `dataset.get_data(target=dataset.default_target_attribute)` returns `X`, `y`, a boolean
   categorical indicator per feature and the attribute names; nominal features arrive as pandas categoricals.
   The data is stored as Parquet.
6. **Tasks.** `openml.tasks.list_tasks(data_id=31, output_format="dataframe")` lists the standardised tasks
   defined on the dataset.

### Fetching with scikit-learn
If only the data is needed, `sklearn.datasets.fetch_openml` is enough (`main_2_fetch_openml_sklearn.py`):

```python
from sklearn.datasets import fetch_openml
bunch = fetch_openml(name="adult", version=2, as_frame=True, data_home="outputs/...", parser="auto")
bunch.frame        # DataFrame incl. target       bunch.data / bunch.target
bunch.details      # metadata: id, version, licence, default target, md5 checksum, ...
bunch.DESCR        # description                  bunch.url
```

Always pass an explicit `version` (or `data_id`): the default `"active"` picks the oldest active version, which
may change over time. Downloads are cached in `data_home` and verified with an MD5 checksum. The script then
checks data quality (missing values, duplicate rows, class balance) and trains a quick baseline: a
majority-class `DummyClassifier` versus a `HistGradientBoostingClassifier`, which handles missing values and
pandas categoricals natively. On *adult* the boosting model reaches roughly 87 % accuracy against 76 % for the
majority class - a sanity check that the fetched data is complete and correctly typed.

### When to use OpenML
- Benchmarking a method on many tabular datasets with comparable evaluation set-ups.
- Searching for datasets by properties (size, number of classes, missing values, feature types).
- Teaching and prototyping with well-documented, citable datasets.

Note that many classic datasets on OpenML are old and some have known issues (e.g. sensitive attributes in
*adult* and *credit-g* that raise fairness questions). The metadata and the original sources help to judge
whether a dataset is appropriate for a given purpose.
